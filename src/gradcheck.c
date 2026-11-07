#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#include "nn.h"
#include "dataset.h"

// ---------- Gradient check settings ----------
#define CHECK_SAMPLES       3           // number of images used (labels 1, 5, 9)
#define PARAMS_PER_GROUP    40          // random parameters checked per group and image
#define STEP                5e-4f       // h of the central difference (float: too small = rounding noise,
                                        //   too large = ReLU kinks distort the numeric value)
#define PASS_THRESHOLD      3e-2        // relative error below this -> OK (float precision)

#define MAX_GROUPS          16

#define CHECK_LEGACY        0           // 1: also run the original (buggy) backprop
#define L2_CHECK_LAMBDA     1.0f        // lambda used for the L2 check
#define DROPOUT_CHECK_P     0.5f        // dropout probability used for the dropout check

typedef struct
{
    char name[24];

    double diff2;       // sum (analytic - numeric)^2
    double a2;          // sum analytic^2
    double n2;          // sum numeric^2

    size_t total;
    size_t analytic_zero_only;  // analytic == 0 although numeric != 0
} GroupStat;

// random number in [0, n) that also works when RAND_MAX is only 32767 (MSVC)
static size_t rand_below(size_t n)
{
    size_t r = ((size_t)rand() << 15) | (size_t)rand();

    return r % n;
}

// (lambda / 2) * sum of squares of all weights, summed in double.
// With a large lambda this penalty is ~100; stored in a float loss its rounding error
// (~1e-5) would swamp the small loss differences the central difference needs.
static double l2_penalty_double(NN nn, ConvLayer conv)
{
    if (NN_l2_lambda == 0.0f)
        return 0.0;

    double sum = 0.0;

    for (size_t l = 0; l < nn.count; l++)
        for (size_t j = 0; j < nn.ws[l].rows * nn.ws[l].cols; j++)
            sum += (double)nn.ws[l].es[j] * (double)nn.ws[l].es[j];

    for (size_t k = 0; k < conv.out_channels * conv.in_channels; k++)
        for (size_t j = 0; j < conv.kernels[k].rows * conv.kernels[k].cols; j++)
            sum += (double)conv.kernels[k].es[j] * (double)conv.kernels[k].es[j];

    return 0.5 * (double)NN_l2_lambda * sum;
}

// forward pass + total loss (cross-entropy + L2 penalty), returned in double
static double eval_loss(NN nn, ConvLayer conv, Mat image, Mat label)
{
    Mat* conv_out = (Mat*)malloc(sizeof(Mat) * conv.out_channels);
    Mat* pooled = (Mat*)malloc(sizeof(Mat) * conv.out_channels);

    if (!conv_out || !pooled)
    {
        free(conv_out);
        free(pooled);

        return -1.0;
    }

    Mat flat;
    CNN_forward(nn, conv, image, conv_out, pooled, &flat);

    float loss = 0.0f;

    for (size_t j = 0; j < NN_OUTPUT(nn).cols; j++)
        loss -= MAT_AT(label, 0, j) * logf(fmaxf(MAT_AT(NN_OUTPUT(nn), 0, j), 1e-7f));

    for (size_t c = 0; c < conv.out_channels; c++)
    {
        Mat_free(conv_out[c]);
        Mat_free(pooled[c]);
    }

    Mat_free(flat);

    free(conv_out);
    free(pooled);

    return (double)loss + l2_penalty_double(nn, conv);   // total loss = cross-entropy + L2 penalty
}

static void record(GroupStat* s, float analytic, float numeric)
{
    double a = (double)analytic;
    double n = (double)numeric;
    double d = a - n;

    s->diff2 += d * d;
    s->a2 += a * a;
    s->n2 += n * n;
    s->total++;

    if (fabs(n) > 1e-6 && fabs(a) < 1e-12)
        s->analytic_zero_only++;
}

static void run_check(
    int legacy, Dataset all,
    NN nn, ConvLayer conv, NN grad_fc, Mat* kernel_grads, float* bias_grads)
{
    NN_legacy_mode = legacy;

    size_t group_count = 2 + 2 * nn.count;

    if (group_count > MAX_GROUPS)
        return;

    GroupStat stats[MAX_GROUPS];
    memset(stats, 0, sizeof(stats));

    snprintf(stats[0].name, sizeof(stats[0].name), "conv kernels");
    snprintf(stats[1].name, sizeof(stats[1].name), "conv biases");

    for (size_t l = 0; l < nn.count; l++)
    {
        snprintf(stats[2 + 2 * l].name, sizeof(stats[0].name), "FC W%zu", l);
        snprintf(stats[3 + 2 * l].name, sizeof(stats[0].name), "FC b%zu", l);
    }

    size_t kernel_count = conv.out_channels * conv.in_channels;
    size_t kernel_elems = conv.kernel_size * conv.kernel_size;

    srand(2024);
    NN_dropout_seed(2024u);

    Mat image = Mat_alloc(DS_IMG_LEN, DS_IMG_LEN);

    for (int s = 0; s < CHECK_SAMPLES; s++)
    {
        size_t sample = (size_t)s * 4 + 1;

        if (sample >= all.count)
            break;

        for (size_t y = 0; y < DS_IMG_LEN; y++)
            for (size_t x = 0; x < DS_IMG_LEN; x++)
                MAT_AT(image, y, x) = MAT_AT(all.inputs, sample, y * DS_IMG_LEN + x);

        Mat label = Mat_row(all.labels, sample);

        // dropout: ONE fixed mask per image, used for the analytic AND the numeric gradient
        if (NN_dropout_p > 0.0f)
        {
            NN_dropout_sample(nn);
            NN_dropout_apply = 1;
        }

        // analytic gradients
        CNN_compute_grads(nn, conv, image, label, grad_fc, kernel_grads, bias_grads);

        for (size_t g = 0; g < group_count; g++)
        {
            for (int k = 0; k < PARAMS_PER_GROUP; k++)
            {
                float* p = NULL;
                float analytic = 0.0f;

                if (g == 0)
                {
                    size_t m = rand_below(kernel_count);
                    size_t e = rand_below(kernel_elems);

                    p = &conv.kernels[m].es[e];
                    analytic = kernel_grads[m].es[e];
                }
                else if (g == 1)
                {
                    size_t m = rand_below(conv.out_channels);

                    p = &conv.biases[m].es[0];
                    analytic = bias_grads[m];
                }
                else
                {
                    size_t l = (g - 2) / 2;
                    int is_bias = (int)((g - 2) % 2);

                    Mat* P = is_bias ? &nn.bs[l] : &nn.ws[l];
                    Mat* G = is_bias ? &grad_fc.bs[l] : &grad_fc.ws[l];

                    size_t e = rand_below(P->rows * P->cols);

                    p = &P->es[e];
                    analytic = G->es[e];
                }

                // numerical gradient (central difference, evaluated in double)
                float original = *p;

                float plus = original + STEP;
                float minus = original - STEP;

                *p = plus;
                double loss_plus = eval_loss(nn, conv, image, label);

                *p = minus;
                double loss_minus = eval_loss(nn, conv, image, label);

                *p = original;

                // divide by the step that was actually applied (float rounding of original +- STEP)
                double numeric = (loss_plus - loss_minus) / ((double)plus - (double)minus);

                record(&stats[g], analytic, (float)numeric);
            }
        }

        NN_dropout_apply = 0;
    }

    Mat_free(image);

    printf("\n=== %s | L2 lambda = %g | dropout p = %g ===\n", legacy ? "ORIGINAL code (legacy)" : "FIXED code", (double)NN_l2_lambda, (double)NN_dropout_p);
    printf("%-13s %12s %12s %11s  %s\n", "group", "|analytic|", "|numeric|", "rel. error", "result");

    for (size_t g = 0; g < group_count; g++)
    {
        double an = sqrt(stats[g].a2);
        double nu = sqrt(stats[g].n2);
        double rel = sqrt(stats[g].diff2) / (an + nu + 1e-12);

        const char* verdict = (an + nu < 1e-6) ? "both ~ 0" : (rel < PASS_THRESHOLD ? "OK" : "MISMATCH");

        printf("%-13s %12.3e %12.3e %11.3e  %s", stats[g].name, an, nu, rel, verdict);

        if (stats[g].analytic_zero_only * 10 > stats[g].total)
            printf("  (analytic = 0 but numeric != 0: %zu / %zu)", stats[g].analytic_zero_only, stats[g].total);

        printf("\n");
    }
}

int main()
{
    Dataset all = Dataset_load("data/number.dat");

    if (all.count < 10)
    {
        printf("Need at least 10 samples in 'data/number.dat'\n");

        Dataset_free(&all);

        return EXIT_FAILURE;
    }

    // same initialization as train.c
    srand(1234);

    size_t conv_out_rows = DS_IMG_LEN - 3 + 1;
    size_t pool_rows = conv_out_rows / 2;
    size_t fc_input_size = 16 * pool_rows * pool_rows;
    size_t fc_arch[] = { fc_input_size, 128, 64, 10 };

    ConvLayer conv = Conv_alloc(1, 16, 3);
    NN nn = NN_alloc(fc_arch, sizeof(fc_arch) / sizeof(*fc_arch));
    NN grad_fc = NN_alloc(fc_arch, sizeof(fc_arch) / sizeof(*fc_arch));

    NN_xavier_init(nn);
    NN_zero_grad(grad_fc);

    for (size_t i = 0; i < nn.count; i++)
        Mat_fill(nn.bs[i], 0.01f);

    for (size_t i = 0; i < conv.out_channels * conv.in_channels; i++)
        Mat_rand(conv.kernels[i], -0.1f, 0.1f);

    // Conv bias 0 makes the ReLU input exactly 0 on empty (black) pixels, where the
    // derivative is undefined and a numerical check is meaningless. Use 0.1 for the check.
    for (size_t i = 0; i < conv.out_channels; i++)
        MAT_AT(conv.biases[i], 0, 0) = 0.1f;

    size_t kernel_count = conv.out_channels * conv.in_channels;

    Mat* kernel_grads = (Mat*)malloc(sizeof(Mat) * kernel_count);
    float* bias_grads = (float*)malloc(sizeof(float) * conv.out_channels);

    if (!kernel_grads || !bias_grads)
    {
        printf("Memory allocation failed\n");

        free(kernel_grads);
        free(bias_grads);

        return EXIT_FAILURE;
    }

    for (size_t k = 0; k < kernel_count; k++)
        kernel_grads[k] = Mat_alloc(conv.kernel_size, conv.kernel_size);

    printf("Gradient check: %d images, %d random parameters per group and image, h = %g\n",
        CHECK_SAMPLES, PARAMS_PER_GROUP, (double)STEP);

#if CHECK_LEGACY
    NN_l2_lambda = 0.0f;
    run_check(1, all, nn, conv, grad_fc, kernel_grads, bias_grads);   // original code
#endif

    NN_l2_lambda = 0.0f;
    run_check(0, all, nn, conv, grad_fc, kernel_grads, bias_grads);   // fixed code, no L2

    NN_l2_lambda = L2_CHECK_LAMBDA;
    run_check(0, all, nn, conv, grad_fc, kernel_grads, bias_grads);   // fixed code + L2

    NN_l2_lambda = 0.0f;
    NN_dropout_p = DROPOUT_CHECK_P;
    run_check(0, all, nn, conv, grad_fc, kernel_grads, bias_grads);   // fixed code + dropout

    NN_l2_lambda = L2_CHECK_LAMBDA;
    run_check(0, all, nn, conv, grad_fc, kernel_grads, bias_grads);   // fixed code + L2 + dropout

    NN_l2_lambda = 0.0f;
    NN_dropout_p = 0.0f;

    for (size_t k = 0; k < kernel_count; k++)
        Mat_free(kernel_grads[k]);

    free(kernel_grads);
    free(bias_grads);

    NN_free(&nn);
    NN_free(&grad_fc);
    Conv_free(&conv);

    Dataset_free(&all);

    return EXIT_SUCCESS;
}