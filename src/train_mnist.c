#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "nn.h"
#include "dataset.h"

#define AUG_DEG         10.0f       // 0 and 0 = augmentation off
#define AUG_SHIFT       3.0f
#define LOAD_LIMIT      0           // 0 = all 60000. Use 6000 for a quick timing run
#define EPOCHS          10
#define LR              0.003f
#define INIT_SEED       1234
#define L2_LAMBDA       0.0f
#define DROPOUT_P       0.0f
#define EVAL_TRAIN_N    5000        // training accuracy is measured on the first 5000 training images

typedef struct
{
    NN nn;
    NN grad;
    ConvLayer conv;
} Model;

static uint32_t shuffle_state = 12345u;

static void shuffle_order(size_t* order, size_t n)
{
    for (size_t i = 0; i < n; i++)
        order[i] = i;

    for (size_t k = n; k > 1; k--)
    {
        uint32_t x = shuffle_state;

        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;

        shuffle_state = x;

        size_t j = x % k;
        size_t tmp = order[k - 1];

        order[k - 1] = order[j];
        order[j] = tmp;
    }
}

static int Model_init(Model* m)
{
    size_t pool_rows = (DS_IMG_LEN - 3 + 1) / 2;
    size_t fc_arch[] = { 16 * pool_rows * pool_rows, 128, 64, 10 };
    size_t arch_count = sizeof(fc_arch) / sizeof(*fc_arch);

    memset(m, 0, sizeof(*m));
    srand(INIT_SEED);

    m->conv = Conv_alloc(1, 16, 3);
    m->nn = NN_alloc(fc_arch, arch_count);
    m->grad = NN_alloc(fc_arch, arch_count);

    if (!m->nn.as || !m->grad.as || !m->conv.kernels)
        return 0;

    NN_xavier_init(m->nn);
    NN_zero_grad(m->grad);

    for (size_t i = 0; i < m->nn.count; i++)
        Mat_fill(m->nn.bs[i], 0.01f);

    for (size_t i = 0; i < m->conv.out_channels * m->conv.in_channels; i++)
        Mat_rand(m->conv.kernels[i], -0.1f, 0.1f);

    for (size_t i = 0; i < m->conv.out_channels; i++)
        MAT_AT(m->conv.biases[i], 0, 0) = 0.0f;

    return 1;
}

static void Model_free(Model* m)
{
    if (m->nn.as) NN_free(&m->nn);
    if (m->grad.as) NN_free(&m->grad);

    Conv_free(&m->conv);
}

// view of the first n rows (no copy)
static void view_first(Dataset d, size_t n, Mat* in, Mat* lb)
{
    *in = d.inputs;
    *lb = d.labels;

    in->rows = n;
    lb->rows = n;
}

// confusion matrix: row = true digit, column = predicted digit
static void print_confusion(NN nn, ConvLayer conv, Dataset d)
{
    int cm[DS_CLASSES][DS_CLASSES];
    memset(cm, 0, sizeof(cm));

    Mat* conv_out = (Mat*)malloc(sizeof(Mat) * conv.out_channels);
    Mat* pooled = (Mat*)malloc(sizeof(Mat) * conv.out_channels);
    Mat image = Mat_alloc(DS_IMG_LEN, DS_IMG_LEN);

    if (!conv_out || !pooled)
    {
        free(conv_out);
        free(pooled);
        Mat_free(image);
        return;
    }

    for (size_t i = 0; i < d.count; i++)
    {
        for (size_t p = 0; p < DS_PIXELS; p++)
            image.es[p] = MAT_AT(d.inputs, i, p);

        Mat flat;
        CNN_forward(nn, conv, image, conv_out, pooled, &flat);

        size_t best = 0;

        for (size_t j = 1; j < NN_OUTPUT(nn).cols; j++)
            if (MAT_AT(NN_OUTPUT(nn), 0, j) > MAT_AT(NN_OUTPUT(nn), 0, best))
                best = j;

        cm[d.digits[i]][best]++;

        for (size_t c = 0; c < conv.out_channels; c++)
        {
            Mat_free(conv_out[c]);
            Mat_free(pooled[c]);
        }

        Mat_free(flat);
    }

    printf("\nConfusion matrix on own handwriting (row = true, col = predicted)\n     ");

    for (int j = 0; j < DS_CLASSES; j++)
        printf("%3d", j);

    printf("\n");

    for (int i = 0; i < DS_CLASSES; i++)
    {
        printf("  %d: ", i);

        for (int j = 0; j < DS_CLASSES; j++)
            printf("%3d", cm[i][j]);

        printf("\n");
    }

    Mat_free(image);
    free(conv_out);
    free(pooled);
}

int main(void)
{
    Dataset all = Dataset_load_mnist("data/mnist/train-images-idx3-ubyte", "data/mnist/train-labels-idx1-ubyte", LOAD_LIMIT);
    Dataset test = Dataset_load_mnist("data/mnist/t10k-images-idx3-ubyte", "data/mnist/t10k-labels-idx1-ubyte", 0);
    Dataset own = Dataset_load("data/number.dat");

    if (all.count == 0 || test.count == 0 || own.count == 0)
        return EXIT_FAILURE;

    size_t n_val = all.count >= 20000 ? 5000 : all.count / 10;

    Dataset train = Dataset_subset(all, 0, all.count - n_val);
    Dataset val = Dataset_subset(all, all.count - n_val, n_val);
    Dataset train_aug = Dataset_subset(train, 0, train.count);

    if (train_aug.count == 0)
        return EXIT_FAILURE;

    Model m;

    if (train.count == 0 || val.count == 0 || !Model_init(&m))
        return EXIT_FAILURE;

    size_t* order = (size_t*)malloc(sizeof(size_t) * train.count);

    if (!order)
        return EXIT_FAILURE;

    size_t eval_n = train.count < EVAL_TRAIN_N ? train.count : EVAL_TRAIN_N;

    Mat tr_in, tr_lb;
    view_first(train, eval_n, &tr_in, &tr_lb);

    printf("[MNIST | train %zu | val %zu | lr %g | %d epochs | L2 %g | dropout %g | aug %g deg %g px]\n",
        train.count, val.count, (double)LR, EPOCHS, (double)L2_LAMBDA, (double)DROPOUT_P, (double)AUG_DEG, (double)AUG_SHIFT);

    NN_l2_lambda = L2_LAMBDA;
    NN_dropout_p = DROPOUT_P;
    NN_dropout_seed(777u);

    FILE* csv = NULL;
#ifdef _MSC_VER
    if (fopen_s(&csv, "data/curve_mnist_aug.csv", "w") != 0)
        csv = NULL;
#else
    csv = fopen("data/curve_mnist_aug.csv", "w");
#endif

    if (csv)
        fprintf(csv, "epoch,train_loss,val_loss,own_loss,train_acc,val_acc,own_acc\n");

    clock_t t0 = clock();

    for (int e = 1; e <= EPOCHS; e++)
    {
        shuffle_order(order, train.count);

        Dataset_random_transform(train, train_aug, AUG_DEG, AUG_SHIFT, 1000u + (uint32_t)e);

        CNN_train_epoch(m.nn, m.grad, &m.conv, train_aug.inputs, train_aug.labels, LR, order);

        float tl, ta, vl, va, ol, oa;

        CNN_evaluate(m.nn, m.conv, tr_in, tr_lb, &tl, &ta);
        CNN_evaluate(m.nn, m.conv, val.inputs, val.labels, &vl, &va);
        CNN_evaluate(m.nn, m.conv, own.inputs, own.labels, &ol, &oa);

        printf("epoch %2d | train %5.1f%% (loss %.4f) | val %5.1f%% (loss %.4f) | own %5.1f%% (loss %.4f) | %.0fs\n",
            e, (double)ta * 100, (double)tl, (double)va * 100, (double)vl, (double)oa * 100, (double)ol,
            (double)(clock() - t0) / CLOCKS_PER_SEC);

        if (csv)
            fprintf(csv, "%d,%.5f,%.5f,%.5f,%.4f,%.4f,%.4f\n", e, (double)tl, (double)vl, (double)ol, (double)ta, (double)va, (double)oa);
    }

    if (csv)
        fclose(csv);

    float xl, xa, ol, oa;

    CNN_evaluate(m.nn, m.conv, test.inputs, test.labels, &xl, &xa);
    CNN_evaluate(m.nn, m.conv, own.inputs, own.labels, &ol, &oa);

    printf("\nFINAL | MNIST test %.2f%% (loss %.4f) | own handwriting %.2f%% (loss %.4f)\n",
        (double)xa * 100, (double)xl, (double)oa * 100, (double)ol);

    print_confusion(m.nn, m.conv, own);

    FILE* out = NULL;
#ifdef _MSC_VER
    if (fopen_s(&out, "data/model_mnist_aug.cnn", "wb") != 0)
        out = NULL;
#else
    out = fopen("data/model_mnist_aug.cnn", "wb");
#endif
    if (out)
    {
        CNN_save(out, m.conv, m.nn);
        fclose(out);
    }

    NN_l2_lambda = 0.0f;
    NN_dropout_p = 0.0f;

    free(order);
    Model_free(&m);

    Dataset_free(&train);
	Dataset_free(&train_aug);
    Dataset_free(&val);
    Dataset_free(&all);
    Dataset_free(&test);
    Dataset_free(&own);

    return EXIT_SUCCESS;
}