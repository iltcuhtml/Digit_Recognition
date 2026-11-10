#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#include "nn.h"
#include "dataset.h"

#define SEEDS           3
#define CHUNK           20000       // SEEDS * CHUNK must be <= 60000
#define EPOCHS          6
#define LR              0.003f
#define AUG_DEG         10.0f
#define AUG_SHIFT       3.0f
#define LAMBDA          0.03f       // chosen in the 1st study
#define DROP_P          0.6f
#define EVAL_TRAIN_N    2000
#define MODELS          4

typedef struct { NN nn; NN grad; ConvLayer conv; } Model;

typedef struct
{
    float train_acc, val_acc;
    float test_acc, test_loss;
    float own_acc, own_loss;
} Result;

static uint32_t shuffle_state = 1u;

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

static int Model_init(Model* m, int init_seed)
{
    size_t pool_rows = (DS_IMG_LEN - 3 + 1) / 2;
    size_t fc_arch[] = { 16 * pool_rows * pool_rows, 128, 64, 10 };
    size_t arch_count = sizeof(fc_arch) / sizeof(*fc_arch);

    memset(m, 0, sizeof(*m));
    srand((unsigned)init_seed);

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

static void mean_std(const float* v, int n, float* mean, float* sd)
{
    float s = 0.0f;

    for (int i = 0; i < n; i++)
        s += v[i];

    *mean = s / (float)n;

    float q = 0.0f;

    for (int i = 0; i < n; i++)
        q += (v[i] - *mean) * (v[i] - *mean);

    *sd = n > 1 ? sqrtf(q / (float)(n - 1)) : 0.0f;
}

static int run_one(float lambda, float dropout, int s,
    Dataset train, Dataset train_aug, Dataset val, Dataset test, Dataset own, Result* r)
{
    Model m;

    memset(r, 0, sizeof(*r));

    if (!Model_init(&m, 1234 + s))
    {
        Model_free(&m);
        return 0;
    }

    size_t* order = (size_t*)malloc(sizeof(size_t) * train.count);

    if (!order)
    {
        Model_free(&m);
        return 0;
    }

    NN_l2_lambda = lambda;
    NN_dropout_p = dropout;
    NN_dropout_seed(777u + (uint32_t)s);

    shuffle_state = 12345u + (uint32_t)s;

    for (int e = 1; e <= EPOCHS; e++)
    {
        shuffle_order(order, train.count);

        Dataset_random_transform(train, train_aug, AUG_DEG, AUG_SHIFT, 1000u + 100u * (uint32_t)s + (uint32_t)e);

        CNN_train_epoch(m.nn, m.grad, &m.conv, train_aug.inputs, train_aug.labels, LR, order);
    }

    // evaluation is always plain cross-entropy, no L2 / dropout
    NN_l2_lambda = 0.0f;
    NN_dropout_p = 0.0f;

    Mat tr_in = train.inputs, tr_lb = train.labels;

    tr_in.rows = EVAL_TRAIN_N;
    tr_lb.rows = EVAL_TRAIN_N;

    CNN_evaluate(m.nn, m.conv, tr_in, tr_lb, NULL, &r->train_acc);
    CNN_evaluate(m.nn, m.conv, val.inputs, val.labels, NULL, &r->val_acc);
    CNN_evaluate(m.nn, m.conv, test.inputs, test.labels, &r->test_loss, &r->test_acc);
    CNN_evaluate(m.nn, m.conv, own.inputs, own.labels, &r->own_loss, &r->own_acc);

    free(order);
    Model_free(&m);

    return 1;
}

int main(void)
{
    Dataset all = Dataset_load_mnist("data/mnist/train-images-idx3-ubyte", "data/mnist/train-labels-idx1-ubyte", 0);
    Dataset test = Dataset_load_mnist("data/mnist/t10k-images-idx3-ubyte", "data/mnist/t10k-labels-idx1-ubyte", 0);
    Dataset own = Dataset_load("data/number.dat");

    if (all.count < (size_t)SEEDS * CHUNK || test.count == 0 || own.count == 0)
        return EXIT_FAILURE;

    const char* names[MODELS] = { "baseline", "l2", "dropout", "both" };
    float lam[MODELS] = { 0.0f, LAMBDA, 0.0f, LAMBDA };
    float drp[MODELS] = { 0.0f, 0.0f, DROP_P, DROP_P };

    Result res[MODELS][SEEDS];

    memset(res, 0, sizeof(res));

    printf("[Compare | MNIST + aug | lr %g | %d epochs | %d chunks of %d | lambda %g | p %g]\n",
        (double)LR, EPOCHS, SEEDS, CHUNK, (double)LAMBDA, (double)DROP_P);

    for (int s = 0; s < SEEDS; s++)
    {
        Dataset chunk = Dataset_subset(all, (size_t)s * CHUNK, CHUNK);
        size_t n_tr = CHUNK / 10 * 9;

        Dataset train = Dataset_subset(chunk, 0, n_tr);
        Dataset val = Dataset_subset(chunk, n_tr, CHUNK - n_tr);
        Dataset train_aug = Dataset_subset(train, 0, train.count);

        if (chunk.count == 0 || train.count == 0 || val.count == 0 || train_aug.count == 0)
            return EXIT_FAILURE;

        for (int c = 0; c < MODELS; c++)
        {
            Result* r = &res[c][s];

            if (!run_one(lam[c], drp[c], s, train, train_aug, val, test, own, r))
                return EXIT_FAILURE;

            printf("  chunk %d | %-8s | train %5.1f%% | val %5.1f%% | test %5.2f%% (loss %.4f) | own %5.1f%% (loss %.4f)\n",
                s, names[c], (double)r->train_acc * 100, (double)r->val_acc * 100,
                (double)r->test_acc * 100, (double)r->test_loss, (double)r->own_acc * 100, (double)r->own_loss);

            fflush(stdout);
        }

        Dataset_free(&chunk);
        Dataset_free(&train);
        Dataset_free(&val);
        Dataset_free(&train_aug);
    }

    FILE* csv = NULL;
#ifdef _MSC_VER
    if (fopen_s(&csv, "data/compare_mnist.csv", "w") != 0)
        csv = NULL;
#else
    csv = fopen("data/compare_mnist.csv", "w");
#endif

    if (csv)
    {
        fprintf(csv, "model,chunk,train_acc,val_acc,test_acc,test_loss,own_acc,own_loss\n");

        for (int c = 0; c < MODELS; c++)
            for (int s = 0; s < SEEDS; s++)
                fprintf(csv, "%s,%d,%.2f,%.2f,%.2f,%.5f,%.2f,%.5f\n", names[c], s,
                    (double)res[c][s].train_acc * 100, (double)res[c][s].val_acc * 100,
                    (double)res[c][s].test_acc * 100, (double)res[c][s].test_loss,
                    (double)res[c][s].own_acc * 100, (double)res[c][s].own_loss);

        fclose(csv);
    }

    printf("\n=========== Table 1: mean +- std over %d chunks ===========\n", SEEDS);
    printf("%-9s %-12s %-14s %-16s %-12s %-12s\n", "model", "train", "test(MNIST)", "test loss", "gap(tr-te)", "own");

    for (int c = 0; c < MODELS; c++)
    {
        float a[SEEDS], b[SEEDS], d[SEEDS], g[SEEDS], o[SEEDS];
        float m1, s1, m2, s2, m3, s3, m4, s4, m5, s5;

        for (int s = 0; s < SEEDS; s++)
        {
            a[s] = res[c][s].train_acc * 100.0f;
            b[s] = res[c][s].test_acc * 100.0f;
            d[s] = res[c][s].test_loss;
            g[s] = a[s] - b[s];
            o[s] = res[c][s].own_acc * 100.0f;
        }

        mean_std(a, SEEDS, &m1, &s1);
        mean_std(b, SEEDS, &m2, &s2);
        mean_std(d, SEEDS, &m3, &s3);
        mean_std(g, SEEDS, &m4, &s4);
        mean_std(o, SEEDS, &m5, &s5);

        printf("%-9s %5.1f +- %-4.1f %5.2f +- %-5.2f %6.4f +- %-6.4f %+5.2f +- %-4.2f %5.1f +- %-4.1f\n",
            names[c], (double)m1, (double)s1, (double)m2, (double)s2,
            (double)m3, (double)s3, (double)m4, (double)s4, (double)m5, (double)s5);
    }

    printf("\n=========== Table 2: difference to baseline on the SAME chunk (model - baseline) ===========\n");
    printf("%-9s %-18s %-20s %-8s %-18s\n", "model", "d test acc (%p)", "d test loss", "lower", "d own acc (%p)");

    for (int c = 1; c < MODELS; c++)
    {
        float dt[SEEDS], dl[SEEDS], dw[SEEDS];
        int lower = 0;
        float m1, s1, m2, s2, m3, s3;

        for (int s = 0; s < SEEDS; s++)
        {
            dt[s] = (res[c][s].test_acc - res[0][s].test_acc) * 100.0f;
            dl[s] = res[c][s].test_loss - res[0][s].test_loss;
            dw[s] = (res[c][s].own_acc - res[0][s].own_acc) * 100.0f;

            if (dl[s] < 0.0f)
                lower++;
        }

        mean_std(dt, SEEDS, &m1, &s1);
        mean_std(dl, SEEDS, &m2, &s2);
        mean_std(dw, SEEDS, &m3, &s3);

        printf("%-9s %+6.2f +- %-6.2f   %+8.4f +- %-8.4f  %d / %d    %+6.1f +- %-6.1f\n",
            names[c], (double)m1, (double)s1, (double)m2, (double)s2, lower, SEEDS, (double)m3, (double)s3);
    }

    Dataset_free(&all);
    Dataset_free(&test);
    Dataset_free(&own);

    return EXIT_SUCCESS;
}