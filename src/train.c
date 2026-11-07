#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#include "nn.h"
#include "dataset.h"

#define CELL_LEN 28

// ---------- Experiment settings ----------
#define TRAIN_RATIO      0.70f
#define VAL_RATIO        0.15f
#define AUGMENT_COPIES   9
#define AUGMENT_SEED     7u

// ---------- Stage 3 parts (run them one by one) ----------
//  Part 1: RUN_LR_SEARCH 1, LR_SEARCH_ONLY 1            -> choose the learning rate
//  Part 2: RUN_LR_SEARCH 0, FIXED_LR <chosen>, SPLIT_REPEATS 1  -> one baseline run
//  Part 3: RUN_LR_SEARCH 0, FIXED_LR <chosen>, SPLIT_REPEATS 5  -> repeated splits
#define RUN_LR_SEARCH    1
#define LR_SEARCH_ONLY   1
#define FIXED_LR         0.03f              // used when RUN_LR_SEARCH is 0
#define SPLIT_REPEATS    1                  // number of different splits (seeds)
#define BASE_SPLIT_SEED  20424u             // seeds: BASE + 0, 1, 2, ...

#define EPOCHS           30                 // epochs of the final baseline runs
#define LR_SEARCH_EPOCHS 15                 // epochs used to compare learning rates
#define INIT_SEED        1234               // weight initialization seed (same for every run)

static const float LR_CANDIDATES[] = { 0.003f, 0.01f, 0.03f, 0.1f };
#define LR_COUNT ((int)(sizeof(LR_CANDIDATES) / sizeof(LR_CANDIDATES[0])))

#define SAVE_MODEL_FILE  "data/model_baseline.cnn"   // seed #0 model (does not overwrite model.cnn)

typedef struct
{
    NN nn;
    NN grad;
    ConvLayer conv;
} Model;

typedef struct
{
    float train_loss, train_acc;    // on the augmented training set
    float orig_acc;                 // on the original (not augmented) training images
    float val_loss, val_acc;
    float test_loss, test_acc;
} Result;

// ---------- small helpers ----------
static uint32_t shuffle_state = 1;

static uint32_t shuffle_next(void)
{
    uint32_t x = shuffle_state;

    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;

    shuffle_state = x;
    return x;
}

static void shuffle_order(size_t* order, size_t n)
{
    for (size_t i = 0; i < n; i++)
        order[i] = i;

    for (size_t k = n; k > 1; k--)
    {
        size_t j = shuffle_next() % k;
        size_t tmp = order[k - 1];

        order[k - 1] = order[j];
        order[j] = tmp;
    }
}

static int Model_init(Model* m)
{
    size_t conv_out_rows = CELL_LEN - 3 + 1;
    size_t pool_rows = conv_out_rows / 2;
    size_t fc_input_size = 16 * pool_rows * pool_rows;
    size_t fc_arch[] = { fc_input_size, 128, 64, 10 };
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

// Train 'epochs' epochs. If csv_path != NULL, write the learning curve there.
// Returns the final-epoch result (train/val/test).
static Result run_training(
    Model* m, Dataset train, Dataset train_raw, Dataset val, Dataset test,
    float lr, int epochs, const char* csv_path, int verbose)
{
    Result r;
    memset(&r, 0, sizeof(r));

    size_t* order = (size_t*)malloc(sizeof(size_t) * train.count);

    if (!order)
    {
        printf("Memory allocation failed\n");
        return r;
    }

    FILE* csv = NULL;

    if (csv_path)
    {
#ifdef _MSC_VER
        if (fopen_s(&csv, csv_path, "w") != 0)
            csv = NULL;
#else
        csv = fopen(csv_path, "w");
#endif
        if (csv)
            fprintf(csv, "epoch,train_loss,val_loss,train_acc,val_acc\n");
    }

    shuffle_state = 12345u;

    for (int e = 1; e <= epochs; e++)
    {
        shuffle_order(order, train.count);

        CNN_train_epoch(m->nn, m->grad, &m->conv, train.inputs, train.labels, lr, order);

        float tl, ta, vl, va;

        CNN_evaluate(m->nn, m->conv, train.inputs, train.labels, &tl, &ta);
        CNN_evaluate(m->nn, m->conv, val.inputs, val.labels, &vl, &va);

        if (csv)
            fprintf(csv, "%d,%.5f,%.5f,%.4f,%.4f\n", e, tl, vl, ta, va);

        if (verbose && (e == 1 || e % 5 == 0 || e == epochs))
            printf("    epoch %2d/%d  train loss %.4f acc %.3f | val loss %.4f acc %.3f\n",
                e, epochs, tl, ta, vl, va);

        r.train_loss = tl;
        r.train_acc = ta;
        r.val_loss = vl;
        r.val_acc = va;
    }

    CNN_evaluate(m->nn, m->conv, train_raw.inputs, train_raw.labels, NULL, &r.orig_acc);
    CNN_evaluate(m->nn, m->conv, test.inputs, test.labels, &r.test_loss, &r.test_acc);

    if (csv)
        fclose(csv);

    free(order);

    return r;
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

int main()
{
    Dataset all = Dataset_load("data/number.dat");

    if (all.count == 0)
        return EXIT_FAILURE;

    printf("Dataset: %zu samples\n", all.count);

    Result results[SPLIT_REPEATS];
    memset(results, 0, sizeof(results));

    float best_lr = FIXED_LR;
    int ok = 1;

    for (int s = 0; s < SPLIT_REPEATS && ok; s++)
    {
        uint32_t seed = BASE_SPLIT_SEED + (uint32_t)s;

        Dataset train_raw = { 0 }, val = { 0 }, test = { 0 }, train = { 0 };

        Dataset_split(all, TRAIN_RATIO, VAL_RATIO, seed, &train_raw, &val, &test);
        train = Dataset_augment(train_raw, AUGMENT_COPIES, AUGMENT_SEED);

        if (train_raw.count == 0 || val.count == 0 || test.count == 0 || train.count == 0)
        {
            printf("Dataset split failed\n");
            ok = 0;
        }

        // ---- learning-rate search (first split only, validation data) ----
        if (ok && s == 0 && RUN_LR_SEARCH)
        {
            printf("\n[Learning-rate search | split seed %u | %d epochs | validation data]\n",
                seed, LR_SEARCH_EPOCHS);

            float best_val_loss = 1e30f;

            for (int k = 0; k < LR_COUNT && ok; k++)
            {
                Model m;

                if (!Model_init(&m))
                {
                    Model_free(&m);
                    ok = 0;
                    break;
                }

                Result r = run_training(&m, train, train_raw, val, test,
                    LR_CANDIDATES[k], LR_SEARCH_EPOCHS, NULL, 0);

                int diverged = !(r.val_loss == r.val_loss) || r.val_loss > 1e6f;

                printf("  lr = %-6g  train loss %.4f  val loss %.4f  val acc %.3f%s\n",
                    (double)LR_CANDIDATES[k], r.train_loss, r.val_loss, r.val_acc,
                    diverged ? "  (diverged)" : "");

                if (!diverged && r.val_loss < best_val_loss)
                {
                    best_val_loss = r.val_loss;
                    best_lr = LR_CANDIDATES[k];
                }

                Model_free(&m);
            }

            printf("  -> selected lr = %g (lowest validation loss)\n", (double)best_lr);

            if (LR_SEARCH_ONLY)
            {
                Dataset_free(&train_raw);
                Dataset_free(&val);
                Dataset_free(&test);
                Dataset_free(&train);
                Dataset_free(&all);

                return EXIT_SUCCESS;
            }
        }

        // ---- baseline training ----
        if (ok)
        {
            printf("\n[Baseline | split seed %u | lr %g | %d epochs]\n", seed, (double)best_lr, EPOCHS);
            Dataset_print_summary("TRAIN (original) ", train_raw);
            Dataset_print_summary("VAL              ", val);
            Dataset_print_summary("TEST             ", test);

            Model m;

            if (!Model_init(&m))
            {
                Model_free(&m);
                ok = 0;
            }
            else
            {
                char path[64];
                snprintf(path, sizeof(path), "data/curve_baseline_seed%d.csv", s);

                results[s] = run_training(&m, train, train_raw, val, test, best_lr, EPOCHS, path, 1);

                if (s == 0)
                {
                    FILE* out = NULL;
#ifdef _MSC_VER
                    if (fopen_s(&out, SAVE_MODEL_FILE, "wb") != 0)
                        out = NULL;
#else
                    out = fopen(SAVE_MODEL_FILE, "wb");
#endif
                    if (out)
                    {
                        CNN_save(out, m.conv, m.nn);
                        fclose(out);
                    }
                }

                Model_free(&m);
            }
        }

        Dataset_free(&train_raw);
        Dataset_free(&val);
        Dataset_free(&test);
        Dataset_free(&train);
    }

    Dataset_free(&all);

    if (!ok)
        return EXIT_FAILURE;

    // ---- summary table ----
    printf("\n=========== Baseline summary (no regularization, lr %g, %d epochs) ===========\n",
        (double)best_lr, EPOCHS);
    printf("%-6s %-9s %-10s %-9s %-9s %-10s\n", "seed", "train(aug)", "train(orig)", "val", "test", "gap(tr-te)");

    float tr[SPLIT_REPEATS], va[SPLIT_REPEATS], te[SPLIT_REPEATS], gap[SPLIT_REPEATS];

    for (int s = 0; s < SPLIT_REPEATS; s++)
    {
        tr[s] = results[s].train_acc * 100.0f;
        va[s] = results[s].val_acc * 100.0f;
        te[s] = results[s].test_acc * 100.0f;
        gap[s] = tr[s] - te[s];

        printf("%-6u %8.1f%% %9.1f%% %8.1f%% %8.1f%% %9.1f%%p\n",
            BASE_SPLIT_SEED + (uint32_t)s, (double)tr[s], (double)(results[s].orig_acc * 100.0f),
            (double)va[s], (double)te[s], (double)gap[s]);
    }

    float m1, s1, m2, s2, m3, s3, m4, s4;

    mean_std(tr, SPLIT_REPEATS, &m1, &s1);
    mean_std(va, SPLIT_REPEATS, &m2, &s2);
    mean_std(te, SPLIT_REPEATS, &m3, &s3);
    mean_std(gap, SPLIT_REPEATS, &m4, &s4);

    printf("mean   %5.1f%%+-%.1f  val %5.1f%%+-%.1f  test %5.1f%%+-%.1f  gap %5.1f%%p+-%.1f\n",
        (double)m1, (double)s1, (double)m2, (double)s2, (double)m3, (double)s3, (double)m4, (double)s4);

    return EXIT_SUCCESS;
}