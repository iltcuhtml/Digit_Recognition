#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#include "nn.h"
#include "dataset.h"

#define CELL_LEN 28

// ---------- Experiment settings (same for every model) ----------
#define TRAIN_RATIO      0.70f
#define VAL_RATIO        0.15f
#define AUGMENT_COPIES   9
#define AUGMENT_SEED     7u

#define SPLIT_REPEATS    5                  // number of different splits (seeds)
#define BASE_SPLIT_SEED  20424u             // seeds: BASE + 0, 1, 2, ...
#define EPOCHS           30
#define LR               0.003f             // chosen in stage 3
#define INIT_SEED        1234               // weight initialization seed (same for every run)

// ---------- Stage 4-3: run the two parts one by one ----------
//  Part 1: RUN_MODE MODE_SEARCH   -> choose lambda and p with VALIDATION data only
//                                    (the test data is never evaluated in this part)
//  Part 2: RUN_MODE MODE_COMPARE  -> set CHOSEN_LAMBDA / CHOSEN_P below first, then
//                                    compare baseline / L2 / dropout / both on the same splits
#define MODE_SEARCH      0
#define MODE_COMPARE     1
#define RUN_MODE         MODE_COMPARE

#define CHOSEN_LAMBDA    0.03f
#define CHOSEN_P         0.6f

static const float L2_CANDIDATES[] = { 0.03f, 0.3f };
static const float DROPOUT_CANDIDATES[] = { 0.6f, 0.7f };

#define L2_COUNT        ((int)(sizeof(L2_CANDIDATES) / sizeof(L2_CANDIDATES[0])))
#define DROPOUT_COUNT   ((int)(sizeof(DROPOUT_CANDIDATES) / sizeof(DROPOUT_CANDIDATES[0])))
#define MAX_CFG         16

typedef struct
{
    NN nn;
    NN grad;
    ConvLayer conv;
} Model;

typedef struct
{
    char name[24];
    float lambda;           // L2 strength, 0 = off
    float dropout;          // dropout probability of the hidden layers, 0 = off
} Config;

typedef struct
{
    float train_loss, train_acc;    // on the augmented training set
    float orig_acc;                 // on the original (not augmented) training images
    float val_loss, val_acc;
    float test_loss, test_acc;      // only filled when eval_test = 1
} Result;

typedef struct
{
    Dataset train_raw, val, test, train;
} Split;

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

static FILE* open_write(const char* path)
{
    FILE* f = NULL;

#ifdef _MSC_VER
    if (fopen_s(&f, path, "w") != 0)
        f = NULL;
#else
    f = fopen(path, "w");
#endif

    return f;
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

static Config make_config(const char* name, float lambda, float dropout)
{
    Config c;

    memset(&c, 0, sizeof(c));
    snprintf(c.name, sizeof(c.name), "%s", name);

    c.lambda = lambda;
    c.dropout = dropout;

    return c;
}

// ---------- model ----------
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

// ---------- data split ----------
static int Split_make(Dataset all, uint32_t seed, Split* sp)
{
    memset(sp, 0, sizeof(*sp));

    Dataset_split(all, TRAIN_RATIO, VAL_RATIO, seed, &sp->train_raw, &sp->val, &sp->test);
    sp->train = Dataset_augment(sp->train_raw, AUGMENT_COPIES, AUGMENT_SEED);

    return sp->train_raw.count > 0 && sp->val.count > 0 && sp->test.count > 0 && sp->train.count > 0;
}

static void Split_free(Split* sp)
{
    Dataset_free(&sp->train_raw);
    Dataset_free(&sp->val);
    Dataset_free(&sp->test);
    Dataset_free(&sp->train);
}

// ---------- training ----------
// Train 'epochs' epochs with the regularization given in cfg.
// eval_test = 0: the test data is NOT touched (used while choosing hyper-parameters).
// If csv_path != NULL, the learning curve is written there.
// Losses are plain cross-entropy (no L2 term, no dropout) so every model is measured the same way.
static Result run_training(
    Model* m, Split* sp, float lr, int epochs, Config cfg, uint32_t seed,
    int eval_test, const char* csv_path)
{
    Result r;
    memset(&r, 0, sizeof(r));

    size_t* order = (size_t*)malloc(sizeof(size_t) * sp->train.count);

    if (!order)
    {
        printf("Memory allocation failed\n");
        return r;
    }

    FILE* csv = csv_path ? open_write(csv_path) : NULL;

    if (csv)
        fprintf(csv, "epoch,train_loss,val_loss,train_acc,val_acc\n");

    NN_l2_lambda = cfg.lambda;
    NN_dropout_p = cfg.dropout;
    NN_dropout_seed(seed + 777u);       // same masks for every model that uses dropout on this split

    shuffle_state = 12345u;

    for (int e = 1; e <= epochs; e++)
    {
        shuffle_order(order, sp->train.count);

        CNN_train_epoch(m->nn, m->grad, &m->conv, sp->train.inputs, sp->train.labels, lr, order);

        float tl, ta, vl, va;

        CNN_evaluate(m->nn, m->conv, sp->train.inputs, sp->train.labels, &tl, &ta);
        CNN_evaluate(m->nn, m->conv, sp->val.inputs, sp->val.labels, &vl, &va);

        if (csv)
            fprintf(csv, "%d,%.5f,%.5f,%.4f,%.4f\n", e, tl, vl, ta, va);

        r.train_loss = tl;
        r.train_acc = ta;
        r.val_loss = vl;
        r.val_acc = va;
    }

    CNN_evaluate(m->nn, m->conv, sp->train_raw.inputs, sp->train_raw.labels, NULL, &r.orig_acc);

    if (eval_test)
        CNN_evaluate(m->nn, m->conv, sp->test.inputs, sp->test.labels, &r.test_loss, &r.test_acc);

    NN_l2_lambda = 0.0f;
    NN_dropout_p = 0.0f;

    if (csv)
        fclose(csv);

    free(order);

    return r;
}

// =====================================================================
// Part 1: choose lambda and p with validation data
// =====================================================================
static int run_search(Dataset all)
{
    Config cfgs[MAX_CFG];
    int cfg_count = 0;

    memset(cfgs, 0, sizeof(cfgs));

    cfgs[cfg_count++] = make_config("baseline", 0.0f, 0.0f);

    for (int i = 0; i < L2_COUNT; i++)
    {
        char name[24];
        snprintf(name, sizeof(name), "L2 %g", (double)L2_CANDIDATES[i]);

        cfgs[cfg_count++] = make_config(name, L2_CANDIDATES[i], 0.0f);
    }

    for (int i = 0; i < DROPOUT_COUNT; i++)
    {
        char name[24];
        snprintf(name, sizeof(name), "dropout %g", (double)DROPOUT_CANDIDATES[i]);

        cfgs[cfg_count++] = make_config(name, 0.0f, DROPOUT_CANDIDATES[i]);
    }

    static float vloss[MAX_CFG][SPLIT_REPEATS], vacc[MAX_CFG][SPLIT_REPEATS], tacc[MAX_CFG][SPLIT_REPEATS];

    memset(vloss, 0, sizeof(vloss));
    memset(vacc, 0, sizeof(vacc));
    memset(tacc, 0, sizeof(tacc));

    printf("\n[Search | lr %g | %d epochs | %d splits | validation data only]\n",
        (double)LR, EPOCHS, SPLIT_REPEATS);

    for (int s = 0; s < SPLIT_REPEATS; s++)
    {
        uint32_t seed = BASE_SPLIT_SEED + (uint32_t)s;

        Split sp;

        if (!Split_make(all, seed, &sp))
        {
            printf("Dataset split failed\n");
            Split_free(&sp);

            return 0;
        }

        for (int c = 0; c < cfg_count; c++)
        {
            Model m;

            if (!Model_init(&m))
            {
                Model_free(&m);
                Split_free(&sp);

                return 0;
            }

            Result r = run_training(&m, &sp, LR, EPOCHS, cfgs[c], seed, 0, NULL);

            vloss[c][s] = r.val_loss;
            vacc[c][s] = r.val_acc * 100.0f;
            tacc[c][s] = r.orig_acc * 100.0f;

            printf("  seed %u | %-14s | train(orig) %5.1f%% | val loss %.4f | val acc %5.1f%%\n",
                seed, cfgs[c].name, (double)tacc[c][s], (double)vloss[c][s], (double)vacc[c][s]);

            Model_free(&m);
        }

        Split_free(&sp);
    }

    // ---- summary ----
    FILE* csv = open_write("data/search_summary.csv");

    if (csv)
        fprintf(csv, "config,lambda,dropout,seed,val_loss,val_acc,train_orig_acc\n");

    float mean_loss[MAX_CFG] = { 0.0f };

    printf("\n=========== Search summary (mean +- std over %d splits, validation data) ===========\n", SPLIT_REPEATS);
    printf("%-14s %-16s %-14s %-12s\n", "config", "val loss", "val acc(%)", "train(orig)(%)");

    for (int c = 0; c < cfg_count; c++)
    {
        float ml, sl, ma, sa, mt, st;

        mean_std(vloss[c], SPLIT_REPEATS, &ml, &sl);
        mean_std(vacc[c], SPLIT_REPEATS, &ma, &sa);
        mean_std(tacc[c], SPLIT_REPEATS, &mt, &st);

        mean_loss[c] = ml;

        printf("%-14s %6.4f +- %-6.4f  %5.1f +- %-5.1f  %5.1f +- %-4.1f\n",
            cfgs[c].name, (double)ml, (double)sl, (double)ma, (double)sa, (double)mt, (double)st);

        if (csv)
            for (int s = 0; s < SPLIT_REPEATS; s++)
                fprintf(csv, "%s,%g,%g,%u,%.5f,%.2f,%.2f\n", cfgs[c].name,
                    (double)cfgs[c].lambda, (double)cfgs[c].dropout, BASE_SPLIT_SEED + (uint32_t)s,
                    (double)vloss[c][s], (double)vacc[c][s], (double)tacc[c][s]);
    }

    if (csv)
        fclose(csv);

    // best of each family = lowest mean validation loss
    int best_l2 = 1, best_dr = 1 + L2_COUNT;

    for (int c = 1; c <= L2_COUNT; c++)
        if (mean_loss[c] < mean_loss[best_l2])
            best_l2 = c;

    for (int c = 1 + L2_COUNT; c < cfg_count; c++)
        if (mean_loss[c] < mean_loss[best_dr])
            best_dr = c;

    printf("\n-> best L2      : lambda = %g (val loss %.4f, baseline %.4f)\n",
        (double)cfgs[best_l2].lambda, (double)mean_loss[best_l2], (double)mean_loss[0]);
    printf("-> best dropout : p      = %g (val loss %.4f, baseline %.4f)\n",
        (double)cfgs[best_dr].dropout, (double)mean_loss[best_dr], (double)mean_loss[0]);

    if (best_l2 == 1 || best_l2 == L2_COUNT)
        printf("   note: the best lambda is at the edge of the candidates -> the range may need to be extended\n");

    if (best_dr == 1 + L2_COUNT || best_dr == cfg_count - 1)
        printf("   note: the best p is at the edge of the candidates -> the range may need to be extended\n");

    return 1;
}

// =====================================================================
// Part 2: baseline / L2 / dropout / both on the same splits (paired)
// =====================================================================
#define MODEL_COUNT 4

static int run_compare(Dataset all)
{
    // read through volatile so the compiler / code analysis does not treat the check as a constant
    volatile float lambda_in = CHOSEN_LAMBDA;
    volatile float p_in = CHOSEN_P;

    float chosen_lambda = lambda_in;
    float chosen_p = p_in;

    if (chosen_lambda <= 0.0f || chosen_p <= 0.0f)
    {
        printf("Set CHOSEN_LAMBDA and CHOSEN_P (results of Part 1) at the top of train.c first.\n");

        return 0;
    }

    Config cfgs[MODEL_COUNT];

    memset(cfgs, 0, sizeof(cfgs));

    cfgs[0] = make_config("baseline", 0.0f, 0.0f);
    cfgs[1] = make_config("l2", chosen_lambda, 0.0f);
    cfgs[2] = make_config("dropout", 0.0f, chosen_p);
    cfgs[3] = make_config("both", chosen_lambda, chosen_p);

    float tr[MODEL_COUNT][SPLIT_REPEATS], va[MODEL_COUNT][SPLIT_REPEATS];
    float te[MODEL_COUNT][SPLIT_REPEATS], gap[MODEL_COUNT][SPLIT_REPEATS];
    float tl[MODEL_COUNT][SPLIT_REPEATS];

    memset(tr, 0, sizeof(tr));
    memset(va, 0, sizeof(va));
    memset(te, 0, sizeof(te));
    memset(gap, 0, sizeof(gap));
    memset(tl, 0, sizeof(tl));

    printf("\n[Compare | lr %g | %d epochs | %d splits | lambda %g | p %g]\n",
        (double)LR, EPOCHS, SPLIT_REPEATS, (double)chosen_lambda, (double)chosen_p);

    for (int s = 0; s < SPLIT_REPEATS; s++)
    {
        uint32_t seed = BASE_SPLIT_SEED + (uint32_t)s;

        Split sp;

        if (!Split_make(all, seed, &sp))
        {
            printf("Dataset split failed\n");
            Split_free(&sp);

            return 0;
        }

        for (int c = 0; c < MODEL_COUNT; c++)
        {
            Model m;

            if (!Model_init(&m))
            {
                Model_free(&m);
                Split_free(&sp);

                return 0;
            }

            char path[64];
            snprintf(path, sizeof(path), "data/curve_%s_seed%d.csv", cfgs[c].name, s);

            Result r = run_training(&m, &sp, LR, EPOCHS, cfgs[c], seed, 1, path);

            tr[c][s] = r.orig_acc * 100.0f;
            va[c][s] = r.val_acc * 100.0f;
            te[c][s] = r.test_acc * 100.0f;
            tl[c][s] = r.test_loss;
            gap[c][s] = tr[c][s] - te[c][s];

            printf("  seed %u | %-8s | train(orig) %5.1f%% | val %5.1f%% | test %5.1f%% | gap %5.1f%%p\n",
                seed, cfgs[c].name, (double)tr[c][s], (double)va[c][s], (double)te[c][s], (double)gap[c][s]);

            if (s == 0)
            {
                char model_path[64];
                snprintf(model_path, sizeof(model_path), "data/model_%s.cnn", cfgs[c].name);

                FILE* out = NULL;
#ifdef _MSC_VER
                if (fopen_s(&out, model_path, "wb") != 0)
                    out = NULL;
#else
                out = fopen(model_path, "wb");
#endif
                if (out)
                {
                    CNN_save(out, m.conv, m.nn);
                    fclose(out);
                }
            }

            Model_free(&m);
        }

        Split_free(&sp);
    }

    // ---- csv for the report graphs ----
    FILE* csv = open_write("data/compare_summary.csv");

    if (csv)
    {
        fprintf(csv, "model,seed,train_orig_acc,val_acc,test_acc,test_loss,gap\n");

        for (int c = 0; c < MODEL_COUNT; c++)
            for (int s = 0; s < SPLIT_REPEATS; s++)
                fprintf(csv, "%s,%u,%.2f,%.2f,%.2f,%.5f,%.2f\n", cfgs[c].name, BASE_SPLIT_SEED + (uint32_t)s,
                    (double)tr[c][s], (double)va[c][s], (double)te[c][s], (double)tl[c][s], (double)gap[c][s]);

        fclose(csv);
    }

    // ---- table 1: mean +- std per model ----
    printf("\n=========== Table 1: mean +- std over %d splits (lr %g, %d epochs) ===========\n",
        SPLIT_REPEATS, (double)LR, EPOCHS);
    printf("%-9s %-13s %-13s %-13s %-13s\n", "model", "train(orig)", "val", "test", "gap(tr-te)");

    for (int c = 0; c < MODEL_COUNT; c++)
    {
        float m1, s1, m2, s2, m3, s3, m4, s4;

        mean_std(tr[c], SPLIT_REPEATS, &m1, &s1);
        mean_std(va[c], SPLIT_REPEATS, &m2, &s2);
        mean_std(te[c], SPLIT_REPEATS, &m3, &s3);
        mean_std(gap[c], SPLIT_REPEATS, &m4, &s4);

        printf("%-9s %5.1f +- %-4.1f  %5.1f +- %-4.1f  %5.1f +- %-4.1f  %5.1f +- %-4.1f\n",
            cfgs[c].name, (double)m1, (double)s1, (double)m2, (double)s2,
            (double)m3, (double)s3, (double)m4, (double)s4);
    }

    // ---- table 2: test accuracy per split ----
    printf("\n=========== Table 2: test accuracy (%%) per split ===========\n");
    printf("%-7s", "seed");

    for (int c = 0; c < MODEL_COUNT; c++)
        printf(" %-9s", cfgs[c].name);

    printf("\n");

    for (int s = 0; s < SPLIT_REPEATS; s++)
    {
        printf("%-7u", BASE_SPLIT_SEED + (uint32_t)s);

        for (int c = 0; c < MODEL_COUNT; c++)
            printf(" %-9.1f", (double)te[c][s]);

        printf("\n");
    }

    // ---- table 3: paired difference to the baseline (same split, same init) ----
    printf("\n=========== Table 3: difference to baseline on the SAME split (model - baseline) ===========\n");
    printf("%-9s %-17s %-17s %-11s %-19s %s\n",
        "model", "d test acc (%p)", "d gap (%p)", "acc b/s/w", "d test loss", "loss lower");

    for (int c = 1; c < MODEL_COUNT; c++)
    {
        float dt[SPLIT_REPEATS] = { 0.0f };
        float dg[SPLIT_REPEATS] = { 0.0f };
        float dl[SPLIT_REPEATS] = { 0.0f };
        int better = 0, same = 0, worse = 0, lower = 0;

        for (int s = 0; s < SPLIT_REPEATS; s++)
        {
            dt[s] = te[c][s] - te[0][s];
            dg[s] = gap[c][s] - gap[0][s];
            dl[s] = tl[c][s] - tl[0][s];

            if (dt[s] > 0.0f) better++;
            else if (dt[s] < 0.0f) worse++;
            else same++;

            if (dl[s] < 0.0f) lower++;
        }

        float mt, st, mg, sg, ml, sl;

        mean_std(dt, SPLIT_REPEATS, &mt, &st);
        mean_std(dg, SPLIT_REPEATS, &mg, &sg);
        mean_std(dl, SPLIT_REPEATS, &ml, &sl);

        printf("%-9s %+6.1f +- %-6.1f  %+6.1f +- %-6.1f  %d / %d / %d   %+7.4f +- %-7.4f  %d / %d\n",
            cfgs[c].name, (double)mt, (double)st, (double)mg, (double)sg, better, same, worse,
            (double)ml, (double)sl, lower, SPLIT_REPEATS);
    }

    return 1;
}

int main()
{
    Dataset all = Dataset_load("data/number.dat");

    if (all.count == 0)
        return EXIT_FAILURE;

    printf("Dataset: %zu samples\n", all.count);

    int ok = (RUN_MODE == MODE_SEARCH) ? run_search(all) : run_compare(all);

    Dataset_free(&all);

    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}