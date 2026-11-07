#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include "nn.h"
#include "dataset.h"

#define CELL_LEN 28

// ---------- Experiment settings ----------
#define TRAIN_RATIO      0.70f      // train 70% / val 15% / test 15%
#define VAL_RATIO        0.15f
#define SPLIT_SEED       20424u     // fixed seed -> the same split every run
#define AUGMENT_COPIES   9          // each training image + 9 rotated/shifted copies
#define AUGMENT_SEED     7u

#define RESUME_TRAINING  0          // 0: always start from a new model (no leakage from old training)
#define DATA_CHECK_ONLY  1          // 1: only print split/augmentation result and exit

int main()
{
    FILE* model_file = NULL;
    FILE* out_file = NULL;

    Dataset all = { 0 }, train_raw = { 0 }, val = { 0 }, test = { 0 }, train = { 0 };
    NN nn = { 0 }, grad_fc = { 0 };
    ConvLayer conv = { 0 };

    // --- Load dataset ---
    all = Dataset_load("data/number.dat");

    if (all.count == 0)
        goto cleanup;

    // --- Split (train / val / test) -> augment ONLY the training set ---
    Dataset_split(all, TRAIN_RATIO, VAL_RATIO, SPLIT_SEED, &train_raw, &val, &test);
    Dataset_free(&all);

    train = Dataset_augment(train_raw, AUGMENT_COPIES, AUGMENT_SEED);

    Dataset_print_summary("TRAIN (original) ", train_raw);
    Dataset_print_summary("TRAIN (augmented)", train);
    Dataset_print_summary("VAL              ", val);
    Dataset_print_summary("TEST             ", test);

#if DATA_CHECK_ONLY
    printf("\n[original image]\n");
    Dataset_print_ascii(train, 0);

    printf("\n[augmented image (rotated/shifted copy of the same sample)]\n");
    Dataset_print_ascii(train, 1);

    goto cleanup;
#endif

    const int num_classes = DS_CLASSES;
    (void)num_classes;

    // --- Load or initialize model ---
#if RESUME_TRAINING
    if (fopen_s(&model_file, "data/model.cnn", "rb") == 0 && model_file)
    {
        CNN_load(&conv, &nn, model_file);

        fclose(model_file);
        model_file = NULL;

        printf("Existing model loaded, continuing training\n");
    }
    else
#endif
    {
        size_t conv_out_rows = CELL_LEN - 3 + 1;
        size_t pool_rows = conv_out_rows / 2;
        size_t fc_input_size = 16 * pool_rows * pool_rows;
        size_t fc_arch[] = { fc_input_size, 128, 64, 10 };

        conv = Conv_alloc(1, 16, 3);
        nn = NN_alloc(fc_arch, sizeof(fc_arch) / sizeof(*fc_arch));

        NN_xavier_init(nn);

        for (size_t i = 0; i < nn.count; i++)
            Mat_fill(nn.bs[i], 0.01f);

        for (size_t i = 0; i < conv.out_channels * conv.in_channels; i++)
            Mat_rand(conv.kernels[i], -0.1f, 0.1f);

        for (size_t i = 0; i < conv.out_channels; i++)
            MAT_AT(conv.biases[i], 0, 0) = 0.0f;
    }

    // Build grad_fc
    if (nn.count > 0 && nn.as != NULL)
    {
        size_t* arch = malloc(sizeof(size_t) * (nn.count + 1));

        if (!arch)
        {
            fprintf(stderr, "Memory allocation failed for arch\n");

            goto cleanup;
        }

        arch[0] = nn.as[0].cols;

        for (size_t i = 0; i < nn.count; i++)
            arch[i + 1] = nn.ws[i].cols;

        grad_fc = NN_alloc(arch, nn.count + 1);

        NN_zero_grad(grad_fc);
        free(arch);
    }
    else
    {
        fprintf(stderr, "Error: NN not initialized!\n");

        goto cleanup;
    }

    // --- Training loop (trains on the augmented TRAIN set only) ---
    const float lr = 3.0f;
    const int epochs = 500;

    for (int e = 1; e <= epochs; e++)
        CNN_train_epoch(nn, grad_fc, &conv, train.inputs, train.labels, lr, e, epochs);

    // --- Save model ---
    if (fopen_s(&out_file, "data/model.cnn", "wb") == 0 && out_file)
    {
        CNN_save(out_file, conv, nn);

        fclose(out_file);
        out_file = NULL;

        printf("Model saved to 'data/model.cnn'\n");
    }

cleanup:
    if (model_file) fclose(model_file);
    if (out_file) fclose(out_file);

    if (nn.as) NN_free(&nn);
    if (grad_fc.as) NN_free(&grad_fc);

    Conv_free(&conv);

    Dataset_free(&all);
    Dataset_free(&train_raw);
    Dataset_free(&train);
    Dataset_free(&val);
    Dataset_free(&test);

    return EXIT_SUCCESS;
}
