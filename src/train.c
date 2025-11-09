#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include "nn.h"

#define CELL_LEN 28

int main()
{
    FILE* file = NULL;
    FILE* model_file = NULL;
    FILE* out_file = NULL;

    Mat train_inputs = { 0 }, train_labels = { 0 };
    NN nn = { 0 }, grad_fc = { 0 };
    ConvLayer conv = { 0 };
    uint8_t* raw_images = NULL;

    // --- Load dataset ---
    if (fopen_s(&file, "data/number.dat", "rb") != 0 || !file)
    {
        printf("Failed to open 'data/number.dat'\n");

        goto cleanup;
    }

    char header[7];

    if (fread(header, sizeof(char), 7, file) != 7 || memcmp(header, "NUMDATA", 7) != 0)
    {
        printf("File header mismatch\n");

        goto cleanup;
    }

    size_t sample_count = 0;

    if (fread(&sample_count, sizeof(size_t), 1, file) != 1 || sample_count == 0)
    {
        printf("No data in 'data/number.dat'\n");

        goto cleanup;
    }

    printf("Dataset loaded, %zu samples\n", sample_count);

    const int input_size = CELL_LEN * CELL_LEN;
    const int num_classes = 10;

    raw_images = (uint8_t*)malloc(sizeof(uint8_t) * sample_count * input_size);
    
    if (!raw_images)
    {
        printf("Memory allocation failed\n");
        
        goto cleanup;
    }

    if (fread(raw_images, sizeof(uint8_t), sample_count * input_size, file) != sample_count * input_size)
    {
        printf("Failed to read image data\n");
        
        goto cleanup;
    }

    fclose(file);
    file = NULL;

    train_inputs = Mat_alloc(sample_count, input_size);

    for (size_t i = 0; i < sample_count; i++)
        for (size_t j = 0; j < input_size; j++)
            MAT_AT(train_inputs, i, j) = raw_images[i * input_size + j] / 255.0f;

    free(raw_images);
    raw_images = NULL;

    train_labels = Mat_alloc(sample_count, num_classes);

    for (size_t i = 0; i < sample_count; i++)
    {
        Mat row = Mat_row(train_labels, i);

        for (int j = 0; j < num_classes; j++)
            MAT_AT(row, 0, j) = 0.0f;

        MAT_AT(row, 0, i % num_classes) = 1.0f;
    }

    // --- Load or initialize model ---
    if (fopen_s(&model_file, "data/model.cnn", "rb") == 0 && model_file)
    {
        CNN_load(&conv, &nn, model_file);

        fclose(model_file);
        model_file = NULL;

        printf("Existing model loaded, continuing training\n");
    }
    else
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

    // --- Training loop ---
    const float lr = 3.0f;
    const int epochs = 500;
    
    for (int e = 1; e <= epochs; e++)
        CNN_train_epoch(nn, grad_fc, &conv, train_inputs, train_labels, lr, e, epochs);

    // --- Save model ---
    if (fopen_s(&out_file, "data/model.cnn", "wb") == 0 && out_file)
    {
        CNN_save(out_file, conv, nn);
        
        fclose(out_file);
        out_file = NULL;
        
        printf("Model saved to 'data/model.cnn'\n");
    }

cleanup:
    if (file) fclose(file);
    if (model_file) fclose(model_file);
    if (out_file) fclose(out_file);
    if (raw_images) free(raw_images);

    NN_free(&nn);
    NN_free(&grad_fc);

    Mat_free(train_inputs);
    Mat_free(train_labels);

    Conv_free(&conv);

    return EXIT_SUCCESS;
}