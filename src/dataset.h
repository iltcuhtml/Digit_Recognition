#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "nn.h"

#define DS_IMG_LEN  28
#define DS_PIXELS   (DS_IMG_LEN * DS_IMG_LEN)
#define DS_CLASSES  10

typedef struct
{
    size_t count;

    Mat inputs;     // count x 784, values 0~1
    Mat labels;     // count x 10, one-hot
    int* digits;    // count, label as integer (0~9)
} Dataset;

// Load 'data/number.dat'. Label of sample i is (i % 10), same rule as before.
// Returns count == 0 on failure.
Dataset Dataset_load(const char* path);

void Dataset_free(Dataset* d);

// Stratified split: every digit is split with the same ratio.
// train_ratio + val_ratio < 1, the rest becomes the test set.
// The same seed always gives the same split.
void Dataset_split(
    Dataset all,
    float train_ratio, float val_ratio, uint32_t seed,
    Dataset* train, Dataset* val, Dataset* test
);

// Returns a new dataset: each original sample + 'copies' rotated/shifted versions.
// Apply this ONLY to the training set (after the split).
Dataset Dataset_augment(Dataset src, int copies, uint32_t seed);

// Debug helpers
void Dataset_print_summary(const char* name, Dataset d);
void Dataset_print_ascii(Dataset d, size_t index);
