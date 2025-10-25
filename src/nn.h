#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <float.h>
#include <assert.h>
#include <string.h>

// -------------------------
// Utilities
// -------------------------
float rand_float(void);

float drelu(float x);

// -------------------------
// Matrix
// -------------------------
typedef struct
{
    size_t rows;
    size_t cols;
    size_t stride;

    float* es;
} Mat;

#define MAT_AT(m,r,c) (m).es[(r)*(m).stride + (c)]

Mat Mat_alloc(size_t rows, size_t cols);
void Mat_free(Mat m);

void Mat_fill(Mat m, float x);
void Mat_copy(Mat dst, Mat src);
void Mat_rand(Mat m, float low, float high);

Mat Mat_row(Mat m, size_t row);
void Mat_dot(Mat dst, Mat m1, Mat m2);
void Mat_outer(Mat dst, Mat a, Mat b);
void Mat_sum(Mat dst, Mat m);
void Mat_resize(Mat* m, size_t rows, size_t cols);
Mat Mat_transpose(Mat m);

void Mat_relu_inplace(Mat m);
void Mat_softmax_inplace(Mat m);

void Mat_save(FILE* out, Mat m);
Mat Mat_load(FILE* in);

// -------------------------
// Convolution Layer
// -------------------------
typedef struct
{
    size_t in_channels;
    size_t out_channels;
    size_t kernel_size;

    Mat* kernels;
    Mat* biases;
} ConvLayer;

ConvLayer Conv_alloc(size_t in_channels, size_t out_channels, size_t kernel_size);
void Conv_free(ConvLayer* conv);

void Conv_compute_kernel_grad(Mat input, Mat d_out, Mat kernel_grad);
void Conv_forward_single(Mat kernel, Mat input, Mat* output);
void Conv_forward(ConvLayer conv, Mat* input_channels, Mat* output_channels);

void Conv_save(FILE* out, ConvLayer conv);
ConvLayer Conv_load(FILE* in);

// -------------------------
// Pooling & Flatten
// -------------------------
Mat Pool2D(Mat input, size_t pool_size, size_t stride);
void MaxPool2D_backprop(Mat pooled_grad, Mat pooled, Mat conv_out, Mat* d_conv_out);

Mat Flatten(Mat* channels, size_t channel_count);

// -------------------------
// Fully Connected NN
// -------------------------
typedef struct
{
    size_t count;

    Mat* ws;
    Mat* bs;
    Mat* as;

    size_t conv_count;
    ConvLayer* convs;
} NN;

#define NN_INPUT(nn) (nn).as[0]
#define NN_OUTPUT(nn) (nn).as[(nn).count]

NN NN_alloc(size_t* arch, size_t arch_count);
void NN_free(NN* nn);

void NN_zero_grad(NN grad);
void NN_xavier_init(NN nn);

void NN_forward(NN nn);
void NN_learn(NN nn, NN grad, float lr);

void NN_save(FILE* out, NN nn);
NN NN_load(FILE* in);

// -------------------------
// CNN API
// -------------------------
void CNN_forward(NN nn, ConvLayer conv, Mat input_image, Mat* conv_out, Mat* pooled, Mat* flat);
void CNN_backprop(NN nn, NN grad, Mat flat, Mat label, Mat* conv_out, Mat* pooled, ConvLayer* conv);

void CNN_train_epoch(NN nn, NN grad_fc, ConvLayer* conv, Mat inputs, Mat labels, float lr, int epoch_num, int total_epochs);

void CNN_save(FILE* file, ConvLayer conv, NN nn);
void CNN_load(ConvLayer* conv, NN* nn, FILE* file);