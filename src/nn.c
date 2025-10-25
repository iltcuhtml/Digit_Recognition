#include "nn.h"

// -------------------------
// Utilities
// -------------------------
float rand_float(void)
{
    return (float)rand() / (float)RAND_MAX;
}

float drelu(float x)
{
    return x > 0.0f ? 1.0f : 0.0f;
}

// -------------------------
// Matrix
// -------------------------
Mat Mat_alloc(size_t rows, size_t cols)
{
    Mat m = { rows, cols, cols, malloc(sizeof(float) * rows * cols) };
    
    assert(m.es != NULL);
    
    return m;
}

void Mat_free(Mat m)
{
    if (m.es)
        free(m.es);
}

void Mat_fill(Mat m, float x)
{
    for (size_t i = 0; i < m.rows; i++)
        for (size_t j = 0; j < m.cols; j++)
            MAT_AT(m, i, j) = x;
}

void Mat_copy(Mat dst, Mat src)
{
    assert(dst.rows == src.rows && dst.cols == src.cols);
    
    for (size_t i = 0; i < dst.rows; i++)
        for (size_t j = 0; j < dst.cols; j++)
            MAT_AT(dst, i, j) = MAT_AT(src, i, j);
}

void Mat_rand(Mat m, float low, float high)
{
    for (size_t i = 0; i < m.rows; i++)
        for (size_t j = 0; j < m.cols; j++)
            MAT_AT(m, i, j) = rand_float() * (high - low) + low;
}

Mat Mat_row(Mat m, size_t row)
{
    return (Mat)
    {
        .rows = 1,
        .cols = m.cols,
        .stride = m.stride,
        .es = &MAT_AT(m, row, 0)
    };
}

void Mat_dot(Mat dst, Mat m1, Mat m2)
{
    assert(m1.cols == m2.rows && dst.rows == m1.rows && dst.cols == m2.cols);
    
    Mat_fill(dst, 0);
    
    for (size_t i = 0; i < dst.rows; i++)
        for (size_t j = 0; j < dst.cols; j++)
            for (size_t k = 0; k < m1.cols; k++)
                MAT_AT(dst, i, j) += MAT_AT(m1, i, k) * MAT_AT(m2, k, j);
}

void Mat_outer(Mat dst, Mat a, Mat b)
{
    assert(a.rows == 1 && b.rows == 1);
    assert(dst.rows == a.cols && dst.cols == b.cols);
    
    for (size_t i = 0; i < dst.rows; i++)
        for (size_t j = 0; j < dst.cols; j++)
            MAT_AT(dst, i, j) = MAT_AT(a, 0, i) * MAT_AT(b, 0, j);
}

void Mat_sum(Mat dst, Mat m)
{
    assert(dst.rows == m.rows && dst.cols == m.cols);
    
    for (size_t i = 0; i < dst.rows; i++)
        for (size_t j = 0; j < dst.cols; j++)
            MAT_AT(dst, i, j) += MAT_AT(m, i, j);
}

void Mat_resize(Mat* m, size_t rows, size_t cols)
{
    if (m->es)
        free(m->es);
    
    m->es = malloc(sizeof(float) * rows * cols);
    
    m->rows = rows;
    m->cols = cols;
    
    m->stride = cols;
}

Mat Mat_transpose(Mat m)
{
    Mat t = Mat_alloc(m.cols, m.rows);
    
    for (size_t i = 0; i < m.rows; i++)
        for (size_t j = 0; j < m.cols; j++)
            MAT_AT(t, j, i) = MAT_AT(m, i, j);
    
    return t;
}

void Mat_relu_inplace(Mat m)
{
    for (size_t i = 0; i < m.rows; i++)
        for (size_t j = 0; j < m.cols; j++)
            if (MAT_AT(m, i, j) < 0)
                MAT_AT(m, i, j) = 0;
}

void Mat_softmax_inplace(Mat m)
{
    for (size_t i = 0; i < m.rows; i++)
    {
        float max_val = -FLT_MAX;
        
        for (size_t j = 0; j < m.cols; j++)
            if (MAT_AT(m, i, j) > max_val)
                max_val = MAT_AT(m, i, j);

        float sum = 0.0f;
        
        for (size_t j = 0; j < m.cols; j++)
        {
            MAT_AT(m, i, j) = expf(MAT_AT(m, i, j) - max_val);
            
            sum += MAT_AT(m, i, j);
        }

        for (size_t j = 0; j < m.cols; j++)
            MAT_AT(m, i, j) /= sum;
    }
}

void Mat_save(FILE* out, Mat m)
{
    fwrite("MATDATA", sizeof(char), 7, out);

    fwrite(&m.rows, sizeof(size_t), 1, out);
    fwrite(&m.cols, sizeof(size_t), 1, out);

    fwrite(m.es, sizeof(float), m.rows * m.cols, out);
}

Mat Mat_load(FILE* in)
{
    char header[7];
    fread(header, sizeof(char), 7, in);

    assert(memcmp(header, "MATDATA", 7) == 0);

    size_t rows, cols;

    fread(&rows, sizeof(size_t), 1, in);
    fread(&cols, sizeof(size_t), 1, in);

    Mat m = Mat_alloc(rows, cols);

    fread(m.es, sizeof(float), rows * cols, in);

    return m;
}

// -------------------------
// Conv Layer
// -------------------------
ConvLayer Conv_alloc(size_t in_channels, size_t out_channels, size_t kernel_size)
{
    ConvLayer conv = { 0 };
    
    conv.in_channels = in_channels;
    conv.out_channels = out_channels;
    conv.kernel_size = kernel_size;

    conv.kernels = malloc(sizeof(Mat) * in_channels * out_channels);
    conv.biases = malloc(sizeof(Mat) * out_channels);

    for (size_t oc = 0; oc < out_channels; oc++)
    {
        conv.biases[oc] = Mat_alloc(1, 1);
        
        MAT_AT(conv.biases[oc], 0, 0) = 0;

        for (size_t ic = 0; ic < in_channels; ic++)
            conv.kernels[oc * in_channels + ic] = Mat_alloc(kernel_size, kernel_size);
    }

    return conv;
}

void Conv_free(ConvLayer* conv)
{
    if (!conv)
        return;

    if (conv->kernels)
    {
        for (size_t i = 0; i < conv->out_channels * conv->in_channels; i++)
            Mat_free(conv->kernels[i]);
        
        free(conv->kernels);
    }

    if (conv->biases)
    {
        for (size_t i = 0; i < conv->out_channels; i++)
            Mat_free(conv->biases[i]);
        
        free(conv->biases);
    }

    conv->kernels = NULL;
    conv->biases = NULL;
    
    conv->in_channels = conv->out_channels = conv->kernel_size = 0;
}

void Conv_compute_kernel_grad(Mat input, Mat d_out, Mat kernel_grad)
{
    // input: original input feature map corresponding to the input channel
    // d_out: gradients for convolution output (out_rows x out_cols)
    // kernel_grad: k x k, will be accumulated

    size_t k = kernel_grad.rows;

    size_t out_rows = d_out.rows;
    size_t out_cols = d_out.cols;

    // zero kernel_grad
    Mat_fill(kernel_grad, 0.0f);

    for (size_t y = 0; y < out_rows; y++)
        for (size_t x = 0; x < out_cols; x++)
        {
            float dout = MAT_AT(d_out, y, x);

            for (size_t ky = 0; ky < k; ky++)
                for (size_t kx = 0; kx < k; kx++)
                    // input patch starts at (y, x)
                    MAT_AT(kernel_grad, ky, kx) += MAT_AT(input, y + ky, x + kx) * dout;
        }
}

void Conv_forward_single(Mat kernel, Mat input, Mat* output)
{
    size_t out_rows = input.rows - kernel.rows + 1;
    size_t out_cols = input.cols - kernel.cols + 1;

    *output = Mat_alloc(out_rows, out_cols);

    for (size_t y = 0; y < out_rows; y++)
        for (size_t x = 0; x < out_cols; x++)
        {
            float sum = 0.0f;
            
            for (size_t ky = 0; ky < kernel.rows; ky++)
                for (size_t kx = 0; kx < kernel.cols; kx++)
                    sum += MAT_AT(input, y + ky, x + kx) * MAT_AT(kernel, ky, kx);

            MAT_AT(*output, y, x) = sum;
        }
}

void Conv_forward(ConvLayer conv, Mat* input_channels, Mat* output_channels)
{
    for (size_t oc = 0; oc < conv.out_channels; oc++)
    {
        output_channels[oc] = Mat_alloc(
            input_channels[0].rows - conv.kernel_size + 1,
            input_channels[0].cols - conv.kernel_size + 1
        );

        Mat_fill(output_channels[oc], MAT_AT(conv.biases[oc], 0, 0));

        for (size_t ic = 0; ic < conv.in_channels; ic++)
        {
            Mat temp;
            
            Conv_forward_single(conv.kernels[oc * conv.in_channels + ic], input_channels[ic], &temp);
            
            Mat_sum(output_channels[oc], temp);
            Mat_free(temp);
        }

        Mat_relu_inplace(output_channels[oc]);
    }
}

void Conv_save(FILE* out, ConvLayer conv)
{
    fwrite("CONVLAYR", sizeof(char), 8, out);

    fwrite(&conv.in_channels, sizeof(size_t), 1, out);
    fwrite(&conv.out_channels, sizeof(size_t), 1, out);
    fwrite(&conv.kernel_size, sizeof(size_t), 1, out);

    for (size_t i = 0; i < conv.out_channels * conv.in_channels; i++)
        Mat_save(out, conv.kernels[i]);

    for (size_t i = 0; i < conv.out_channels; i++)
        Mat_save(out, conv.biases[i]);
}

ConvLayer Conv_load(FILE* in)
{
    char header[8];
    fread(header, sizeof(char), 8, in);

    assert(memcmp(header, "CONVLAYR", 8) == 0);

    ConvLayer conv = { 0 };

    fread(&conv.in_channels, sizeof(size_t), 1, in);
    fread(&conv.out_channels, sizeof(size_t), 1, in);
    fread(&conv.kernel_size, sizeof(size_t), 1, in);

    conv.kernels = malloc(sizeof(Mat) * conv.in_channels * conv.out_channels);
    conv.biases = malloc(sizeof(Mat) * conv.out_channels);

    for (size_t i = 0; i < conv.out_channels * conv.in_channels; i++)
        conv.kernels[i] = Mat_load(in);

    for (size_t i = 0; i < conv.out_channels; i++)
        conv.biases[i] = Mat_load(in);

    return conv;
}

Mat Pool2D(Mat input, size_t pool_size, size_t stride)
{
    size_t out_rows = (input.rows - pool_size) / stride + 1;
    size_t out_cols = (input.cols - pool_size) / stride + 1;

    Mat out = Mat_alloc(out_rows, out_cols);

    for (size_t y = 0; y < out_rows; y++)
        for (size_t x = 0; x < out_cols; x++)
        {
            float max_val = -FLT_MAX;
            
            for (size_t py = 0; py < pool_size; py++)
                for (size_t px = 0; px < pool_size; px++)
                {
                    float v = MAT_AT(input, y * stride + py, x * stride + px);
                    
                    if (v > max_val)
                        max_val = v;
                }

            MAT_AT(out, y, x) = max_val;
        }

    return out;
}

void MaxPool2D_backprop(Mat pooled_grad, Mat pooled, Mat conv_out, Mat* d_conv_out)
{
    // Assumes pool_size = 2 and stride = 2 (matches Pool2D usage in this code).
    Mat_fill(*d_conv_out, 0.0f);

    size_t pool_size = 2;
    size_t stride = 2;

    for (size_t y = 0; y < pooled.rows; y++)
        for (size_t x = 0; x < pooled.cols; x++)
        {
            float max_val = -FLT_MAX;
            size_t max_i = 0, max_j = 0;

            for (size_t py = 0; py < pool_size; py++)
                for (size_t px = 0; px < pool_size; px++)
                {
                    size_t iy = y * stride + py;
                    size_t ix = x * stride + px;

                    float v = MAT_AT(conv_out, iy, ix);

                    if (v > max_val)
                    {
                        max_val = v;
                        max_i = iy;
                        max_j = ix;
                    }
                }

            // add pooled gradient to the location of the max
            MAT_AT(*d_conv_out, max_i, max_j) += MAT_AT(pooled_grad, y, x);
        }
}

Mat Flatten(Mat* channels, size_t channel_count)
{
    size_t total = 0;
    
    for (size_t c = 0; c < channel_count; c++)
        total += channels[c].rows * channels[c].cols;

    Mat out = Mat_alloc(1, total);

    size_t idx = 0;
    
    for (size_t c = 0; c < channel_count; c++)
        for (size_t i = 0; i < channels[c].rows; i++)
            for (size_t j = 0; j < channels[c].cols; j++)
                MAT_AT(out, 0, idx++) = MAT_AT(channels[c], i, j);

    return out;
}

// -------------------------
// Fully Connected NN
// -------------------------
NN NN_alloc(size_t* arch, size_t arch_count)
{
    assert(arch_count >= 2);

    NN nn = { 0 };

    nn.count = arch_count - 1;

    nn.ws = malloc(nn.count * sizeof(Mat));
    nn.bs = malloc(nn.count * sizeof(Mat));
    nn.as = malloc(arch_count * sizeof(Mat));

    nn.as[0] = Mat_alloc(1, arch[0]);

    for (size_t i = 0; i < nn.count; i++)
    {
        nn.ws[i] = Mat_alloc(arch[i], arch[i + 1]);
        nn.bs[i] = Mat_alloc(1, arch[i + 1]);
        nn.as[i + 1] = Mat_alloc(1, arch[i + 1]);
    }

    nn.conv_count = 0;
    nn.convs = NULL;

    return nn;
}

void NN_free(NN* nn)
{
    if (!nn)
        return;

    for (size_t i = 0; i < nn->count; i++)
    {
        Mat_free(nn->ws[i]);
        Mat_free(nn->bs[i]);
        Mat_free(nn->as[i + 1]);
    }

    Mat_free(nn->as[0]);

    free(nn->ws);
    free(nn->bs);
    free(nn->as);

    if (nn->convs)
        free(nn->convs);

    nn->ws = nn->bs = nn->as = NULL;
    nn->convs = NULL;
    nn->count = nn->conv_count = 0;
}

void NN_zero_grad(NN grad)
{
    for (size_t i = 0; i < grad.count; i++)
    {
        Mat_fill(grad.ws[i], 0.0f);
        Mat_fill(grad.bs[i], 0.0f);
        Mat_fill(grad.as[i + 1], 0.0f);
    }

    Mat_fill(grad.as[0], 0.0f);
}

void NN_xavier_init(NN nn)
{
    for (size_t i = 0; i < nn.count; i++)
    {
        float limit = sqrtf(6.0f / (nn.ws[i].rows + nn.ws[i].cols));

        for (size_t j = 0; j < nn.ws[i].rows * nn.ws[i].cols; j++)
            nn.ws[i].es[j] = rand_float() * 2.0f * limit - limit;

        Mat_fill(nn.bs[i], 0.0f);
    }
}

void NN_forward(NN nn)
{
    Mat x = NN_INPUT(nn);

    for (size_t i = 0; i < nn.count; i++)
    {
        Mat_fill(nn.as[i + 1], 0);

        Mat_dot(nn.as[i + 1], x, nn.ws[i]);
        Mat_sum(nn.as[i + 1], nn.bs[i]);

        Mat_relu_inplace(nn.as[i + 1]);

        x = nn.as[i + 1];
    }

    Mat_softmax_inplace(NN_OUTPUT(nn));
}

void NN_learn(NN nn, NN grad, float lr)
{
    for (size_t i = 0; i < nn.count; i++)
    {
        for (size_t j = 0; j < nn.ws[i].rows * nn.ws[i].cols; j++)
            nn.ws[i].es[j] -= lr * grad.ws[i].es[j];

        for (size_t j = 0; j < nn.bs[i].rows * nn.bs[i].cols; j++)
            nn.bs[i].es[j] -= lr * grad.bs[i].es[j];
    }
}

void NN_save(FILE* out, NN nn)
{
    fwrite("NNMODEL", sizeof(char), 7, out);

    fwrite(&nn.count, sizeof(size_t), 1, out);
    fwrite(&nn.conv_count, sizeof(size_t), 1, out);

    for (size_t i = 0; i < nn.count; i++)
    {
        Mat_save(out, nn.ws[i]);
        Mat_save(out, nn.bs[i]);
    }

    for (size_t i = 0; i < nn.conv_count; i++)
        Conv_save(out, nn.convs[i]);
}

NN NN_load(FILE* in)
{
    char header[7];
    fread(header, sizeof(char), 7, in);

    assert(memcmp(header, "NNMODEL", 7) == 0);

    NN nn = { 0 };

    fread(&nn.count, sizeof(size_t), 1, in);
    fread(&nn.conv_count, sizeof(size_t), 1, in);

    size_t * arch = malloc(sizeof(size_t) * (nn.count + 1));

    nn.ws = malloc(sizeof(Mat) * nn.count);
    nn.bs = malloc(sizeof(Mat) * nn.count);
    nn.as = malloc(sizeof(Mat) * (nn.count + 1));

    for (size_t i = 0; i < nn.count; i++)
    {
        nn.ws[i] = Mat_load(in);
        nn.bs[i] = Mat_load(in);

        arch[i + 1] = nn.ws[i].cols;
    }

    arch[0] = nn.ws[0].rows;

    nn.as[0] = Mat_alloc(1, arch[0]);

    for (size_t i = 0; i < nn.count; i++)
        nn.as[i + 1] = Mat_alloc(1, arch[i + 1]);

    if (nn.conv_count > 0)
    {
        nn.convs = malloc(sizeof(ConvLayer) * nn.conv_count);
        
        for (size_t i = 0; i < nn.conv_count; i++)
            nn.convs[i] = Conv_load(in);
    }

    free(arch);

    return nn;
}

// -------------------------
// CNN API
// -------------------------
void CNN_forward(NN nn, ConvLayer conv, Mat input_image, Mat* conv_out, Mat* pooled, Mat* flat)
{
    Mat input_channels[1] = { input_image };
    
    Conv_forward(conv, input_channels, conv_out);

    for (size_t c = 0; c < conv.out_channels; c++)
        pooled[c] = Pool2D(conv_out[c], 2, 2);

    *flat = Flatten(pooled, conv.out_channels);

    if (NN_INPUT(nn).cols != flat->cols)
    {
        Mat_free(NN_INPUT(nn));
        
        NN_INPUT(nn) = Mat_alloc(1, flat->cols);
    }

    Mat_copy(NN_INPUT(nn), *flat);
    
    NN_forward(nn);
}

void CNN_backprop(NN nn, NN grad, Mat flat, Mat label, Mat* conv_out, Mat* pooled, ConvLayer* conv)
{
    Mat_copy(NN_INPUT(nn), flat);
    NN_forward(nn);

    for (size_t j = 0; j < NN_OUTPUT(nn).cols; j++)
        MAT_AT(grad.as[nn.count], 0, j) = MAT_AT(NN_OUTPUT(nn), 0, j) - MAT_AT(label, 0, j);

    for (size_t l = nn.count; l-- > 0; )
    {
        Mat* dA = &grad.as[l + 1];
        Mat* A_prev = &nn.as[l];
        Mat* W = &nn.ws[l];
        Mat* dW = &grad.ws[l];
        Mat* db = &grad.bs[l];

        Mat A_prev_T = Mat_transpose(*A_prev);
        
        Mat_dot(*dW, A_prev_T, *dA);
        Mat_free(A_prev_T);

        Mat_copy(*db, *dA);

        Mat W_T = Mat_transpose(*W);
        Mat dA_prev = Mat_alloc(A_prev->rows, A_prev->cols);
        
        Mat_dot(dA_prev, *dA, W_T);
        Mat_free(W_T);

        for (size_t i = 0; i < A_prev->rows; i++)
            for (size_t j = 0; j < A_prev->cols; j++)
                MAT_AT(dA_prev, i, j) *= MAT_AT(*A_prev, i, j) > 0 ? 1.0f : 0.0f;

        Mat_copy(*dA, dA_prev);
        Mat_free(dA_prev);
    }

    Mat* d_conv_out = malloc(sizeof(Mat) * conv->out_channels);

    for (size_t c = 0; c < conv->out_channels; c++)
        d_conv_out[c] = Mat_alloc(conv_out[c].rows, conv_out[c].cols);

    for (size_t c = 0; c < conv->out_channels; c++)
    {
        Mat pooled_grad = Mat_alloc(pooled[c].rows, pooled[c].cols);

        for (size_t i = 0; i < pooled[c].rows; i++)
            for (size_t j = 0; j < pooled[c].cols; j++)
                MAT_AT(pooled_grad, i, j) = MAT_AT(NN_INPUT(nn), 0, c * pooled[c].rows * pooled[c].cols + i * pooled[c].cols + j);

            MaxPool2D_backprop(pooled_grad, pooled[c], conv_out[c], &d_conv_out[c]);
            
            Mat_free(pooled_grad);
        }

    for (size_t oc = 0; oc < conv->out_channels; oc++)
    {
        for (size_t ic = 0; ic < conv->in_channels; ic++)
        {
            Mat* kernel = &conv->kernels[oc * conv->in_channels + ic];
            Mat kernel_grad = Mat_alloc(kernel->rows, kernel->cols);
            
            Mat_fill(kernel_grad, 0.0f);
            
            Conv_compute_kernel_grad(conv_out[ic], d_conv_out[oc], kernel_grad);

            for (size_t k = 0; k < kernel->rows * kernel->cols; k++)
                kernel->es[k] -= 0.01f * kernel_grad.es[k];

            Mat_free(kernel_grad);
        }

        float bias_grad = 0.0f;
        
        for (size_t y = 0; y < d_conv_out[oc].rows; y++)
            for (size_t x = 0; x < d_conv_out[oc].cols; x++)
                bias_grad += MAT_AT(d_conv_out[oc], y, x);

        MAT_AT(conv->biases[oc], 0, 0) -= 0.01f * bias_grad;
    }

    for (size_t c = 0; c < conv->out_channels; c++)
        Mat_free(d_conv_out[c]);

    free(d_conv_out);
}

void CNN_train_epoch(NN nn, NN grad_fc, ConvLayer* conv, Mat inputs, Mat labels, float lr, int epoch_num, int total_epochs)
{
    size_t samples = inputs.rows;
    size_t img_size = (size_t)sqrt((double)inputs.cols);
    
    size_t conv_out_h = img_size - conv->kernel_size + 1;
    size_t conv_out_w = conv_out_h;
    
    size_t pooled_h = conv_out_h / 2;
    size_t pooled_w = conv_out_w / 2;

    Mat* conv_out = malloc(sizeof(Mat) * conv->out_channels);
    Mat* pooled = malloc(sizeof(Mat) * conv->out_channels);
    Mat* d_conv_out = malloc(sizeof(Mat) * conv->out_channels);

    for (size_t c = 0; c < conv->out_channels; c++)
    {
        conv_out[c] = Mat_alloc(conv_out_h, conv_out_w);
        pooled[c] = Mat_alloc(pooled_h, pooled_w);
        d_conv_out[c] = Mat_alloc(conv_out_h, conv_out_w);
    }

    float epoch_cost = 0.0f;
    size_t correct = 0;

    for (size_t i = 0; i < samples; i++)
    {
        Mat input_row = Mat_row(inputs, i);
        Mat input_image = Mat_alloc(img_size, img_size);

        for (size_t y = 0; y < img_size; y++)
            for (size_t x = 0; x < img_size; x++)
                MAT_AT(input_image, y, x) = MAT_AT(input_row, 0, y * img_size + x);

        Mat label_row = Mat_row(labels, i);

        // forward
        Mat flat;
        CNN_forward(nn, *conv, input_image, conv_out, pooled, &flat);
        
        // zero grad
        NN_zero_grad(grad_fc);

        // output gradient (cross-entropy with softmax)
        for (size_t j = 0; j < NN_OUTPUT(nn).cols; j++)
            MAT_AT(grad_fc.as[nn.count], 0, j) = MAT_AT(NN_OUTPUT(nn), 0, j) - MAT_AT(label_row, 0, j);

        // backprop through FC layers (standard)
        for (size_t l = nn.count; l-- > 0; )
        {
            Mat* dA = &grad_fc.as[l + 1];
            Mat* A_prev = &nn.as[l];
            Mat* W = &nn.ws[l];
            Mat* dW = &grad_fc.ws[l];
            Mat* db = &grad_fc.bs[l];

            // dW = A_prev^T dot dA   (A_prev: 1 x n_prev => transpose n_prev x 1)
            Mat A_prev_T = Mat_transpose(*A_prev);
            Mat_dot(*dW, A_prev_T, *dA);
            Mat_free(A_prev_T);

            // db = dA (since batch size = 1)
            Mat_copy(*db, *dA);

            // dA_prev = dA dot W^T
            Mat W_T = Mat_transpose(*W);
            Mat dA_prev = Mat_alloc(A_prev->rows, A_prev->cols);
            Mat_dot(dA_prev, *dA, W_T);
            Mat_free(W_T);

            // apply ReLU derivative on A_prev
            for (size_t y = 0; y < A_prev->rows; y++)
                for (size_t x = 0; x < A_prev->cols; x++)
                    MAT_AT(dA_prev, y, x) *= (MAT_AT(*A_prev, y, x) > 0.0f) ? 1.0f : 0.0f;

            Mat_copy(*dA, dA_prev);
            Mat_free(dA_prev);
        }

        // map fc input gradients (grad_fc.as[0]) back to pooled channel grads
        size_t idx = 0;

        for (size_t c = 0; c < conv->out_channels; c++)
        {
            Mat pooled_grad = Mat_alloc(pooled[c].rows, pooled[c].cols);

            for (size_t y = 0; y < pooled[c].rows; y++)
                for (size_t x = 0; x < pooled[c].cols; x++)
                    MAT_AT(pooled_grad, y, x) = MAT_AT(grad_fc.as[0], 0, idx++);

            // unpool -> d_conv_out[c]
            MaxPool2D_backprop(pooled_grad, pooled[c], conv_out[c], &d_conv_out[c]);

            Mat_free(pooled_grad);
        }

        // update conv kernels using original input_image as input to conv (single-layer conv)
        for (size_t oc = 0; oc < conv->out_channels; oc++)
        {
            for (size_t ic = 0; ic < conv->in_channels; ic++)
            {
                // for single-layer conv with raw image inputs, input is input_image
                // if multi-channel input, the calling code must provide correct input channels
                Mat* kernel = &conv->kernels[oc * conv->in_channels + ic];
                Mat kernel_grad = Mat_alloc(kernel->rows, kernel->cols);
                
                Mat_fill(kernel_grad, 0.0f);

                // IMPORTANT: use the original input for kernel gradient.
                // If conv->in_channels == 1 => use input_image
                // If >1, user must pass per-channel inputs; here assume ic==0 uses input_image,
                // for other channels we treat as zero (or duplicate). For single-layer MNIST this suffices.
                Conv_compute_kernel_grad(input_image, d_conv_out[oc], kernel_grad);

                for (size_t k = 0; k < kernel->rows * kernel->cols; k++)
                    kernel->es[k] -= lr * kernel_grad.es[k];

                Mat_free(kernel_grad);
            }

            // bias update
            float bias_grad = 0.0f;

            for (size_t y = 0; y < d_conv_out[oc].rows; y++)
                for (size_t x = 0; x < d_conv_out[oc].cols; x++)
                    bias_grad += MAT_AT(d_conv_out[oc], y, x);

            MAT_AT(conv->biases[oc], 0, 0) -= lr * bias_grad;
        }

        // update FC params
        NN_learn(nn, grad_fc, lr);

        // cost + accuracy
        for (size_t j = 0; j < NN_OUTPUT(nn).cols; j++)
        {
            float y = MAT_AT(label_row, 0, j);
            float p = MAT_AT(NN_OUTPUT(nn), 0, j);

            epoch_cost -= y * logf(fmaxf(p, 1e-7f));
        }

        size_t max_idx = 0;
        float max_val = MAT_AT(NN_OUTPUT(nn), 0, 0);

        for (size_t j = 1; j < NN_OUTPUT(nn).cols; j++)
            if (MAT_AT(NN_OUTPUT(nn), 0, j) > max_val)
            {
                max_val = MAT_AT(NN_OUTPUT(nn), 0, j);

                max_idx = j;
            }

        if (MAT_AT(label_row, 0, max_idx) == 1.0f)
            correct++;

        Mat_free(input_image);
        Mat_free(flat);
    }

    epoch_cost /= (float)samples;
    float acc = (float)correct / (float)samples;

    printf("Epoch %d/%d, cost = %.4f, accuracy = %.2f\n", epoch_num, total_epochs, epoch_cost, acc);

    for (size_t c = 0; c < conv->out_channels; c++)
    {
        Mat_free(conv_out[c]);
        Mat_free(pooled[c]);
        Mat_free(d_conv_out[c]);
    }

    free(conv_out);
    free(pooled);
    free(d_conv_out);
}

void CNN_save(FILE* file, ConvLayer conv, NN nn)
{
    if (!file)
        return;

    fwrite("CNNMODEL", sizeof(char), 8, file);

    fwrite(&conv.in_channels, sizeof(size_t), 1, file);
    fwrite(&conv.out_channels, sizeof(size_t), 1, file);
    fwrite(&conv.kernel_size, sizeof(size_t), 1, file);

    for (size_t i = 0; i < conv.out_channels * conv.in_channels; i++)
        Mat_save(file, conv.kernels[i]);

    for (size_t i = 0; i < conv.out_channels; i++)
        Mat_save(file, conv.biases[i]);

    NN_save(file, nn);
}

void CNN_load(ConvLayer* conv, NN* nn, FILE* file)
{
    if (!file)
        return;

    char header[8];
    fread(header, sizeof(char), 8, file);

    if (memcmp(header, "CNNMODEL", 8) != 0)
    {
        fprintf(stderr, "Invalid CNN model file\n");

        return;
    }

    fread(&conv->in_channels, sizeof(size_t), 1, file);
    fread(&conv->out_channels, sizeof(size_t), 1, file);
    fread(&conv->kernel_size, sizeof(size_t), 1, file);

    size_t kernel_count = conv->out_channels * conv->in_channels;

    conv->kernels = malloc(sizeof(Mat) * kernel_count);
    conv->biases = malloc(sizeof(Mat) * conv->out_channels);

    if (!conv->kernels || !conv->biases)
    {
        fprintf(stderr, "Memory allocation failed for conv layer\n");
        
        if (conv->kernels)
            free(conv->kernels);
        
        if (conv->biases)
            free(conv->biases);
        
        conv->kernels = NULL;
        conv->biases = NULL;
        
        return;
    }

    for (size_t i = 0; i < kernel_count; i++)
        conv->kernels[i] = Mat_load(file);

    for (size_t i = 0; i < conv->out_channels; i++)
        conv->biases[i] = Mat_load(file);

    *nn = NN_load(file);
}