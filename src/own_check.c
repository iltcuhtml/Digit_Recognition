#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "nn.h"
#include "dataset.h"

#define IMG DS_IMG_LEN

typedef void (*Fn)(const float*, float*);

static float at(const float* img, int x, int y)
{
    if (x < 0 || x >= IMG || y < 0 || y >= IMG)
        return 0.0f;

    return img[y * IMG + x];
}

static float bilin(const float* img, float x, float y)
{
    int x0 = (int)floorf(x), y0 = (int)floorf(y);
    float fx = x - (float)x0, fy = y - (float)y0;

    return (at(img, x0, y0) * (1.0f - fx) + at(img, x0 + 1, y0) * fx) * (1.0f - fy)
        + (at(img, x0, y0 + 1) * (1.0f - fx) + at(img, x0 + 1, y0 + 1) * fx) * fy;
}

// dst(x, y) = src(sx0 + (x - 13.5) / scale, sy0 + (y - 13.5) / scale)
static void resample(const float* src, float* dst, float sx0, float sy0, float scale)
{
    const float c = (IMG - 1) / 2.0f;

    for (int y = 0; y < IMG; y++)
        for (int x = 0; x < IMG; x++)
            dst[y * IMG + x] = bilin(src, sx0 + ((float)x - c) / scale, sy0 + ((float)y - c) / scale);
}

static void center_of_mass(const float* img, float* cx, float* cy)
{
    double s = 0.0, sx = 0.0, sy = 0.0;

    for (int y = 0; y < IMG; y++)
        for (int x = 0; x < IMG; x++)
        {
            double v = img[y * IMG + x];

            s += v;
            sx += v * x;
            sy += v * y;
        }

    *cx = s > 0.0 ? (float)(sx / s) : (IMG - 1) / 2.0f;
    *cy = s > 0.0 ? (float)(sy / s) : (IMG - 1) / 2.0f;
}

static void norm_shift(const float* src, float* dst)
{
    float cx, cy;

    center_of_mass(src, &cx, &cy);
    resample(src, dst, cx, cy, 1.0f);
}

static void norm_box(const float* src, float* dst)
{
    int x0 = IMG, x1 = -1, y0 = IMG, y1 = -1;

    for (int y = 0; y < IMG; y++)
        for (int x = 0; x < IMG; x++)
            if (src[y * IMG + x] > 0.2f)
            {
                if (x < x0) x0 = x;
                if (x > x1) x1 = x;
                if (y < y0) y0 = y;
                if (y > y1) y1 = y;
            }

    if (x1 < x0 || y1 < y0)
    {
        memcpy(dst, src, sizeof(float) * IMG * IMG);
        return;
    }

    float w = (float)(x1 - x0 + 1), h = (float)(y1 - y0 + 1);
    float m = w > h ? w : h;

    float tmp[IMG * IMG];

    resample(src, tmp, (float)(x0 + x1) / 2.0f, (float)(y0 + y1) / 2.0f, 20.0f / m);
    norm_shift(tmp, dst);
}

static void erode(const float* src, float* dst)
{
    for (int y = 0; y < IMG; y++)
        for (int x = 0; x < IMG; x++)
        {
            float m = 1.0f;

            for (int dy = -1; dy <= 1; dy++)
                for (int dx = -1; dx <= 1; dx++)
                {
                    float v = at(src, x + dx, y + dy);

                    if (v < m) m = v;
                }

            dst[y * IMG + x] = m;
        }
}

static Dataset make_variant(Dataset src, Fn f1, Fn f2)
{
    Dataset d = Dataset_subset(src, 0, src.count);

    float a[IMG * IMG] = { 0.0f };
    float b[IMG * IMG] = { 0.0f };

    for (size_t i = 0; i < d.count; i++)
    {
        for (size_t p = 0; p < DS_PIXELS; p++)
            a[p] = MAT_AT(d.inputs, i, p);

        if (f1) { f1(a, b); memcpy(a, b, sizeof(a)); }
        if (f2) { f2(a, b); memcpy(a, b, sizeof(a)); }

        for (size_t p = 0; p < DS_PIXELS; p++)
            MAT_AT(d.inputs, i, p) = a[p];
    }

    return d;
}

// average shape statistics of the first n images
static void shape_stats(Dataset d, size_t n, double* ink, double* w, double* h, double* run)
{
    if (n > d.count) n = d.count;

    double s_ink = 0, s_w = 0, s_h = 0, s_run = 0;

    for (size_t i = 0; i < n; i++)
    {
        const float* img = &MAT_AT(d.inputs, i, 0);

        int x0 = IMG, x1 = -1, y0 = IMG, y1 = -1;
        double runs = 0, run_cnt = 0;

        for (int y = 0; y < IMG; y++)
        {
            int len = 0;

            for (int x = 0; x < IMG; x++)
            {
                float v = img[y * IMG + x];

                s_ink += v;

                if (v > 0.2f)
                {
                    if (x < x0) x0 = x;
                    if (x > x1) x1 = x;
                    if (y < y0) y0 = y;
                    if (y > y1) y1 = y;
                }

                if (v > 0.5f)
                    len++;
                else if (len > 0)
                {
                    runs += len;
                    run_cnt++;
                    len = 0;
                }
            }

            if (len > 0)
            {
                runs += len;
                run_cnt++;
            }
        }

        s_w += x1 >= x0 ? x1 - x0 + 1 : 0;
        s_h += y1 >= y0 ? y1 - y0 + 1 : 0;
        s_run += run_cnt > 0 ? runs / run_cnt : 0;
    }

    *ink = s_ink / (double)n;
    *w = s_w / (double)n;
    *h = s_h / (double)n;
    *run = s_run / (double)n;
}

static void report(const char* name, NN nn, ConvLayer conv, Dataset d, int evaluate)
{
    double ink, w, h, run;
    float loss = 0.0f, acc = 0.0f;

    shape_stats(d, d.count < 2000 ? d.count : 2000, &ink, &w, &h, &run);

    if (evaluate)
        CNN_evaluate(nn, conv, d.inputs, d.labels, &loss, &acc);

    printf("%-10s | ink %6.1f | box %4.1f x %4.1f | stroke run %4.2f | ", name, ink, w, h, run);

    if (evaluate)
        printf("acc %5.1f%% | loss %.3f\n", (double)acc * 100, (double)loss);
    else
        printf("(reference)\n");
}

int main(int argc, char** argv)
{
    FILE* f = fopen(argc > 1 ? argv[1] : "data/model_mnist_aug.cnn", "rb");

    if (!f)
    {
        printf("data/model_mnist.cnn not found\n");
        return EXIT_FAILURE;
    }

    ConvLayer conv = { 0 };
    NN nn = { 0 };

    CNN_load(&conv, &nn, f);
    fclose(f);

    if (!conv.kernels || !nn.as)
        return EXIT_FAILURE;

    Dataset own = Dataset_load("data/number.dat");
    Dataset mn = Dataset_load_mnist("data/mnist/t10k-images-idx3-ubyte", "data/mnist/t10k-labels-idx1-ubyte", 2000);

    if (own.count == 0 || mn.count == 0)
        return EXIT_FAILURE;

    Dataset v_shift = make_variant(own, norm_shift, NULL);
    Dataset v_box = make_variant(own, norm_box, NULL);
    Dataset v_thin = make_variant(own, erode, NULL);
    Dataset v_both = make_variant(own, norm_box, erode);

    printf("\n");
    report("MNIST", nn, conv, mn, 0);
    report("own raw", nn, conv, own, 1);
    report("shift", nn, conv, v_shift, 1);
    report("box", nn, conv, v_box, 1);
    report("thin", nn, conv, v_thin, 1);
    report("box+thin", nn, conv, v_both, 1);

    printf("\n--- own[1] raw ---\n");
    Dataset_print_ascii(own, 1);

    printf("\n--- own[1] box+thin ---\n");
    Dataset_print_ascii(v_both, 1);

    Dataset_free(&v_shift);
    Dataset_free(&v_box);
    Dataset_free(&v_thin);
    Dataset_free(&v_both);
    Dataset_free(&own);
    Dataset_free(&mn);

    return EXIT_SUCCESS;
}