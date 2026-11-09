#include "dataset.h"

#include <math.h>
#include <string.h>

#define PI_F 3.14159265f

// -------------------------
// Own RNG (so the split does not depend on rand() used for weight init)
// -------------------------
typedef struct { uint32_t s; } Rng;

static Rng Rng_make(uint32_t seed)
{
    Rng r = { seed ? seed : 0x9E3779B9u };
    return r;
}

static uint32_t Rng_next(Rng* r)
{
    uint32_t x = r->s;

    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;

    r->s = x;
    return x;
}

static float Rng_range(Rng* r, float lo, float hi)
{
    float u = (float)(Rng_next(r) >> 8) / 16777216.0f; // [0, 1)
    return lo + (hi - lo) * u;
}

// -------------------------
// Basic helpers
// -------------------------
static Dataset Dataset_alloc(size_t count)
{
    Dataset d = { 0 };

    if (count == 0)
        return d;

    d.count = count;
    d.inputs = Mat_alloc(count, DS_PIXELS);
    d.labels = Mat_alloc(count, DS_CLASSES);
    d.digits = (int*)malloc(sizeof(int) * count);

    if (!d.digits)
    {
        printf("Memory allocation failed in Dataset_alloc\n");

        Mat_free(d.inputs);
        Mat_free(d.labels);

        memset(&d, 0, sizeof(d));

        return d;
    }

    Mat_fill(d.labels, 0.0f);

    return d;
}

void Dataset_free(Dataset* d)
{
    if (!d)
        return;

    Mat_free(d->inputs);
    Mat_free(d->labels);
    free(d->digits);

    memset(d, 0, sizeof(*d));
}

static void Dataset_set_label(Dataset* d, size_t i, int digit)
{
    d->digits[i] = digit;

    for (int j = 0; j < DS_CLASSES; j++)
        MAT_AT(d->labels, i, j) = (j == digit) ? 1.0f : 0.0f;
}

static void Dataset_copy_sample(Dataset* dst, size_t di, Dataset src, size_t si)
{
    for (size_t j = 0; j < DS_PIXELS; j++)
        MAT_AT(dst->inputs, di, j) = MAT_AT(src.inputs, si, j);

    Dataset_set_label(dst, di, src.digits[si]);
}

static FILE* open_rb(const char* path)
{
    FILE* f = NULL;

#ifdef _MSC_VER
    if (fopen_s(&f, path, "rb") != 0)
        f = NULL;
#else
    f = fopen(path, "rb");
#endif

    return f;
}

static int read_be32(FILE* f, uint32_t* out)
{
    uint8_t b[4];

    if (fread(b, 1, 4, f) != 4)
        return 0;

    *out = ((uint32_t)b[0] << 24) | ((uint32_t)b[1] << 16) | ((uint32_t)b[2] << 8) | (uint32_t)b[3];
    return 1;
}

Dataset Dataset_subset(Dataset src, size_t start, size_t count)
{
    Dataset d = Dataset_alloc(count);

    if (d.count == 0 || start + count > src.count)
    {
        Dataset_free(&d);
        return d;
    }

    for (size_t i = 0; i < count; i++)
        Dataset_copy_sample(&d, i, src, start + i);

    return d;
}

Dataset Dataset_load_mnist(const char* image_path, const char* label_path, size_t limit)
{
    Dataset d = { 0 };

    FILE* fi = open_rb(image_path);
    FILE* fl = open_rb(label_path);

    uint8_t* raw = NULL;
    uint8_t* lab = NULL;

    uint32_t magic_i = 0, n_i = 0, rows = 0, cols = 0, magic_l = 0, n_l = 0;
    size_t n = 0;

    if (!fi || !fl)
    {
        printf("Failed to open MNIST files\n");
        goto done;
    }

    if (!read_be32(fi, &magic_i) || !read_be32(fi, &n_i) || !read_be32(fi, &rows) || !read_be32(fi, &cols) ||
        !read_be32(fl, &magic_l) || !read_be32(fl, &n_l))
    {
        printf("Failed to read MNIST header\n");
        goto done;
    }

    if (magic_i != 2051u || magic_l != 2049u || rows != (uint32_t)DS_IMG_LEN || cols != (uint32_t)DS_IMG_LEN || n_i != n_l)
    {
        printf("MNIST header mismatch\n");
        goto done;
    }

    n = n_i;

    if (limit > 0 && limit < n)
        n = limit;

    raw = (uint8_t*)malloc(n * DS_PIXELS);
    lab = (uint8_t*)malloc(n);

    if (!raw || !lab ||
        fread(raw, 1, n * DS_PIXELS, fi) != n * DS_PIXELS ||
        fread(lab, 1, n, fl) != n)
    {
        printf("Failed to read MNIST data\n");
        goto done;
    }

    d = Dataset_alloc(n);

    if (d.count == 0)
        goto done;

    for (size_t i = 0; i < n; i++)
    {
        if (lab[i] >= DS_CLASSES)
        {
            printf("Invalid MNIST label\n");
            Dataset_free(&d);
            goto done;
        }

        for (size_t j = 0; j < DS_PIXELS; j++)
            MAT_AT(d.inputs, i, j) = raw[i * DS_PIXELS + j] / 255.0f;

        Dataset_set_label(&d, i, (int)lab[i]);
    }

done:
    free(raw);
    free(lab);

    if (fi) fclose(fi);
    if (fl) fclose(fl);

    return d;
}

// -------------------------
// Load
// -------------------------
Dataset Dataset_load(const char* path)
{
    Dataset d = { 0 };
    FILE* f = NULL;

#ifdef _MSC_VER
    if (fopen_s(&f, path, "rb") != 0)
        f = NULL;
#else
    f = fopen(path, "rb");
#endif

    if (!f)
    {
        printf("Failed to open '%s'\n", path);
        return d;
    }

    char header[7];
    size_t n = 0;

    if (fread(header, sizeof(char), 7, f) != 7 || memcmp(header, "NUMDATA", 7) != 0)
    {
        printf("File header mismatch\n");
        fclose(f);
        return d;
    }

    if (fread(&n, sizeof(size_t), 1, f) != 1 || n == 0)
    {
        printf("No data in '%s'\n", path);
        fclose(f);
        return d;
    }

    uint8_t* raw = (uint8_t*)malloc(sizeof(uint8_t) * n * DS_PIXELS);

    if (!raw || fread(raw, sizeof(uint8_t), n * DS_PIXELS, f) != n * DS_PIXELS)
    {
        printf("Failed to read image data\n");
        free(raw);
        fclose(f);
        return d;
    }

    fclose(f);

    d = Dataset_alloc(n);

    if (d.count == 0)
    {
        free(raw);

        return d;
    }

    for (size_t i = 0; i < n; i++)
    {
        for (size_t j = 0; j < DS_PIXELS; j++)
            MAT_AT(d.inputs, i, j) = raw[i * DS_PIXELS + j] / 255.0f;

        Dataset_set_label(&d, i, (int)(i % DS_CLASSES));
    }

    free(raw);

    return d;
}

// -------------------------
// Stratified split
// -------------------------
void Dataset_split(
    Dataset all,
    float train_ratio, float val_ratio, uint32_t seed,
    Dataset* train, Dataset* val, Dataset* test)
{
    Rng rng = Rng_make(seed);

    size_t* idx = (size_t*)malloc(sizeof(size_t) * all.count);
    uint8_t* role = (uint8_t*)calloc(all.count, sizeof(uint8_t)); // 0 train, 1 val, 2 test

    if (!idx || !role)
    {
        printf("Memory allocation failed in Dataset_split\n");

        free(idx);
        free(role);

        memset(train, 0, sizeof(*train));
        memset(val, 0, sizeof(*val));
        memset(test, 0, sizeof(*test));

        return;
    }

    size_t total[3] = { 0, 0, 0 };

    for (int c = 0; c < DS_CLASSES; c++)
    {
        size_t n = 0;

        for (size_t i = 0; i < all.count; i++)
            if (all.digits[i] == c)
                idx[n++] = i;

        // Fisher-Yates shuffle
        for (size_t k = n; k > 1; k--)
        {
            size_t j = Rng_next(&rng) % k;
            size_t tmp = idx[k - 1];

            idx[k - 1] = idx[j];
            idx[j] = tmp;
        }

        size_t n_val = 0, n_test = 0;

        if (n >= 3)
        {
            n_val = (size_t)(n * val_ratio + 0.5f);
            n_test = (size_t)(n * (1.0f - train_ratio - val_ratio) + 0.5f);

            if (n_val < 1) n_val = 1;
            if (n_test < 1) n_test = 1;

            while (n_val + n_test >= n)
            {
                if (n_val > n_test) n_val--;
                else n_test--;
            }
        }

        for (size_t k = 0; k < n; k++)
        {
            if (k < n_val) role[idx[k]] = 1;
            else if (k < n_val + n_test) role[idx[k]] = 2;
            else role[idx[k]] = 0;
        }
    }

    for (size_t i = 0; i < all.count; i++)
        total[role[i]]++;

    *train = Dataset_alloc(total[0]);
    *val = Dataset_alloc(total[1]);
    *test = Dataset_alloc(total[2]);

    size_t pos[3] = { 0, 0, 0 };

    for (size_t i = 0; i < all.count; i++)
    {
        Dataset* dst = (role[i] == 0) ? train : (role[i] == 1) ? val : test;

        Dataset_copy_sample(dst, pos[role[i]]++, all, i);
    }

    free(idx);
    free(role);
}

// -------------------------
// Augmentation (rotation + shift, bilinear interpolation)
// -------------------------
static float pixel_at(const float* img, int x, int y)
{
    if (x < 0 || x >= DS_IMG_LEN || y < 0 || y >= DS_IMG_LEN)
        return 0.0f;

    return img[y * DS_IMG_LEN + x];
}

static float sample_bilinear(const float* img, float x, float y)
{
    int x0 = (int)floorf(x);
    int y0 = (int)floorf(y);

    float fx = x - (float)x0;
    float fy = y - (float)y0;

    float v00 = pixel_at(img, x0, y0);
    float v10 = pixel_at(img, x0 + 1, y0);
    float v01 = pixel_at(img, x0, y0 + 1);
    float v11 = pixel_at(img, x0 + 1, y0 + 1);

    return (v00 * (1.0f - fx) + v10 * fx) * (1.0f - fy)
        + (v01 * (1.0f - fx) + v11 * fx) * fy;
}

// dst(p) = src( R(-angle) * (p - center - shift) + center )
static void transform_image(const float* src, float* dst, float angle, float dx, float dy)
{
    const float c = cosf(angle);
    const float s = sinf(angle);
    const float center = (DS_IMG_LEN - 1) / 2.0f;

    for (int y = 0; y < DS_IMG_LEN; y++)
        for (int x = 0; x < DS_IMG_LEN; x++)
        {
            float xr = (float)x - center - dx;
            float yr = (float)y - center - dy;

            float sx = c * xr + s * yr + center;
            float sy = -s * xr + c * yr + center;

            dst[y * DS_IMG_LEN + x] = sample_bilinear(src, sx, sy);
        }
}

void Dataset_random_transform(Dataset src, Dataset dst, float max_deg, float max_shift, uint32_t seed)
{
    Rng rng = Rng_make(seed);

    for (size_t i = 0; i < src.count && i < dst.count; i++)
    {
        float angle = Rng_range(&rng, -max_deg, max_deg) * PI_F / 180.0f;
        float dx = Rng_range(&rng, -max_shift, max_shift);
        float dy = Rng_range(&rng, -max_shift, max_shift);

        transform_image(&MAT_AT(src.inputs, i, 0), &MAT_AT(dst.inputs, i, 0), angle, dx, dy);
    }
}

Dataset Dataset_augment(Dataset src, int copies, uint32_t seed)
{
    Rng rng = Rng_make(seed);

    Dataset out = Dataset_alloc(src.count * (size_t)(1 + copies));

    size_t o = 0;

    for (size_t i = 0; i < src.count; i++)
    {
        Dataset_copy_sample(&out, o++, src, i);

        for (int k = 0; k < copies; k++)
        {
            float angle = Rng_range(&rng, -15.0f, 15.0f) * PI_F / 180.0f;
            float dx = Rng_range(&rng, -2.0f, 2.0f);
            float dy = Rng_range(&rng, -2.0f, 2.0f);

            transform_image(&MAT_AT(src.inputs, i, 0), &MAT_AT(out.inputs, o, 0), angle, dx, dy);

            Dataset_set_label(&out, o, src.digits[i]);
            o++;
        }
    }

    return out;
}

// -------------------------
// Debug helpers
// -------------------------
void Dataset_print_summary(const char* name, Dataset d)
{
    size_t per_class[DS_CLASSES] = { 0 };

    for (size_t i = 0; i < d.count; i++)
        per_class[d.digits[i]]++;

    printf("[%s] %zu samples | per digit:", name, d.count);

    for (int c = 0; c < DS_CLASSES; c++)
        printf(" %d:%zu", c, per_class[c]);

    printf("\n");
}

void Dataset_print_ascii(Dataset d, size_t index)
{
    static const char* ramp = " .:-=+*#%@";

    printf("label = %d\n", d.digits[index]);

    for (int y = 0; y < DS_IMG_LEN; y++)
    {
        for (int x = 0; x < DS_IMG_LEN; x++)
        {
            float v = MAT_AT(d.inputs, index, y * DS_IMG_LEN + x);
            int level = (int)(v * 9.0f + 0.5f);

            if (level < 0) level = 0;
            if (level > 9) level = 9;

            putchar(ramp[level]);
            putchar(ramp[level]);
        }

        putchar('\n');
    }
}