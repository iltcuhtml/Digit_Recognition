#include <stdio.h>
#include <stdlib.h>

#include "nn.h"
#include "dataset.h"

static float mean_pixel(Dataset d)
{
    double s = 0.0;

    for (size_t i = 0; i < d.count; i++)
        for (size_t j = 0; j < DS_PIXELS; j++)
            s += MAT_AT(d.inputs, i, j);

    return d.count ? (float)(s / ((double)d.count * DS_PIXELS)) : 0.0f;
}

int main(void)
{
    Dataset tr = Dataset_load_mnist("data/mnist/train-images-idx3-ubyte", "data/mnist/train-labels-idx1-ubyte", 0);
    Dataset te = Dataset_load_mnist("data/mnist/t10k-images-idx3-ubyte", "data/mnist/t10k-labels-idx1-ubyte", 0);
    Dataset own = Dataset_load("data/number.dat");

    if (tr.count == 0 || te.count == 0)
        return EXIT_FAILURE;

    Dataset_print_summary("mnist train", tr);
    Dataset_print_summary("mnist test", te);
    Dataset_print_summary("own", own);

    printf("mean pixel | mnist train %.4f | mnist test %.4f | own %.4f\n",
        (double)mean_pixel(tr), (double)mean_pixel(te), (double)mean_pixel(own));

    Dataset_print_ascii(tr, 0);     // expected label: 5
    Dataset_print_ascii(te, 0);     // expected label: 7

    Dataset_free(&tr);
    Dataset_free(&te);
    Dataset_free(&own);

    return EXIT_SUCCESS;
}