#include "utils.h"
#include <math.h>
#include <stdio.h>

int check(const char *test, const char *label,
          double got_re, double got_im,
          double exp_re, double exp_im, double eps)
{
    // This is effectively spanning a diamond-like shape around got_re + I*got_im
    double diff = fabs(got_re - exp_re) + fabs(got_im - exp_im);
    if (diff > eps) {
        fprintf(stderr, "FAIL %s [%s]: got (%.17g, %.17g), expected (%.17g, %.17g)\n",
                test, label, got_re, got_im, exp_re, exp_im);
        return 1;
    }
    return 0;
}

int check_len(const char *label, size_t got, size_t expected)
{
    if (got != expected) {
        fprintf(stderr, "FAIL lengths [%s]: got %zu, expected %zu\n", label, got, expected);
        return 1;
    }
    return 0;
}

int check_val(const char *name, size_t actual, size_t expected)
{
    if (actual != expected) {
        fprintf(stderr, "FAIL: %s - expected %zu, got %zu\n", name, expected, actual);
        return 1;
    }
    return 0;
}
