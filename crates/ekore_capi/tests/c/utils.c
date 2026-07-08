#include "utils.h"
#include <math.h>
#include <stdio.h>

int check(const char *test, const char *label,
          double got_re, double got_im,
          double exp_re, double exp_im, double eps)
{
    double diff = fabs(got_re - exp_re) + fabs(got_im - exp_im);
    if (diff > eps) {
        fprintf(stderr, "FAIL %s [%s]: got (%.17g, %.17g), expected (%.17g, %.17g)\n",
                test, label, got_re, got_im, exp_re, exp_im);
        return 1;
    }
    return 0;
}
