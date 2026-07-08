#include "utils.hpp"
#include <complex>
#include <cmath>
#include <cstdio>

using namespace std;

int check(const char *test, const char *label,
          ComplexF64 got, double exp_re, double exp_im, double eps)
{
    double diff = abs(complex<double>(got.re, got.im) -
                           complex<double>(exp_re, exp_im));
    if (diff > eps) {
        fprintf(stderr, "FAIL %s [%s]: got (%.17g, %.17g), expected (%.17g, %.17g)\n",
                     test, label, got.re, got.im, exp_re, exp_im);
        return 1;
    }
    return 0;
}
