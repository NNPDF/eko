#ifndef EKORE_TEST_UTILS_HPP
#define EKORE_TEST_UTILS_HPP

#include <ekore_capi.h>

static constexpr double DEFAULT_EPS = 1e-15;

/* Returns 1 on failure, 0 on success. Prints a diagnostic to stderr on failure. */
int check(const char *test, const char *label,
          ComplexF64 got, double exp_re, double exp_im, double eps);

#endif
