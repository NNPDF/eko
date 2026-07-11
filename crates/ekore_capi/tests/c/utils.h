#ifndef EKORE_TEST_UTILS_H
#define EKORE_TEST_UTILS_H

#define DEFAULT_EPS 1e-15

#include <stddef.h>

/* Returns 1 on failure, 0 on success. Prints a diagnostic to stderr on failure. */
int check(const char *test, const char *label,
          double got_re, double got_im,
          double exp_re, double exp_im, double eps);

int check_len(const char *label, size_t got, size_t expected);

int check_val(const char *name, size_t actual, size_t expected);

#endif
