#include "utils.h"
#include <ekore_capi.h>
#include <stdio.h>
#include <stdlib.h>

static int check_len(const char *label, size_t got, size_t expected)
{
    if (got != expected) {
        fprintf(stderr, "FAIL lengths [%s]: got %zu, expected %zu\n", label, got, expected);
        return 1;
    }
    return 0;
}

static int test_lengths(void)
{
    int fail = 0;

    fail |= check_len("ome_us_A_singlet_result_len(0)", ome_us_A_singlet_result_len(0), 0);
    fail |= check_len("ome_us_A_singlet_result_len(1)", ome_us_A_singlet_result_len(1), 9);
    fail |= check_len("ome_us_A_singlet_result_len(2)", ome_us_A_singlet_result_len(2), 18);
    fail |= check_len("ome_us_A_singlet_result_len(3)", ome_us_A_singlet_result_len(3), 0);

    fail |= check_len("ome_us_A_non_singlet_result_len(0)", ome_us_A_non_singlet_result_len(0), 0);
    fail |= check_len("ome_us_A_non_singlet_result_len(1)", ome_us_A_non_singlet_result_len(1), 4);
    fail |= check_len("ome_us_A_non_singlet_result_len(2)", ome_us_A_non_singlet_result_len(2), 8);
    fail |= check_len("ome_us_A_non_singlet_result_len(3)", ome_us_A_non_singlet_result_len(3), 0);

    if (!fail) printf("PASS test_lengths\n");
    return fail;
}

static int test_a_non_singlet(void)
{
    int fail = 0;
    const uint8_t nf = 5;
    const double L = 0.0;
    Cache *c = cache_new(1.0, 0.0);

    const size_t len = ome_us_A_non_singlet_result_len(2);
    ComplexF64 a[len];

    ome_us_A_non_singlet(2, c, nf, L, a, len);

    /* LO */
    for (int k = 0; k < 4; k++)
        fail |= check("a_non_singlet", "LO", a[k].re, a[k].im, 0., 0., 1e-14);
    /* NNLO */
    for (int k = 4; k < 8; k++)
        fail |= check("a_non_singlet", "NNLO", a[k].re, a[k].im, 0., 0., 1e-14);

    cache_delete(c);
    if (!fail) printf("PASS test_a_non_singlet\n");
    return fail;
}

static int test_a_singlet(void)
{
    int fail = 0;
    const uint8_t nf = 5;
    const double L = 100.0;
    Cache *c = cache_new(2.0, 0.0);

    const size_t len = ome_us_A_singlet_result_len(2);
    ComplexF64 a[len];

    ome_us_A_singlet(2, c, nf, L, a, len);

    /* LO */
    fail |= check("a_singlet", "LO col2 sum",
                  a[2].re + a[5].re + a[8].re, a[2].im + a[5].im + a[8].im, 0., 0., 1e-10);
    fail |= check("a_singlet", "LO col0 sum",
                  a[0].re + a[3].re + a[6].re, a[0].im + a[3].im + a[6].im, 0., 0., 1e-12);

    /* NNLO */
    fail |= check("a_singlet", "NNLO col0 sum",
                  a[9].re + a[12].re + a[15].re, a[9].im + a[12].im + a[15].im, 0., 0., 2e-6);
    fail |= check("a_singlet", "NNLO col1 sum",
                  a[10].re + a[13].re + a[16].re, a[10].im + a[13].im + a[16].im, 0., 0., 1e-11);

    cache_delete(c);
    if (!fail) printf("PASS test_a_singlet\n");
    return fail;
}

static int test_guards(void)
{
    int fail = 0;
    const uint8_t nf = 4;
    const double L = 0.0;
    const ComplexF64 sentinel = {123.456, -654.321};
    Cache *c = cache_new(1.234, 0.0);

    ComplexF64 a_s[9];
    for (int i = 0; i < 9; i++) a_s[i] = sentinel;
    ome_us_A_singlet(3, c, nf, L, a_s, 9);
    for (int i = 0; i < 9; i++)
        fail |= check("guards", "A_singlet order3 untouched", a_s[i].re, a_s[i].im, sentinel.re, sentinel.im, 0.);

    ComplexF64 a_ns[4] = {sentinel, sentinel, sentinel, sentinel};
    ome_us_A_non_singlet(3, c, nf, L, a_ns, 4);
    for (int i = 0; i < 4; i++)
        fail |= check("guards", "A_non_singlet order3 untouched", a_ns[i].re, a_ns[i].im, sentinel.re, sentinel.im, 0.);

    cache_delete(c);
    if (!fail) printf("PASS test_guards\n");
    return fail;
}

int main(void)
{
    int fail = 0;
    fail |= test_lengths();
    fail |= test_a_non_singlet();
    fail |= test_a_singlet();
    fail |= test_guards();
    return fail ? EXIT_FAILURE : EXIT_SUCCESS;
}
