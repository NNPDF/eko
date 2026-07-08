#include "utils.hpp"
#include <cstdio>
#include <cstdlib>

using namespace std;

static int check_len(const char *label, size_t got, size_t expected)
{
    if (got != expected) {
        fprintf(stderr, "FAIL lengths [%s]: got %zu, expected %zu\n", label, got, expected);
        return 1;
    }
    return 0;
}

static int test_lengths()
{
    int fail = 0;

    fail |= check_len("ad_ps_gamma_ns_qcd_n3lo_len", ad_ps_gamma_ns_qcd_n3lo_len(), 3);
    fail |= check_len("ad_ps_gamma_ns_qcd_result_len(0)", ad_ps_gamma_ns_qcd_result_len(0), 0);
    fail |= check_len("ad_ps_gamma_ns_qcd_result_len(1)", ad_ps_gamma_ns_qcd_result_len(1), 1);
    fail |= check_len("ad_ps_gamma_ns_qcd_result_len(2)", ad_ps_gamma_ns_qcd_result_len(2), 2);
    fail |= check_len("ad_ps_gamma_ns_qcd_result_len(3)", ad_ps_gamma_ns_qcd_result_len(3), 0);

    fail |= check_len("ad_ps_gamma_singlet_qcd_n3lo_len", ad_ps_gamma_singlet_qcd_n3lo_len(), 4);
    fail |= check_len("ad_ps_gamma_singlet_qcd_result_len(0)", ad_ps_gamma_singlet_qcd_result_len(0), 0);
    fail |= check_len("ad_ps_gamma_singlet_qcd_result_len(1)", ad_ps_gamma_singlet_qcd_result_len(1), 4);
    fail |= check_len("ad_ps_gamma_singlet_qcd_result_len(2)", ad_ps_gamma_singlet_qcd_result_len(2), 8);
    fail |= check_len("ad_ps_gamma_singlet_qcd_result_len(3)", ad_ps_gamma_singlet_qcd_result_len(3), 0);

    if (!fail) printf("PASS test_lengths\n");
    return fail;
}

static int test_gamma_ns_qcd()
{
    int fail = 0;
    const uint8_t nf = 3;
    const uint8_t var[3] = {0, 0, 0};
    Cache *c = cache_new(1.0, 0.0);

    const size_t n3lo_len = ad_ps_gamma_ns_qcd_n3lo_len();
    const size_t len = ad_ps_gamma_ns_qcd_result_len(2);
    ComplexF64 r[len];

    ad_ps_gamma_ns_qcd(2, PID_NSP, c, nf, var, n3lo_len, r, len);
    fail |= check("gamma_ns_qcd", "LO [0]", r[0], 0., 0., 1e-14);
    for (size_t i = 0; i < len; i++)
        fail |= check("gamma_ns_qcd", "NLO", r[i], 0., 0., 2e-6);

    cache_delete(c);
    if (!fail) printf("PASS test_gamma_ns_qcd\n");
    return fail;
}

static int test_gamma_singlet_qcd()
{
    int fail = 0;
    const uint8_t nf = 5;
    const uint8_t var[4] = {0, 0, 0, 0};
    Cache *c = cache_new(2.0, 0.0);

    const double CF = 4.0 / 3.0;
    const double CA = 3.0;
    const double TR = 1.0 / 2.0;
    const double PI = 3.14159265358979323846;

    const size_t n3lo_len = ad_ps_gamma_singlet_qcd_n3lo_len();
    const size_t len = ad_ps_gamma_singlet_qcd_result_len(2);
    ComplexF64 g[len];

    ad_ps_gamma_singlet_qcd(2, c, nf, var, n3lo_len, g, len);

    /* LO */
    ComplexF64 lo0 = {g[0 * 4 + 0 * 2 + 0].re + g[0 * 4 + 1 * 2 + 0].re,
                       g[0 * 4 + 0 * 2 + 0].im + g[0 * 4 + 1 * 2 + 0].im};
    fail |= check("gamma_singlet_qcd", "LO qq+gq", lo0, 4. * CF / 3., 0., 1e-12);

    ComplexF64 lo1 = {g[0 * 4 + 0 * 2 + 1].re + g[0 * 4 + 1 * 2 + 1].re,
                       g[0 * 4 + 0 * 2 + 1].im + g[0 * 4 + 1 * 2 + 1].im};
    fail |= check("gamma_singlet_qcd", "LO qg+gg", lo1, 3. + (double)nf / 3., 0., 1e-12);

    /* NLO */
    double expected_nlo = 4. * (double)nf *
        (0.574074074 * CF - 2. * CA * (-7. / 18. + 1. / 6. * (5. - PI * PI / 3.))) * TR;
    ComplexF64 nlo = {-g[1 * 4 + 0 * 2 + 1].re, -g[1 * 4 + 0 * 2 + 1].im};
    fail |= check("gamma_singlet_qcd", "NLO", nlo, expected_nlo, 0., 1e-9);

    cache_delete(c);
    if (!fail) printf("PASS test_gamma_singlet_qcd\n");
    return fail;
}

static int test_guards()
{
    int fail = 0;
    const uint8_t nf = 4;
    const uint8_t var3[3] = {0, 0, 0};
    const uint8_t var4[4] = {0, 0, 0, 0};
    const ComplexF64 sentinel = {123.456, -654.321};
    Cache *c = cache_new(1.234, 0.0);

    ComplexF64 r3[3] = {sentinel, sentinel, sentinel};
    ad_ps_gamma_ns_qcd(3, PID_NSP, c, nf, var3, 3, r3, 3);
    for (int i = 0; i < 3; i++)
        fail |= check("guards", "ns order_qcd=3 untouched", r3[i], sentinel.re, sentinel.im, 0.);

    ComplexF64 g[8];
    for (int i = 0; i < 8; i++) g[i] = sentinel;
    ad_ps_gamma_singlet_qcd(3, c, nf, var4, 4, g, 8);
    for (int i = 0; i < 8; i++)
        fail |= check("guards", "singlet order_qcd=3 untouched", g[i], sentinel.re, sentinel.im, 0.);

    ComplexF64 r2[2] = {sentinel, sentinel};
    ad_ps_gamma_ns_qcd(2, PID_NSM_U, c, nf, var3, 3, r2, 2);
    for (int i = 0; i < 2; i++)
        fail |= check("guards", "unknown mode untouched", r2[i], sentinel.re, sentinel.im, 0.);

    cache_delete(c);
    if (!fail) printf("PASS test_guards\n");
    return fail;
}

int main()
{
    int fail = 0;
    fail |= test_lengths();
    fail |= test_gamma_ns_qcd();
    fail |= test_gamma_singlet_qcd();
    fail |= test_guards();
    return fail ? EXIT_FAILURE : EXIT_SUCCESS;
}
