#include "utils.h"
#include <ekore_capi.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

static int test_lengths(void)
{
    int fail = 0;

    fail |= check_len("ad_us_gamma_ns_qcd_result_len(4)", ad_us_gamma_ns_qcd_result_len(4), 4);
    fail |= check_len("ad_us_gamma_ns_qcd_result_len(5)", ad_us_gamma_ns_qcd_result_len(5), 0);

    fail |= check_len("ad_us_gamma_singlet_qcd_result_len(4)", ad_us_gamma_singlet_qcd_result_len(4), 16);
    fail |= check_len("ad_us_gamma_singlet_qcd_result_len(5)", ad_us_gamma_singlet_qcd_result_len(5), 0);

    fail |= check_len("ad_us_gamma_ns_qed_result_len(3,2)", ad_us_gamma_ns_qed_result_len(3, 2), 12);
    fail |= check_len("ad_us_gamma_ns_qed_result_len(5,1)", ad_us_gamma_ns_qed_result_len(5, 1), 0);
    fail |= check_len("ad_us_gamma_ns_qed_result_len(1,3)", ad_us_gamma_ns_qed_result_len(1, 3), 0);

    fail |= check_len("ad_us_gamma_singlet_qed_result_len(3,2)", ad_us_gamma_singlet_qed_result_len(3, 2), 192);
    fail |= check_len("ad_us_gamma_singlet_qed_result_len(5,1)", ad_us_gamma_singlet_qed_result_len(5, 1), 0);
    fail |= check_len("ad_us_gamma_singlet_qed_result_len(1,3)", ad_us_gamma_singlet_qed_result_len(1, 3), 0);

    fail |= check_len("ad_us_gamma_valence_qed_result_len(3,2)", ad_us_gamma_valence_qed_result_len(3, 2), 48);
    fail |= check_len("ad_us_gamma_valence_qed_result_len(5,1)", ad_us_gamma_valence_qed_result_len(5, 1), 0);
    fail |= check_len("ad_us_gamma_valence_qed_result_len(1,3)", ad_us_gamma_valence_qed_result_len(1, 3), 0);

    if (!fail) printf("PASS test_lengths\n");
    return fail;
}

static int test_gamma_ns_qcd(void)
{
    int fail = 0;
    const uint8_t nf = 3;
    const uint8_t var[3] = {0, 0, 0};
    Cache *c = cache_new(1.0, 0.0);
    const size_t len2 = ad_us_gamma_ns_qcd_result_len(2);
    const size_t len3 = ad_us_gamma_ns_qcd_result_len(3);
    const size_t len4 = ad_us_gamma_ns_qcd_result_len(4);
    ComplexF64 r2[len2], r3[len3], r4[len4];

    ad_us_gamma_ns_qcd(3, PID_NSP, c, nf, var, r3, len3);
    fail |= check("gamma_ns_qcd", "NSP order3 [0]", r3[0].re, r3[0].im, 0., 0., 1e-14);

    ad_us_gamma_ns_qcd(2, PID_NSM, c, nf, var, r2, len2);
    for (int i = 0; i < 2; i++)
        fail |= check("gamma_ns_qcd", "NSM order2", r2[i].re, r2[i].im, 0., 0., 2e-6);

    ad_us_gamma_ns_qcd(3, PID_NSM, c, nf, var, r3, len3);
    for (int i = 0; i < 3; i++)
        fail |= check("gamma_ns_qcd", "NSM order3", r3[i].re, r3[i].im, 0., 0., 2e-4);

    ad_us_gamma_ns_qcd(3, PID_NSV, c, nf, var, r3, len3);
    for (int i = 0; i < 3; i++)
        fail |= check("gamma_ns_qcd", "NSV order3", r3[i].re, r3[i].im, 0., 0., 8e-4);

    ad_us_gamma_ns_qcd(4, PID_NSP, c, nf, var, r4, len4);
    int any_nonzero = 0;
    for (int i = 0; i < 4; i++)
        if (r4[i].re * r4[i].re + r4[i].im * r4[i].im > 1e-12) any_nonzero = 1;
    if (!any_nonzero) {
        fprintf(stderr, "FAIL gamma_ns_qcd [NSP order4]: expected a non-trivial N3LO entry\n");
        fail = 1;
    }

    cache_delete(c);
    if (!fail) printf("PASS test_gamma_ns_qcd\n");
    return fail;
}

static int test_gamma_ns_qed(void)
{
    int fail = 0;
    const uint8_t nf = 3;
    const uint8_t var[3] = {0, 0, 0};
    Cache *c = cache_new(1.0, 0.0);
    const size_t len11 = ad_us_gamma_ns_qed_result_len(1, 1);
    const size_t len12 = ad_us_gamma_ns_qed_result_len(1, 2);
    const size_t len21 = ad_us_gamma_ns_qed_result_len(2, 1);
    const size_t len31 = ad_us_gamma_ns_qed_result_len(3, 1);
    ComplexF64 r1[len11], r2[len12], r8[len31];

    uint16_t nsm_pids[2] = {PID_NSM_U, PID_NSM_D};
    for (int p = 0; p < 2; p++) {
        ad_us_gamma_ns_qed(1, 1, nsm_pids[p], c, nf, var, r1, len11);
        for (int i = 0; i < 4; i++)
            fail |= check("gamma_ns_qed", "ns-", r1[i].re, r1[i].im, 0., 0., 1e-5);
    }

    ad_us_gamma_ns_qed(1, 1, PID_NSP_U, c, nf, var, r1, len11);
    fail |= check("gamma_ns_qed", "ns+ [0][0]", r1[0].re, r1[0].im, 0., 0., DEFAULT_EPS);
    fail |= check("gamma_ns_qed", "ns+ [0][1]", r1[1].re, r1[1].im, 0., 0., 1e-5);
    ad_us_gamma_ns_qed(1, 1, PID_NSP_D, c, nf, var, r1, len11);
    fail |= check("gamma_ns_qed", "ns+ [0][0]", r1[0].re, r1[0].im, 0., 0., DEFAULT_EPS);
    fail |= check("gamma_ns_qed", "ns+ [0][1]", r1[1].re, r1[1].im, 0., 0., 1e-5);

    for (int p = 0; p < 2; p++) {
        ad_us_gamma_ns_qed(1, 2, nsm_pids[p], c, nf, var, r2, len12);
        for (int i = 0; i < 6; i++)
            fail |= check("gamma_ns_qed", "aem2+as1aem1", r2[i].re, r2[i].im, 0., 0., 1e-5);
    }

    for (int p = 0; p < 2; p++) {
        ad_us_gamma_ns_qed(2, 1, nsm_pids[p], c, nf, var, r2, len21);
        for (int i = 0; i < 6; i++)
            fail |= check("gamma_ns_qed", "as2", r2[i].re, r2[i].im, 0., 0., 1e-5);
    }

    for (int p = 0; p < 2; p++) {
        ad_us_gamma_ns_qed(3, 1, nsm_pids[p], c, nf, var, r8, len31);
        for (int i = 0; i < 8; i++)
            fail |= check("gamma_ns_qed", "as3", r8[i].re, r8[i].im, 0., 0., 1e-3);
    }

    cache_delete(c);
    if (!fail) printf("PASS test_gamma_ns_qed\n");
    return fail;
}

static int test_gamma_valence_qed(void)
{
    int fail = 0;
    const uint8_t nf = 3;
    const uint8_t var[3] = {0, 0, 0};
    Cache *c = cache_new(2.0, 0.0);
    const size_t len = ad_us_gamma_valence_qed_result_len(3, 2);
    ComplexF64 g[len];

    ad_us_gamma_valence_qed(3, 2, c, nf, var, g, len);

    for (int k = 0; k < 4; k++)
        fail |= check("gamma_valence_qed", "g[0][0]", g[k].re, g[k].im, 0., 0., DEFAULT_EPS);

    int base = (3 * 3 + 0) * 4;
    fail |= check("gamma_valence_qed", "g[3][0][0][0]", g[base+0].re, g[base+0].im, 459.646893789751, 0., 1e-5);
    fail |= check("gamma_valence_qed", "g[3][0][0][1]", g[base+1].re, g[base+1].im, 0.,               0., 1e-5);
    fail |= check("gamma_valence_qed", "g[3][0][1][0]", g[base+2].re, g[base+2].im, 0.,               0., 1e-5);
    fail |= check("gamma_valence_qed", "g[3][0][1][1]", g[base+3].re, g[base+3].im, 437.60340375,     0., 1e-5);

    cache_delete(c);
    if (!fail) printf("PASS test_gamma_valence_qed\n");
    return fail;
}

static int test_gamma_singlet_qed(void)
{
    int fail = 0;
    const uint8_t nf = 3;
    const uint8_t var[7] = {0, 0, 0, 0, 0, 0, 0};
    Cache *c = cache_new(2.0, 0.0);
    const size_t len = ad_us_gamma_singlet_qed_result_len(3, 2);
    ComplexF64 g[len];

    ad_us_gamma_singlet_qed(3, 2, c, nf, var, g, len);

    for (int k = 0; k < 16; k++)
        fail |= check("gamma_singlet_qed", "g[0][0]", g[k].re, g[k].im, 0., 0., DEFAULT_EPS);

    int base = (3 * 3 + 0) * 16;
    double ref[4][4][2] = {
        {{ 3.857918949669738,  0.}, {0., 0.}, {-290.42193908689745, 0.}, {0., 0.}},
        {{0., 0.}, {0., 0.}, {0., 0.}, {0., 0.}},
        {{-3.859554320251334,  0.}, {0., 0.}, { 290.4252052962147,  0.}, {0., 0.}},
        {{0., 0.}, {0., 0.}, {0., 0.}, {448.0752570151872,  0.}},
    };
    for (int r = 0; r < 4; r++)
        for (int col = 0; col < 4; col++)
            fail |= check("gamma_singlet_qed", "g[3][0]",
                          g[base + r * 4 + col].re, g[base + r * 4 + col].im,
                          ref[r][col][0], ref[r][col][1], 1e-5);

    cache_delete(c);
    if (!fail) printf("PASS test_gamma_singlet_qed\n");
    return fail;
}

static int test_guards(void)
{
    int fail = 0;
    const uint8_t nf = 4;
    const uint8_t var3[3] = {0, 0, 0};
    const uint8_t var4[4] = {0, 0, 0, 0};
    const uint8_t var7[7] = {0, 0, 0, 0, 0, 0, 0};
    const ComplexF64 sentinel = {123.456, -654.321};
    Cache *c = cache_new(1.234, 0.0);

    ComplexF64 r5[5] = {sentinel, sentinel, sentinel, sentinel, sentinel};
    ad_us_gamma_ns_qcd(5, PID_NSP, c, nf, var3, r5, 5);
    for (int i = 0; i < 5; i++)
        fail |= check("guards", "ns_qcd order5 untouched", r5[i].re, r5[i].im, sentinel.re, sentinel.im, 0.);

    ComplexF64 gs[20];
    for (int i = 0; i < 20; i++) gs[i] = sentinel;
    ad_us_gamma_singlet_qcd(5, c, nf, var4, gs, 20);
    for (int i = 0; i < 20; i++)
        fail |= check("guards", "singlet_qcd order5 untouched", gs[i].re, gs[i].im, sentinel.re, sentinel.im, 0.);

    ComplexF64 rq[12];
    for (int i = 0; i < 12; i++) rq[i] = sentinel;
    ad_us_gamma_ns_qed(5, 1, PID_NSP_U, c, nf, var3, rq, 12);
    for (int i = 0; i < 12; i++)
        fail |= check("guards", "ns_qed order_qcd=5 untouched", rq[i].re, rq[i].im, sentinel.re, sentinel.im, 0.);
    for (int i = 0; i < 12; i++) rq[i] = sentinel;
    ad_us_gamma_ns_qed(1, 3, PID_NSP_U, c, nf, var3, rq, 12);
    for (int i = 0; i < 12; i++)
        fail |= check("guards", "ns_qed order_qed=3 untouched", rq[i].re, rq[i].im, sentinel.re, sentinel.im, 0.);

    ComplexF64 r3g[3] = {sentinel, sentinel, sentinel};
    ad_us_gamma_ns_qed(2, 0, 10106 /* not a valid non-singlet mode */, c, nf, var3, r3g, ad_us_gamma_ns_qed_result_len(2, 0));
    for (int i = 0; i < 3; i++)
        fail |= check("guards", "unknown mode untouched", r3g[i].re, r3g[i].im, sentinel.re, sentinel.im, 0.);

    ComplexF64 rv[16];
    for (int i = 0; i < 16; i++) rv[i] = sentinel;
    ad_us_gamma_valence_qed(5, 1, c, nf, var3, rv, 16);
    for (int i = 0; i < 16; i++)
        fail |= check("guards", "valence_qed order_qcd=5 untouched", rv[i].re, rv[i].im, sentinel.re, sentinel.im, 0.);

    ComplexF64 rsq[64];
    for (int i = 0; i < 64; i++) rsq[i] = sentinel;
    ad_us_gamma_singlet_qed(1, 3, c, nf, var7, rsq, 64);
    for (int i = 0; i < 64; i++)
        fail |= check("guards", "singlet_qed order_qed=3 untouched", rsq[i].re, rsq[i].im, sentinel.re, sentinel.im, 0.);

    cache_delete(c);
    if (!fail) printf("PASS test_guards\n");
    return fail;
}

int main(void)
{
    int fail = 0;
    fail |= test_lengths();
    fail |= test_gamma_ns_qcd();
    fail |= test_gamma_ns_qed();
    fail |= test_gamma_valence_qed();
    fail |= test_gamma_singlet_qed();
    fail |= test_guards();
    return fail ? EXIT_FAILURE : EXIT_SUCCESS;
}
