// g++ -shared -o ekorepp.so -fPIC src/ekorepp/cinterface.cpp

#include <exception>
#include "./types.hpp"
#include "./harmonics/cache.hpp"
#include "./anomalous_dimensions/unpolarized/space_like/init.hpp"

extern "C" double c_getdouble(double* ptr) {
    return *ptr;
}

extern "C" int c_ad_us_gamma_ns(const unsigned int order_qcd, const unsigned int mode,
        const double re_in, const double im_in, const unsigned int nf, double* re_out, double* im_out) {
    const ekorepp::cmplx n (re_in, im_in);
    try {
        ekorepp::harmonics::Cache c(n);
        const ekorepp::vctr res = ekorepp::anomalous_dimensions::unpolarized::spacelike::gamma_ns(order_qcd, mode, c, nf);
        for (unsigned int j = 0; j < res.size(); ++j) {
            re_out[j] = res[j].real();
            im_out[j] = res[j].imag();
        }
    } catch (ekorepp::harmonics::InvalidPolygammaOrder &e) {
        return -2;
    } catch (ekorepp::harmonics::InvalidPolygammaArgument &e) {
        return -3;
    }
    return 0;
}

extern "C" int c_ad_us_gamma_singlet(const unsigned int order_qcd, const double re_in, const double im_in, const unsigned int nf,
        double* re_out, double* im_out) {
    const ekorepp::cmplx n (re_in, im_in);
    try {
        ekorepp::harmonics::Cache c(n);
        const ekorepp::tnsr res = ekorepp::anomalous_dimensions::unpolarized::spacelike::gamma_singlet(order_qcd, c, nf);
        unsigned int j = 0;
        for (unsigned int k = 0; k < res.size(); ++k) {
            for (unsigned int l = 0; l < res[k].size(); ++l) {
                for (unsigned int m = 0; m < res[k][l].size(); ++m) {
                    re_out[j] = res[k][l][m].real();
                    im_out[j] = res[k][l][m].imag();
                    ++j;
                }
            }
        }
    } catch (ekorepp::harmonics::InvalidPolygammaOrder &e) {
        return -2;
    } catch (ekorepp::harmonics::InvalidPolygammaArgument &e) {
        return -3;
    }
    return 0;
}
