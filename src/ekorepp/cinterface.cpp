// g++ -shared -o ekorepp.so -fPIC src/ekorepp/cinterface.cpp

#include <math.h>
#include <exception>
#include <complex>
#include <vector>
#include "./harmonics/polygamma.hpp"
#include "./anomalous_dimensions/unpolarized/space_like/init.hpp"

extern "C" double c_getdouble(double* ptr) {
    return *ptr;
}

extern "C" int c_cern_polygamma(const double re_in, const double im_in, const unsigned int K, double* re_out, double* im_out) {
    const std::complex<double> Z (re_in, im_in);
    try {
        const std::complex<double> H = ekorepp::harmonics::cern_polygamma(Z, K);
        *re_out = H.real();
        *im_out = H.imag();
    } catch (std::invalid_argument &e) {
        return -1;
    }
    return 0;
}


extern "C" int c_ad_us_gamma_ns(const unsigned int order_qcd, const unsigned int mode,
    const double re_in, const double im_in, const unsigned int nf, double* re_out, double* im_out) {
    const std::complex<double> n (re_in, im_in);
    try {
        const std::vector<std::complex<double>> res = ekorepp::anomalous_dimensions::unpolarized::spacelike::gamma_ns(order_qcd, mode, n, nf);
        for (unsigned int j = 0; j < res.size(); ++j) {
            re_out[j] = res[j].real();
            im_out[j] = res[j].imag();
        }
    } catch (std::invalid_argument &e) {
        return -1;
    }
    return 0;
}
