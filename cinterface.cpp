// g++ -shared -o cernlibc2.so -fPIC cinterface.cpp

#include <math.h>
#include <exception>
#include <complex>
#include "cern_polygamma.hpp"

extern "C" double c_getdouble(double* ptr) {
    return *ptr;
}

extern "C" int c_cern_polygamma(const double re_in, const double im_in, const unsigned int K, double* re_out, double* im_out) {
    const std::complex<double> Z (re_in, im_in);
    try {
        const std::complex<double> H = cern_polygamma(Z, K);
        *re_out = H.real();
        *im_out = H.imag();
    } catch (std::invalid_argument &e) {
        return -1;
    }
    return 0;
}
