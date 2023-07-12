#ifndef EKOREPP_HARMONICS_POLYGAMMA_HPP_
#define EKOREPP_HARMONICS_POLYGAMMA_HPP_

#include <complex>
#include <exception>
#include "../types.hpp"

namespace ekorepp {
namespace harmonics {

/** @brief Wrong order k of :math:`\psi_k(z)` */
struct InvalidPolygammaOrder : public std::invalid_argument { using std::invalid_argument::invalid_argument; };

/** @brief Wrong argument z of :math:`\psi_k(z)` */
struct InvalidPolygammaArgument : public std::invalid_argument { using std::invalid_argument::invalid_argument; };

/**
 * @brief Compute the polygamma functions :math:`\psi_k(z)`.
 * Reimplementation of ``WPSIPG`` (C317) in `CERNlib <http://cernlib.web.cern.ch/cernlib/>`_ :cite:`KOLBIG1972221`.
 * @param Z argument of polygamma function
 * @param K order of polygamma function
 * @return k-th polygamma function :math:`\psi_k(z)`
 */
cmplx cern_polygamma(const cmplx Z, const unsigned int K) {
    const double DELTA = 5e-13;
    const double R1 = 1;
    const double HF = R1/2;
    const double C1 = pow(M_PI,2);
    const double C2 = 2*pow(M_PI,3);
    const double C3 = 2*pow(M_PI,4);
    const double C4 = 8*pow(M_PI,5);

    // SGN is originally indexed 0:4 -> no shift
    const double SGN[] = {-1,+1,-1,+1,-1};
    // FCT is originally indexed -1:4 -> shift +1
    const double FCT[] = {0,1,1,2,6,24};

    // C is originally indexed 1:6 x 0:4 -> swap indices and shift new last -1
    const double C[5][6] = {
           {
            8.33333333333333333e-2,
           -8.33333333333333333e-3,
            3.96825396825396825e-3,
           -4.16666666666666667e-3,
            7.57575757575757576e-3,
           -2.10927960927960928e-2
           },
           {
            1.66666666666666667e-1,
           -3.33333333333333333e-2,
            2.38095238095238095e-2,
           -3.33333333333333333e-2,
            7.57575757575757576e-2,
           -2.53113553113553114e-1
           },
           {
            5.00000000000000000e-1,
           -1.66666666666666667e-1,
            1.66666666666666667e-1,
           -3.00000000000000000e-1,
            8.33333333333333333e-1,
           -3.29047619047619048e+0
           },
           {
            2.00000000000000000e+0,
           -1.00000000000000000e+0,
            1.33333333333333333e+0,
           -3.00000000000000000e+0,
            1.00000000000000000e+1,
           -4.60666666666666667e+1
           },
           {10., -7., 12., -33., 130., -691.}
    };
    cmplx U=Z;
    double X=U.real();
    double A=fabs(X);
    if (K < 0 || K > 4)
        throw InvalidPolygammaOrder("Order K has to be in [0:4]");
    const int A_as_int = int(A);
    if (fabs(U.imag()) < DELTA && fabs(X+A_as_int) < DELTA){
        throw InvalidPolygammaArgument("Argument Z equals non-positive integer");
    }
    const unsigned int K1=K+1;
    if (X < 0)
         U=-U;
    cmplx V=U;
    cmplx H=0;
    if (A < 15) {
        H=1./pow(V,K1);
        for (int i = 1; i < 14 - A_as_int + 1 ; ++i) {
            V=V+1.;
            H=H+1./pow(V,K1);
        }
        V=V+1.;
    }
    cmplx R=1./pow(V,2);
    cmplx P=R*C[K][6-1];
    for (int i = 5; i>1-1; i--)
         P=R*(C[K][i-1]+P);
    H=SGN[K]*(FCT[K+1]*H+(V*(FCT[K-1+1]+P)+HF*FCT[K+1])/pow(V,K1));
    if (0 == K)
        H=H+log(V);
    if (X < 0){
        V=M_PI*U;
        X=V.real();
        const double Y=V.imag();
        const double A=sin(X);
        const double B=cos(X);
        const double T=tanh(Y);
        P=cmplx(B,-A*T)/cmplx(A,+B*T);
        if (0 == K)
            H=H+1./U+M_PI*P;
        else if (1 == K)
            H=-H+1./pow(U,2)+C1*(pow(P,2)+1.);
        else if (2 == K)
            H=H+2./pow(U,3)+C2*P*(pow(P,2)+1.);
        else if (3 == K) {
            R=pow(P,2);
            H=-H+6./pow(U,4)+C3*((3.*R+4.)*R+1.);
        } else if (4 == K) {
            R=pow(P,2);
            H=H+24./pow(U,5)+C4*P*((3.*R+5.)*R+2.);
        }
    }
    return H;
}

cmplx recursive_harmonic_sum(const cmplx base_value, const cmplx n, const unsigned int iterations, const unsigned int weight) {
    cmplx fact = 0.0;
    for (unsigned int i = 1; i <= iterations + 1; ++i)
        fact += pow(1.0 / (n + double(i)), weight);
    return base_value + fact;
}

} // namespace harmonics
} // namespace ekorepp

#endif // EKOREPP_HARMONICS_POLYGAMMA_HPP_
