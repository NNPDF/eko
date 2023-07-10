#include <complex>
#include <exception>

std::complex<double> cern_polygamma(const std::complex<double> Z, const unsigned int K) {
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
    std::complex<double> U=Z;
    double X=U.real();
    double A=abs(X);
    if (K < 0 || K > 4)
        throw std::invalid_argument("Order K has to be in [0:4]");
    const int A_as_int = A;
    if (abs(U.imag()) < DELTA && abs(X+A_as_int) < DELTA)
        throw std::invalid_argument("Argument Z equals non-positive integer");
    const unsigned int K1=K+1;
    if (X < 0)
         U=-U;
    std::complex<double> V=U;
    std::complex<double> H=0;
    if (A < 15) {
        H=1./pow(V,K1);
        for (int i = 1; i < 14 - A_as_int + 1 ; ++i) {
            V=V+1.;
            H=H+1./pow(V,K1);
        }
        V=V+1.;
    }
    std::complex<double> R=1./pow(V,2);
    std::complex<double> P=R*C[K][6-1];
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
        P=std::complex<double>(B,-A*T)/std::complex<double>(A,+B*T);
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
