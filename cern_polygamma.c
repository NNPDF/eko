// gcc -shared -o cernlibc.so -fPIC cern_polygamma.c

#include <math.h>
#include <complex.h>

double getdouble(double* ptr) {
    return *ptr;
}

int cern_polygamma(const double re_in, const double im_in, const unsigned int K, double* re_out, double* im_out) {
    const double _Complex Z = re_in + im_in * _Complex_I;
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
    double _Complex U=Z;
    double X=creal(U);
    double A=cabs(X);
    if (K < 0 || K > 4)
        return -1;
    //     raise NotImplementedError("Order K has to be in [0:4]")
    const int A_as_int = A;
    if (cabs(cimag(U)) < DELTA && cabs(X+A_as_int) < DELTA)
        return -2;
    //     raise ValueError("Argument Z equals non-positive integer")
    const unsigned int K1=K+1;
    if (X < 0)
         U=-U;
    double _Complex V=U;
    double _Complex H=0;
    if (A < 15) {
        H=1/cpow(V,K1);
        for (int i = 1; i < 14 - A_as_int + 1 ; ++i) {
            V=V+1;
            H=H+1/cpow(V,K1);
        }
        V=V+1;
    }
    double _Complex R=1/cpow(V,2);
    double _Complex P=R*C[K][6-1];
    for (int i = 5; i>1-1; i--)
         P=R*(C[K][i-1]+P);
    H=SGN[K]*(FCT[K+1]*H+(V*(FCT[K-1+1]+P)+HF*FCT[K+1])/cpow(V,K1));
    if (0 == K)
        H=H+clog(V);
    if (X < 0){
        V=M_PI*U;
        X=creal(V);
        const double Y=cimag(V);
        const double A=sin(X);
        const double B=cos(X);
        const double T=tanh(Y);
        P=(B-A*T*_Complex_I)/(A+B*T*_Complex_I);
        if (0 == K)
            H=H+1/U+M_PI*P;
        else if (1 == K)
            H=-H+1/cpow(U,2)+C1*(cpow(P,2)+1);
        else if (2 == K)
            H=H+2/cpow(U,3)+C2*P*(cpow(P,2)+1);
        else if (3 == K) {
            R=cpow(P,2);
            H=-H+6/cpow(U,4)+C3*((3*R+4)*R+1);
        } else if (4 == K) {
            R=cpow(P,2);
            H=H+24/cpow(U,5)+C4*P*((3*R+5)*R+2);
        }
    }
    //return creal(H);
    *re_out = creal(H);
    *im_out = cimag(H);
    return 0;
}
