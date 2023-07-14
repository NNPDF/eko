// g++ -shared -o c_quad_ker.so -fPIC src/eko/evolution_operator/c_quad_ker.cpp

#include <exception>
#include <vector>
#include "./../../ekorepp/types.hpp"
#include "./../../ekorepp/harmonics/cache.hpp"
#include "./../../ekorepp/anomalous_dimensions/unpolarized/space_like.hpp"

class TalbotPath {
    double t;
    double r;
    double o;
public:
    TalbotPath(const double t, const double logx, const bool is_singlet) : t(t), r(0.4 * 16.0 / (1.0 - logx)), o(is_singlet ? 1. : 0.) {}
    double theta() const { return M_PI * (2.0 * t - 1.0); }
    ekorepp::cmplx n() const {
        const double theta = this->theta();
        // treat singular point separately
        const double re = 0.5 == t ? 1.0 : theta / tan(theta);
        const double im = theta;
        return o + r * ekorepp::cmplx(re, im);
    }
    ekorepp::cmplx jac() const {
        const double theta = this->theta();
        // treat singular point separately
        const double re = 0.5 == t ? 0.0 : 1.0 / tan(theta) - theta / pow(sin(theta), 2);
        const double im = 1.0;
        return r * M_PI * 2.0 * ekorepp::cmplx(re, im);
    }
    ekorepp::cmplx prefactor() const {
        return ekorepp::cmplx(0,-1./M_PI);
    }
};

typedef double (*py_quad_ker_qcd)(
                        double* re_gamma, double* im_gamma,
                        double re_n, double im_n,
                        double re_jac, double im_jac,
                        unsigned int order_qcd,
                        bool is_singlet,
                        unsigned int mode0,
                        unsigned int mode1,
                        unsigned int nf,
                        bool is_log,
                        double logx,
                        double* areas,
                        unsigned int areas_x,
                        unsigned int areas_y,
                        double L,
                        unsigned int method_num,
                        double as1, double as0,
                        unsigned int ev_op_iterations, unsigned int ev_op_max_order_qcd,
                        unsigned int sv_mode_num, bool is_threshold);

struct QuadCargs {
    unsigned int order_qcd;
    unsigned int mode0;
    unsigned int mode1;
    bool is_polarized;
    bool is_time_like;
    unsigned int nf;
    py_quad_ker_qcd py;
    bool is_log;
    double logx;
    double* areas;
    unsigned int areas_x;
    unsigned int areas_y;
    double L;
    unsigned int method_num;
    double as1;
    double as0;
    unsigned int ev_op_iterations;
    unsigned int ev_op_max_order_qcd;
    unsigned int sv_mode_num;
    bool is_threshold;
};

extern "C" double c_quad_ker_qcd(const double u, void* rargs) {
    QuadCargs args = *(QuadCargs* )rargs;
    const bool is_singlet = (100 == args.mode0) || (21 == args.mode0) || (90 == args.mode0);
    // prepare gamma
    const TalbotPath path(u, args.logx, is_singlet);
    const ekorepp::cmplx jac = path.jac() * path.prefactor();
    ekorepp::harmonics::Cache c(path.n());
    const size_t size = args.order_qcd * (is_singlet ? 4 : 1);
    std::vector<double> re(size);
    std::vector<double> im(size);
    if (is_singlet) {
        const ekorepp::tnsr res = ekorepp::anomalous_dimensions::unpolarized::spacelike::gamma_singlet(args.order_qcd, c, args.nf);
        unsigned int j = 0;
        for (unsigned int k = 0; k < res.size(); ++k) {
            for (unsigned int l = 0; l < res[k].size(); ++l) {
                for (unsigned int m = 0; m < res[k][l].size(); ++m) {
                    re[j] = res[k][l][m].real();
                    im[j] = res[k][l][m].imag();
                    ++j;
                }
            }
        }
    } else {
        const ekorepp::vctr res = ekorepp::anomalous_dimensions::unpolarized::spacelike::gamma_ns(args.order_qcd, args.mode0, c, args.nf);
        for (unsigned int j = 0; j < res.size(); ++j) {
            re[j] = res[j].real();
            im[j] = res[j].imag();
        }
    }
    // pass on
    return args.py(re.data(), im.data(),
            c.N().real(), c.N().imag(),
            jac.real(),jac.imag(),
            args.order_qcd,
            is_singlet,
            args.mode0, args.mode1,
            args.nf,
            args.is_log, args.logx,
            args.areas, args.areas_x, args.areas_y,
            args.L, args.method_num, args.as1, args.as0,
            args.ev_op_iterations, args.ev_op_max_order_qcd,
            args.sv_mode_num, args.is_threshold);
}
