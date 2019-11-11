from cffi import FFI
ffibuilder = FFI()

ffibuilder.set_source("_gsl_digamma",
   r""" // passed to the real C compiler,
        // contains implementation of things declared in cdef()
        #include <gsl/gsl_sf_psi.h>
        struct complex {
            double r;
            double i;
        };

        static struct complex digamma(double x, double y) {
            gsl_sf_result r, i;
            struct complex res;
            gsl_sf_complex_psi_e(x, y, &r, &i);
            res.r = r.val;
            res.i = i.val;
            return res;
        }
    """,
    libraries=["gsl", "gslcblas"])   # or a list of libraries to link with
    # (more arguments like setup.py's Extension class:
    # include_dirs=[..], extra_objects=[..], and so on)

ffibuilder.cdef("""
    // declarations that are shared between Python and C
    struct complex {
        double r;
        double i;
        };
    struct complex digamma(double x, double y);
""")

#TODO: numba is not able to do nopython mode with structs

if __name__ == "__main__":
    ffibuilder.compile(verbose=True)
