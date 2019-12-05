from cffi import FFI
ffibuilder = FFI()

ffibuilder.set_source("_gsl_digamma",
   r""" // passed to the real C compiler,
        // contains implementation of things declared in cdef()
        #include <gsl/gsl_sf_psi.h>

        static void digamma(double x, double y, double out[2]) {
            gsl_sf_result r, i;
            gsl_sf_complex_psi_e(x, y, &r, &i);
            out[0] = r.val;
            out[1] = i.val;
        }
    """,
    libraries=["gsl", "gslcblas"])   # or a list of libraries to link with
    # (more arguments like setup.py's Extension class:
    # include_dirs=[..], extra_objects=[..], and so on)

ffibuilder.cdef("""
    // declarations that are shared between Python and C
    void digamma(double x, double y, double out[2]);
""")

if __name__ == "__main__":
    ffibuilder.compile(verbose=True)
