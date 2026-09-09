//////////////////////////////////////////////////////////////////////
// hell_shim.cc -- minimal C shim between eko (python/numba, via ctypes)
// and the HELLN library (small-x resummed splitting functions and
// heavy-quark matching functions, arXiv:1708.07510).
//
// Design constraints (see mytests/HELL_EKO_INTERFACE.md):
//  * numba can call ctypes functions from nopython code ONLY with
//    scalar arguments/returns -- no complex numbers, no strings, no
//    arrays.  Hence the compute-then-get pattern: hell_shim_dp()
//    evaluates everything for one (order, a_s, N) point and stores it;
//    hell_shim_get() returns one real scalar at a time.
//  * String arguments (the data path) are only needed at
//    initialisation, which happens from plain python -- so
//    hell_shim_load() may take a char*.
//  * One HELLNnf instance per nf is cached (table construction reads
//    ~MB of data and dominates otherwise; same design as eMELA's
//    HellnEngine).  hell_shim_init_nf() switches the active nf.
//  * HELLN's damping parameters are file-scope globals of the library
//    (HELLN::_damping/_dampingsqrt, read at evaluation time); they are
//    (re)stamped in hell_shim_load().  This is single-thread safe; eko
//    parallelises with multiprocessing (fork), where each worker gets
//    its own copy of the whole state -- load/init BEFORE the pool
//    forks (eko does: Operator.integrate initialises before creating
//    the pool).
//
// Build: see build_shim.sh (compiles HELLN from its source tree, or
// links an installed libhell-N.a if it was built with -fPIC).
//////////////////////////////////////////////////////////////////////

#include <complex>
#include <map>
#include <memory>
#include <string>

#include "hell-N.hh"

namespace {

  std::string g_datapath;
  std::map<int, std::shared_ptr<HELLN::HELLNnf>> g_cache;
  HELLN::HELLNnf* g_cur = nullptr;

  // last evaluated point: DeltaP entries + matching functions
  // idx: 0 = dPgg, 1 = dPgq, 2 = dPqg, 3 = dPqq, 4 = dKhg, 5 = dKhq
  dcomplex g_val[6];

}  // namespace

extern "C" {

// Store the data-table directory and the damping parameters.
// Returns 0 on success.  Must be called before hell_shim_init_nf.
int hell_shim_load(const char* datapath, int damping, int dampingsqrt) {
  g_datapath = datapath ? datapath : "";
  HELLN::_damping = damping;
  HELLN::_dampingsqrt = dampingsqrt;
  return g_datapath.empty() ? 1 : 0;
}

// Select (and lazily construct) the tables for nf active flavours.
// NLL is the only log order provided by the tables we use (as in
// MELA/eMELA).  Returns 0 on success.  NOTE: HELLNnf aborts/exits by
// itself when the data files are missing.
int hell_shim_init_nf(int nf) {
  auto it = g_cache.find(nf);
  if (it == g_cache.end()) {
    auto ptr = std::make_shared<HELLN::HELLNnf>(nf, HELLN::NLL, g_datapath);
    g_cache[nf] = ptr;
    g_cur = ptr.get();
  } else {
    g_cur = it->second.get();
  }
  return g_cur ? 0 : 1;
}

// Evaluate DeltaP (2x2, matched to fixed order fo: 1 = NLO, 2 = NNLO)
// and the heavy-quark matching functions DeltaKhg / DeltaKhq at
// (a_s, N), storing the six complex results for hell_shim_get().
//
//  * as is the PHYSICAL alpha_s (not a_s/4pi).
//  * (n_re, n_im) is the Mellin variable IN HELL'S CONVENTION: the
//    small-x pole sits at N = 0, i.e. one unit below the standard
//    Mellin variable used by eko (pass N_std - 1; see
//    arXiv:1708.07510 eqs. (2.40), (4.28) and the caller comments).
//  * DeltaP column-charge structure: dPgq = CF/CA dPgg,
//    dPqq = CF/CA dPqg (ibid. eq. (4.41)); dKhq = CF/CA dKhg
//    (ibid. eq. (2.28)) -- as returned by HELLN itself.
void hell_shim_dp(int fo, double as, double n_re, double n_im) {
  const dcomplex N(n_re, n_im);
  dcomplex dKhg;
  HELLN::Order ord = static_cast<HELLN::Order>(fo);
  HELLN::sqmatrix<dcomplex> m = g_cur->DeltaP(as, N, ord, &dKhg);
  g_val[0] = m.gg();
  g_val[1] = m.gq();
  g_val[2] = m.qg();
  g_val[3] = m.qq();
  g_val[4] = dKhg;
  g_val[5] = 4.0 / 9.0 * dKhg;  // CF/CA for SU(3)
}

// Return one real scalar of the last hell_shim_dp evaluation.
// idx: 0 dPgg, 1 dPgq, 2 dPqg, 3 dPqq, 4 dKhg, 5 dKhq; im: 0 real, 1 imag.
double hell_shim_get(int idx, int im) {
  if (idx < 0 || idx > 5) return 0.0;
  return im ? g_val[idx].imag() : g_val[idx].real();
}

}  // extern "C"
