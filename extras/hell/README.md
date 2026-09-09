# Small-x (NLL) resummation in eko via HELLN
# First implementation by Giovanni Stagnitto, with the help of Claude.

This directory contains the glue between eko and **HELLN**, the N-space
incarnation of the [HELL](https://www.roma1.infn.it/~bonvini/hell/) library,
which provides the small-x resummed contributions to the singlet DGLAP
evolution and to the heavy-quark matching conditions.  HELLN is C++ and
is **not shipped with eko**: the user compiles the thin C shim in this directory
against their own HELLN installation once, and eko loads the resulting
shared library at runtime.

## What HELLN provides

For a given number of active flavours `nf`, a value of the physical
strong coupling and a complex Mellin variable, HELLN returns

- the NLL-resummed corrections to the singlet splitting-function
  matrix, `Delta P_ij(N, alpha_s)`, i.e. the resummed result *minus*
  its fixed-order expansion, matched to NLO, NNLO or N3LO.
- the resummed correction to the heavy-quark matching function,
  `Delta K_hg` (with `Delta K_hq = CF/CA Delta K_hg`),
  normalised to the heavy-quark **pair** `h + hbar` and defined at the
  matching scale equal to the heavy-quark mass.

These are all-orders functions of `alpha_s`, obtained by interpolating
pre-computed tables (the `data/` directory of HELLN).  The tables cover
a limited `alpha_s` window per `nf` and HELLN aborts the process outside it.

## Building the shim

```sh
HELLN_DIR=/path/to/HELLN bash extras/hell/build_shim.sh
```

`HELLN_DIR` must contain `include/hell-N.hh` and either the HELLN
source tree (`src/*.cc`, compiled here with `-fPIC`) or a
`libhell-N.a` that was itself compiled with `-fPIC` (a plain static
build cannot be linked into a shared object on x86-64).  The result is
`extras/hell/libhell_shim.so`; the data tables are expected in
`$HELLN_DIR/data` (the path is passed at runtime, see below).

## Usage

The interface has two independent switches:

1. **the physics switch** is the theory-card field `smallx_res`
   (`0` = off, `1` = NLL resummation).  It is serialised with the
   output like any other theory parameter, and defaults to `0` so
   existing runcards are unaffected;
2. **the library wiring** (paths, damping) is process configuration
   done through `eko.hell.configure`.

```python
import eko
import eko.hell
from ekobox.cards import example

# 1) wire up the library -- BEFORE the first eko run of the process
eko.hell.configure(
    shim="/path/to/eko/extras/hell/libhell_shim.so",
    data="/path/to/HELLN/data",
    damping=(2, 4),   # (1-x)^2 (1-sqrt(x))^4, the HELL default
)

# 2) request the resummation in the theory card
th = example.theory()
op = example.operator()
th.order = (2, 0)                 # NLO (or (3, 0): NNLO, (4, 0): N3LO) + NLL
th.matching_order = (1, 0)        # resp. (2, 0), (3, 0)
th.smallx_res = 1
op.configs.evolution_method = "iterate-exact"
op.configs.ev_op_iterations = 240   # see "Discretisation" below

eko.solve(th, op, "resummed.tar")
```

Alternatively, set the environment variable `EKO_HELL_SHIM` to the
path of the shim: the library is then bound when `eko.hell` is
imported, and only the data path has to be given,
`eko.hell.configure(data=...)`.  This removes the ordering constraint
and is the recommended setup when eko is used from several scripts.

Supported configurations:

- QCD orders NLO, NNLO and N3LO (`order[0] = 2, 3, 4`), matched
  resummation `fo = order[0] - 1`; LO raises (nothing to match to);
- pure-QCD and unified QED x QCD evolution (`order[1] > 0`);
- evolution methods `iterate-exact` and `iterate-expanded` only (the
  resummed correction depends on `alpha_s` to all orders, so it has to
  be evaluated inside a discretised path ordering; the closed-form and
  expanded solutions raise);
- forward heavy-quark matching and its **exact** backward inverse, at
  `mu_F = m_h` (`L = 0`): other configurations raise, see the
  limitations below.

Discretisation: the resummed correction is added at every step of the
`ev_op_iterations` segments of the iterated solution (the fixed-order
singlet path ordering itself needs a few hundred segments to converge
at the 1e-4 level at large x, so `ev_op_iterations = 240` was used in
the validation runs).

Runtime: each Mellin-space evaluation of the singlet kernel calls
HELLN once per iteration step, so the resummed operators are noticeably
slower than the fixed-order ones; the table construction is done once
per `nf` in each process.

## Implementation

### The C shim (`hell_shim.cc`)

numba can only call external C functions with scalar arguments and
return values (no complex numbers, strings or arrays), hence the
compute-then-read pattern:

| symbol | role |
|---|---|
| `hell_shim_load(datapath, damping, dampingsqrt)` | stores the table directory and stamps HELLN's (file-scope, library-wide) damping parameters |
| `hell_shim_init_nf(nf)` | constructs the `HELLNnf` object for `nf` flavours (cached per `nf`) and makes it current |
| `hell_shim_dp(fo, alpha_s, n_re, n_im)` | evaluates `Delta P` (matched to `fo` = 1 NLO / 2 NNLO / 3 N3LO) and `Delta K` at one point and stores the six complex results; returns 0, or 1 (all results NaN) for an unsupported `fo` |
| `hell_shim_get(idx, im)` | returns one real scalar of the last evaluation: `idx` = 0 `Delta P_gg`, 1 `Delta P_gq`, 2 `Delta P_qg`, 3 `Delta P_qq`, 4 `Delta K_hg`, 5 `Delta K_hq`; `im` = 0 real, 1 imaginary part |

The state is process-global and not thread safe; eko parallelises with
`multiprocessing` (fork), where every worker inherits its own copy.
For this to work the tables must be loaded **before** the pool is
created, which is what `Operator.integrate` does (see below).

### The python bridge (`src/eko/hell.py`)

- The shim symbols are declared as `numba.types.ExternalFunction`
  objects and resolved *by name* by LLVM from libraries loaded with
  `llvmlite.binding.load_library_permanently`.  Unlike `ctypes` /
  cffi-ABI function pointers (which numba treats as dynamic globals and
  refuses to cache), this keeps every kernel fully cacheable
  (`cache=True` stays on everywhere) and requires no compiled python
  extension.
- `hell.dp(fo, alpha_s, n_re, n_im)` is the numba-callable evaluator
  used by the kernels; it returns a `complex128[6]` array in the order
  above.  `configure()` loads the library and binds `dp` to the real
  implementation; `init_nf(nf)` selects the tables; `is_active()`
  reports whether the bridge is wired.
- numba freezes the module-level `dp` binding at the **first
  compilation** of the calling kernels, and typing an
  `ExternalFunction` whose library is not loaded aborts hard.  Before
  `configure()` (or import with `EKO_HELL_SHIM` set), `dp` is a
  NaN-returning stub: a mis-ordered setup fails loudly, with NaN in
  every output, instead of silently dropping the resummation.
- **Cache consistency caveat**: because the stub/real binding is part
  of the compiled kernels, numba's on-disk cache must not be shared
  between processes that configured the bridge and processes that did
  not.  A stale cache shows up as NaN outputs (or a missing-symbol
  error at load time): delete the `*.nbi`/`*.nbc` files under
  `src/**/__pycache__` when switching, or set `EKO_HELL_SHIM` globally
  so that every process binds the real bridge at import.
- The runcard flag travels `TheoryCard.smallx_res` ->
  `runner/parts.py` (`_evolve_configs`) -> `Operator.config` ->
  `Operator.use_hell`, which also checks that the bridge is configured
  when the card asks for resummation (a `RuntimeError` otherwise).
  `Operator.integrate` calls `hell.init_nf(self.nf)` before forking
  the integration pool; for a matching operator `self.nf` is the number
  of flavours *below* the threshold, which is the right table both for
  the evolution segment ending there and for `Delta K`.

### Where the corrections enter

**Evolution, pure QCD** (`eko.kernels.singlet.eko_iterate`).  The
iterated solution splits `[a_0, a_1]` into `ev_op_iterations` steps
and in each step exponentiates `gamma(a_half)/beta(a_half) * delta_a`
at the mid-point coupling.  With `use_hell` the 2x2 resummed correction
evaluated at `alpha_s = 4 pi a_half` is added to `gamma` in the same
step:

```
gamma_summed -= Delta P(N - 1, 4 pi a_half) / a_half
```

(`gamma_summed` there is `gamma(a)/a`, one power of `a` having been
cancelled against `beta`, hence the division).  The `dispatcher`
forwards the Mellin variable and the switch and enforces the method /
order restrictions above.

**Evolution, unified QED x QCD** (`eko.kernels.singlet_qed.eko_iterate`).
The same 2x2 matrix is embedded in the 4x4
`(g, gamma, Sigma, Sigma_Delta)` block:

- the photon row and column receive nothing;
- the quark row feeds `Sigma` only.  eko defines
  `Sigma_Delta = n_d/n_u Sigma_u - Sigma_d` (see the flavour-space
  documentation), a combination built so that a flavour-democratic
  source cancels exactly (`n_d/n_u * n_u/n_f - n_d/n_f = 0`); since
  HELL's quark production is democratic, the `Sigma_Delta` row is
  empty;
- the quark column feeds from `Sigma` only: `Delta P_xq = CF/CA
  Delta P_xg` is blind to the incoming flavour.
- Normalisation: this kernel builds the *physical* `gamma(a_s, a_em)`
  and `beta` (LO QCD at `a_s^1`), so here `gamma -= Delta P` with no
  division.

The QED valence block and all non-singlet sectors are untouched.

**Non-singlet: no change.**  HELL's corrections are entirely of
singlet / pure-singlet nature, so the `T`/`V` sectors receive nothing
and keep their exact closed-form solutions.

**Heavy-quark matching** (`eko.evolution_operator.quad_ker.quad_ker_ome`).
After `build_ome` assembles the fixed-order operator matrix element,
the resummed matching functions are added to the `h+ = h + hbar` row
(index 2 of the singlet OME), fed by the gluon (index 0) and by `Sigma`
(index 1):

```
A[h+, g]     += Delta K_hg(N - 1, 4 pi a_s)
A[h+, Sigma] += Delta K_hq = CF/CA Delta K_hg
```

with `fo` equal to the QCD matching order (1, 2 or 3).  `Delta K_hg` is normalised
to the pair (1708.07510 eq. 2.15; its `O(alpha_s)` term, eq. 2.40, is
the pair matching kernel), which is precisely eko's `h+` row, so it
enters with unit weight.  For the exact backward inversion the resummed
*forward* operator is rebuilt and inverted numerically, so no extra
HELL ingredient is needed; the expanded (order-by-order) inverse cannot
accommodate an all-orders piece and raises.

### Conventions

- **Mellin variable**: HELL's `N` has its small-x pole at `N = 0`, one
  unit below eko's convention (the resummed objects are expansions in
  `1/N`, 1708.07510 eqs. 2.40 and 4.28).  Every call passes `N - 1`,
  in the evolution *and* in the matching.
- **Coupling**: HELLN takes the physical `alpha_s`, not eko's
  `a_s = alpha_s / 4 pi`.
- **Sign**: `Delta P` are splitting-function corrections,
  `df/dln mu^2 += Delta P . f`, while eko's anomalous dimensions obey
  `df/dln mu^2 = -gamma . f`: the correction is *subtracted* from
  `gamma`.
- **Singlet ordering**: eko's QCD singlet is `(Sigma, g)`; the
  unified block is `(g, gamma, Sigma, Sigma_Delta)`.

## N3LO and the Q0MSbar scheme

HELLN can match its NLL resummation to N3LO: `DeltaP(as, N, N3LO)`
returns the NNLO-matched tables minus the `O(alpha_s^4)` expansion of
the resummation (`gamma3NLL`, `gammaqg3NLL` in `expansionSFs.cc`, with
`mcPgg3NLL` restoring momentum conservation), and analogously the
`O(alpha_s^3)` term for `Delta K_hg`.  The interface simply forwards
`fo = 3`.

HELL performs the resummation in the Q0MSbar scheme
(1708.07510, 1805.06460 sect. 2.3), and its fixed-order expansion is
therefore the Q0MSbar one.  Up to NNLO the splitting functions
coincide with MSbar, but at N3LO the two schemes differ at NLL: only
`P_gg` is affected among the splitting functions (1805.06460 eq. 2.30),

```
gamma_gg^MSbar(N) = gamma_gg^Q0MSbar(N) + alpha_s^4 beta0 8 zeta3 gamma_0^3(N) + O(1/N^2),   gamma_0 = CA/(pi N)
```

and the `O(alpha_s^3)` heavy-quark matching function differs by the
first term of the scheme-change factor `R(M) = 1 + 8/3 zeta3 M^3`
entering through `Lambda_qg` (1708.07510 sect. 2.2.2 mentions that this
conversion is needed to reproduce the MSbar three-loop OME):

```
K_hg^(3),MSbar(N) = K_hg^(3),Q0MSbar(N) - 8/3 zeta3 (CA/pi)^2 / (3 pi N^2) + O(1/N)
```

(pair normalisation, HELL's `N`).  eko's N3LO ingredients (FHMRUVV
splitting functions, three-loop OMEs) are MSbar.  Consequently, an
N3LO+NLL run with `smallx_res = 1` produces

```
P = P^MSbar_N3LO + [P_res - P_res^exp,Q0MSbar]_{O(as^4)}
```

which is neither MSbar nor Q0MSbar: the mismatch is an
`O(alpha_s^4) ln^2(1/x)/x` term in `P_gg` (and `CF/CA` of it in `P_gq`)
and an `O(alpha_s^3) ln(1/x)/x` term in the matching function -- i.e.
NLL terms, formally within the claimed accuracy.  eko logs a warning
(`eko.hell.warn_n3lo_scheme`).  A consistent N3LO+NLL evolution
requires an MSbar implementation of the resummation, not available yet.

## Known limitations

- Resummed matching at `mu_F != m_h` (`L != 0`) is not available: the
  HELLN tables provide `Delta K` at the mass only.
- The expanded backward matching is not supported (see above).
- Only NLL is available (the log order is fixed in the shim, matching
  the tables).
- N3LO+NLL is in a mixed MSbar/Q0MSbar scheme (see above).
- Developer note: numba's on-disk cache is not invalidated when a
  *callee* changes.  After editing any jitted kernel on this path,
  delete the `*.nbi`/`*.nbc` files under `src/`.