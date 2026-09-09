r"""Bridge to the |HELL| library for small-x (|NLL| |BFKL|) resummation.
First implementation by Giovanni Stagnitto, with the help of Claude.

This module connects eko to HELLN, the N-space incarnation of the library
`HELL <https://www.roma1.infn.it/~bonvini/hell/>`, which provides the
resummed contributions to the singlet splitting-function matrix,
:math:`\Delta_n P_{ij}(N, a_s)` matched to (N)NLO, and to the
heavy-quark matching functions, :math:`\Delta K_{hg}` (and
:math:`\Delta K_{hq} = C_F/C_A\,\Delta K_{hg}`).

HELLN is C++ and lives *outside* eko: the user compiles the thin C shim
in ``extras/hell/`` against their HELLN installation once
(``HELLN_DIR=... bash extras/hell/build_shim.sh``) and wires it up::

    import eko
    eko.hell.configure(
        shim="/path/to/eko/extras/hell/libhell_shim.so",
        data="/path/to/HELLN/data",
    )
    theory_card.smallx_res = 1     # the physics switch lives in the card
    eko.solve(theory_card, operator_card, path)

The *physics* switch is the theory-card field ``smallx_res`` (0 = off,
1 = NLL); :func:`configure` only wires up the library (paths are process
configuration, not physics).

Implementation notes (see also extras/hell/README.md):

* The bridge uses :class:`numba.types.ExternalFunction`: the shim
  symbols are referenced *by name* and resolved by LLVM from libraries
  loaded with :func:`llvmlite.binding.load_library_permanently`.  Unlike
  ctypes/cffi-ABI function pointers, this keeps every kernel fully
  **numba-cacheable** (verified: no caching warnings, ``.nbi/.nbc``
  written), and it needs no compiled python extension.
* numba resolves the module-level :data:`dp` at the *first compilation*
  of the calling kernels, and typing an ExternalFunction whose library
  is not loaded aborts hard.  Therefore :data:`dp` is bound to a safe
  NaN stub until the library is loaded — either by :func:`configure`
  (call it **before the first eko run of the process**) or automatically
  at import when the environment variable ``EKO_HELL_SHIM`` points to
  the shim.  The NaN stub makes a mis-ordered setup fail loudly instead
  of silently dropping the resummation.
* **Cache consistency caveat**: because the stub/real binding is part of
  the compiled kernels, numba's on-disk cache must not be shared between
  HELL-enabled and HELL-less processes.  A stale cache manifests as NaN
  outputs (the canary) — delete the ``__pycache__`` ``*.nbi/*.nbc``
  files when switching.  Setting ``EKO_HELL_SHIM`` globally (so every
  process binds the real bridge at import) avoids the issue entirely.
* The resummed corrections are **singlet only**: HELL provides
  :math:`\Delta P_+` and :math:`\Delta P_{qg}`, from which the 2x2
  matrix follows via the colour-charge relations (1708.07510 eq. 4.41).
  The quark entry is pure-singlet, so the non-singlet sectors keep
  their exact solutions.  In the unified (QED) evolution the same 2x2
  is embedded into the (g, gamma, Sigma, Sigma_Delta) block; the photon
  row/column receive nothing, and so does :math:`\Sigma_\Delta`: eko
  defines it as :math:`n_d/n_u\,\Sigma_u - \Sigma_d`, which a
  flavour-democratic source cancels exactly.
* HELL's Mellin variable has its small-x pole at :math:`N = 0`, one
  unit below eko's convention (1708.07510 eqs. 2.40, 4.28): every call
  passes ``N - 1``.
"""

import ctypes
import logging
import os

import numba as nb
import numpy as np
from llvmlite import binding as _llvm
from numba import types as nbtypes

logger = logging.getLogger(__name__)

_ACTIVE = False
_DSO = None

# ---- numba-side externals (symbols resolved at load time by name) ----
_c_dp = nbtypes.ExternalFunction(
    "hell_shim_dp",
    nbtypes.intc(nbtypes.intc, nbtypes.float64, nbtypes.float64, nbtypes.float64),
)
_c_get = nbtypes.ExternalFunction(
    "hell_shim_get", nbtypes.float64(nbtypes.intc, nbtypes.intc)
)


@nb.njit(cache=True)
def _dp_real(fo, as_phys, n_re, n_im):  # pragma: no cover - needs the DSO
    """Evaluate HELLN at one point (library-backed implementation).

    Parameters
    ----------
    fo : int
        matched fixed order (1 = |NLO|, 2 = |NNLO|, 3 = |N3LO|); an
        unsupported value yields NaN (the shim refuses it instead of
        letting HELLN terminate the process)
    as_phys : float
        physical strong coupling :math:`\\alpha_s` (NOT :math:`a_s/4\\pi`)
    n_re : float
        real part of the Mellin variable, already shifted to HELL's
        convention (pole at N = 0, i.e. pass ``N - 1``)
    n_im : float
        imaginary part of the (shifted) Mellin variable

    Returns
    -------
    numpy.ndarray
        complex128[6]: (dPgg, dPgq, dPqg, dPqq, dKhg, dKhq)
    """
    _c_dp(fo, as_phys, n_re, n_im)
    out = np.empty(6, dtype=np.complex128)
    for i in range(6):
        out[i] = complex(_c_get(i, 0), _c_get(i, 1))
    return out


@nb.njit(cache=True)
def _dp_stub(_fo, _as_phys, _n_re, _n_im):
    """Fail loudly when the bridge is used without :func:`configure`.

    Returning NaN guarantees a mis-ordered setup (kernels compiled
    before the shim was wired up, or a stale numba cache) is visible in
    every output instead of silently losing the resummation.
    """
    return np.full(6, np.nan, dtype=np.complex128)


#: numba-callable evaluator used by the kernels; rebound to
#: :func:`_dp_real` once the shim library is loaded.
dp = _dp_stub


def is_active():
    """Return whether the HELL bridge has been wired up.

    Returns
    -------
    bool
        True after a successful :func:`configure`.
    """
    return _ACTIVE


def _load_shim(shim):
    """Load the shim both for LLVM (kernels) and ctypes (python side)."""
    global _DSO, dp
    # LLVM: makes hell_shim_* resolvable by name inside jitted kernels
    _llvm.load_library_permanently(str(shim))
    # ctypes: python-side calls (string arguments, initialisation)
    dso = ctypes.CDLL(str(shim))
    dso.hell_shim_load.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
    dso.hell_shim_load.restype = ctypes.c_int
    dso.hell_shim_init_nf.argtypes = [ctypes.c_int]
    dso.hell_shim_init_nf.restype = ctypes.c_int
    _DSO = dso
    dp = _dp_real


def configure(shim=None, data=None, damping=(2, 4)):
    """Wire up the HELLN shim.

    Must be called before the first operator computation of the process
    (numba freezes the :data:`dp` binding at first compilation, see the
    module docstring) — unless ``EKO_HELL_SHIM`` was set, in which case
    the library is already bound and only ``data`` is needed here.

    Parameters
    ----------
    shim : str or None
        path to ``libhell_shim.so`` (see ``extras/hell/build_shim.sh``);
        None = already loaded via the ``EKO_HELL_SHIM`` environment
        variable
    data : str
        path to the HELLN data tables (the ``data/`` directory of the
        HELLN installation)
    damping : tuple(int, int)
        the :math:`(1-x)^{k}(1-\\sqrt{x})^{j}` damping of the resummed
        contributions (1708.07510 eq. 4.41); the HELL default is (2, 4)
    """
    global _ACTIVE
    if _DSO is None:
        if shim is None:
            raise RuntimeError(
                "hell: no shim loaded -- pass shim=... or set EKO_HELL_SHIM"
            )
        _load_shim(shim)
    if data is None:
        raise RuntimeError("hell: the HELLN data-table path is required")
    if _DSO.hell_shim_load(str(data).encode(), damping[0], damping[1]) != 0:
        raise RuntimeError(f"hell: cannot initialise shim with data path {data}")
    _ACTIVE = True
    logger.info("hell: small-x NLL resummation wired up (data=%s)", data)


def init_nf(nf):
    """Load (or select) the HELLN tables for ``nf`` active flavours.

    Called by :meth:`eko.evolution_operator.Operator.integrate` before
    the integration pool is created, so forked workers inherit the
    tables.  Table construction is cached per nf inside the shim.

    Parameters
    ----------
    nf : int
        number of active flavours of the segment (for a matching
        operator: the flavours *below* the threshold)
    """
    if not _ACTIVE:
        raise RuntimeError("hell: init_nf called before configure")
    if _DSO.hell_shim_init_nf(int(nf)) != 0:
        raise RuntimeError(f"hell: cannot initialise HELLN tables for nf={nf}")


# automatic wiring at import: with EKO_HELL_SHIM set, the real bridge is
# bound before any kernel can be compiled -- no ordering constraint and
# a numba cache that is consistent by construction.
_env_shim = os.environ.get("EKO_HELL_SHIM")
if _env_shim:  # pragma: no cover - environment dependent
    _load_shim(_env_shim)
    logger.info("hell: shim preloaded from EKO_HELL_SHIM=%s", _env_shim)
