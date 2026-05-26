# EKO Architecture

---

## 1. Build system

The patch file switches the root build backend from Poetry to Maturin. This switch is not needed. Maturin is designed to be the root build tool for projects that are primarily or completely Rust (example you told me, [pineappl](https://github.com/NNPDF/pineappl)). Currently, `eko` is still predominantly Python. Using Maturin as the root builder produces broken wheels and complicates the standard `poetry install` workflow without any benefit.

We shoulld keep Poetry as the single build tool for the `eko` Python package. Maturin's role is narrower - it compiles the Rust extension crate and installs it into the active venv. Poetry then treats that compiled extension as a path dependency. The transition to Maturin as root builder will happen only once the project is heavily or completely Rust.

The necessary changes in the current architecture for this to take place are:

- Change build backend to Poetry.
- Add eko-rs as a dependency in pyproject.toml.
- Add another .github/workflows file which publish the crates to PyPI on commit.
- The file will take care of building, publishing using maturin. The version published will be taken care by `crates/bump-versions.py`.

---

## 2. User entry point

**File:** `src/eko/runner/managed.py` : `solve(theory, operator, path)`

This is the single public function users call. It loads the two cards (`TheoryCard`, `OperatorCard`), builds an `Atlas`, and delegates to `runner/parts.py` for each segment and matching.

### High-level data flow

```text
User
 └─ runner.solve()                         [managed.py]
     └─ for each evolution segment / matching
         ├─ Operator.compute()             [evolution_operator/__init__.py]
         └─ OperatorMatrixElement.compute() [operator_matrix_element.py]
             └─ Operator.integrate()
                 └─ for each target x-grid point
                     └─ run_op_integration()
                         └─ for each source basis function j
                             └─ for each flavor label
                                 └─ scipy.integrate.quad(func, 0.5, 1-ε)
                                     └─ func called O(100) times per quad
                                         └─ quad_ker  (Python/Numba path)
                                            rust_quad_ker (Rust path)
```

---

## 3. Operator vs OperatorMatrixElement

After the path is decomposed into segments and matchings by the `Atlas`, two
different compute objects handle the integration:

| Class | File | Purpose |
| --- | --- | --- |
| `Operator` | `evolution_operator/__init__.py` | DGLAP evolution between two scales within a fixed-nf region |
| `OperatorMatrixElement` | `evolution_operator/operator_matrix_element.py` | Heavy-quark matching condition at a flavor threshold |

`OperatorMatrixElement` inherits `Operator` and they share the same `integrate()` / `run_op_integration()` machinery; they differ only in which kernel function and labels they use.

---

## 4. Integration loops

### 4.1 Outer loop

`Operator.integrate()` iterates over every point `(k, logx)` in the output x-grid. Each point is independent and hence the problem is embarrassingly parallel (currently via `multiprocessing.Pool`).

```python
# evolution_operator/__init__.py  (line 997)
with pool:
    results = pool.map(self.run_op_integration, log_grid)
```

### 4.2 Inner loop

Inside `run_op_integration`, for each target point the code iterates:

1. **Source basis function `j`** — each `BasisFunction` carries the polynomial coefficients for one source x-node (`areas_representation`).
2. **Flavor label** — a `(mode0, mode1)` pair identifying which element of the operator matrix is being computed (e.g. `(100, 100)` for quark-singlet → quark-singlet).

For each `(j, label)` pair a separate `scipy.integrate.quad` call is made.
I had a thought to parallelize this (i.e. remove the two loops), but the thing is python has GIL, and outer loop is already parallelized, so it may not improve / might deteriorate the performance. Also, we would have to create a separate cfg for each process, thus increasing the memory usage.

---

## 5. scipy.integrate.quad

### 5.1 quad_ker

```text
scipy.integrate.quad(quad_ker_partial, 0.5, 1-ε)
       ↓  calls func at each quadrature node
quad_ker(u, order, mode0, ...) [quad_ker.py, @nb.njit]
       ↓
QuadKerBase  →  integrand
       ↓
quad_ker_qcd / quad_ker_qed  →  anomalous dimensions via ekore
       ↓
kernels/singlet.py | non_singlet.py | ...  →  evolution operator matrix
       ↓
np.real(ker * integrand)  →  returned float
```

`quad_ker` is decorated `@nb.njit`, meaning Numba compiles it to machine code.
`scipy` calls the resulting function through Python's normal calling convention on every quadrature node this carries Python overhead even with Numba.

### 5.2 rust_quad_ker

```text
scipy.integrate.quad(LowLevelCallable(rust_quad_ker, &cfg), 0.5, 1-ε)
       ↓  scipy's Fortran/C backend calls the C function pointer directly
rust_quad_ker(u, *args)  [crates/eko/src/lib.rs]
       ↓
TalbotPath  →  Mellin contour + Jacobian
       ↓
ekore Rust  →  anomalous dimensions
       ↓
Python callback  →  cb_quad_ker_qcd / cb_quad_ker_qed
       ↓
Rust
       ↓
f64 returned to scipy
```

**Current transitional state:** `scipy → Rust → Python callback → Rust → scipy`

**Target state:** `scipy → Rust → Rust → scipy` (no Python in the hot loop)

### Why LowLevelCallable, not PyO3?

`scipy.integrate.quad` accepts a `LowLevelCallable`, a raw C function pointer paired with a user-data pointer.
When given an LLC, scipy's underlying Fortran/C QUADPACK routines call the function pointer directly, without ever entering the Python runtime.
There is no GIL acquisition, no Python frame allocation, and no argument marshalling through Python objects on each of the ~100–10000 evaluations per integral.

PyO3 wraps Rust functions as ordinary Python callables. If the integration
kernel were exposed via PyO3, the call chain on every point would be:

```text
scipy Fortran  →  acquire GIL  →  Python call dispatch  →  PyO3 wrapper  →  Rust  →  release GIL  →  scipy Fortran
```

This overhead per node dominates at the scale of `O(grid_size²) × O(flavors) × O(100)` calls made in a typical EKO run.

**Conclusion:** PyO3 is the right tool for the Python-Rust API boundary (configuration, result retrieval). LLC is the right tool for the integration loop. The architecture uses both in their appropriate roles.

---

## 6. ekore

I am not sure what you want me to document exactly, so I'll give my best assumptions.

### 6.1 access

In python workflow, the ekore library is a dependency of several files in the eko folder.
In rust workflow, the ekore library is completely in Rust. There's a lib.rs which acts as a bridge between python and rust. All the integrands which are to be calculated and handled pretty neatly using `cfg` on the python side, and let Rust decide what the integrand should be based on the config. I would like to keep this system as it is.

### 6.2 bottom

I am not sure what this meant. Maybe that I got to check and document if python to rust is implemented correctly or not (i.e. any mistypings), or document how much of the functions are converted and what's remaining. Please do provide more info on this.

---

## 7. Linear algebra and interpolation

### 7.1 Linear algebra for the singlet evolution solution

The non-singlet sector is a scalar equation, its solution is a scalar exponential, no matrix operations needed.

The singlet sector is different. Quarks and gluons mix under evolution, so the singlet DGLAP equation is a 2×2 matrix ODE:

$$\frac{d}{d(\ln \mu^2)} \begin{bmatrix} \Sigma \\ g \end{bmatrix} = \begin{bmatrix} \gamma_{qq} & \gamma_{qg} \\ \gamma_{gq} & \gamma_{gg} \end{bmatrix} \begin{bmatrix} \Sigma \\ g \end{bmatrix}$$

Solving this is the job of `kernels/singlet.py`, entered via `s.dispatcher(...)`. The dispatcher selects one of several solution strategies depending on `method` and `order`. Each uses different linear algebra. I won't go into detail as to what all linear algebra the functions do.

Thing to note is every method ultimately calls `ekore.anomalous_dimensions.exp_matrix_2D`, which computes a 2×2 complex matrix exponential by diagonalisation: find eigenvalues λ₊, λ₋ and projectors e₊, e₋, then:

$$\exp(M) = \exp(\lambda_{+}) \cdot e_{+} + \exp(\lambda_{-}) \cdot e_{-}$$

### 7.2 Interpolation polynomial

The inverse Mellin transform that produces one entry `E[i][j]` of the operator matrix is:

$$E[i][j] = \int E(N) \cdot p_j(N) \cdot \text{jac}(u) du$$

Here, $p_j(N)$ is the Mellin transform of the $j$-th basis polynomial evaluated at complex point $N$. This evaluation happens on every single call to `quad_ker` (~100 times per integral).

EKO works on a discrete $x$-grid using piecewise Lagrange basis polynomials. These are pre-computed once during setup, and their coefficients are stored in a compact flat float array (`areas_representation`).

During the integral, `evaluate_grid` computes $p_j(N)$ analytically piece by piece. Depending on the `is_log` flag, it delegates to one of two functions:

- **`log_evaluate_Nx`** for logarithmic interpolation.
- **`evaluate_Nx`** for linear interpolation.

I won't go into detail on the exact formulas these functions execute.

**What stays in Python vs what must move to Rust:**

The architecture splits into two layers:

1. **Setup (Stays in Python):** Classes like `BasisFunction`, `XGrid`, and `InterpolatorDispatcher` run once at startup to build the `areas_representation` array.
2. **Hot path (Must move to Rust):** `evaluate_grid`, `log_evaluate_Nx`, and `evaluate_Nx`. Once the coefficient array exists and is passed into the kernel, the Python class hierarchy is never touched again during integration.

## 8. Versioning

The project has two parallel version streams, the Python package and the Rust crates.

### 8.1 poetry-dynamic-versioning

```toml
# pyproject.toml
[build-system]
requires = ["poetry-core>=1.0.0", "poetry-dynamic-versioning"]
build-backend = "poetry_dynamic_versioning.backend"

[tool.poetry-dynamic-versioning]
format-jinja = "{% if distance == 0 %}{{ base }}{% else %}0.0.0-post.{{ distance }}+{{ commit }}{% endif %}"
```

`poetry-dynamic-versioning` reads the current git tag at build time and injects it into the package metadata. No `version` field in `pyproject.toml` is ever manually edited, it stays as the placeholder `"0.0.0"`. On a tagged commit the version becomes the tag (e.g. `1.2.0`); on an untagged commit it becomes `0.0.0-post.N+<hash>` so development builds are always distinguishable.

The version string is also written into three Python source files at build time via substitution rules:

```toml
[tool.poetry-dynamic-versioning.substitution]
files = ["src/eko/version.py", "src/ekomark/version.py", "src/ekobox/version.py"]
```

### 8.2 bump-versions.py

The Rust workspace and all crates inside `crates/` have their own `version` fields in their respective `Cargo.toml` files. They must be bumped manually before a release using:

```bash
poe bump-version   # runs: python crates/bump-versions.py $(git describe --tags)
```

`bump-versions.py` does two things:

1. Sets `workspace.package.version` to the new version.
2. Updates the `version` field of any internal cross-dependencies (e.g. ekore inside eko's dependencies) to the same version string.

The script strips the leading `v` from the git tag since Cargo does not use the `v` prefix.

---

## 10. Files targeted for Rust conversion (GSoC)

The following files contain logic that currently runs inside or is directly called by the integration callback. They must be ported to Rust. Files are listed in recommended porting order (each row depends on those above it).

### Pure arithmetic, no matrix ops (port first)

| File | What it contains | Rust notes |
| --- | --- | --- |
| `src/eko/constants.py` | Physical constants | Already ported to Rust |
| `src/eko/beta.py` | QCD/QED beta function coefficients | Tuple, `match (k.0, k.1)`, clean 1:1 port |
| `src/eko/kernels/evolution_integrals.py` | Scalar Mellin integrals | `num::Complex` arithmetic, no external deps |
| `src/eko/kernels/as4_evolution_integrals.py` | N3LO scalar integrals | Same, complex sqrt/arctan via `num::Complex` |
| `src/eko/scale_variations/exponentiated.py` | In-place gamma variation | Needs `beta.rs`, array slicing via `ndarray` |
| `src/ekore/anomalous_dimensions/__init__.py` | `exp_matrix_2D`, `exp_matrix` | `exp_matrix_2D` is pure arithmetic, `exp_matrix` needs nalgebra |

### Matrix operations (port second)

| File | What it contains | Rust notes |
| --- | --- | --- |
| `src/eko/scale_variations/expanded.py` | Scalar + 2D/4D matrix sv kernels | Generic dispatch over dim=2 and dim=4 |
| `src/eko/kernels/non_singlet.py` | Non-singlet evolution dispatcher | No matrix ops, nested `match` on `EvoMethods × order` |
| `src/eko/kernels/non_singlet_qed.py` | QED non-singlet | `np.geomspace` → manual `exp(linspace(…))`, 2D dot product |
| `src/eko/kernels/singlet_qed.py` | QED singlet iterate | Requires `exp_matrix` for the 4×4 QED case, `exp_matrix_2D` for the 2×2 QCD singlet case, const generics for dim |
| `src/eko/kernels/valence_qed.py` | Thin QED valence dispatcher | ~10 lines once `singlet_qed.rs` exists |

### Most complex (port last)

| File | What it contains | Rust notes |
| --- | --- | --- |
| `src/eko/kernels/singlet.py` | Full singlet: iterate, perturbative, truncated, decompose | `np.linalg.inv` → `nalgebra`, rank-3 complex array `(k,2,2)` with runtime `k` |
| `src/eko/interpolation.py` | **Only** `evaluate_grid`, `log_evaluate_Nx`, `evaluate_Nx` | `math.gamma` → `libm::tgamma`, rest of file stays Python |
| `src/eko/evolution_operator/quad_ker.py` | `cb_quad_ker_qcd`, `cb_quad_ker_qed`, `cb_quad_ker_ome` | The callbacks themselves, port last once all above exist |

### Files that stay in Python (do not port)

| File | Reason |
| --- | --- |
| `src/eko/kernels/__init__.py` | `EvoMethods` enum → already `method_num: u8` in `QuadArgs` |
| `src/eko/scale_variations/__init__.py` | `Modes` enum → already `sv_mode_num: u8` in `QuadArgs` |
| `src/eko/io/types.py` | Config type aliases, never on hot path |
| `src/eko/io/dictlike.py` | Serialisation framework, Python-only concern |
| `src/eko/quantities/heavy_quarks.py` | Config data container, not in hot path |
| `src/eko/matchings.py` | Orchestration stays Python, `lepton_number` is NOT in Rust. Inline in the QED callback as `if mu2_to > MTAU.powi(2) { 3 } else { 2 } |
