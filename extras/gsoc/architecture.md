# EKO Architecture

---

## 1. Build system

`eko` uses **Poetry** as its root build backend. Running `rustify.sh` switches the root build backend to **Maturin**, which is designed for projects that are primarily or entirely Rust (e.g., [pineappl](https://github.com/NNPDF/pineappl)).

Maturin's role is narrower than the root builder: it compiles the Rust extension crate and installs it into the active virtual environment. Poetry then treats the compiled extension as a path dependency.

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
quad_ker_qcd / quad_ker_qed  →  anomalous dimensions via ekore [Numba]
       ↓
kernels/singlet.py | non_singlet.py | ...  →  evolution operator matrix
       ↓
np.real(ker * integrand)  →  returned float
```

**Flow:** `scipy → Numba → scipy`

`quad_ker` is decorated `@nb.njit`; Numba compiles it to machine code. The anomalous dimensions called from within `quad_ker` are also Numba-compiled (via ekore). `scipy` calls the resulting function through Python's normal calling convention on every quadrature node, carrying Python overhead at the entry point.

### 5.2 rust_quad_ker

```text
scipy.integrate.quad(LowLevelCallable(rust_quad_ker, &cfg), 0.5, 1-ε)
       ↓  scipy's Fortran/C backend calls the C function pointer directly
rust_quad_ker(u, *args)  [crates/eko/src/lib.rs]
       ↓
ekore Rust  →  anomalous dimensions
       ↓
Python callback  →  cb_quad_ker_qcd / cb_quad_ker_qed  [Numba]
       ↓
kernels/singlet.py | non_singlet.py | ... →  evolution operator matrix
       ↓
f64 returned to scipy
```

**Flow:** `scipy → Numba → Rust → Numba → scipy` *(intermediate state)*

The mid-/long-term direction is to grow the middle Rust portion until it replaces the Numba callbacks entirely: `scipy → Rust → scipy`.

---

## 6. ekore

`ekore` provides user-facing entry points into the anomalous dimensions and operator matrix elements, for example [`ad.u.s.gamma_ns_qcd`](https://github.com/NNPDF/eko/blob/ba40d2be721b647b82259bb998c69bc7feedd1dd/crates/ekore/src/anomalous_dimensions/unpolarized/spacelike.rs#L22), which assembles the underlying mathematical ingredients (the "bottom" layer of actual math implementations).

The `eko` Python library accesses the `ekore` Rust library through the `eko` Rust library. In the Rust workflow, `lib.rs` acts as the bridge between Python and Rust, with the integrand selection delegated to Rust via `cfg`.

The user-facing entry points of `ekore` are the primary focus of [#519](https://github.com/NNPDF/eko/issues/519), which tracks adding and testing C/C++ and Fortran interfaces to the Rust library.

---

## 7. Linear algebra and interpolation

### 7.1 Linear algebra for the singlet evolution solution

The non-singlet sector is a scalar equation, its solution is a scalar exponential, no matrix operations needed.

The singlet sector is different. Quarks and gluons mix under evolution, so the singlet DGLAP equation is a 2×2 matrix ODE:

$$\frac{d}{d(\ln \mu^2)} \begin{bmatrix} \Sigma \\ g \end{bmatrix} = \begin{bmatrix} \gamma_{qq} & \gamma_{qg} \\ \gamma_{gq} & \gamma_{gg} \end{bmatrix} \begin{bmatrix} \Sigma \\ g \end{bmatrix}$$

Solving this is the job of `kernels/singlet.py`, entered via `s.dispatcher(...)`. The dispatcher selects one of several solution strategies depending on `method` and `order`. Each uses different linear algebra.

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
