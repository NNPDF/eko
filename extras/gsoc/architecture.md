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

**Flow:** `scipy → Rust → Numba → Rust → scipy`

`scipy` skips Python overhead entirely by calling `rust_quad_ker` via a `LowLevelCallable` C function pointer. Rust handles the anomalous dimensions but then delegates back out to Numba callbacks for the evolution operator matrix, before returning to Rust and finally to `scipy`.

The long-term goal is to grow the Rust portion until it replaces Numba entirely: `scipy → Rust → scipy`.

### 5.3 Attempted optimisation: LowLevelCallable with nb.cfunc (#526)

During [526](https://github.com/NNPDF/eko/pull/526) an attempt was made to invert the call chain, i.e. replacing the architecture with `scipy → Numba → Rust → Numba` by wrapping the top-level kernel in `nb.cfunc` and passing it to `scipy` via `LowLevelCallable`. The intended chain was:

```text
scipy.integrate.quad(LowLevelCallable(quad_ker_llc, &cfg), 0.5, 1-ε)
       ↓  scipy's Fortran/C backend calls the C function pointer directly
quad_ker_llc(u, *args)  [nb.cfunc]
       ↓
QuadKerBase  →  integrand
       ↓  ctypes call into Rust
ekors.qcd_gamma_singlet / ekors.qcd_gamma_ns  [crates/eko/src/lib.rs]
       ↓
ekore Rust  →  anomalous dimensions
       ↓  back to Numba
kernels/singlet.py | non_singlet.py | ...  →  evolution operator matrix
       ↓
f64 returned to scipy
```

**Flow:** `scipy → Numba → Rust → Numba → scipy`

Benchmarked against the master baseline using `poe lha -m nnlo and sv`, this approach was significantly worse on both metrics. See [performance.md](./performance.md) for the full numbers; the key comparison is:

| Metric | Master (rs → nb → rs) | #526 (nb → rs → nb) | Ratio |
| --- | --- | --- | --- |
| Wall clock | 11:16 | 20:24 | ~1.8× |
| Peak RSS | 616 MB | 1953 MB | ~3.2× |
| Avg. quad cost/call | ~1.5 ms | ~5 ms | ~3.3× |

Two root causes were identified.

#### Memory regression (~3×)

When Numba compiles `_quad_ker_llc` it compiles the entire call graph in one pass, producing a compiled unit of approximately 1 GB. More critically, any function that holds a `ctypes` function pointer cannot use `cache=True` because `ctypes` addresses are process-specific.

In the master architecture all hot-path `cfunc`s have `cache=True`, so Numba writes compiled artifacts to disk and releases the in-memory copies after startup. With the LLC path, all LLVM IR, typed IR, and machine code stay resident for the full process lifetime.

#### Time regression (~3× per quad call)

The overhead is structural. Every `integrate.quad` node evaluation incurs substantial fixed overhead (see [performance.md](./performance.md)), adding several milliseconds per node which amplifies into a large total regression.

In the master architecture, QUADPACK calls a pure Rust C function directly via LLC. Rust pre-computes the Talbot path and anomalous dimensions before handing already-prepared values to the Numba callback, so the per-call cost stays low.

**Decision:** [526](https://github.com/NNPDF/eko/pull/526) was not merged. The `Numba → Rust → Numba` pattern is not viable for this workload.

---

## 6. ekore

`ekore` is the core computational engine in Rust providing the underlying physics calculations for anomalous dimensions and operator matrix elements (OMEs), along with Mellin-space harmonic sum evaluation and caching. For example, [`ad.u.s.gamma_ns_qcd`](https://github.com/NNPDF/eko/blob/ba40d2be721b647b82259bb998c69bc7feedd1dd/crates/ekore/src/anomalous_dimensions/unpolarized/spacelike.rs#L22) assembles the analytical and numerical formulas for non-singlet QCD anomalous dimensions.

To make `ekore` accessible across various ecosystems and languages without reimplementing the physics kernels, the repository provides dedicated interface crates:

- **`eko` (Rust crate):** The internal bridge connecting Python EKO to `ekore` during operator integration (`rust_quad_ker`).
- **`ekore_capi`:** Exposes `ekore` via a C-compatible ABI for C, C++, and Fortran callers.
- **`ekore_py`:** Exposes `ekore` directly to Python using PyO3 bindings.

---

### 6.1 eko crate (Internal Python/Rust bridge)

**Directory:** `crates/eko`

The `eko` Python library accesses the `ekore` Rust library through the `eko` Rust library. In the Rust workflow, `lib.rs` acts as the bridge between Python and Rust, with the integrand selection delegated to Rust via `cfg`.

This internal crate exports C function pointers (such as `rust_quad_ker`) that `scipy.integrate.quad` calls via `LowLevelCallable` without interpreter overhead at every quadrature node. It coordinates between `ekore` (for anomalous dimensions) and Numba callbacks (for the evolution operator matrix).

---

### 6.2 ekore_capi (C, C++, and Fortran Interface)

**Directory:** `crates/ekore_capi`

`ekore_capi` exposes `ekore`'s anomalous dimensions and operator matrix elements through a stable, standard C Application Binary Interface (ABI) (`#[no_mangle]`, `extern "C"`).

#### Architecture & Design

- **C-Compatible Types:** Complex numbers are exposed using `#[repr(C)] ComplexF64` (consisting of adjacent `re` and `im` double-precision floats), matching standard C99 `double complex` and `num::Complex<f64>`.
- **Opaque Handles:** The Mellin-space harmonic cache is managed through opaque heap pointers (`Cache`), initialized with `cache_new(n_re, n_im)` and explicitly freed with `cache_delete(c)`.
- **Buffer Convention:** Functions return perturbative series as flattened arrays. Each calculation `<name>` is paired with `<name>_result_len(...)` returning the required buffer size, and `<name>(..., result)` which writes into the caller-allocated array.
- **Header & Metadata Generation:** Uses [`cargo-c`](https://crates.io/crates/cargo-c) and [`cbindgen`](https://github.com/mozilla/cbindgen) to generate the C header (`ekore_capi.h`) and `pkg-config` file (`ekore_capi.pc`).

#### Multi-Language Support

- **C & C++:** Directly includable via `#include <ekore_capi/ekore_capi.h>` and linkable via `pkg-config --cflags --libs ekore_capi`.
- **Fortran:** Consumable via Fortran 2003+ `iso_c_binding` interoperability definitions, allowing Fortran codes to bind to C function symbols and struct memory layouts.
- **Other FFI Consumers:** Usable in any language or environment supporting C dynamic library loading (e.g., Julia, Python `ctypes`/`cffi`, Mathematica).

#### Distribution & Tooling

- **Pre-built binaries:** An installer script `crates/ekore_capi/install-capi.sh` allows downloading and installing pre-built shared/static libraries for Linux and macOS.
- **Poe tasks:**
  - `poe build-capi`: Compiles and packages the C library into `crates/ekore_capi/dist/`.
  - `poe ctest`: Runs the C test runner script (`crates/ekore_capi/tests/run_tests.sh`).

#### Testing Status

- **C Tests:** A suite of C tests exists under `crates/ekore_capi/tests/c/` (covering unpolarized anomalous dimensions, polarized anomalous dimensions, OMEs, and constants), compiled and verified via `run_tests.sh`.
- **Fortran / C++ Tests:** While Fortran and C++ interoperability is architecturally supported via the C ABI, dedicated automated test suites (especially for Fortran) are not yet present in the repository (tracked under [#519](https://github.com/NNPDF/eko/issues/519)).

---

### 6.3 ekore_py (Python Bindings via PyO3)

**Directory:** `crates/ekore_py`

`ekore_py` provides direct, idiomatic Python bindings to `ekore` built using [PyO3](https://pyo3.rs/) and [Maturin](https://www.maturin.rs/).

#### Purpose & Distinctions

- **Standalone Python Access:** Unlike the internal `eko` crate (`crates/eko`), which serves as a specialized integration bridge for EKO's evolution runner and `scipy.integrate.quad` callbacks (`rust_quad_ker`), `ekore_py` provides direct, general-purpose Python access to the raw anomalous dimensions, operator matrix elements, and harmonic cache.
- **Native Types:** Functions automatically convert between Rust and Python numeric and array types (e.g., returning NumPy-compatible arrays or Python complex types) with minimal overhead.

---

### 6.4 Summary of Physics Engine Crates

| Crate | Directory | Target Audience / Language | Interface Mechanism |
| --- | --- | --- | --- |
| `ekore` | `crates/ekore` | Pure Rust | Native Rust API |
| `eko` | `crates/eko` | Python (`eko` evolution runner) | Internal C FFI / `LowLevelCallable` bridge |
| `ekore_capi` | `crates/ekore_capi` | C, C++, Fortran, FFI consumers | C ABI (`extern "C"`, `cbindgen`, `cargo-c`) |
| `ekore_py` | `crates/ekore_py` | Python (standalone physics API) | PyO3 / Maturin extension module |

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

---

## 8. Output format and dekoder

### 8.1 EKO on-disk archive format

When `eko` finishes computation, the result is written into an uncompressed `.tar` archive containing metadata headers and operator tensors. The archive unpacks to the following layout:

```text
<eko_output.tar>/
├── metadata.yaml
└── operators/
    ├── <evolution_point_1>.yaml     # header: target scale + active flavor count (nf)
    ├── <evolution_point_1>.npz.lz4  # LZ4-compressed operator and error tensors
    ├── <evolution_point_2>.yaml
    └── <evolution_point_2>.npz.lz4
```

Inside each `<evolution_point>.npz.lz4`, two rank-4 NumPy arrays are stored:

| Array | Description |
| --- | --- |
| `operator.npy` | The evolution kernel operator (rank-4 tensor: $E_{ij,kl}$) |
| `error.npy` | Element-wise numerical error estimates from the quadrature integration |

### 8.2 dekoder crate

**Directory:** `crates/dekoder`

`dekoder` is a standalone, pure-Rust I/O crate for reading, inspecting, and writing EKO output files.

#### Architecture & Role

- **Independent I/O Layer:** Unlike `ekore` or `eko`, `dekoder` does not participate in the physics computation or numerical integration. It has no dependency on `ekore`, depending solely on I/O and serialization crates (`tar`, `lz4_flex`, `ndarray`, `ndarray-npy`, `yaml-rust2`).
- **Downstream Interoperability:** It allows downstream physics tools and frameworks (such as [PineAPPL](https://github.com/NNPDF/pineappl) or external C++/Rust tools) to natively load, manipulate, and write EKO evolution operators without requiring a Python runtime or the full EKO framework.
- **Key API Constructs:**
  - `EKO`: Represents an extracted EKO archive in a working directory; provides methods like `EKO::extract`, `EKO::load_opened`, `load_operator`, `write`, and `write_and_destroy`.
  - `EvolutionPoint`: Defines a query point with target `scale` ($\mu^2$) and active flavor count `nf`.
  - `Operator`: Holds the loaded 4D `ndarray::Array4<f64>` operator and error tensors.

---

## 9. Versioning

The project has two parallel version streams, the Python package and the Rust crates.

### 9.1 poetry-dynamic-versioning

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

### 9.2 bump-versions.py

The Rust workspace and all crates inside `crates/` have their own `version` fields in their respective `Cargo.toml` files. They must be bumped manually before a release using:

```bash
poe bump-version   # runs: python crates/bump-versions.py $(git describe --tags)
```

`bump-versions.py` does two things:

1. Sets `workspace.package.version` to the new version.
2. Updates the `version` field of any internal cross-dependencies (e.g. ekore inside eko's dependencies) to the same version string.

The script strips the leading `v` from the git tag since Cargo does not use the `v` prefix.
