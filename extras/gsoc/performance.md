# Performance Statistics

This file tracks benchmark results across commits and architectural experiments.
The reference benchmark is:

```bash
poe lha -m nnlo and sv
```

This runs 3 integration tests filtered to `nnlo and sv`, which is sufficiently complex to serve as a realistic benchmark. Timing is collected with `/usr/bin/time -v` (or equivalent) to capture wall clock time, CPU time, and peak memory.

---

## Baseline: `2c49156022235da1bc3582b4b61bd323e42d8d7f`

[After 522](https://github.com/NNPDF/eko/tree/2c49156022235da1bc3582b4b61bd323e42d8d7f). Original architecture with LowLevelCallable being passed a Rust binary with python callback.

### Full-suite timing (`/usr/bin/time -v`)

```text
====================== 3 passed, 12 deselected, 1 warning in 517.12s (0:08:37) ======================
        Command being timed: "poe lha -m nnlo and sv"
        User time (seconds): 667.09
        System time (seconds): 6.06
        Percent of CPU this job got: 99%
        Elapsed (wall clock) time (h:mm:ss or m:ss): 11:16.11
        Average shared text size (kbytes): 0
        Average unshared data size (kbytes): 0
        Average stack size (kbytes): 0
        Average total size (kbytes): 0
        Maximum resident set size (kbytes): 616324
        Average resident set size (kbytes): 0
        Major (requiring I/O) page faults: 1410
        Minor (reclaiming a frame) page faults: 573231
        Voluntary context switches: 3869
        Involuntary context switches: 55162
        Swaps: 0
        File system inputs: 351544
        File system outputs: 193368
        Socket messages sent: 0
        Socket messages received: 0
        Signals delivered: 0
        Page size (bytes): 4096
        Exit status: 0
```

### Per-column integration timing

`run_op_integration` is called once per output x-grid point. The column index corresponds to the position in the 60-point x-grid; fewer `quad` calls per column toward the end reflects the near-triangular sparsity of the operator matrix (a mathematical feature of the DGLAP master equation combined with Lagrange interpolation, see [architecture.md §7](./architecture.md#7-linear-algebra-and-interpolation)).

```text
Evolution: col 1/60 | total=0.743s quad=0.728s(420 calls, 1.73ms/call) setup=0.000s misc=0.014s
Evolution: col 2/60 | total=0.632s quad=0.623s(420 calls, 1.48ms/call) setup=0.000s misc=0.008s
Evolution: col 3/60 | total=0.635s quad=0.626s(413 calls, 1.51ms/call) setup=0.000s misc=0.009s
Evolution: col 4/60 | total=0.639s quad=0.629s(406 calls, 1.55ms/call) setup=0.000s misc=0.008s
Evolution: col 5/60 | total=0.616s quad=0.608s(399 calls, 1.52ms/call) setup=0.000s misc=0.008s
...
Evolution: col 56/60 | total=0.054s quad=0.052s(42 calls, 1.25ms/call) setup=0.000s misc=0.001s
Evolution: col 57/60 | total=0.035s quad=0.033s(35 calls, 0.95ms/call) setup=0.000s misc=0.001s
Evolution: col 58/60 | total=0.019s quad=0.017s(35 calls, 0.49ms/call) setup=0.000s misc=0.001s
Evolution: col 59/60 | total=0.021s quad=0.019s(35 calls, 0.53ms/call) setup=0.000s misc=0.002s
Evolution: col 60/60 | total=0.001s quad=0.000s(0 calls, 0.00ms/call) setup=0.000s misc=0.001s
```

**Average per-call cost:** ~1.5 ms/call

---

## 526:`ev_op/init` split, Numba cfunc + LowLevelCallable

After [526](https://github.com/NNPDF/eko/pull/526). `ev_op/init.py` was split. Locally, the quad kernels were wrapped in `nb.cfunc` and passed to `scipy.integrate.quad` via `scipy.LowLevelCallable`. This introduces a `scipy → Numba → Rust → Numba` call chain.

### Full-suite timing (`/usr/bin/time -v`)

```text
====================== 3 passed, 12 deselected, 3 warnings in 828.57s (0:13:48)  ======================
        Command being timed: "poe lha -m nnlo and sv"
        User time (seconds): 1214.81
        System time (seconds): 11.79
        Percent of CPU this job got: 100%
        Elapsed (wall clock) time (h:mm:ss or m:ss): 20:24.52
        Average shared text size (kbytes): 0
        Average unshared data size (kbytes): 0
        Average stack size (kbytes): 0
        Average total size (kbytes): 0
        Maximum resident set size (kbytes): 1952660
        Average resident set size (kbytes): 0
        Major (requiring I/O) page faults: 18
        Minor (reclaiming a frame) page faults: 3282977
        Voluntary context switches: 530
        Involuntary context switches: 14346
        Swaps: 0
        File system inputs: 4104
        File system outputs: 312184
        Socket messages sent: 0
        Socket messages received: 0
        Signals delivered: 0
        Page size (bytes): 4096
        Exit status: 0
```

### Per-column integration timing

```text
Evolution: col 1/60 | total=2.157s quad=2.153s(420 calls, 5.13ms/call) setup=0.001s misc=0.002s
Evolution: col 2/60 | total=2.181s quad=2.177s(420 calls, 5.18ms/call) setup=0.000s misc=0.002s
Evolution: col 3/60 | total=2.039s quad=2.035s(413 calls, 4.93ms/call) setup=0.000s misc=0.002s
Evolution: col 4/60 | total=2.020s quad=2.016s(406 calls, 4.97ms/call) setup=0.000s misc=0.002s
Evolution: col 5/60 | total=2.003s quad=2.000s(399 calls, 5.01ms/call) setup=0.000s misc=0.002s
...
Evolution: col 56/60 | total=0.181s quad=0.180s(42 calls, 4.29ms/call) setup=0.000s misc=0.001s
Evolution: col 57/60 | total=0.115s quad=0.114s(35 calls, 3.25ms/call) setup=0.000s misc=0.001s
Evolution: col 58/60 | total=0.082s quad=0.081s(35 calls, 2.32ms/call) setup=0.000s misc=0.001s
Evolution: col 59/60 | total=0.062s quad=0.061s(35 calls, 1.75ms/call) setup=0.000s misc=0.001s
Evolution: col 60/60 | total=0.001s quad=0.000s(0 calls, 0.00ms/call) setup=0.000s misc=0.001s
```

**Average per-call cost:** ~5 ms/call (~3× slower than baseline)

### Comparison with baseline

| Metric | Baseline (`2c49156...`) | [526](https://github.com/NNPDF/eko/pull/526) + cfunc/LLC | Ratio |
| ------------------------------- | ---------------------- | ---------------------- | --------- |
| Wall clock time | 11:16 | 20:24 | ~1.8× |
| User CPU time | 667 s | 1215 s | ~1.8× |
| Peak RSS | 616 MB | 1953 MB | ~3.2× |
| Avg. `quad` cost per call | ~1.5 ms | ~5 ms | ~3.3× |

### Root-cause analysis

Two separate issues contribute to the regression, one in memory and one in time.

#### Memory (~3× increase)

When Numba compiles `_quad_ker_llc` (the top-level `nb.cfunc` that `scipy` calls), it compiles the entire call graph in one shot. The resulting compiled unit is large (~1 GB of LLVM IR, typed IR, and machine code).

More importantly, any function that references a `ctypes` function pointer cannot use `cache=True`. Numba refuses because `ctypes` addresses are process-specific and emits:

```text
NumbaWarning: Cannot cache compiled function as it uses dynamic globals (such as ctypes pointers and large global arrays)
```

In the baseline architecture all hot-path `cfunc`s had `cache=True`, so Numba wrote compiled artifacts to disk and released the in-memory copies after startup. With the new LLC path those artifacts stay resident for the full process lifetime, accounting for the extra ~1.4 GB.

#### Time (~3× increase per call)

The overhead comes from the call-chain architecture itself. In each `integrate.quad` evaluation the following happens:

1. Numba's runtime entry overhead for the top-level `cfunc`.
2. Unpacking ~37 `int64` fields from the `args` array passed through the LLC.
3. Constructing ~11 `carray` views over the unpacked data.
4. Two separate `ctypes` round-trips into Rust (one for the Mellin path, one for the anomalous dimensions).

This adds a fixed overhead of several milliseconds on every `integrate.quad` node evaluation, regardless of how much mathematical work is actually done. This is in contrast to the earlier `scipy → Rust → Numba → Rust` architecture, where QUADPACK called a pure Rust C function directly via LLC and the Rust side pre-computed the Talbot path and anomalous dimensions before delegating to the Numba callback with already-computed values.

**Summary:** the `Numba → Rust → Numba` call pattern is inherently more expensive than `Rust → Numba → Rust` for this workload. The per-node overhead dominates because `integrate.quad` calls the kernel O(100) times per integral point, amplifying even small per-call costs.

### Decision

The `nb → rs → nb` architecture introduced in [526](https://github.com/NNPDF/eko/pull/526) was not viable, hence the PR was closed. The preferred short-term path remains `scipy → Rust → Numba → Rust → scipy`. See [architecture.md §5](./architecture.md#5-scipyintegratequad) for the full discussion.

---

## 542: Remove mallocs, single shared buffer

After [542](https://github.com/NNPDF/eko/pull/542), all mallocs were removed from `eko` and `ekore`, the gamma output is now written directly into a single Python-owned `f64` buffer that is pre-allocated once and passed through `QuadArgs`, minor python loops were removed.

### Full-suite timing (`/usr/bin/time -v`)

```text
====================== 3 passed, 12 deselected, 1 warning in 219.57s (0:03:39) ======================
        Command being timed: "poe lha -m nnlo and sv"
        User time (seconds): 342.81
        System time (seconds): 2.75
        Percent of CPU this job got: 101%
        Elapsed (wall clock) time (h:mm:ss or m:ss): 5:40.23
        Average shared text size (kbytes): 0
        Average unshared data size (kbytes): 0
        Average stack size (kbytes): 0
        Average total size (kbytes): 0
        Maximum resident set size (kbytes): 931916
        Average resident set size (kbytes): 0
        Major (requiring I/O) page faults: 1681
        Minor (reclaiming a frame) page faults: 679694
        Voluntary context switches: 3758
        Involuntary context switches: 3360
        Swaps: 0
        File system inputs: 366568
        File system outputs: 179328
        Socket messages sent: 0
        Socket messages received: 0
        Signals delivered: 0
        Page size (bytes): 4096
        Exit status: 0
```

### Comparison with master

| | Wall clock | pytest time |
| --- | ---------- | ----------- |
| [master](https://github.com/NNPDF/eko/tree/5aebe0abdea3e75ed7761b5ad4ce31d7fd227cd4) | 6:33 | 4:16 |
| PR #542 | 5:40 | 3:39 |
| **improvement** | | **−14.5%** |

### Rust/Python time split and memory analysis

Some debugging revealed where time is actually spent per integration:

```text
Evolution: Rust/Python split over 891240 quad evaluations — Rust (anomalous dims): 1.573 s (9.5%), Python (numba callback): 15.015 s (90.5%)
Matching:  Rust/Python split over 1847076 quad evaluations — Rust (anomalous dims): 2.654 s (11.4%), Python (numba callback): 20.665 s (88.6%)
```

~90% of integration time is spent in the Python/Numba callback. This sets a hard ceiling: even eliminating the Rust side entirely could only improve total runtime by ~10%.

Regarding the peak RSS (~910 MB): this is not a maturin issue. Switching the compile step to `pip install -e . crates/eko` produces nearly identical RSS. The process sits at ~800 MB for the entire run; the peak is caused by a transient spike at the end when computed results are compared against reference values. The 800 MB baseline itself is dominated by Numba `cfunc` compilation at import time:

```text
[MEM] eko.__init__: START                                      RSS=   58.1 MB  (baseline reset)
[MEM] interpolation: loaded (all @njit functions)              RSS=  147.3 MB  delta=+89.2 MB
[MEM] beta: loaded (all @njit functions)                       RSS=  186.4 MB  delta=+38.8 MB
[MEM] quad_ker: cb_quad_ker_qcd @cfunc loaded                  RSS=  539.9 MB  delta=+353.0 MB
[MEM] quad_ker: cb_quad_ker_ome @cfunc loaded                  RSS=  557.1 MB  delta=+17.1 MB
[MEM] quad_ker: cb_quad_ker_qed @cfunc loaded (all cfuncs done)  RSS=  722.9 MB  delta=+165.9 MB
[MEM] eko.__init__: END (all submodules loaded)                RSS=  722.9 MB
```

Of the 722.9 MB loaded at startup, only ~58 MB (~8%) is non-Numba code; the three `cfunc`s alone account for ~74%. The long-term fix is to port them to Rust (see [#383](https://github.com/NNPDF/eko/issues/383)), at which point the memory footprint will drop substantially.

### Further optimisation attempts and conclusions

- **quad_vec**: tried passing multiple integrand points at once; failed because `scipy.LowLevelCallable` cannot be used with `quad_vec`, causing a regression to 4:19.
- **SIMD / FMA**: passing batches of points to `rust_quad_ker` could allow LLVM to emit SIMD instructions, but this requires significant changes for a gain capped at ~5–10% on the Rust share (i.e. ≤1% overall).
