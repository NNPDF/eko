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

`run_op_integration` is called once per output x-grid point. The column index corresponds to the position in the 60-point x-grid; fewer `quad` calls per column toward the end reflects the near-triangular sparsity of the operator matrix (a mathematical feature of the DGLAP master equation combined with Lagrange interpolation, see `architecture.md §7`).

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

| Metric | Baseline (`2c49156`) | [526](https://github.com/NNPDF/eko/pull/526) + cfunc/LLC | Ratio |
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

The `nb → rs → nb` architecture introduced in [526](https://github.com/NNPDF/eko/pull/526) was not viable, hence the PR was closed. The preferred short-term path remains `scipy → Rust → Numba → Rust → scipy`. See `architecture.md §5` for the full discussion.
