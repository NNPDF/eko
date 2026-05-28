# dekoder

Rust crate for reading and writing [EKO](https://github.com/NNPDF/eko) output files.

EKO produces **Evolution Kernel Operators** (EKOs) which are rank-4 tensors used in perturbative QCD calculations. This crate handles the on-disk format: a tar archive containing YAML metadata headers and LZ4-compressed NumPy (`.npz.lz4`) operator arrays.

## File format

An EKO archive (`.tar`) unpacks to a directory with the following layout:

```text
<eko>/
├── metadata.yaml
└── operators/
    ├── <evolution_point>.yaml     # header: scale + nf
    └── <evolution_point>.npz.lz4  # operator + error tensors
```

Each operator file stores two rank-4 arrays:

| Array | Description |
| --- | --- |
| `operator.npy` | The evolution kernel tensor |
| `error.npy` | Element-wise numerical error estimate |

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
dekoder = "0.0.1"
```

### Open an archive and inspect available operators

```rust
use std::path::PathBuf;
use dekoder::eko::{EvolutionPoint, EKO};

let eko = EKO::extract(
    PathBuf::from("my_eko.tar"),
    PathBuf::from("/tmp/eko_workdir"),
)?;

println!("Available operators: {}", eko.available_operators().len());
```

### Load a specific operator

```rust
let ep = EvolutionPoint { scale: 10000.0, nf: 4 };

if eko.has_operator(&ep) {
    let op = eko.load_operator(&ep)?;
    let tensor = op.op.unwrap();
    println!("Operator shape: {:?}", tensor.dim());
}
```

### Write back and clean up

```rust
// Write to a new archive, keep the working directory
eko.write(PathBuf::from("output.tar"))?;

// Or write and remove the working directory in one step
eko.write_and_destroy(PathBuf::from("output.tar"))?;
```

### Work with an already-extracted directory

```rust
let eko = EKO::load_opened(PathBuf::from("/tmp/eko_workdir"))?;
```

## API overview

### `EKO`

The central handle to an EKO output.

| Method | Description |
| --- | --- |
| `EKO::extract(src, dst)` | Unpack a tar archive into `dst` and load it |
| `EKO::load_opened(path)` | Load from an already-extracted directory |
| `available_operators()` | List all `EvolutionPoint`s present on disk |
| `has_operator(ep)` | Check whether a given evolution point exists |
| `load_operator(ep)` | Read and decompress the operator for `ep` |
| `write(dst)` | Pack the working directory into a tar archive |
| `write_and_destroy(dst)` | Pack, then remove the working directory |
| `destroy()` | Remove the working directory |

### `EvolutionPoint`

Identifies a point in the QCD evolution atlas by a scale `Q²` and a number of active flavors `nf`. Float comparisons use tolerances matching NumPy's `np.isclose` defaults (`rtol = 1e-5`, `atol = 1e-3`).

```rust
pub struct EvolutionPoint {
    pub scale: f64,
    pub nf: i64,
}
```

### `Operator`

A loaded rank-4 evolution operator.

```rust
pub struct Operator {
    pub op:  Option<Array4<f64>>,  // evolution kernel
    pub err: Option<Array4<f64>>,  // numerical error
}
```

## License

GPL-3.0-or-later
