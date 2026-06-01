# dekoder

Rust crate for reading and writing [EKO](https://github.com/NNPDF/eko) output files.

EKO produces **Evolution Kernel Operators** (EKOs) which are rank-4 tensors used in perturbative QCD calculations. This crate handles the on-disk format: a tar archive containing YAML metadata headers and LZ4-compressed NumPy (`.npz.lz4`) operator arrays.

## License

GPL-3.0-or-later
