# dekoder

Rust crate for reading and writing [EKO](https://github.com/NNPDF/eko) output files.

EKO produces **Evolution Kernel Operators** (EKOs) which are rank-4 tensors used in perturbative QCD calculations. This crate handles the on-disk format: a tar archive containing YAML metadata headers and LZ4-compressed NumPy (`.npz.lz4`) operator arrays.

## Citation policy

When using our code please cite

- our DOI: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3874237.svg)](https://doi.org/10.5281/zenodo.3874237)
- our paper: [![arXiv](https://img.shields.io/badge/arXiv-2202.02338-b31b1b?labelColor=222222)](https://arxiv.org/abs/2202.02338)

## Crates in the eko framework

- [dekoder](https://crates.io/crates/dekoder) - Reading and writing EKO output files
- [eko](https://crates.io/crates/eko) - Core EKO utilities
- [ekore](https://crates.io/crates/ekore) - Anomalous dimensions and operator matrix elements

## License

[GPL-3.0-or-later](https://github.com/NNPDF/eko/blob/master/LICENSE)
