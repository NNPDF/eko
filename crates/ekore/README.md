# ekore

Rust crate containing the core expressions used in the [EKO](https://github.com/NNPDF/eko) library.

This crate collects the anomalous dimensions and operator matrix elements currently available in the literature - please see our [list of references](https://eko.readthedocs.io/en/latest/zzz-refs.html).
The crate also provides the necessary mathematical tools, such as harmonic sums, to deal with the relevant expressions.

## Citation policy

When using our code please cite

- our DOI: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3874237.svg)](https://doi.org/10.5281/zenodo.3874237)
- our paper: [![arXiv](https://img.shields.io/badge/arXiv-2202.02338-b31b1b?labelColor=222222)](https://arxiv.org/abs/2202.02338)

## Crates in the eko framework

- [dekoder](https://crates.io/crates/dekoder) - Reading and writing EKO output files
- [eko](https://crates.io/crates/eko) - Core EKO utilities
- [ekore](https://crates.io/crates/ekore) - Anomalous dimensions and operator matrix elements
- [ekore_capi](https://crates.io/crates/ekore_capi) - C API for ekore

## License

[GPL-3.0-or-later](https://github.com/NNPDF/eko/blob/master/LICENSE)
