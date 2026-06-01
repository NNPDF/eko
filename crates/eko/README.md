# eko

Rust crate for the [EKO](https://github.com/NNPDF/eko) project, which links the Rust part to Python.

This crate is the glue between the complicated math in the [ekore](https://crates.io/crates/ekore) crate, written in Rust, and the main EKO library written in Python. This crate is only intended for internal use by the EKO Python library. See the [related crates](#crates-in-the-eko-framework) for other parts of the Rust ecosystem.

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
