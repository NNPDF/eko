# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

We may prefix the items to signal the scope:

- py: Python library
- rust: Rust library
- data: Data related changes

Items without prefix refer to a global change.

## [Unreleased](https://github.com/NNPDF/eko/compare/v0.15.2...HEAD)

### Added
- rust: N3LO unpolarized, space-like anomalous dimensions from FHMRUVV translated ([#443](https://github.com/NNPDF/eko/pull/443))

## [0.15.2](https://github.com/NNPDF/eko/compare/v0.15.1...v0.15.2) - 2025-07-14

### Changed
- data: Change version naming for unreleased eko versions from `0.0.0` to `0.0.0-post.{distance}+{commit hash}` ([#448](https://github.com/NNPDF/eko/pull/448)) ([#465](https://github.com/NNPDF/eko/pull/465))
- py: `dump_blocks` and `dump_info` in `ekobox.genpdf.export` now accept paths pointing to the actual output file. The actual output path is now also returned from these functions (soft breaking change) ([#469](https://github.com/NNPDF/eko/pull/469))
- py: Add support for Python 3.13, lift minimum NumPy version to 2

### Fixed
- py: Remove usage of `ev_op_iterations` from truncated evolution as this is an inconsistent choice ([#459](https://github.com/NNPDF/eko/pull/459))

## [0.15.1](https://github.com/NNPDF/eko/compare/v0.15.0...v0.15.1) - 2025-03-20

### Fixed
- py: Fix Python release workflow by explicitly downloading the test data
- py: Resolve files in output to actually edit the true file

## [0.15.0](https://github.com/NNPDF/eko/compare/v0.14.6...v0.15.0) - 2025-03-06
