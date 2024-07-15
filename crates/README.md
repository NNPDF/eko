# Welcome to the rusty side of EKO!

Here, we develop the Rust components of the EKO library

## Crates

- `ekore` contains the underlying collinear anomalous dimensions and the operator matrix elements
- `eko` is the glue between the Python side and the `ekore` crate

## Files

- `release.json` defines the releasing order of crates
  - only listed crates will be released
  - dependent crates should follow those they are depending on
- `katex-header.html` is an HTML snippet to be included in every docs page to inject
  KaTeX support
- `bump-versions.py` increases the Rust versions in all crates consistently
- `make_bib.py` generates the Rust function stubs which serve as fake bibliography system
