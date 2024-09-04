# Welcome to the rusty side of EKO!

Here, we develop the Rust components of the EKO library

## Crates

- `dekoder` handles the output file
- `ekore` contains the underlying collinear anomalous dimensions and the operator matrix elements
- `eko` is the glue between the Python side and the `ekore` crate

## Files

- `release.json` defines the releasing order of crates
  - only listed crates will be released
  - dependent crates should follow those they are depending on
- `doc-header.html` is an HTML snippet to be included in every docs page to inject
  KaTeX support and abbreviation support
- `bump-versions.py` increases the Rust versions in all crates consistently
- `make_bib.py` generates the Rust function stubs which serve as fake bibliography system
