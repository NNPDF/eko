# Crates

...

## Files

- `release.json` defines the releasing order of crates
  - only listed crates will be released
  - dependent crates should follow those they are depending on
- `katex-header.html` is an HTML snippet to be included in every docs page to inject
  KaTeX support
