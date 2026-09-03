# Mock libome for Mellin N-space OMEs

This directory contains a mock C++ library exposing a C ABI for $\mathcal{O}(\alpha_s^3)$ (N3LO) space-like unpolarized Operator Matrix Elements in Mellin $N$-space.

## Interface

The C ABI declarations are provided in `ome.h`.
Currently, the functions return dummy complex values (`{0.0, 0.0}`) for testing Rust FFI integration.

## Future Integration

In the future, this mock library is intended to be replaced with the upstream implementation from [libome](https://gitlab.com/libome/libome).

Integration can be done either by:

1. **Dynamic Linking**: Linking directly against a pre-installed `libome.so`.
2. **Git Submodule**: Adding the upstream `libome` repository as a submodule and compiling it via CMake in `crates/ekore/build.rs`.
