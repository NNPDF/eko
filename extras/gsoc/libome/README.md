# Mock libome for Mellin N-space OMEs

This directory contains a mock C++ library exposing a C ABI for $\mathcal{O}(\alpha_s^3)$ (N3LO) space-like unpolarized Operator Matrix Elements in Mellin $N$-space.

## Interface

The C ABI declarations are provided in `ome.h`.
Currently, the functions return dummy complex values (`{0.0, 0.0}`) for testing Rust FFI integration.
