#!/usr/bin/bash

patch -p1 <pyproject.toml.patch
patch -p1 <src/eko/evolution_operator/__init__.py.patch
patch -p1 <tests/eko/evolution_operator/test_init.py.patch
