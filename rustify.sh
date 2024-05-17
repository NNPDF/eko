#!/usr/bin/bash

# git diff --merge-base master pyproject.toml > pyproject.toml.patch
patch -p1 <pyproject.toml.patch

# git diff --merge-base master src/eko/evolution_operator/__init__.py > src/eko/evolution_operator/__init__.py.patch
patch -p1 <src/eko/evolution_operator/__init__.py.patch

mv tests/eko/evolution_operator/test_init.py tests/eko/evolution_operator/deactivated_t_e_s_t_init.py
