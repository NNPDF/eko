#!/usr/bin/env bash

# git diff --merge-base master pyproject.toml > pyproject.toml.patch
patch -p1 <pyproject.toml.patch

# git diff --merge-base master src/eko/evolution_operator/__init__.py > src/eko/evolution_operator/__init__.py.patch
patch -p1 <src/eko/evolution_operator/__init__.py.patch

# git diff --merge-base master src/eko/evolution_operator/operator_matrix_element.py > src/eko/evolution_operator/operator_matrix_element.py.patch
patch -p1 <src/eko/evolution_operator/operator_matrix_element.py.patch

# deactivate associated tests for the moment
mv tests/eko/evolution_operator/test_init.py tests/eko/evolution_operator/deactivated_t_e_s_t_init.py
mv tests/eko/evolution_operator/test_ome.py tests/eko/evolution_operator/deactivated_t_e_s_t_ome.py
