#!/bin/bash
#Find conda
source ~/.bashrc
set -e
set -o pipefail
set -u
set -v

#Build package
CONDA_PY=$CONDA_PY conda build -q conda-recipe
if [ $? != 0 ]; then
	echo failed to build
	exit 1
fi