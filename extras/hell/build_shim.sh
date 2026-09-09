#!/usr/bin/env bash
#######################################################################
# build_shim.sh -- build libhell_shim.so against an external HELLN.
#
# Usage:  HELLN_DIR=/path/to/HELLN bash build_shim.sh
#
# HELLN_DIR must contain include/hell-N.hh and either
#   * a source tree src/*.cc (preferred: compiled here with -fPIC), or
#   * a libhell-N.a compiled with -fPIC (plain static builds cannot be
#     linked into a shared object on x86-64).
# The data tables are expected in $HELLN_DIR/data (pass that path to
# eko.hell.configure at runtime).
#######################################################################
set -eu
HELLN_DIR="${HELLN_DIR:?set HELLN_DIR to the HELLN installation}"
HERE="$(cd "$(dirname "$0")" && pwd)"
CXX="${CXX:-g++}"
if [ -e "$HELLN_DIR/src/hell-N.cc" ]; then
  $CXX -O2 -fPIC -shared -std=c++11 \
    "$HERE/hell_shim.cc" \
    "$HELLN_DIR/src/hell-N.cc" \
    "$HELLN_DIR/src/math/special_functions.cc" \
    "$HELLN_DIR/src/expansionSFs.cc" \
    -I"$HELLN_DIR/include" \
    -o "$HERE/libhell_shim.so"
else
  $CXX -O2 -fPIC -shared -std=c++11 \
    "$HERE/hell_shim.cc" \
    -I"$HELLN_DIR/include" \
    "$HELLN_DIR/libhell-N.a" \
    -o "$HERE/libhell_shim.so"
fi
echo "built $HERE/libhell_shim.so"
