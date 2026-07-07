#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INCLUDE_DIR="$SCRIPT_DIR/../dist/include/ekore_capi"
LIB_DIR="$SCRIPT_DIR/../dist/lib"
RPATH="$(realpath "$LIB_DIR")"

run_section() {
    local lang="$1" compiler="$2" ext="$3"
    local dir="$SCRIPT_DIR/$lang"
    local extra=("-I$INCLUDE_DIR")

    echo "=== Running $lang tests ==="

    for src in "$dir"/*."$ext"; do
        [ -f "$src" ] || { echo "  No $ext files found in $lang/"; continue; }
        name="$(basename "$src" ".$ext")"
        bin="$dir/$name"

        echo "  Compiling $name.$ext..."
        "$compiler" "$src" "${extra[@]}" \
            -L"$LIB_DIR" -lekore_capi -o "$bin" -Wl,-rpath,"$RPATH"

        echo "  Running $name..."
        "$bin"
        rm -f "$bin"
    done
}

run_section c    cc   c
run_section cpp  c++  cpp

echo "=== All tests passed ==="
