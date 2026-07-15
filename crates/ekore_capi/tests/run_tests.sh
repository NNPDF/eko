#!/usr/bin/env bash
set -e

# setup environment
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INCLUDE_DIR="$SCRIPT_DIR/../dist/include/ekore_capi"
LIB_DIR="$SCRIPT_DIR/../dist/lib"
RPATH="$(realpath "$LIB_DIR")"

run_section() {
    local lang="$1" compiler="$2" ext="$3"
    local dir="$SCRIPT_DIR/$lang"
    local utils_obj="" extra=()

    echo "=== Running $lang tests ==="

    # compile utils for everybody
    if [ -f "$dir/utils.$ext" ]; then
        utils_obj="$dir/utils.o"
        echo "  Compiling utils.$ext..."
        "$compiler" -c "$dir/utils.$ext" -I"$INCLUDE_DIR" -o "$utils_obj"
        extra+=("$utils_obj")
    fi

    extra+=("-I$INCLUDE_DIR")

    # iterate test files
    for src in "$dir"/*."$ext"; do
        [ -f "$src" ] || { echo "  No $ext files found in $lang/"; continue; }
        name="$(basename "$src" ".$ext")"
        [ "$name" = "utils" ] && continue
        bin="$dir/$name"
        echo "  Compiling $name.$ext..."
        "$compiler" "$src" "${extra[@]}" \
            -L"$LIB_DIR" -lekore_capi -o "$bin" -Wl,-rpath,"$RPATH"
        echo "  Running $name..."
        "$bin"
        rm -f "$bin"
    done

    rm -f "$utils_obj"
}

run_section c       cc       c

echo "=== All tests passed ==="
