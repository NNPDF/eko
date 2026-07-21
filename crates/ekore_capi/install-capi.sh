#!/bin/sh

# WARNING: do not commit changes to this file unless you've checked it against
# `shellcheck` (https://www.shellcheck.net/); run `shellcheck install-capi.sh`
# to make sure this script is POSIX shell compatible; we cannot rely on bash
# being present

set -eu

prefix=
version=
target=

while [ $# -gt 0 ]; do
    case $1 in
        --version)
            if [ $# -lt 2 ]; then
                echo "Error: --version requires a value" >&2
                exit 1
            fi
            version=$2
            shift
            shift
            ;;
        --version=*)
            version=${1#--version=}
            shift
            ;;
        --prefix)
            if [ $# -lt 2 ]; then
                echo "Error: --prefix requires a value" >&2
                exit 1
            fi
            prefix=$2
            shift
            shift
            ;;
        --prefix=*)
            prefix=${1#--prefix=}
            shift
            ;;
        --target)
            if [ $# -lt 2 ]; then
                echo "Error: --target requires a value" >&2
                exit 1
            fi
            target=$2
            shift
            shift
            ;;
        --target=*)
            target=${1#--target=}
            shift
            ;;
        *)
            echo "Error: argument '$1' unknown" >&2
            exit 1
            ;;
    esac
done

if [ -z "${target}" ]; then
    case $(uname -m):$(uname -s) in
        arm64:Darwin)
            target=macos-aarch64;;
        x86_64:Darwin)
            target=macos-x86_64;;
        aarch64:Linux)
            target=linux-aarch64;;
        x86_64:Linux)
            target=linux-x86_64;;
        *)
            echo "Error: unknown target, uname = '$(uname -a)'"
            exit 1;;
    esac
fi

# if no prefix is given, prompt for one
if [ -z "${prefix}" ]; then
    # read from stdin (`<&1`), even if piped into a shell
    printf "Enter installation path: "
    read -r <&1 prefix
    echo
fi

# we need the absolute path; use `eval` to expand possible tilde `~`
eval mkdir -p "${prefix}"
eval cd "${prefix}"
prefix=$(pwd)
cd - >/dev/null

# if no version is given, use the latest version tag
if [ -z "${version}" ]; then
    version=$(curl -s -o /dev/null -w '%{redirect_url}' \
        "https://github.com/NNPDF/eko/releases/latest" | sed 's:.*/tag/::')
fi

url="https://github.com/NNPDF/eko/releases/download/${version}/ekore_capi-${version}-${target}.tar.gz"

echo "prefix:  '${prefix}'"
echo "target:  '${target}'"
echo "version: '${version}'"
echo "URL:     '${url}'"

curl -fsSL "${url}" | tar xzf - -C "${prefix}"

# Patch the pkg-config file
sed "s:prefix=/:prefix=${prefix}/:" "${prefix}"/lib/pkgconfig/ekore_capi.pc > \
    "${prefix}"/lib/pkgconfig/ekore_capi.pc.new
mv "${prefix}"/lib/pkgconfig/ekore_capi.pc.new "${prefix}"/lib/pkgconfig/ekore_capi.pc

pcbin=

if command -v pkg-config >/dev/null; then
    pcbin=$(command -v pkg-config)
elif command -v pkgconf >/dev/null; then
    pcbin=$(command -v pkgconf)
else
    echo
    echo "Error: neither \`pkg-config\` nor \`pkgconf\` found. At least one is needed for the CAPI to be found"
    exit 1
fi

# check whether the library can be found
if "${pcbin}" --exists ekore_capi; then
    found_prefix=$(cd "$("${pcbin}" --variable=prefix ekore_capi)" && pwd)

    if [ "${prefix}" != "${found_prefix}" ]; then
        echo
        echo "Warning: Your PKG_CONFIG_PATH environment variable isn't properly set."
        echo "It appears a different installation of Ekore C-API is found:"
        echo
        echo "  ${found_prefix}"
        echo
        echo "Remove this installation or reorder your PKG_CONFIG_PATH"
    fi
else
    echo
    echo "Warning: Your PKG_CONFIG_PATH environment variable isn't properly set."
    echo "Try adding"
    echo
    echo "  export PKG_CONFIG_PATH=${prefix}/lib/pkgconfig:\"\${PKG_CONFIG_PATH:-}\""
    echo
    echo "to your shell configuration file"
fi
