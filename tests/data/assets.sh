# This script downloads the test assets to run both Rust and Python unit tests.
# The script is to be executed in the root directory of the repository.

# Server path
URL="https://data.nnpdf.science/eko/test-data/"
# local paths
PYDATADIR="tests/data/"
RUSTDATADIR="crates/dekoder/tests/data/"

# download from the server
curl -s -C - -o "./${PYDATADIR}v1-0.13.tar" "${URL}v1-0.13.tar"
curl -s -C - -o "./${PYDATADIR}v1-0.14.tar" "${URL}v1-0.14.tar"
curl -s -C - -o "./${PYDATADIR}v3.tar" "${URL}v3.tar"
# expose to Rust
if [ ! -f "./${RUSTDATADIR}v3.tar" ]; then
    ln -s "./../../../../${PYDATADIR}v3.tar" "./${RUSTDATADIR}"
fi
