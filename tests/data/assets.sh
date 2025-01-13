#!/usr/bin/bash
# upload: scp v0.15.tar nnpdf@data.nnpdf.science:WEB/eko/test-data

# Server path
URL="https://data.nnpdf.science/eko/test-data/"
# local paths
PYDATADIR="tests/data/"
RUSTDATADIR="crates/dekoder/tests/data/"

# download from the server
curl -s -C - -o "./${PYDATADIR}v0.13.tar" "${URL}ekov013.tar"
curl -s -C - -o "./${PYDATADIR}v0.14.tar" "${URL}ekov014.tar"
curl -s -C - -o "./${PYDATADIR}v0.15.tar" "${URL}v0.15.tar"
# expose to Rust
if [ ! -f "./${RUSTDATADIR}v0.15.tar" ]; then
    ln -s "./../../../../${PYDATADIR}v0.15.tar" "./${RUSTDATADIR}"
fi
