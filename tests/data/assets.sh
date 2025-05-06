#!/usr/bin/bash
# This script downloads the test assets to run both Rust and Python unit tests.
# The script is to be executed in the root directory of the repository.
# Currently these assets are only older and current eko objects, which we test
# for readability. The number of versions here determines how backward
# compatible we are. The eko objects have to correspond all to the LO FFNS
# Les Houches benchmark. To generate a new file proceed as follows:
# 1. Invoke the benchmark.
#    This can be easily achieved with Poe, i.e. execute (in the repo root dir):
#      `poe lha -m "ffns and lo"`
# 2. Move the generated object locally.
#    The above command will create a file in `benchmarks/data/`. Move the file
#    here and include it into the unit tests (consider both Rust and Python).
# 3. Upload the new file to the server.
#    This can be done e.g. with `scp`:
#      `scp v0.0.tar nnpdf@data.nnpdf.science:WEB/eko/test-data`
#    Note that this may overwrite an existing file, so be sure on what you do!
#    If necessary, add the file to the list in this script.
# 4. Update the data version in the workflows.
#    Update both Rust and Python workflows, by incrementing the data version,
#    such that the new files are also triggered in the CI.

# Server path
URL="https://data.nnpdf.science/eko/test-data/"
# local paths
PYDATADIR="tests/data/"
RUSTDATADIR="crates/dekoder/tests/data/"

# download from the server
curl -s -C - -o "./${PYDATADIR}v0.13.tar" "${URL}v0.13.tar"
curl -s -C - -o "./${PYDATADIR}v0.14.tar" "${URL}v0.14.tar"
curl -s -C - -o "./${PYDATADIR}v0.15.tar" "${URL}v0.15.tar"
# expose to Rust
if [ ! -f "./${RUSTDATADIR}v0.15.tar" ]; then
    ln -s "./../../../../${PYDATADIR}v0.15.tar" "./${RUSTDATADIR}"
fi
