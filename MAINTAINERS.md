# Instructions for maintainers

## API

- Remember to always update the `Changelog.md`

### Release workflow

- Adjust the `Changelog.md` to the new version
- Commit this change and make that commit the new version (using `git tag`)

## Data format

### Instructions on what to do when we break the data format:

We need to ensure backward compatibility and thus we need to make some changes.

1. Create a file `vX.py` with X the data version (analogous to `v1.py` and `v2.py`) in `src/eko/io`.
2. Make the corresponding changes to `metadata.py` and `struct.py` (also in `src/eko/io`).
3. Modify `src/eko/version.py` to the new data version.
4. During the development of this new data version (i.e. until the moment the data version breaks again and we move to version X+1), fill up `vX.py` with all breaking changes.
5. Also implement these changes in the old `vX.py` scripts (at the moment these are `v1.py` and `v2.py`).

> **Note**: We call the unreleased eko version `v0.0`. EKO's created with unpublished versions (`0.0.0` versions) are not part of this backward compatibility system.
> They have version `0.0.0-post.{distance}+{commit hash}` and they cannot necessarily be read by newer eko versions.

### Instructions on how to update the test assets:

Using unit tests we check if eko handles legacy EKO's in the correct way. When we move to a new eko version, it is important to keep these test assets up to date.

In the directory `tests/data`, you will find the script `assets.sh` that downloads the currently existing test assets from the NNPDF server.
The script is to be executed in the root directory of the repository. Currently, these assets are only older and current eko objects, which we test for readability. The number of versions here determines how backward compatible we are. The eko objects must all correspond to the LO FFNS Les Houches benchmark.

To generate a new file, proceed as follows:

1. **Invoke the benchmark**
   This can be easily achieved with Poe. Execute (in the repo root dir):
   ```bash
   poe lha -m "ffns and lo"
   ```

2. **Move the generated object locally**
   The above command will create a file in `benchmarks/data/`. Move the file to `tests/data`, call it `vX.tar`, and include it into the unit tests (consider both Rust and Python).

3. **Upload the new file to the server**
   This can be done, e.g., with `scp`:
   ```bash
   scp vX.tar nnpdf@data.nnpdf.science:WEB/eko/test-data
   ```
   > Note that this may overwrite an existing file, so be sure about what you're doing!
   > If necessary, add the file to the list in this script.

4. **Update the data version in the workflows**
   Update both Rust and Python workflows by incrementing the data version, so the new files are also triggered in the CI.

Finally, you can test the compatibility of your eko version and the test assets in the usual way with `pytest`, specifically with the `test_legacy.py` script.
