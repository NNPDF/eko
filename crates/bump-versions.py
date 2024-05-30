import json
import sys
from pathlib import Path

import tomlkit

HERE = Path(__file__).parent
CRATES = json.loads((HERE / "release.json").read_text())


def workspace(manifest, version):
    manifest["workspace"]["package"]["version"] = version
    return manifest


def crate(manifest, version):
    internals = set(manifest["dependencies"].keys()).intersection(CRATES)
    for dep in internals:
        manifest["dependencies"][dep]["version"] = version
    return manifest


def update(path, version, edit):
    path = HERE / Path(path) / "Cargo.toml"
    manifest = tomlkit.parse(path.read_text())
    manifest = edit(manifest, version)
    path.write_text(tomlkit.dumps(manifest))


def main(version):
    update("..", version, workspace)
    for name in CRATES:
        update(name, version, crate)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError(f"Pass a version (e.g. v0.0.0) to {sys.argv[0]}")
    # `git describe` starts with a `v` which we need to remove again
    main(sys.argv[1][1:])
