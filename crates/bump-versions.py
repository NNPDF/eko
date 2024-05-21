import json
from pathlib import Path

import tomlkit

HERE = Path(__file__).parent
CRATES = json.loads((HERE / "release.json").read_text())

VERSION = "0.1.1-alpha.5"


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


def main():
    update("..", VERSION, workspace)
    for name in CRATES:
        update(name, VERSION, crate)


if __name__ == "__main__":
    main()
