import os
import tarfile

import yaml


def theory(card: dict) -> int:
    if "_version" not in card:
        return 0

    return card["_version"]


def operator(card: dict) -> int:
    if "_version" not in card:
        return 0

    return card["_version"]


def output(path: os.PathLike) -> int:
    with tarfile.open(path, "r") as tar:
        try:
            metafile = tar.extractfile("metadata.yaml")
        except KeyError:
            return 0

    if metafile is None:
        return 0

    metadata = yaml.safe_load(metafile)
    return metadata["data_version"]
