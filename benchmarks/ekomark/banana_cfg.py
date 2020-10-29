# -*- coding: utf-8 -*-
import pathlib
import yaml

pkg_path = pathlib.Path(__file__).parents[1]

banana_cfg = {}
with open(pkg_path / "banana.yaml", "r") as o:
    banana_cfg = yaml.safe_load(o)

banana_cfg["dir"] = pkg_path
