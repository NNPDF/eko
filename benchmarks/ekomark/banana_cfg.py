# -*- coding: utf-8 -*-
import pathlib
from banana import load_config

pkg_path = pathlib.Path(__file__).parents[1]
banana_cfg = load_config(pkg_path)
