# -*- coding: utf-8 -*-
import pathlib

from banana import load_config

banana_cfg = None

def register(path):
    path = pathlib.Path(path)
    if path.is_file():
        path = path.parent

    banana_cfg = load_config(path)
