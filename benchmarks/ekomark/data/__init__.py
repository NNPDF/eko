# -*- coding: utf-8 -*-
from banana.data import theories
from .. import banana_cfg

generate_theories = theories.TheoriesGenerator.get_run_parser(banana_cfg.banana_cfg)
