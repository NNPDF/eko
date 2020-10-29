# -*- coding: utf-8 -*-
from banana.data import theories
from .. import banana_cfg

from . import operators

generate_theories = theories.TheoriesGenerator.get_run_parser(banana_cfg.banana_cfg)

generate_operators = operators.OperatorsGenerator.get_run_parser(banana_cfg.banana_cfg)
