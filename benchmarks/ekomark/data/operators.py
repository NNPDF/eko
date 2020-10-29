# -*- coding: utf-8 -*-
import copy

from banana.data import power_set
from banana.data.card_generator import CardGenerator
from .. import banana_cfg

class OperatorsGenerator(CardGenerator):

    def __init__(self, mode, external=None):
        super().__init__(banana_cfg.banana_cfg,mode, external)

    table_name = "operators"

    def get_all(self):
        defaults = dict()
        defaults["debug_skip_non_singlet"] = False
        defaults["debug_skip_singlet"] = False
        matrix = self.mode_cfg["operators"]["matrix"]
        full = power_set(matrix)
        cards = []
        for c in full:
            defaults.update(c)
            cards.append(copy.copy(defaults))
        return cards