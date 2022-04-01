# -*- coding: utf-8 -*-
from dataclasses import dataclass


@dataclass
class TheoryCard:
    pto: int


@dataclass
class OperatorCard:
    xgrid: list
