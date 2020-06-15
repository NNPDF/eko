# -*- coding: utf-8 -*-
from bowler import Query

q = (
    Query("src/eko/")
    .select_class("Threshold")
    .rename("ThresholdsConfig")
)
