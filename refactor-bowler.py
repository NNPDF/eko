# -*- coding: utf-8 -*-
from bowler import Query

q = (
    Query("src/eko/")
    .select_module("alpha_s")
    .rename("strong_coupling")
)
