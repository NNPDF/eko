# -*- coding: utf-8 -*-
"""
Database tables
"""

from banana.data.db import Base
from sqlalchemy import Boolean, Column, Integer, Text


class Operator(Base):  # pylint: disable=too-few-public-methods
    """
    Operator cards tbale
    """

    __tablename__ = "operators"

    interpolation_is_log = Column(Text)
    interpolation_polynomial_degree = Column(Integer)
    interpolation_xgrid = Column(Text)
    debug_skip_non_singlet = Column(Boolean)
    debug_skip_singlet = Column(Boolean)
    ev_op_max_order = Column(Integer)
    ev_op_iterations = Column(Integer)
    Q2grid = Column(Text)
