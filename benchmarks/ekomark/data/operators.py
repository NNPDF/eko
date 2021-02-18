# -*- coding: utf-8 -*-
from eko import interpolation

from banana.data import power_set, sql

default_card = dict(
    #interpolation_xgrid=interpolation.make_grid(30, 20).tolist(),
    interpolation_xgrid=interpolation.make_grid(2, 2).tolist(),

    interpolation_polynomial_degree=1,
    interpolation_is_log=True,
    debug_skip_non_singlet=False,
    debug_skip_singlet=False,
    ev_op_max_order=10,
    ev_op_iterations=30,
    Q2grid=[1.0e4],
)

default_card = dict(sorted(default_card.items()))


lha_config = {
    #"ev_op_max_order": [10],
    #"ev_op_iterations": [2, 10, 30],
    "Q2grid": [[1.0e4]],
}

lhapdf_config = {
    #"ev_op_max_order": [10],
    #"ev_op_iterations": [2, 10, 30],
    "Q2grid": [[20, 1.0e2, 1.0e3, 1.0e4]],
}

apfel_config = {
    "ev_op_max_order": [10],
    "ev_op_iterations": [2, 10, 30],
    "Q2grid": [[1.0e3, 1.0e4]],
}


def build(update=None):
    """
    Generate all operator card updates

    Parameters
    ----------
        update : dict
            base modifiers

    Returns
    -------
        cards : list(dict)
            list of update
    """
    cards = []
    if update is None:
        update = {}
    for c in power_set(update):
        card = dict()
        card.update(c)
        cards.append(card)
    return cards


# db interface
def load(conn, updates):
    """
    Load operator records from the DB.

    Parameters
    ----------
        conn : sqlite3.Connection
            DB connection
        update : dict
            modifiers

    Returns
    -------
        cards : list(dict)
            list of records
    """
    # add hash
    raw_records, rf = sql.prepare_records(default_card, updates)

    # insert new ones
    sql.insertnew(conn, "operators", rf)
    return raw_records
