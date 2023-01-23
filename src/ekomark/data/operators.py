"""Operator card configurations."""
from banana.data import cartesian_product, sql

from eko import interpolation

from . import db

default_card = dict(
    sorted(
        dict(
            interpolation_xgrid=interpolation.make_grid(30, 20).tolist(),
            interpolation_polynomial_degree=4,
            interpolation_is_log=True,
            ev_op_max_order=10,
            ev_op_iterations=10,
            backward_inversion="expanded",
            n_integration_cores=0,
            debug_skip_non_singlet=False,
            debug_skip_singlet=False,
            Q2grid=[100],
            inputgrid=None,
            targetgrid=None,
            inputpids=None,
            targetpids=None,
            polarized=False,
            time_like=False,
        ).items()
    )
)


lhapdf_config = {
    # "ev_op_max_order": [10],
    # "ev_op_iterations": [2, 10, 30],
    "Q2grid": [[20, 1.0e2, 1.0e3, 1.0e4]],
}

apfel_config = {
    # "ev_op_max_order": [10],
    # "ev_op_iterations": [2, 10, 30],
    "Q2grid": [[1.0e3, 1.0e4]],
}

pegasus_config = {
    "ev_op_max_order": [10],
    "ev_op_iterations": [10],
    "Q2grid": [[1.0e3]],
}

pegasus_exact_config = {
    "ev_op_max_order": [15],
    "ev_op_iterations": [20],
    "Q2grid": [[1.0e3]],
}


def build(update=None):
    """
    Generate all operator card updates.

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
    for c in cartesian_product(update):
        card = {}
        card.update(c)
        cards.append(card)
    return cards


# db interface
def load(session, updates):
    """
    Load operator records from the DB.

    Parameters
    ----------
    session : sqlalchemy.session.Session
        DB ORM session
    updates : dict
        modifiers

    Returns
    -------
    cards : list(dict)
        list of records
    """
    # add hash
    raw_records, df = sql.prepare_records(default_card, updates)

    # insert new ones
    sql.insertnew(session, db.Operator, df)
    return raw_records
