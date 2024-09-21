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
            n_integration_cores=1,
            debug_skip_non_singlet=False,
            debug_skip_singlet=False,
            mugrid=[10],
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
    "mugrid": [[4.4, 10, 31, 100]],
}

apfel_config = {
    # "ev_op_max_order": [10],
    # "ev_op_iterations": [2, 10, 30],
    "mugrid": [[31, 100]],
}

pegasus_config = {
    "ev_op_max_order": [10],
    "ev_op_iterations": [10],
    "mugrid": [[31]],
}

pegasus_exact_config = {
    "ev_op_max_order": [15],
    "ev_op_iterations": [20],
    "mugrid": [[31]],
}


def build(update=None):
    """Generate all operator card updates.

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
    """Load operator records from the DB.

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
