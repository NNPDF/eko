"""Tools to generate runcards."""
import os
from math import nan

import numpy as np
import yaml

from eko import basis_rotation as br
from eko.io import runcards
from eko.io.types import RawCard

_theory = dict(
    order=[1, 0],
    couplings=dict(
        alphas=[0.118, 91.2],
        alphaem=[0.007496252, nan],
        num_flavs_ref=None,
        max_num_flavs=6,
    ),
    num_flavs_init=None,
    num_flavs_max_pdf=6,
    intrinsic_flavors=[4],
    quark_masses={q: [mq, nan] for mq, q in zip((2.0, 4.5, 173.07), "cbt")},
    quark_masses_scheme="POLE",
    matching=[1.0, 1.0, 1.0],
    xif=1.0,
)

_operator = dict(
    mu0=1.65,
    _mugrid=[100.0],
    configs=dict(
        evolution_method="iterate-exact",
        ev_op_max_order=[10, 0],
        ev_op_iterations=10,
        interpolation_polynomial_degree=4,
        interpolation_is_log=True,
        scvar_method=None,
        inversion_method=None,
        n_integration_cores=0,
        polarized=False,
        time_like=False,
    ),
    debug=dict(
        skip_singlet=False,
        skip_non_singlet=False,
    ),
    rotations=dict(
        xgrid=np.geomspace(1e-7, 1.0, 50).tolist(),
        pids=list(br.flavor_basis_pids),
    ),
)


class example:
    """Provide runcards examples."""

    @classmethod
    def theory(cls) -> runcards.TheoryCard:
        """Provide example theory card object."""
        return runcards.TheoryCard.from_dict(_theory)

    @classmethod
    def operator(cls) -> runcards.OperatorCard:
        """Provide example operator card object."""
        return runcards.OperatorCard.from_dict(_operator)

    @classmethod
    def raw_theory(cls):
        """Provide example theory card unstructured."""
        # TODO: consider to return instead `cls.theory().raw`
        return _theory.copy()

    @classmethod
    def raw_operator(cls):
        """Provide example operator card unstructured."""
        return _theory.copy()


def dump(card: RawCard, path: os.PathLike):
    """Export the operators card.

    Parameters
    ----------
    card : dict
        card to dump
    path : str
        destination of the dumped card

    """
    with open(path, "w", encoding="utf-8") as fd:
        yaml.safe_dump(card, fd)


def load(path) -> RawCard:
    """Import the theory card specified by path.

    Parameters
    ----------
    path : str
        path to theory card in yaml format

    Returns
    -------
    dict
        loaded card

    """
    with open(path, encoding="utf-8") as fd:
        card = yaml.safe_load(fd)

    return card
