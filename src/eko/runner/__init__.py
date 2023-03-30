"""Manage steps to DGLAP solution, and operator creation."""
import os
from typing import Union

from ..io.runcards import OperatorCard, TheoryCard
from ..io.types import RawCard
from . import legacy

# Mathematica snippet to compute the scale variation terms
# (* define beta fnc - note: convention is different from AB, but matches with EKO *)
# \[Beta]f[k_][a_]:=Sum[-\[Beta][n]*a^(2+n),{n,0,k}]
# (*\[Beta]f[2][a]*)
# (* spell out the all order expansion with taylor + diff eq. *)
# aExpanded[m_,k_]:=Module[{v},
# v=Sum[1/n!t^n (D[a[t],{t,n}]/.{t->0}),{n,0,m}];
# v=v//.{Derivative[n_][a][u_]:>(D[\[Beta]f[k][a[t]],{t,n-1}]/.{t->u})};
# Collect[v,{a0,t},FullSimplify]
# ]
# Print["Alpha_s evolution"]
# Series[aExpanded[5,5],{a[0],0,5}]
# Module[{s,ss,gg},
# (* get alphas series *)
# s=Series[aExpanded[4,4],{a[0],0,4+1}];
# (*Print@s;*)
# (* we can use this to get the gamma series. We need to invert the arguments *)
# ss=s/.{t->-tt};
# gg=Series[Sum[(ss)^k  g[k-1],{k,1,4}],{a[0],0,4}];
# Print["Scheme A transformation"];
# Print@Collect[gg,{a[0],tt}];
# ]
# (* define gamma fnc *)
# \[Gamma]f[k_][a_]:=Sum[\[Gamma][n]*a^(1+n),{n,0,k}]
# Module[{v,m,w,ww},
# m=3;
# v=Sum[1/(n!)t^n (D[f[t],{t,n}]/.{t->0}),{n,0,m}];
# v=v//. {Derivative[n_][f][u_]:>(D[-\[Gamma]f[m+1][a[t]]f[t],{t,n-1}]/.{t->u})};
# v=v//. {Derivative[n_][a][u_]:>(D[\[Beta]f[m+1][a[t]],{t,n-1}]/.{t->u})};
# v=v/.{f[0]-> 1};
# Print["EKO evolution"];
# w=Collect[v+O[a[0]]^(m+1),{a[0],t},Simplify];
# Print[w];
# Print["Scheme B transformation"];
# ww = Collect[w/.{t->-t},{a[0],t},Simplify];
# Print@ww;
# ]


def solve(
    theory_card: Union[RawCard, TheoryCard],
    operators_card: Union[RawCard, OperatorCard],
    path: os.PathLike,
):
    r"""Solve DGLAP equations in terms of evolution kernel operators (EKO).

    The EKO :math:`\mathbf E_{k,j}(a_s^1\leftarrow a_s^0)` is determined in order
    to fullfill the following evolution

    .. math::
        \mathbf f(x_k,a_s^1) = \mathbf E_{k,j}(a_s^1\leftarrow a_s^0) \mathbf f(x_j,a_s^0)

    The configuration is split between the theory settings, representing
    Standard Model parameters and other defining features of the theory
    calculation, and the operator settings, those that are more closely related
    to the solution of the |DGLAP| equation itself, and determine the resulting
    operator features.

    Parameters
    ----------
    theory_card :
        theory parameters and related settings
    operator_card :
        solution configurations, and further EKO options
    path :
        path where to store the computed operator

    Note
    ----
    For further information about EKO inputs and output see :doc:`/code/IO`

    """
    legacy.Runner(theory_card, operators_card, path).compute()
