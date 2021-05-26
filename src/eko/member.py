# -*- coding: utf-8 -*-
from numbers import Number
import numpy as np


class OpMember:
    """
    A single operator for a specific element in evolution basis.

    This class provide some basic mathematical operations such as products.
    It can also be applied to a pdf vector via :meth:`apply_pdf`.
    This class will never be exposed to the outside, but will be an internal member
    of the :class:`Operator` and :class:`PhysicalOperator` instances.

    Parameters
    ----------
        value : np.array
            operator matrix
        error : np.array
            operator error matrix
    """

    def __init__(self, value, error):
        self.value = np.array(value)
        self.error = np.array(error)

    def copy(self):
        return self.__class__(self.value.copy(), self.error.copy())

    def __matmul__(self, operator_member):
        rval = operator_member.value
        rerror = operator_member.error
        lval = self.value
        ler = self.error
        new_val = np.matmul(lval, rval)
        # TODO check error propagation
        new_err = np.abs(np.matmul(lval, rerror)) + np.abs(np.matmul(ler, rval))
        return self.__class__(new_val, new_err)

    def __mul__(self, other):
        if not isinstance(other, Number):
            raise NotImplementedError(f"Can't multiply OpMember and {type(other)}")
        return self.__class__(other * self.value, other * self.error)

    def __add__(self, operator_member):
        if isinstance(operator_member, Number):
            # we only allow the integer 0 as alias for the true zero operator
            if operator_member != 0:
                raise ValueError(
                    "The only number we can sum to is 0 (as alias for the zero operator)"
                )
            rval = operator_member
            rerror = 0.0
        elif isinstance(operator_member, self.__class__):
            rval = operator_member.value
            rerror = operator_member.error
        else:
            raise NotImplementedError(f"Can't sum OpMember and {type(operator_member)}")
        new_val = self.value + rval
        new_err = self.error + rerror
        return self.__class__(new_val, new_err)

    def __neg__(self):
        return self.__class__(-self.value.copy(), self.error.copy())

    def __eq__(self, operator_member):
        return np.allclose(self.value, operator_member.value)

    def __sub__(self, operator_member):
        return self.__add__(-operator_member)

    def __radd__(self, operator_member):
        return self.__add__(operator_member)

    def __rsub__(self, operator_member):
        return self.__radd__(-operator_member)

    def __rmul__(self, other):
        return self.__mul__(other)


class MemberName:
    """
    Operator member name in operator evolution space

    Parameters
    ----------
        name : str
            operator name
    """

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return self.name

    def __eq__(self, other):
        return self.name == other.name

    def __hash__(self):
        return hash(str(self))

    def _split_name(self):
        """Splits the name according to target.input"""
        # we need to do this late, as in raw mode the name to not follow this principle
        name_spl = self.name.split(".")
        if len(name_spl) != 2:
            raise ValueError("The operator name has no valid format: target.input")
        for k in [0, 1]:
            name_spl[k] = name_spl[k].strip()
            if len(name_spl[k]) <= 0:
                raise ValueError("The operator name has no valid format: target.input")
        return name_spl

    @property
    def target(self):
        """Returns target flavor name (given by the first part of the name)"""
        return self._split_name()[0]

    @property
    def input(self):
        """Returns input flavor name (given by the second part of the name)"""
        return self._split_name()[1]


class OperatorBase:
    """
    Abstract base class to hold a dictionary of interpolation matrices.

    Parameters
    ----------
        op_members : dict
            mapping of :class:`MemberName` onto :class:`OpMember`
        q2_final : float
            final scale
    """

    def __init__(self, op_members, q2_final):
        self.op_members = op_members
        self.q2_final = q2_final
