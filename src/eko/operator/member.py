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
