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
        name : str
            operator name
    """

    def __init__(self, value, error, name):
        self.value = np.array(value)
        self.error = np.array(error)
        self.name = name

    def copy(self, new_name):
        return self.__class__(self.value.copy(), self.error.copy(), new_name)

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
        """Returns target flavour name (given by the first part of the name)"""
        return self._split_name()[0]

    @property
    def input(self):
        """Returns input flavour name (given by the second part of the name)"""
        return self._split_name()[1]

    @property
    def is_physical(self):
        """Lives inside a :class:`PhysicalOperator`? determined by name"""
        for n in ["NS_p", "NS_m", "NS_v", "S_qq", "S_qg", "S_gq", "S_gg"]:
            if self.name.find(n) >= 0:
                return False
        return True

    def apply_pdf(self, pdf_member):
        """
        The operator member can act on a pdf member.

        Parameters
        ----------
            pdf_member : numpy.ndarray
                pdf vector

        Returns
        -------
            result : float
                higher scale pdf
            error : float
                evolution uncertainty to pdf at higher scale
        """
        result = np.dot(self.value, pdf_member)
        error = np.dot(self.error, pdf_member)
        return result, error

    def __str__(self):
        return self.name

    def __matmul__(self, operator_member):
        # check compatibility
        if self.is_physical != operator_member.is_physical:
            raise ValueError("Operators do not live in the same space!")
        if self.is_physical:
            if self.input != operator_member.target:
                raise ValueError(
                    f"Can not sum {operator_member.name} and {self.name} OpMembers!"
                )
            new_name = f"{self.target}.{operator_member.input}"
        else:
            new_name = f"{self.name}.{operator_member.name}"
        rval = operator_member.value
        rerror = operator_member.error
        lval = self.value
        ler = self.error
        new_val = np.matmul(lval, rval)
        # TODO check error propagation
        new_err = np.abs(np.matmul(lval, rerror)) + np.abs(np.matmul(ler, rval))
        return OpMember(new_val, new_err, new_name)

    def __add__(self, operator_member):
        if isinstance(operator_member, Number):
            # we only allow the integer 0 as alias for the true zero operator
            if operator_member != 0:
                raise ValueError(
                    "The only integer we can sum to is 0 (as alias for the zero operator)"
                )
            rval = operator_member
            rerror = 0.0
            new_name = self.name
        elif isinstance(operator_member, OpMember):
            # check compatibility
            if self.is_physical != operator_member.is_physical:
                raise ValueError("Operators do not live in the same space!")
            if self.is_physical and operator_member.name != self.name:
                raise ValueError(
                    f"Can not sum {operator_member.name} and {self.name} OpMembers!"
                )
            rval = operator_member.value
            rerror = operator_member.error
            new_name = self.name
        else:
            raise NotImplementedError(f"Can't sum OpMember and {type(operator_member)}")
        new_val = self.value + rval
        new_err = self.error + rerror
        return OpMember(new_val, new_err, new_name)

    def __neg__(self):
        return self.__matmul__(-1)

    def __eq__(self, operator_member):
        return np.allclose(self.value, operator_member.value)

    def __sub__(self, operator_member):
        return self.__add__(-operator_member)

    def __radd__(self, operator_member):
        return self.__add__(operator_member)

    def __rsub__(self, operator_member):
        return self.__radd__(-operator_member)

    def __rmatmul__(self, operator_member):
        if isinstance(operator_member, Number):
            return self.__matmul__(operator_member)
        raise NotImplementedError(
            f"Can't multiply OpMember and {type(operator_member)}"
        )
