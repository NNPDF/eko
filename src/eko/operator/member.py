# -*- coding: utf-8 -*-
from numbers import Number
import numpy as np


class OpMember:
    """
    A single operator for a specific element in evolution basis.

    This class provide some basic mathematical operations such as products.
    It can also be applied to a pdf vector by the `__call__` method.
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

    def __mul__(self, operator_member):
        # scalar multiplication
        if isinstance(operator_member, Number):
            one = np.identity(len(self.value))
            rval = operator_member * one
            rerror = 0.0 * one
            new_name = self.name
        # matrix multiplication
        elif isinstance(operator_member, OpMember):
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
        else:
            raise NotImplementedError(
                f"Can't multiply OpMember and {type(operator_member)}"
            )
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
            raise NotImplementedError(
                f"Can't sum OpMember and {type(operator_member)}"
            )
        new_val = self.value + rval
        new_err = self.error + rerror
        return OpMember(new_val, new_err, new_name)

    def __neg__(self):
        return self.__mul__(-1)

    def __eq__(self, operator_member):
        return np.allclose(self.value, operator_member.value)

    def __sub__(self, operator_member):
        return self.__add__(-operator_member)

    def __radd__(self, operator_member):
        return self.__add__(operator_member)

    def __rsub__(self, operator_member):
        return self.__radd__(-operator_member)

    def __rmul__(self, operator_member):
        return self.__mul__(operator_member)

    @staticmethod
    def join(steps, list_of_paths):
        """
        Multiply a list of :class:`OpMember` using the given paths.

        Parameters
        ----------
            steps : list(list(OpMember))
                list of raw operators, with the lowest scale to the right
            list_of_paths : list(list(str))
                list of paths

        Returns
        -------
            final_op : OpMember
                joined operator
        """
        final_op = 0
        for path in list_of_paths:
            cur_op = None
            for step, member in zip(steps, path):
                new_op = step[member]
                if cur_op is None:
                    cur_op = new_op
                else:
                    cur_op = cur_op * new_op
            final_op += cur_op
        return final_op
