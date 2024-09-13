"""Atomic operator member."""

import copy
import operator
from numbers import Number

import numpy as np

from . import basis_rotation as br
from .evolution_operator import flavors


class OpMember:
    """A single operator for a specific element in evolution basis.

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
        """Copy implementation."""
        return self.__class__(self.value.copy(), self.error.copy())

    @classmethod
    def id_like(cls, other):
        """Create an identity operator.

        Parameters
        ----------
        other : OpMember
            reference member

        Returns
        -------
        cls :
            1 all spaces
        """
        len_xgrid = other.value.shape[0]
        return cls(np.eye(len_xgrid), np.zeros((len_xgrid, len_xgrid)))

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
    """Operator member name in operator evolution space.

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
        """Split the name according to target.input."""
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
        """Returns target flavor name (given by the first part of the name)."""
        return self._split_name()[0]

    @property
    def input(self):
        """Returns input flavor name (given by the second part of the name)."""
        return self._split_name()[1]


class OperatorBase:
    """Abstract base class to hold a dictionary of interpolation matrices.

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

    def __getitem__(self, key):
        if not isinstance(key, MemberName):
            key = MemberName(key)
        return self.op_members[key]

    @classmethod
    def promote_names(cls, op_members, q2_final):
        """Promote string keys to MemberName.

        Parameters
        ----------
        op_members : dict
            mapping of :data:`str` onto :class:`OpMember`
        q2_final : float
            final scale
        """
        # map key to MemberName
        opms = {}
        for k, v in op_members.items():
            opms[MemberName(k)] = copy.copy(v)
        return cls(opms, q2_final)

    def __matmul__(self, other):
        """Multiply ``other`` to self.

        Parameters
        ----------
        other : OperatorBase
            second factor with a lower initial scale

        Returns
        -------
        p : PhysicalOperator
            self @ other
        """
        if not isinstance(other, OperatorBase):
            raise ValueError("Can only multiply with another OperatorBase")
        # if a ScalarOperator is multiplied by an OperatorBase, the result is an OperatorBase
        # only the result of ScalarOperator @ ScalarOperator is another ScalarOperator
        cls = self.__class__
        if isinstance(self, ScalarOperator):
            cls = other.__class__
        return cls(
            self.operator_multiply(self, other, self.operation(other)), self.q2_final
        )

    def operation(self, other):
        """Choose mathematical operation by rank.

        Parameters
        ----------
        other : OperatorBase
            operand

        Returns
        -------
        callable :
            operation to perform (np.matmul or np.multiply)
        """
        if isinstance(self, ScalarOperator) or isinstance(other, ScalarOperator):
            return operator.mul
        return operator.matmul

    @staticmethod
    def operator_multiply(left, right, operation):
        """Multiply two operators.

        Parameters
        ----------
        left : OperatorBase
            left operand
        right : OperatorBase
            right operand
        operation : callable
            operation to perform (np.matmul or np.multiply)

        Returns
        -------
        dict
            new operator members dictionary
        """
        # prepare paths
        new_oms = {}
        for l_key, l_op in left.op_members.items():
            for r_key, r_op in right.op_members.items():
                # ops match?
                if l_key.input != r_key.target:
                    continue
                new_key = MemberName(l_key.target + "." + r_key.input)
                # new?
                if new_key not in new_oms:
                    new_oms[new_key] = operation(l_op, r_op)
                else:  # add element
                    new_oms[new_key] += operation(l_op, r_op)
        return new_oms

    def to_flavor_basis_tensor(self, qed: bool):
        """Convert the computations into an rank 4 tensor.

        A sparse tensor defined with dot-notation (e.g. ``S.g``) is converted
        to a plain rank-4 array over flavor operator space and momentum
        fraction operator space.

        If `qed` is passed, the unified intrinsic basis is used.
        """
        nf_in, nf_out = flavors.get_range(self.op_members.keys(), qed)
        len_pids = len(br.flavor_basis_pids)
        len_xgrid = list(self.op_members.values())[0].value.shape[0]
        # dimension will be pids^2 * xgrid^2
        value_tensor = np.zeros((len_pids, len_xgrid, len_pids, len_xgrid))
        error_tensor = value_tensor.copy()
        for name, op in self.op_members.items():
            if not qed:
                in_pids = flavors.pids_from_intrinsic_evol(name.input, nf_in, False)
                out_pids = flavors.pids_from_intrinsic_evol(name.target, nf_out, True)
            else:
                in_pids = flavors.pids_from_intrinsic_unified_evol(
                    name.input, nf_in, False
                )
                out_pids = flavors.pids_from_intrinsic_unified_evol(
                    name.target, nf_out, True
                )
            for out_idx, out_weight in enumerate(out_pids):
                for in_idx, in_weight in enumerate(in_pids):
                    # keep the outer index to the left as we're multiplying from the right
                    value_tensor[
                        out_idx,  # output pid (position)
                        :,  # output momentum fraction
                        in_idx,  # input pid (position)
                        :,  # input momentum fraction
                    ] += out_weight * (op.value * in_weight)
                    error_tensor[
                        out_idx,  # output pid (position)
                        :,  # output momentum fraction
                        in_idx,  # input pid (position)
                        :,  # input momentum fraction
                    ] += out_weight * (op.error * in_weight)
        return value_tensor, error_tensor


class ScalarOperator(OperatorBase):
    """Operator above space of real numbers."""
