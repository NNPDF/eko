"""Abstraction for serializations.

A few known types are directly registered here, in order to be
transparently codified in more native structures.
"""

import copy
import dataclasses
import enum
import inspect
import typing

import numpy as np

from .. import interpolation

# TODO: use typing.dataclass_transform, new in Python 3.11, since all child
# classes are supposed to be dataclasses


@dataclasses.dataclass
class DictLike:
    """Dictionary compatibility base class, for dataclasses.

    This class add compatibility to import and export from Python :class:`dict`,
    in such a way to support serialization interfaces working with them.

    Some collections and scalar objects are normalized to native Python
    structures, in order to simplify the on-disk representation.
    """

    @classmethod
    def _from_dict(cls, dictionary):
        """Initialize dataclass object from raw dictionary.

        Parameters
        ----------
        dictionary : dict
            the dictionary to be converted to :class:`DictLike`

        Returns
        -------
        DictLike
            instance with `dictionary` content loaded as attributes
        """
        dictionary = copy.deepcopy(dictionary)

        # allow initialization from tuple, useful for simple dataclasses
        # NOTE: no further nesting will be supported, they are supposed to be
        # simple
        # TODO: update some names consistently, `DictLike` nad `from_dict` are
        # not any longer the best. And it is good in any case to drop the old
        # `from_dict` name, and solve part of the ambiguity with existing
        # methods
        if any(isinstance(dictionary, cls) for cls in [tuple, list]):
            return cls(*dictionary)

        # support nesting, modifying the copied dictionary
        for field in dataclasses.fields(cls):
            # dictionary does not provide any type information we can rely on
            # to load better, so we will assume that it is fully loaded inside
            if field.type in [dict, typing.Dict]:
                continue
            # allow for default values: is sufficient to leave it empty, the
            # default is already in the class constructor
            if (
                field.name not in dictionary
                and field.default is not dataclasses.MISSING
            ):
                continue

            # load nested: the field has a definite type, and we will use this
            # information to automatically implement the loader
            dictionary[field.name] = load_field(field.type, dictionary[field.name])

        # finally construct the class, just passing the arguments by name
        return cls(**dictionary)

    @classmethod
    def from_dict(cls, dictionary):
        """Deserialize, overwritable interface.

        The default implementation is just :meth:`DictLike._from_dict`, but it
        can be safely overwritten (usually transforming the input before a call
        to :meth:`DictLike._from_dict` itself).
        """
        return cls._from_dict(dictionary)

    def _raw(self):
        """Convert dataclass object to raw dictionary.

        Normalize:

            - :class:`np.ndarray` to lists (possibly nested)
            - scalars to the corresponding built-in type (e.g. :class:`float`)
            - :class:`tuple` to lists
            - :class:`interpolation.XGrid` to the intrinsic serialization format

        Returns
        -------
        dict
            dictionary representation
        """
        dictionary = {}

        for field in dataclasses.fields(self):
            dictionary[field.name] = raw_field(getattr(self, field.name))

        return dictionary

    @property
    def raw(self):
        """Overwritable serialization.

        The default implementation is just :meth:`DictLike._raw`, but it can be
        safely overwritten (usually starting from :meth:`DictLike._raw`
        itself).
        """
        return self._raw()

    @property
    def public_raw(self):
        """Serialize only public attributes."""
        return {k: v for k, v in self._raw().items() if not k.startswith("_")}


def load_field(type_, value):
    """Deserialize dataclass field."""
    # TODO: nice place for a match statement...
    if type(type_) is typing.NewType:
        return load_field(type_.__supertype__, value)
    try:
        # before py3.10 typing.NewType was just a function, so the check above
        # would  fail
        return load_field(type_.__supertype__, value)
    except AttributeError:
        pass

    if typing.get_origin(type_) is not None:
        # this has to go first since for followin ones I will assume they are
        # valid classes, cf. the module docstring
        return load_typing(type_, value)

    # FIXME: python 3.8 does not recognize the origin of `npt.NDArray`, but at
    # the same time it is not a proper class since `numpy==1.23.2`, since it is
    # not recognized by `inspect.isclass`, and `type(npt.NDArray)` is not
    # inheriting from `type`
    # However, this check should be matched by every version of NumPy for every
    # version of Python supported
    # Once py3.8 will be dropped, move this back below `inspect.isclass`
    # assertion
    if np.ndarray in type_.__mro__ or issubclass(type_, np.ndarray):
        # do not apply array on scalars
        if isinstance(value, list):
            return np.array(value)
        return value

    assert inspect.isclass(type_)

    if issubclass(type_, DictLike):
        return type_.from_dict(value)
    if issubclass(type_, enum.Enum):
        return load_enum(type_, value)
    if isinstance(value, dict):
        return type_(**value)

    return type_(value)


def load_enum(type_, value):
    """Deserialize enum variant.

    Accepts both the name and value of variants, attempted in this order.

    Raises
    ------
    ValueError
        if `value` is not the name nor the value of any enum variant
    """
    try:
        return type_[value]
    except KeyError:
        return type_(value)


def load_typing(type_, value):
    """Deserialize type hint associated field."""
    origin = typing.get_origin(type_)
    assert origin is not None

    # unions have to be explored over their variants
    if origin is typing.Union:
        for variant in typing.get_args(type_):
            try:
                return load_field(variant, value)
            except (TypeError, ValueError):
                ...

        if type(None) in type_.__args__:
            return None
        raise TypeError

    if issubclass(origin, (list, typing.Generic)):
        T = typing.get_args(type_)[0]
        return origin([load_field(T, x) for x in value])

    return load_field(origin, value)


def raw_field(value):
    """Serialize DictLike field."""
    # replace numpy arrays with lists
    if isinstance(value, np.ndarray):
        return value.tolist()
    # replace numpy scalars with python ones
    if isinstance(value, float):
        return float(value)
    if isinstance(value, interpolation.XGrid):
        return value.dump()["grid"]
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, DictLike):
        return value.raw
    if dataclasses.is_dataclass(value):
        # not supporting nested DictLike inside nested plain dataclasses
        return dataclasses.asdict(value)
    if isinstance(value, list):
        return [raw_field(el) for el in value]

    return value
