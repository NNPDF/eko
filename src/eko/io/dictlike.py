"""Abstraction for serializations.

A few known types are directly registered here, in order to be transparently
codified in more native structures.

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


class DictLike:
    """Dictionary compatibility base class, for dataclasses.

    This class add compatibility to import and export from Python :class:`dict`,
    in such a way to support serialization interfaces working with them.

    Some collections and scalar objects are normalized to native Python
    structures, in order to simplify the on-disk representation.

    """

    def __init__(self, **kwargs):
        """Empty initializer."""

    @classmethod
    def from_dict(cls, dictionary):
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
        if isinstance(dictionary, tuple):
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
            dictionary[field.name] = load_field(field, dictionary[field.name])

        # finally construct the class, just passing the arguments by name
        return cls(**dictionary)

    @property
    def raw(self):
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


def load_field(field, value):
    # TODO: nice place for a match statement...
    if inspect.isclass(field.type) and issubclass(field.type, DictLike):
        return field.type.from_dict(value)
    if typing.get_origin(field.type) is not None:
        return load_typing(field, value)
    if issubclass(field.type, enum.Enum):
        return load_enum(field, value)

    return field.type(value)


def load_enum(field, value):
    try:
        return field.type(value)
    except ValueError:
        return field.type[value]


def load_typing(field, value):
    origin = typing.get_origin(field.type)
    assert origin is not None

    if origin is typing.Union:
        for variant in typing.get_args(field.type):
            try:
                loaded = variant(value)
                break
            except (TypeError, ValueError):
                ...
        else:
            if type(None) in field.type.__args__:
                loaded = None
            else:
                raise TypeError
    else:
        if isinstance(value, dict):
            loaded = origin(**value)
        elif issubclass(np.ndarray, origin):
            loaded = np.array(value)
        else:
            loaded = origin(value)

    return loaded


def raw_field(value):
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
    if isinstance(value, DictLike):
        return value.raw

    return value
