"""Abstraction for serializations.

A few known types are directly registered here, in order to be transparently
codified in more native structures.

"""
import dataclasses

import numpy as np

from .. import interpolation


# TODO: add typing.dataclass_transform, new in Python 3.11
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
            value = getattr(self, field.name)

            # replace numpy arrays with lists
            if isinstance(value, np.ndarray):
                value = value.tolist()
            # replace numpy scalars with python ones
            elif isinstance(value, float):
                value = float(value)
            elif isinstance(value, interpolation.XGrid):
                value = value.dump()["grid"]
            elif isinstance(value, tuple):
                value = list(value)

            dictionary[field.name] = value

        return dictionary
