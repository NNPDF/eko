"""Physical and internal objects.

This subpackage contains the definitions of many objects, with the purpose of
using them as type hints, but not only (at least for some of them).

Few related functions are also defined here.

Note
----
Unfortunately, Python has still some discrepancies between runtime classes and
type hints, so it is better not to mix dataclasses and generics.

E.g. before it was implemented::

    @dataclasses.dataclass
    class RunningReference(DictLike, Generic[Quantity]):
        value: Quantity
        scale: float

but in this way it is not possible to determine that ``RunningReference`` is
subclassing ``DictLike``, indeed::

    inspect.isclass(RunningReference)       # False
    issubclass(RunningReference, DictLike)  # raise an error, since
                                            # RunningReference is not a class

Essentially classes can be used for type hints, but types are not all classes,
especially when they involve generics.

For this reason I prefer the less elegant dynamic generation, that seems to
preserve type hints.

"""
