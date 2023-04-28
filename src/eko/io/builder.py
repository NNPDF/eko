"""Build an EKO from scratch."""
import pathlib
from dataclasses import dataclass
from typing import Optional

from . import exceptions
from .access import AccessConfigs
from .runcards import OperatorCard, TheoryCard
from .struct import EKO, InternalPaths, Metadata, Rotations


@dataclass
class Builder:
    """Build EKO instances."""

    path: pathlib.Path
    """Path on disk to ."""
    access: AccessConfigs
    """Access related configurations."""

    # optional arguments, required at build time
    theory: Optional[TheoryCard] = None
    operator: Optional[OperatorCard] = None

    eko: Optional[EKO] = None

    def __post_init__(self):
        """Validate paths."""
        if self.access.path.suffix != ".tar":
            raise exceptions.OutputNotTar(self.access.path)
        if self.access.path.exists():
            raise exceptions.OutputExistsError(self.access.path)

    def load_cards(self, theory: TheoryCard, operator: OperatorCard):
        """Load both theory and operator card."""
        self.theory = theory
        self.operator = operator

        return self

    def build(self) -> EKO:
        """Build EKO instance.

        Returns
        -------
        EKO
            the constructed instance

        Raises
        ------
        RuntimeError
            if not enough information is available (at least one card missing)

        """
        missing = []
        for card in ["theory", "operator"]:
            if getattr(self, card) is None:
                missing.append(card)

        if len(missing) > 0:
            raise RuntimeError(
                f"Can not build an EKO, since following cards are missing: {missing}"
            )

        # tell the static analyzer as well
        assert self.theory is not None
        assert self.operator is not None

        self.access.open = True
        metadata = Metadata(
            _path=self.path,
            mu20=(self.operator.mu20, self.theory.heavy.num_flavs_init),
            rotations=Rotations(xgrid=self.operator.xgrid),
        )
        InternalPaths(self.path).bootstrap(
            theory=self.theory.raw,
            operator=self.operator.raw,
            metadata=metadata.raw,
        )

        self.eko = EKO(
            _operators={ep: None for ep in self.operator.evolgrid},
            metadata=metadata,
            access=self.access,
        )

        return self.eko

    def __enter__(self):
        """Allow Builder to be used in :obj:`with` statements."""
        return self

    def __exit__(self, exc_type: type, _exc_value, _traceback):
        """Ensure EKO to be closed properly."""
        if exc_type is not None:
            return

        # assign to variable to help type checker, otherwise self.eko might be
        # a property, and its type can change at every evaluation
        eko = self.eko
        if eko is not None:
            eko.close()
