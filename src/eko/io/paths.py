"""Define paths inside an `eko.EKO` object."""

import pathlib
from dataclasses import dataclass

import yaml

THEORYFILE = "theory.yaml"
OPERATORFILE = "operator.yaml"
METADATAFILE = "metadata.yaml"
RECIPESDIR = "recipes"
PARTSDIR = "parts"
MATCHINGDIR = "matching"
OPERATORSDIR = "operators"


@dataclass
class InternalPaths:
    """Paths inside an EKO folder.

    This structure exists to locate in a single place the internal structure of
    an EKO folder.

    The only value required is the root path, everything else is computed
    relative to this root.
    In case only the relative paths are required, just create this structure
    with :attr:`root` equal to emtpty string or ``"."``.
    """

    root: pathlib.Path
    "The root of the EKO folder (use placeholder if not relevant)"

    @property
    def metadata(self):
        """Metadata file."""
        return self.root / METADATAFILE

    @property
    def recipes(self):
        """Recipes folder."""
        return self.root / RECIPESDIR

    @property
    def recipes_matching(self):
        """Matching recipes folder."""
        return self.root / RECIPESDIR / MATCHINGDIR

    @property
    def parts(self):
        """Parts folder."""
        return self.root / PARTSDIR

    @property
    def parts_matching(self):
        """Matching parts folder."""
        return self.root / PARTSDIR / MATCHINGDIR

    @property
    def operators(self):
        """Operators folder.

        This is the one containing the actual EKO components, after
        computation has been performed.
        """
        return self.root / OPERATORSDIR

    @property
    def theory_card(self):
        """Theory card dump."""
        return self.root / THEORYFILE

    @property
    def operator_card(self):
        """Operator card dump."""
        return self.root / OPERATORFILE

    def bootstrap(self, theory: dict, operator: dict, metadata: dict):
        """Create directory structure."""
        self.metadata.write_text(yaml.dump(metadata), encoding="utf-8")
        self.theory_card.write_text(yaml.dump(theory), encoding="utf-8")
        self.operator_card.write_text(yaml.dump(operator), encoding="utf-8")
        self.recipes.mkdir()
        self.recipes_matching.mkdir()
        self.parts.mkdir()
        self.parts_matching.mkdir()
        self.operators.mkdir()
