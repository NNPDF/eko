"""Compute parts from recipes."""
from .parts import Part
from .recipes import Recipe


def compute(recipe: Recipe) -> Part:
    """Compute EKO component in isolation."""
    return Part("ciao")
