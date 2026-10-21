"""PDF set elements loaders."""

import pathlib

import yaml

from .parser import LhapdfDataFile

here = pathlib.Path(__file__).parent
# Expose the default template
with open(here / "templatePDF.info", encoding="utf-8") as o:
    template_info = yaml.safe_load(o)
with open(here / "Toy.info", encoding="utf-8") as t:
    Toy_info = yaml.safe_load(t)


def load_info_from_file(pdfset_name: str):
    """Load the info file from a parent pdf.

    Parameters
    ----------
    pdfset_name : str
        parent pdf name

    Returns
    -------
    dict
        info dictionary
    """
    import lhapdf  # pylint: disable=import-error, import-outside-toplevel

    src = pathlib.Path(lhapdf.paths()[0]) / pdfset_name
    with open(src / f"{pdfset_name}.info", encoding="utf-8") as o:
        info = yaml.safe_load(o)
    return info


def load_blocks_from_file(pdfset_name: str, member: int = 0) -> LhapdfDataFile:
    """Load a pdf from a parent pdf.

    Parameters
    ----------
    pdfset_name : str
        parent pdf name
    member : int
        pdf member

    Returns
    -------
    LhapdfDataFile:
        data file
    """
    import lhapdf  # pylint: disable=import-error, import-outside-toplevel

    src = pathlib.Path(lhapdf.paths()[0])
    return LhapdfDataFile.read_with_set(src, pdfset_name, member)
