import io
import pathlib

import yaml


def create_info_file(theory_card, operators_card, NumMembers, info_update):
    """
    Generate a lhapdf info file from theory and operators card

    Parameters
    ----------
        theory_card : dict
            theory card
        operators_card : dict
            operators_card
        NumMembers : int
            number of members of the PDF set
        info_update : dict
            info to update

    Returns
    -------
        : dict
        info file in lhapdf format
    """
    here = pathlib.Path(__file__).parent
    with open(here / "templatePDF.info", "r") as o:
        template_info = yaml.safe_load(o)
    # TODOs: Updating the info using the theory and operator cards
    template_info.update(info_update)
    return template_info
