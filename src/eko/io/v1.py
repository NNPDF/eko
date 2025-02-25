"""Legacy interface to files created with v0.13.

Although the data version v1 was already before v0.13 we only support
that API version.
"""

from .paths import InternalPaths
import numpy as np
import yaml
import os
from pathlib import Path

def update_metadata(paths: InternalPaths, raw: dict) -> dict:
    """Modify the raw metadata to the new format.

    Parameters
    ----------
    paths:
        base paths to the EKO
    raw:
        raw yaml content

    Returns
    -------
    dict
        compatible raw yaml content
    """
    raw["data_version"] = 1
    raw["xgrid"] = raw["rotations"]["xgrid"]
    del raw["rotations"]
    raw_theory = yaml.safe_load(paths.theory_card.read_text(encoding="utf-8"))
    raw["origin"] = (np.sqrt(raw["mu20"]), raw_theory["heavy"]["num_flavs_init"])
    del raw["mu20"]
    return raw


def update_theory(raw: dict) -> dict:
    """Modify the raw theory card to the new format.

    Parameters
    ----------
    raw:
        raw yaml content

    Returns
    -------
    dict
        compatible raw yaml content
    """
    raw["couplings"]["ref"] = (
        raw["couplings"]["scale"],
        raw["couplings"]["num_flavs_ref"],
    )
    raw['matching_order'] = [0,0]
    # adjust couplings
    for key in ["num_flavs_ref", "max_num_flavs", "scale"]:
        del raw["couplings"][key]
    # adjust heavy
    for key in ["intrinsic_flavors", "num_flavs_init", "num_flavs_max_pdf"]:
        del raw["heavy"][key]

    # update old names
    if "use_fhmv" in raw:
        raw["use_fhmruvv"] = raw["use_fhmv"]
        del raw["use_fhmv"]

    return raw


def update_operator(raw_op: dict, raw_th) -> dict:
    """Modify the raw operator card to the new format.

    Parameters
    ----------
    raw_op:
        raw operator yaml content
    raw_th:
        raw theory yaml content
    Returns
    -------
    dict
        compatible raw yaml content
    """
    raw_op['configs']['n_integration_cores'] = 1
    raw_op["init"] = (raw_op["mu0"], raw_th["heavy"]["num_flavs_init"])
    del raw_op["mu0"]
    return raw_op

def update_eko(paths: InternalPaths, raw_op):
    """Make a .yaml file in the directory v0.13.tar/operators
    with the evolution point in it

    Also rename the object to match the name given in v0.0
    """

    paths = Path(paths)

    file_name = "EFMRliCNGks=.yaml"
    file_path = os.path.join(paths, file_name)
    
    os.makedirs(paths, exist_ok=True)

    data = {
        "nf": raw_op["init"][1],
        "scale": 10000#raw_op["mugrid"][0]**2   # need to define this from the operator card eventually of course
    }

    with open(file_path, "w") as yaml_file:
        yaml.dump(data, yaml_file, default_flow_style=False)
    
    with open(file_path, "r") as evas_file:
        print("evolution point file:", yaml.safe_load(evas_file))    # temporary line to test if indeed the yaml file was created
        
    
    # Rename EKO operator to the name used by v0.0
    old_file = os.path.join(paths, "AAAAAACIw0A=.npz.lz4")  # old filename
    new_file = os.path.join(paths, "EFMRliCNGks=.npz.lz4")  # new filename

    # Check if the old file exists before renaming
    if os.path.exists(old_file):
        os.rename(old_file, new_file)
        print(f"Renamed '{old_file}' to '{new_file}'")
    else:
        print(f"File '{old_file}' does not exist, skipping rename.")
    
    return file_path



