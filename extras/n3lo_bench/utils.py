import pathlib
import yaml
import itertools
import cycler
import matplotlib as mpl
import pathlib

here = pathlib.Path(__file__).parent

def flatten(d):
    newd = {}
    for k, v in d.items():
        if isinstance(v, dict):
            for ik, iv in flatten(v).items():
                newd[f"{k}.{ik}"] = iv
        else:
            newd[k] = v
    return newd

def load_style():
    path = here / "style.yaml"
    style = flatten(yaml.safe_load(pathlib.Path(path).read_text()))

    capstyle = "lines.solid_capstyle"
    prop_cycle = "axes.prop_cycle"

    if capstyle in style:
        style[capstyle] = mpl._enums.CapStyle(style[capstyle])
    pcd = {k: v for k, v in style.items() if prop_cycle in k}
    if len(pcd) > 0:
        length = max((len(l) for l in pcd.values()))
        for k, v in pcd.items():
            del style[k]
            cyc = cycler.cycler(
                k.split(".")[-1], itertools.islice(itertools.cycle(v), length)
            )
            if prop_cycle not in style:
                style[prop_cycle] = cyc
            else:
                style[prop_cycle] += cyc

    return style
