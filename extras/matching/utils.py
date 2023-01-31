import itertools
import pathlib

import cycler
import matplotlib as mpl
import numpy as np
import yaml


def flatten(d):
    newd = {}
    for k, v in d.items():
        if isinstance(v, dict):
            for ik, iv in flatten(v).items():
                newd[f"{k}.{ik}"] = iv
        else:
            newd[k] = v
    return newd


def load_style(path):
    style = flatten(yaml.safe_load(pathlib.Path(path).read_text()))

    capstyle = "lines.solid_capstyle"
    prop_cycle = "axes.prop_cycle"

    if capstyle in style:
        style[capstyle] = mpl._enums.CapStyle(style[capstyle])
    pcd = {k: v for k, v in style.items() if prop_cycle in k}
    if len(pcd) > 0:
        length = max(len(l) for l in pcd.values())
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


def log_minor_ticks(min, max, base=10.0, scale=1.0, n=10, majors=None):
    if majors is None:
        lmin = np.ceil(np.log(min / scale) / np.log(base))
        lmax = np.floor(np.log(max / scale) / np.log(base))

        majors = scale * base ** np.arange(lmin - 1.0, lmax + 2.0)
    else:
        majors = list(np.unique(majors))
        while np.min(majors) > min:
            majors.insert(0, majors[0] / (majors[1] / majors[0]))
        while np.max(majors) < max:
            majors.append(majors[-1] / (majors[-2] / majors[-1]))

    ranges = []
    x1 = majors[0]
    for x2 in majors[1:]:
        ranges.append(
            np.array(
                list(filter(lambda x: min < x < max, np.linspace(x1, x2, n)[1:-1]))
            )
        )
        x1 = x2

    return np.concatenate(ranges)
