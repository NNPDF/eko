"""Ekomark specialization of the navigator."""

import argparse
import pathlib

from banana import cfg as banana_cfg
from banana import navigator as bnav

from . import glob, navigator


def yelp(*args):
    """Help function (renamed to avoid clash of names) - short cut: h."""
    if len(args) == 0:
        print(
            f"""Welcome to ekomark navigator - the eko benchmark skript!
Available variables:
    {bnav.help_vars}
    o = "{bnav.o}" -> query operator
Available functions:
    {bnav.help_fncs}
    dfl(id) - log as DataFrame
    simlogs(id) - find similar logs
    diff(id,id) - subtractig logs
    check_log(id) - check logs passed
    crashed_log(id) - print crashed logs
    plot_pdfs(id) - plot pdfs to final scale
    display_pdfs(id) - open pdfs plots
    compare(id,id) - compare externals
"""
        )
    elif len(args) == 1:
        return help(*args)
    return None


def register_globals(configpath):
    """Expose global variables."""
    app = navigator.NavigatorApp(configpath, "sandbox")
    glob.app = app

    glob.glob["yelp"] = yelp
    glob.glob["h"] = yelp

    # register banana functions
    bnav.register_globals(glob.glob, glob.app)

    # add my functions
    glob.glob["check_log"] = app.check_log
    glob.glob["plot_pdfs"] = app.plot_pdfs
    glob.glob["display_pdfs"] = app.display_pdfs


def launch_navigator():
    """CLI Entry point."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-c", "--config", type=pathlib.Path, default=None, help="Path to config file"
    )

    args = parser.parse_args()

    register_globals(banana_cfg.detect(args.config))

    # ekomark.navigator makes the globals here (e.g. app, ls, t) available inside IPython
    return bnav.launch_navigator(
        ["eko", "ekomark", "ekomark.navigator.glob"], skip_cfg=True
    )
