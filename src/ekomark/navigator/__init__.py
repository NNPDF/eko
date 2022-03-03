# -*- coding: utf-8 -*-
"""
ekomark specialization of the navigator
"""
import pathlib

from banana import cfg as banana_cfg
from banana import navigator as bnav
from banana import register

from . import navigator

benchmarks_path = pathlib.Path(__file__).parents[3] / "benchmarks"
register(benchmarks_path)


def yelp(*args):
    """
    Help function (renamed to avoid clash of names) - short cut: h.
    """
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


h = yelp

app = navigator.NavigatorApp(benchmarks_path / banana_cfg.name, "sandbox")

# register banana functions
bnav.register_globals(globals(), app)

# add my functions
dfl = app.log_as_dfd
check_log = app.check_log
plot_pdfs = app.plot_pdfs
display_pdfs = app.display_pdfs
compare = app.compare_external


def launch_navigator():
    """CLI Entry point"""
    # ekomark.navigator makes the globals here (e.g. app, ls, t) available inside IPython
    return bnav.launch_navigator(
        ["eko", "ekomark", "ekomark.navigator"], benchmarks_path
    )
