# -*- coding: utf-8 -*-
"""
This script contains a specialization of the Ekomark navigator
"""
import numpy as np
import pandas as pd

from banana import navigator as bnav
from banana import load_config
from banana.data import dfdict


from ekomark.navigator.navigator import NavigatorApp as Ekonavigator

from config import pkg_path
from runner import rotate_to_pm_basis as to_pm
from plots import plot_pdf


class NavigatorApp(Ekonavigator):
    """
    Specialization of the Ekomark navigator.
    """

    def __init__(self, banana_cfg, external):
        super().__init__(banana_cfg, external)

    def fill_logs(self, lg, obj):
        """
        Collect important information of the log record.

        Parameters
        ----------
            lg : dict
                database record
            obj : dict
                to be updated pandas record
        """
        obj["theory"] = lg["t_hash"][: self.hash_len]
        obj["ocard"] = lg["o_hash"][: self.hash_len]
        for f in ["pdf", "external"]:
            obj[f] = lg[f]

    def plot_logs(self, hashes, key_to_plot, rotate_to_pm_basis=True, skip=None):
        """
        Plot two different logs with the same x grid

        Parameters
        ----------
            hashes : list
                log hash list to plot
            key_to_plot: str
                key to display in the plot legend: 'pdf' or theory/operator key
            rotate_to_pm_basis : bool
                if True rotate to plus minus basis
            skip : str
                skip 'plus' or 'minus' distribution

        """

        dfds = []
        labels = []
        fig_name = ""
        for h in hashes:
            log = self.get(bnav.l, h)
            dfds.append(log["log"])
            fig_name += f"{log['hash'][: self.hash_len]}_"

            # search the label
            if key_to_plot == "pdf":
                labels.append(log["pdf"])
            else:
                try:
                    theory = self.get(bnav.t, log["t_hash"])
                    labels.append(theory[key_to_plot])
                except KeyError:
                    try:
                        operators = self.get(bnav.o, log["o_hash"])
                        labels.append(operators[key_to_plot])
                    except KeyError as err:
                        raise KeyError(
                            f"{key_to_plot} is neither in operator card neither in theory card"
                        ) from err
        # build a total log table with new keys
        fig_name = fig_name[:-1]
        total_log = dfdict.DFdict()
        for n, dfd in enumerate(dfds):
            for pid, tab in dfd.items():
                # set first table
                if n == 0:
                    total_log[pid] = pd.DataFrame(
                        {k.replace("EKO", f"{labels[n]}"): v for k, v in tab.items()}
                    )
                # set the other tables
                else:
                    for key, vals in tab.items():
                        if key == "x":
                            np.testing.assert_allclose(tab.x, total_log[pid].x)
                        else:
                            new_key = key.replace("EKO", f"{labels[n]}")
                            total_log[pid][new_key] = vals

        plot_pdf(to_pm(total_log, skip) if rotate_to_pm_basis else total_log, fig_name)


def launch_navigator():
    """CLI Entry point"""
    return bnav.launch_navigator("eko")


app = NavigatorApp(load_config(pkg_path), "sandbox")

# register banana functions
bnav.register_globals(globals(), app)

# add my functions
dfl = app.log_as_dfd
plot_logs = app.plot_logs

# check_log = app.check_log
# plot_pdfs = app.plot_pdfs
# display_pdfs = app.display_pdfs
# compare = app.compare_external


if __name__ == "__main__":
    launch_navigator()
