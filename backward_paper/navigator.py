# -*- coding: utf-8 -*-
"""
This script contains a specialization of the Ekomark navigator
"""
import numpy as np
import pandas as pd
from scipy import integrate
import json

from banana import navigator as bnav
from banana import load_config
from banana.data import dfdict


from ekomark.navigator.navigator import NavigatorApp as Ekonavigator

from config import pkg_path
from runner import rotate_to_pm_basis as to_pm
from plots import plot_pdf


def replace_label(new_label):
    return lambda s: s.replace("EKO", f"{new_label}")


class NavigatorApp(Ekonavigator):
    """
    Specialization of the Ekomark navigator.
    """

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

    @staticmethod
    def get_replica(tab, n):
        """Get the n-th replica"""
        nrep = tab["x"].value_counts().iloc[0]
        ngrid = int(len(tab) / nrep)
        return tab.iloc[n * ngrid : (n + 1) * ngrid, :]

    def search_label(self, log, key):
        """
        Search the label to display

        Parameters
        ----------
            log : dict
                log dictionary
            key : str
                key of which value will be used as label

        Returns
        -------
            label: str
                value to use in the legend
        """
        if key == "pdf":
            label = log["pdf"]
            if label == "210701-n3fit-data-014":
                label = "NNPDF40_nnlo_as_01180_w_EMC"
        elif key == "external":
            label = log["external"]
        elif key is not None:
            try:
                theory = self.get(bnav.t, log["t_hash"])
                label = theory[key]
                if key == "PTO":
                    pto_dict = {0: "LO", 1: "NLO", 2: "NNLO", 3: "N3LO"}
                    label = pto_dict[label]
            except KeyError:
                try:
                    operators = self.get(bnav.o, log["o_hash"])
                    label = operators[key]
                except KeyError as err:
                    raise KeyError(
                        f"{key} is neither in operator card neither in theory card"
                    ) from err
        else:
            label = "EKO"
        return label

    def collect_logs(
        self,
        hashes,
        label_to_display=None,
    ):
        """
        Collect logs with the same x grid and raname labels

        Parameters
        ----------
            hashes : list
                log hash list to plot
            label_to_display: str
                key to display in the plot legend: 'pdf' or theory/operator key
        """

        dfds = []
        labels = []
        log_name = ""
        for h in hashes:
            log = self.get(bnav.l, h)
            dfds.append(log["log"])
            log_name += f"{log['hash'][: self.hash_len]}_"
            # search the label
            labels.append(self.search_label(log, label_to_display))

        # build a total log table with new keys
        log_name = log_name[:-1]
        total_log = dfdict.DFdict()
        for n, dfd in enumerate(dfds):
            for pid, tab in dfd.items():
                # set first table
                if n == 0:
                    total_log[pid] = pd.DataFrame(tab).rename(
                        columns=replace_label(labels[n])
                    )
                # set the other tables
                else:
                    np.testing.assert_allclose(tab.x, total_log[pid].x)
                    for key, vals in tab.iloc[:, 1:].items():
                        new_key = key.replace("EKO", f"{labels[n]}")
                        total_log[pid][new_key] = vals

        return total_log, log_name

    def plot_logs(
        self,
        hashes,
        label_to_display=None,
        rotate_to_pm_basis=True,
        skip=None,
        plot_pull=False,
        plot_reldiff=False,
    ):
        """
        Plot two different logs with the same x grid

        Parameters
        ----------
            hashes : list
                log hash list to plot
            label_to_display: str
                key to display in the plot legend: 'pdf' or theory/operator key
            rotate_to_pm_basis : bool
                if True rotate to plus minus basis
            skip : str
                skip '+' or '-' distribution
        """
        total_log, fig_name = self.collect_logs(hashes, label_to_display)

        plot_pdf(
            to_pm(total_log, skip) if rotate_to_pm_basis else total_log,
            fig_name,
            plot_reldiff=plot_reldiff,
            plot_pull=plot_pull,
        )

    def dump_logs(
        self,
        hashes,
        label_to_display=None,
        rotate_to_pm_basis=True,
        skip=None,
    ):
        """
        Dump logs with the same x grid to a json file

        Parameters
        ----------
            hashes : list
                log hash list to plot
            label_to_display: str
                key to display in the plot legend: 'pdf' or theory/operator key
            rotate_to_pm_basis : bool
                if True rotate to plus minus basis
            skip : str
                skip '+' or '-' distribution
        """
        total_log, log_name = self.collect_logs(hashes, label_to_display)
        if rotate_to_pm_basis:
            total_log = to_pm(total_log, skip)

        with open(f"{log_name}.json", "w") as f:
            for pdf, table in total_log.items():
                json.dump({pdf: table.to_dict()}, f)
        f.close()

    def compute_momentum_fraction(
        self, hashes, label_to_display=None, rotate_to_pm_basis=True, skip=None
    ):
        """
        Compute the momentum fraction for each PDF

        Parameters
        ----------
            hashes : list
                log hash list to plot
            label_to_display: str
                key to display in the plot legend: 'pdf' or theory/operator key
            rotate_to_pm_basis : bool
                if True rotate to plus minus basis
            skip : str
                skip '+' or '-' distribution
        """

        def integrand(delta_x):
            return lambda y: integrate.trapz(y, delta_x)

        for h in hashes:
            log = self.get(bnav.l, h)
            dfd = to_pm(log["log"], skip) if rotate_to_pm_basis else log["log"]
            momentum_log = dfdict.DFdict()

            # loop on pdfs
            for pid, tab in dfd.items():

                # compute integral replica by replica
                nrep = tab["x"].value_counts().iloc[0]
                mom_df = pd.DataFrame()
                for n in range(nrep):
                    replica_tab = get_replica(tab, n)
                    replica_mom = replica_tab.iloc[:, 1:].apply(
                        integrand(replica_tab.x)
                    )
                    mom_df = mom_df.append(pd.DataFrame(replica_mom).T)

                # average and std and rename if necessary
                momentum_log[pid] = pd.concat(
                    [
                        mom_df.mean().rename("momentum fraction"),
                        mom_df.std().rename("std"),
                        mom_df.mean().rename("momentum fraction (%)") * 100,
                        mom_df.mean().div(mom_df.std()).rename("Evidence"),
                    ],
                    axis=1,
                ).rename(index=replace_label(self.search_label(log, label_to_display)))
            print("Log:", log["hash"])
            print(momentum_log)


def launch_navigator():
    """CLI Entry point"""
    return bnav.launch_navigator("eko")


app = NavigatorApp(load_config(pkg_path), "sandbox")

# register banana functions
bnav.register_globals(globals(), app)

# add my functions
dfl = app.log_as_dfd
plot_logs = app.plot_logs
compute_mom = app.compute_momentum_fraction
get_replica = app.get_replica
dump_logs = app.dump_logs

# check_log = app.check_log
# plot_pdfs = app.plot_pdfs
# display_pdfs = app.display_pdfs
# compare = app.compare_external


if __name__ == "__main__":
    launch_navigator()
