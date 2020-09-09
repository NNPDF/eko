# -*- coding: utf-8 -*-
"""
toyLHPDFs from APFEL:
--------------------
This routine returns the toyLH PDFs at the intitial scale
which is supposed to be Q = sqrt(2) GeV.
"""

import atexit

def toyLHPDFs(pid, x): # pylint: disable=too-many-locals
    N_uv = 5.107200e0
    auv = 0.8e0
    buv = 3e0
    N_dv = 3.064320e0
    adv = 0.8e0
    bdv = 4e0
    N_g = 1.7e0
    ag = -0.1e0
    bg = 5e0
    N_db = 0.1939875e0
    adb = -0.1e0
    bdb = 6e0
    fs = 0.2e0

    # User defined PDFs

    xuv = N_uv * x ** auv * (1e0 - x) ** buv
    xdv = N_dv * x ** adv * (1e0 - x) ** bdv
    xg = N_g * x ** ag * (1e0 - x) ** bg
    xdbar = N_db * x ** adb * (1e0 - x) ** bdb
    xubar = xdbar * (1e0 - x)
    xs = fs * (xdbar + xubar)
    xsbar = xs

    # Initialize PDFs to zero

    xpdf = {i: 0 for i in range(-6, 7)}

    if x > 1e0:
        return 0.0

    xpdf[3] = xs
    xpdf[2] = xuv + xubar
    xpdf[1] = xdv + xdbar
    xpdf[21] = xpdf[0] = xg
    xpdf[-1] = xdbar
    xpdf[-2] = xubar
    xpdf[-3] = xsbar
    xpdf[22] = 0

    return xpdf[pid]


class toyPDFSet:
    """Fake PDF set"""
    name = "ToyLH"


class toyPDF:
    """Imitates lhapdf"""

    def xfxQ2(self, pid, x, _Q2):
        """Get the PDF xf(x) value at (x,q2) for the given PID.

        Parameters
        ----------

        Parameters
        ----------
        pid : int
            PDG parton ID.
        x : float
            Momentum fraction.
        Q : float
            Squared energy (renormalization) scale.

        Returns
        -------
        float
            The value of xf(x,q2).

        """

        return toyLHPDFs(pid, x)

    def xfxQ(self, pid, x, _Q):
        """Get the PDF xf(x) value at (x,q) for the given PID.

        Parameters
        ----------
        pid : int
            PDG parton ID.
        x : float
            Momentum fraction.
        Q : float
            Energy (renormalization) scale.

        Returns
        -------
        type
            The value of xf(x,q2).

        """

        return toyLHPDFs(pid, x)

    def alphasQ(self, q):
        "Return alpha_s at q"
        return self.alphasQ2(q**2)

    def alphasQ2(self, _q2):
        "Return alpha_s at q2"
        return 0.35

    def set(self):
        "Return the corresponding PDFSet"
        return toyPDFSet()


def mkPDF(_setname, _member):
    """Factory functions for making single PDF members.

    Create a new PDF with the given PDF set name and member ID.

    Parameters
    ----------
    setname : type
        PDF set name.
    member : type
        Member ID.

    Returns
    -------
    toyPDF
        PDF object.

    """

    return toyPDF()

def ciao():
    """Print at exit."""
    print(
        """\nThanks for using toyPDF. Please make sure to close the door.\n"""
        + "\t" * 7
        + "__Alessandro"
    )

atexit.register(ciao)
