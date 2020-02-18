# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import lhapdf
lhapdf.pathsAppend("/usr/local/share/LHAPDF/")

if __name__ == "__main__":
    mu2 = 25
    xs = np.logspace(-6,-4,20)
    p = lhapdf.mkPDF("NNPDF31_lo_as_0118", 0)
    #plt.semilogx(xs, [-(p.xfxQ2(1,x,mu2) - p.xfxQ2(-1,x,mu2)) + (p.xfxQ2(2,x,mu2) - p.xfxQ2(-2,x,mu2)) for x in xs])
    #plt.semilogx(xs, [p.xfxQ2(1,x,mu2) - p.xfxQ2(-1,x,mu2) for x in xs])
    #plt.semilogx(xs, [p.xfxQ2(2,x,mu2) - p.xfxQ2(-2,x,mu2) for x in xs])
    for k in range(1,3):
        plt.loglog(xs, [p.xfxQ2(k,x,mu2) for x in xs])
        plt.loglog(xs, [p.xfxQ2(-k,x,mu2) for x in xs])
    plt.savefig("test.png")
