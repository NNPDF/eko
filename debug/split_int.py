# -*- coding: utf-8 -*-
import numpy as np
import mpmath as mp
import scipy.integrate as integrate
import matplotlib.pyplot as plt

import eko.alpha_s as alpha_s
from eko.constants import Constants
import eko.splitting_functions_LO as sf_LO

def gamma_ns_0_mp(n, nf, CA, CF):
    gamma = 2 * (mp.harmonic(n - 1) + mp.harmonic(n + 1)) - 3
    result = CF * gamma
    return result

def get_kernel_ns_mp():
    """Returns the non-singlet integration kernel"""
    constants = Constants()
    nf = 3
    CA = constants.CA
    CF = constants.CF
    beta_0 = alpha_s.beta_0(nf,CA,CF,constants.TF)
    delta_t = 1.0

    def ker(n, lnxm, lnx):
        """true non-siglet integration kernel"""
        ln = -delta_t * gamma_ns_0_mp(n, nf, CA, CF) / beta_0
        interpoln = mp.exp(n * (lnxm - lnx))/n
        return mp.exp(ln) * interpoln

    return ker

def get_kernel_ns():
    """Returns the non-singlet integration kernel"""
    constants = Constants()
    nf = 3
    CA = constants.CA
    CF = constants.CF
    beta_0 = alpha_s.beta_0(nf,CA,CF,constants.TF)
    delta_t = 1.0

    def ker(n, lnxm, lnx):
        """true non-siglet integration kernel"""
        ln = -delta_t * sf_LO.gamma_ns_0(n, nf, CA, CF) / beta_0
        interpoln = np.exp(n * (lnxm - lnx))/n
        return np.exp(ln) * interpoln

    return ker

def get_kernel_ns_p():
    """Returns the non-singlet integration kernel"""
    constants = Constants()
    nf = 3
    CA = constants.CA
    CF = constants.CF
    beta_0 = alpha_s.beta_0(nf,CA,CF,constants.TF)
    delta_t = 1.0

    def ker(n, lnxm, lnx):
        """true non-siglet integration kernel"""
        ln = -delta_t * sf_LO.gamma_ns_0(n, nf, CA, CF) / beta_0
        interpoln = 2.*np.exp(np.real(n * (lnxm - lnx)))/n
        return np.exp(ln) * interpoln

    return ker

if __name__ == "__main__":
    f = get_kernel_ns()
    f_p = get_kernel_ns_p()
    #xs = np.logspace(-3,0,3)
    lnxm = -14.
    lnx = -16.
    omega = lnxm - lnx
    n_steps = 5
    r_steps = np.pi/omega

    def g(t):
        return np.real(f(1+1j*t,lnxm,lnx)/np.pi)
    def g_p(t):
        return np.real(f_p(1+1j*t,lnxm,lnx)/np.pi)
    i = 0
    itot = 0
    if True:
        i = integrate.quad(g_p,0,np.inf,weight='cos',wvar=omega)
        print("i=",i)
        itot = 0
        for k in range(n_steps):
            i_s = integrate.quad(g, k*r_steps, (k+1)*r_steps,  full_output=1 )
            itot += i_s[0]
            print(i_s[0:2])
        print("itot=",itot, " -> ",itot/i[0])

    f_mp = get_kernel_ns_mp()
    def g_mp(t):
        return mp.re(f_mp(1+1j*t,lnxm,lnx)/np.pi)
    i_mp = 0
    itot_mp = 0
    if True:
        i_mp = mp.quadosc(g_mp,[0,mp.inf],omega=omega)
        print("mp=",i_mp)
        itot_mp = 0
        for k in range(n_steps):
            i_mp_s = mp.quad(g_mp,[k*r_steps, (k+1)*r_steps],error=True)
            itot_mp += i_mp_s[0]
            print(i_mp_s[0:2])
        print("itot=",itot_mp," -> ",itot_mp/i_mp)
    
    if i_mp > 0 or i_mp < 0:
        print("r=",i[0]/i_mp)

    if True:
        xs = []
        vals = []
        for k in range(250):
            x = k/10.
            xs.append(x)
            vals.append(g(x))
        plt.plot(xs,vals)
        plt.savefig("test.png")
