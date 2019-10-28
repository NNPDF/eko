# -*- coding: utf-8 -*-
from multiprocessing import Pool
import numpy as np
import scipy.integrate as integrate
from scipy import optimize


import eko.interpolation as interpolation

# toy PDF
def toy_uv0(x):
    return 5.107200 * np.exp((0.8-1.0)*np.log(x)) * np.exp(3 * np.log(1.0 - x))

# map a list of [0:1]-variables to a real grid
def get_xgrid_order(order_grid,xmin=1e-7):
    xmin_cur = xmin
    res = np.array([])
    for a in order_grid:
        nex = np.exp((1. - a)*np.log(xmin_cur))
        res = np.append(res,nex)
        xmin_cur = nex
    return res

# measure distance between f and interpolation
# TODO small x is not taken into account ...
def get_norm(f,xgrid):
    n = len(xgrid)
    fs = [f(x) for x in xgrid]
    p = lambda xx: np.sum([fs[j] * interpolation.get_Lagrange_interpolators_log_x(xx,xgrid,j) for j in range(n)])
    ker = lambda xx: np.abs(f(xx) - p(xx))
    res = integrate.quad(ker,0,1)
    #print(xgrid,res)
    return res

# test
n_params = 3
eps = 1e-2
bounds = [(eps,1.-eps) for j in range(n_params)]

print("n_params = ",n_params)

# still too large ...
#opt = optimize.shgo(lambda g: get_norm(toy_uv0,get_xgrid_order(g)),bounds, sampling_method='sobol')
#print(opt)

# let's use brute force instead (this makes everything better (-; )
n_points = 5
target_list = []
single_grid = np.linspace(eps,1.-eps,num=n_points)
print(single_grid)
for j in range((n_points)**(n_params)):
    ks = []
    k = j
    for l in range(n_params):
        ks.append(k % n_points)
        k = k // n_points
    element = []
    for k in ks:
        element.append(single_grid[k])
    target_list.append(element)

if __name__ == '__main__':
    def plot_uv0(g):
        return get_norm(toy_uv0,get_xgrid_order(g))
    with Pool(5) as p:
        m = p.map(plot_uv0, target_list)
    for j in range(len(target_list)):
        params = "\t".join(["% e"%e for e in target_list[j]])
        variables = "\t".join(["% e"%e for e in m[j]])
        print(params,"\t",variables)
