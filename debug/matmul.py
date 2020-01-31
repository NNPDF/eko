# -*- coding: utf-8 -*-
import numpy as np
import mpmath as mp
import yaml

import eko.utils as utils

assets = "../benchmarks/assets/"

if __name__ == "__main__":
    with open(assets+"LHA-LO-FFNS-twostep-ops-l30m20r4-br-step1.yaml") as o:
        step1 = yaml.load(o,Loader=yaml.BaseLoader)
    with open(assets+"LHA-LO-FFNS-twostep-ops-l30m20r4-br-step2.yaml") as o:
        step2 = yaml.load(o,Loader=yaml.BaseLoader)
    with open(assets+"LHA-LO-FFNS-twostep-ops-l30m20r4-br-step2t1.yaml") as o:
        step2t1 = yaml.load(o,Loader=yaml.BaseLoader)
    
    for k in ["V.V"]:
        print(k)
        # load
        o1 = step1["operators"][k]
        o1_np = np.array(o1)
        o1_np = o1_np.astype(float)
        o1_mp = mp.matrix(o1)
        o2 = step2["operators"][k]
        o2_np = np.array(o2)
        o2_np = o2_np.astype(float)
        o2_mp = mp.matrix(o2)
        o2t1 = step2t1["operators"][k]
        o2t1_np = np.array(o2t1)
        o2t1_np = o2t1_np.astype(float)
        o2t1_mp = mp.matrix(o2t1)
        # check internal
        diff_np = np.matmul(o2_np,o1_np)-o2t1_np
        print("max(np) = ",np.max(np.abs(diff_np)),"norm(np,2)=",np.linalg.norm(diff_np,2))
        # check to mp
        diff_mp = o2_mp*o1_mp - o2t1_mp
        print("max(mp) = ",mp.norm(diff_mp,mp.inf),"norm(mp,2)=",mp.norm(diff_mp))

    for k in ["S.S","S.g","g.S","g.g"]:
        print(k)
        to = "q" if k[0] == "S" else "g"
        fromm = "q" if k[-1] == "S" else "g"
        paths = utils.get_singlet_paths(to,fromm,2)
        t_np = 0
        t_mp = 0
        for path in paths:
            to2 = "S" if path[0][-2] == "q" else "g"
            fromm2 = "S" if path[0][-1] == "q" else "g"
            o2 = step2["operators"][f"{to2}.{fromm2}"]
            o2_np = np.array(o2)
            o2_np = o2_np.astype(float)
            o2_mp = mp.matrix(o2)
            to1 = "S" if path[1][-2] == "q" else "g"
            fromm1 = "S" if path[1][-1] == "q" else "g"
            o1 = step1["operators"][f"{to1}.{fromm1}"]
            o1_np = np.array(o1)
            o1_np = o1_np.astype(float)
            o1_mp = mp.matrix(o1)
            print(path,f"-> {to2}.{fromm2}*{to1}.{fromm1}")
            t_np += np.matmul(o2_np,o1_np)
            t_mp += o2_mp*o1_mp
        o2t1 = step2t1["operators"][k]
        o2t1_np = np.array(o2t1)
        o2t1_np = o2t1_np.astype(float)
        o2t1_mp = mp.matrix(o2t1)
        # check internal
        diff_np = t_np-o2t1_np
        print("max(np) = ",np.max(np.abs(diff_np)),"norm(np,2)=",np.linalg.norm(diff_np,2))
        # check to mp
        diff_mp = t_mp - o2t1_mp
        print("max(mp) = ",mp.norm(diff_mp,mp.inf),"norm(mp,2)=",mp.norm(diff_mp))
