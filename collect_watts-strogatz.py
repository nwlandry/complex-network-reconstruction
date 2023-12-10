import json
import os

import numpy as np
from joblib import Parallel, delayed

from lcs import *

data_dir = "Data/watts-strogatz/"


def collect_parameters(dir):
    clist = set()
    blist = set()
    plist = set()
    rlist = set()

    for f in os.listdir(dir):
        d = f.split(".json")[0].split("_")

        c = int(d[0])
        b = float(d[1])
        p = float(d[2])
        r = int(d[3])

        clist.add(c)
        blist.add(b)
        plist.add(p)
        rlist.add(r)

    c_dict = {c: i for i, c in enumerate(sorted(clist))}
    b_dict = {b: i for i, b in enumerate(sorted(blist))}
    p_dict = {p: i for i, p in enumerate(sorted(plist))}
    r_dict = {r: i for i, r in enumerate(sorted(rlist))}

    return c_dict, b_dict, p_dict, r_dict


def get_metrics(f, dir, c_dict, b_dict, p_dict, r_dict):
    fname = os.path.join(dir, f)
    d = f.split(".json")[0].split("_")
    c = int(d[0])
    b = float(d[1])
    p = float(d[2])
    r = int(d[3])

    i = c_dict[c]
    j = b_dict[b]
    k = p_dict[p]
    l = r_dict[r]

    with open(fname, "r") as file:
        data = json.loads(file.read())

    A = np.array(data["A"])
    samples = np.array(data["samples"])

    ps = posterior_similarity(samples, A)
    sps = samplewise_posterior_similarity(samples, A)
    fc = fraction_of_correct_entries(samples, A)
    print((i, j, k, l), flush=True)

    c = np.array(data["c"])
    if np.all(c[:3] == 0):
        ctype = 2
    elif np.all(c[:2] == 0):
        ctype = 1
    else:
        ctype = 0

    return i, j, k, l, ps, sps, fc, ctype


# get number of available cores
n_processes = len(os.sched_getaffinity(0))

c_dict, b_dict, p_dict, r_dict = collect_parameters(data_dir)

n_c = len(c_dict)
n_b = len(b_dict)
n_p = len(p_dict)
n_r = len(r_dict)

ps = np.zeros((n_c, n_b, n_p, n_r))
sps = np.zeros((n_c, n_b, n_p, n_r))
fce = np.zeros((n_c, n_b, n_p, n_r))
cmat = np.zeros((n_c, n_b, n_p, n_r))

arglist = []
for f in os.listdir(data_dir):
    arglist.append((f, data_dir, c_dict, b_dict, p_dict, r_dict))

data = Parallel(n_jobs=n_processes)(delayed(get_metrics)(*arg) for arg in arglist)

for i, j, k, l, pos_sim, s_pos_sim, frac_corr, c in data:
    ps[i, j, k, l] = pos_sim
    sps[i, j, k, l] = s_pos_sim
    fce[i, j, k, l] = frac_corr
    cmat[i, j, k, l] = c

data = {}
data["beta"] = list(b_dict)
data["p"] = list(p_dict)
data["sps"] = sps.tolist()
data["ps"] = ps.tolist()
data["fce"] = fce.tolist()
data["c"] = cmat.tolist()
datastring = json.dumps(data)

with open("Data/watts-strogatz.json", "w") as output_file:
    output_file.write(datastring)
