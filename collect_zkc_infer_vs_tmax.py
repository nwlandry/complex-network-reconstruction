import json
import os

import numpy as np
from joblib import Parallel, delayed

from lcs import *

data_dir = "Data/zkc_infer_vs_tmax/"


def collect_parameters(dir):
    clist = set()
    tlist = set()
    rlist = set()

    for f in os.listdir(dir):
        d = f.split(".json")[0].split("_")

        c = int(d[0])
        t = float(d[1])
        r = float(d[2])

        clist.add(c)
        tlist.add(t)
        rlist.add(r)

    c_dict = {c: i for i, c in enumerate(sorted(clist))}
    t_dict = {t: i for i, t in enumerate(sorted(tlist))}
    r_dict = {r: i for i, r in enumerate(sorted(rlist))}

    return c_dict, t_dict, r_dict


def get_metrics(f, dir, c_dict, t_dict, r_dict):
    fname = os.path.join(dir, f)
    d = f.split(".json")[0].split("_")
    c = int(d[0])
    t = float(d[1])
    r = float(d[2])

    i = c_dict[c]
    j = t_dict[t]
    k = r_dict[r]

    with open(fname, "r") as file:
        data = json.loads(file.read())

    A = np.array(data["A"])
    samples = np.array(data["samples"])

    ps = posterior_similarity(samples, A)
    sps = samplewise_posterior_similarity(samples, A)
    fc = fraction_of_correct_entries(samples, A)
    print((i, j, k), flush=True)

    return i, j, k, ps, sps, fc


# get number of available cores
n_processes = len(os.sched_getaffinity(0))

c_dict, t_dict, r_dict = collect_parameters(data_dir)

n_c = len(c_dict)
n_t = len(t_dict)
n_r = len(r_dict)

ps = np.zeros((n_c, n_t, n_r))
sps = np.zeros((n_c, n_t, n_r))
fce = np.zeros((n_c, n_t, n_r))

arglist = []
for f in os.listdir(data_dir):
    arglist.append((f, data_dir, c_dict, t_dict, r_dict))

data = Parallel(n_jobs=n_processes)(delayed(get_metrics)(*arg) for arg in arglist)

for i, j, k, pos_sim, s_pos_sim, frac_corr in data:
    ps[i, j, k] = pos_sim
    sps[i, j, k] = s_pos_sim
    fce[i, j, k] = frac_corr

data = {}
data["tmax"] = list(t_dict)
data["sps"] = sps.tolist()
data["ps"] = ps.tolist()
data["fce"] = fce.tolist()
datastring = json.dumps(data)

with open("Data/zkc_infer_vs_tmax.json", "w") as output_file:
    output_file.write(datastring)
