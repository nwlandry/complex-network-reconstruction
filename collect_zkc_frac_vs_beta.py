import json
import os

import numpy as np
from joblib import Parallel, delayed

from lcs import *

data_dir = "Data/zkc_frac_vs_beta/"


def collect_parameters(dir):
    blist = set()
    flist = set()
    rlist = set()

    for f in os.listdir(dir):
        d = f.split(".json")[0].split("_")

        b = float(d[0])
        f = float(d[1])
        r = int(d[2])

        blist.add(b)
        flist.add(f)
        rlist.add(r)

    b_dict = {b: i for i, b in enumerate(sorted(blist))}
    f_dict = {p: i for i, p in enumerate(sorted(flist))}
    r_dict = {r: i for i, r in enumerate(sorted(rlist))}

    return b_dict, f_dict, r_dict


def get_metrics(f, dir, b_dict, f_dict, r_dict):
    fname = os.path.join(dir, f)
    d = f.split(".json")[0].split("_")
    b = float(d[0])
    f = float(d[1])
    r = int(d[2])

    i = b_dict[b]
    j = f_dict[f]
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

b_dict, f_dict, r_dict = collect_parameters(data_dir)

n_b = len(b_dict)
n_f = len(f_dict)
n_r = len(r_dict)

ps = np.zeros((n_f, n_b, n_r))
sps = np.zeros((n_f, n_b, n_r))
fce = np.zeros((n_f, n_b, n_r))

arglist = []
for f in os.listdir(data_dir):
    arglist.append((f, data_dir, b_dict, f_dict, r_dict))

data = Parallel(n_jobs=n_processes)(delayed(get_metrics)(*arg) for arg in arglist)

for i, j, k, pos_sim, s_pos_sim, frac_corr in data:
    ps[i, j, k] = pos_sim
    sps[i, j, k] = s_pos_sim
    fce[i, j, k] = frac_corr

data = {}
data["fraction"] = list(f_dict)
data["beta"] = list(b_dict)
data["sps"] = sps.tolist()
data["ps"] = ps.tolist()
data["fce"] = fce.tolist()

datastring = json.dumps(data)

with open("Data/zkc_frac_vs_beta.json", "w") as output_file:
    output_file.write(datastring)
