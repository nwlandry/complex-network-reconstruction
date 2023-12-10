import json
import multiprocessing as mp
import os

import numpy as np

from lcs import *

data_dir = "Data/sbm/"


def collect_parameters(dir):
    clist = set()
    blist = set()
    elist = set()
    rlist = set()

    for f in os.listdir(dir):
        d = f.split(".json")[0].split("_")

        c = int(d[0])
        b = float(d[1])
        e = float(d[2])
        r = int(d[3])

        clist.add(c)
        blist.add(b)
        elist.add(e)
        rlist.add(r)

    c_dict = {c: i for i, c in enumerate(sorted(clist))}
    b_dict = {b: i for i, b in enumerate(sorted(blist))}
    e_dict = {e: i for i, e in enumerate(sorted(elist))}
    r_dict = {r: i for i, r in enumerate(sorted(rlist))}

    return c_dict, b_dict, e_dict, r_dict


def get_metrics(f, dir, c_dict, b_dict, e_dict, r_dict):
    fname = os.path.join(dir, f)
    d = f.split(".json")[0].split("_")
    c = int(d[0])
    b = float(d[1])
    e = float(d[2])
    r = int(d[3])

    i = c_dict[c]
    j = b_dict[b]
    k = e_dict[e]
    l = r_dict[r]

    with open(fname, "r") as file:
        data = json.loads(file.read())

    A = np.array(data["A"])
    samples = np.array(data["samples"])

    ps = posterior_similarity(samples, A)
    sps = samplewise_posterior_similarity(samples, A)
    fc = fraction_of_correct_entries(samples, A)
    print((i, j, k, l), flush=True)

    return i, j, k, l, ps, sps, fc


# get number of available cores
n_processes = len(os.sched_getaffinity(0))

c_dict, b_dict, e_dict, r_dict = collect_parameters(data_dir)

n_c = len(c_dict)
n_b = len(b_dict)
n_e = len(e_dict)
n_r = len(r_dict)

ps = np.zeros((n_c, n_b, n_e, n_r))
sps = np.zeros((n_c, n_b, n_e, n_r))
fce = np.zeros((n_c, n_b, n_e, n_r))

arglist = []
for f in os.listdir(data_dir):
    arglist.append((f, data_dir, c_dict, b_dict, e_dict, r_dict))

with mp.Pool(processes=n_processes) as pool:
    data = pool.starmap(get_metrics, arglist)

for i, j, k, l, pos_sim, s_pos_sim, frac_corr in data:
    ps[i, j, k, l] = pos_sim
    sps[i, j, k, l] = s_pos_sim
    fce[i, j, k, l] = frac_corr

data = {}
data["beta"] = list(b_dict)
data["epsilon"] = list(e_dict)
data["sps"] = sps.tolist()
data["ps"] = ps.tolist()
data["fce"] = fce.tolist()
datastring = json.dumps(data)

with open("Data/sbm.json", "w") as output_file:
    output_file.write(datastring)
