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

    checked_matrix_size = False

    for f in os.listdir(dir):
        if not checked_matrix_size:
            fname = os.path.join(dir, f)
            with open(fname, "r") as file:
                data = json.loads(file.read())
            n = np.array(data["A"]).shape[0]
            checked_matrix_size = True

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

    return c_dict, t_dict, r_dict, n


def get_matrices(f, dir, c_dict, t_dict, r_dict):
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

    A = np.array(data["A"], dtype=float)
    samples = np.array(data["samples"], dtype=float)

    print((i, j, k), flush=True)

    return i, j, k, A, samples.mean(axis=0)


# get number of available cores
n_processes = len(os.sched_getaffinity(0))

c_dict, t_dict, r_dict, n = collect_parameters(data_dir)

n_c = len(c_dict)
n_t = len(t_dict)
n_r = len(r_dict)

Qsamples = np.zeros((n_c, n_t, n_r, n, n))

arglist = []
for f in os.listdir(data_dir):
    arglist.append((f, data_dir, c_dict, t_dict, r_dict))

data = Parallel(n_jobs=n_processes)(delayed(get_matrices)(*arg) for arg in arglist)

node_performance_simple = np.zeros([n, n_t, n_r])
node_performance_complex = np.zeros([n, n_t, n_r])
for i, j, k, A, Q in data:
    if i == 0:
        node_performance_simple[:, j, k] = nodal_performance(Q, A)
    if i == 1:
        node_performance_complex[:, j, k] = nodal_performance(Q, A)

data = {}
data["tmax"] = list(t_dict)
data["A"] = A.tolist()
data["node-performance-simple"] = node_performance_simple.tolist()
data["node-performance-complex"] = node_performance_complex.tolist()


datastring = json.dumps(data)

with open("Data/zkc_tmax_comparison.json", "w") as output_file:
    output_file.write(datastring)
