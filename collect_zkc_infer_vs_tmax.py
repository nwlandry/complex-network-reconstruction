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

    m = dict()
    m["rho"] = density(A)
    m["rho-samples"] = density(samples.mean(axis=0))
    m["ps"] = posterior_similarity(samples, A)
    m["sps"] = samplewise_posterior_similarity(samples, A)
    m["fs"] = f_score(samples, A)
    m["fs-norm-random"] = f_score(samples, A, normalize=True, rho_guess=0.5)
    m["fs-norm-density"] = f_score(samples, A, normalize=True, rho_guess=m["rho"])
    m["fce"] = fraction_of_correct_entries(samples, A)
    m["fce-norm-random"] = fraction_of_correct_entries(
        samples, A, normalize=True, rho_guess=0.5
    )
    m["fce-norm-density"] = fraction_of_correct_entries(
        samples, A, normalize=True, rho_guess=m["rho"]
    )
    m["precision"] = precision(samples, A)
    m["recall"] = recall(samples, A)
    m["auroc"] = auroc(samples, A)
    m["auprc"] = auprc(samples, A)
    print((i, j, k), flush=True)

    return i, j, k, m


# get number of available cores
n_processes = len(os.sched_getaffinity(0))

c_dict, t_dict, r_dict = collect_parameters(data_dir)

n_c = len(c_dict)
n_t = len(t_dict)
n_r = len(r_dict)

data = {}
data["tmax"] = list(t_dict)
data["rho"] = np.zeros((n_c, n_t, n_r))
data["rho-samples"] = np.zeros((n_c, n_t, n_r))
data["ps"] = np.zeros((n_c, n_t, n_r))
data["sps"] = np.zeros((n_c, n_t, n_r))
data["fs"] = np.zeros((n_c, n_t, n_r))
data["fs-norm-random"] = np.zeros((n_c, n_t, n_r))
data["fs-norm-density"] = np.zeros((n_c, n_t, n_r))
data["fce"] = np.zeros((n_c, n_t, n_r))
data["fce-norm-random"] = np.zeros((n_c, n_t, n_r))
data["fce-norm-density"] = np.zeros((n_c, n_t, n_r))
data["precision"] = np.zeros((n_c, n_t, n_r))
data["recall"] = np.zeros((n_c, n_t, n_r))
data["auroc"] = np.zeros((n_c, n_t, n_r))
data["auprc"] = np.zeros((n_c, n_t, n_r))

arglist = []
for f in os.listdir(data_dir):
    arglist.append((f, data_dir, c_dict, t_dict, r_dict))

results = Parallel(n_jobs=n_processes)(delayed(get_metrics)(*arg) for arg in arglist)

for i, j, k, l, m in results:
    for key, val in m.items():
        data[key][i, j, k, l] = val

for key, val in data.items():
    if not isinstance(val, list):
        data[key] = val.tolist()

datastring = json.dumps(data)

with open("Data/zkc_infer_vs_tmax.json", "w") as output_file:
    output_file.write(datastring)
