import json
import os

import numpy as np
from joblib import Parallel, delayed

from lcs import *

data_dir = "Data/erdos-renyi/"


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

    print((i, j, k, l), flush=True)

    return i, j, k, l, m


# get number of available cores
n_processes = len(os.sched_getaffinity(0))

c_dict, b_dict, p_dict, r_dict = collect_parameters(data_dir)

n_c = len(c_dict)
n_b = len(b_dict)
n_p = len(p_dict)
n_r = len(r_dict)

data = {}
data["beta"] = list(b_dict)
data["p"] = list(p_dict)
data["rho"] = np.zeros((n_c, n_b, n_p, n_r))
data["rho-samples"] = np.zeros((n_c, n_b, n_p, n_r))
data["ps"] = np.zeros((n_c, n_b, n_p, n_r))
data["sps"] = np.zeros((n_c, n_b, n_p, n_r))
data["fs"] = np.zeros((n_c, n_b, n_p, n_r))
data["fs-norm-random"] = np.zeros((n_c, n_b, n_p, n_r))
data["fs-norm-density"] = np.zeros((n_c, n_b, n_p, n_r))
data["fce"] = np.zeros((n_c, n_b, n_p, n_r))
data["fce-norm-random"] = np.zeros((n_c, n_b, n_p, n_r))
data["fce-norm-density"] = np.zeros((n_c, n_b, n_p, n_r))
data["precision"] = np.zeros((n_c, n_b, n_p, n_r))
data["recall"] = np.zeros((n_c, n_b, n_p, n_r))
data["auroc"] = np.zeros((n_c, n_b, n_p, n_r))
data["auprc"] = np.zeros((n_c, n_b, n_p, n_r))

arglist = []
for f in os.listdir(data_dir):
    arglist.append((f, data_dir, c_dict, b_dict, p_dict, r_dict))

results = Parallel(n_jobs=n_processes)(delayed(get_metrics)(*arg) for arg in arglist)

for i, j, k, l, m in results:
    for key, val in m.items():
        data[key][i, j, k, l] = val

for key, val in data.items():
    if not isinstance(val, list):
        data[key] = val.tolist()

datastring = json.dumps(data)

with open("Data/erdos-renyi.json", "w") as output_file:
    output_file.write(datastring)
