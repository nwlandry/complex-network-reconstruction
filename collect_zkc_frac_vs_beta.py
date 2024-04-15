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

b_dict, f_dict, r_dict = collect_parameters(data_dir)

n_b = len(b_dict)
n_f = len(f_dict)
n_r = len(r_dict)

data = {}
data["fraction"] = list(f_dict)
data["beta"] = list(b_dict)
data["rho"] = np.zeros((n_f, n_b, n_r))
data["rho-samples"] = np.zeros((n_f, n_b, n_r))
data["ps"] = np.zeros((n_f, n_b, n_r))
data["sps"] = np.zeros((n_f, n_b, n_r))
data["fs"] = np.zeros((n_f, n_b, n_r))
data["fs-norm-random"] = np.zeros((n_f, n_b, n_r))
data["fs-norm-density"] = np.zeros((n_f, n_b, n_r))
data["fce"] = np.zeros((n_f, n_b, n_r))
data["fce-norm-random"] = np.zeros((n_f, n_b, n_r))
data["fce-norm-density"] = np.zeros((n_f, n_b, n_r))
data["precision"] = np.zeros((n_f, n_b, n_r))
data["recall"] = np.zeros((n_f, n_b, n_r))
data["auroc"] = np.zeros((n_f, n_b, n_r))
data["auprc"] = np.zeros((n_f, n_b, n_r))

arglist = []
for f in os.listdir(data_dir):
    arglist.append((f, data_dir, b_dict, f_dict, r_dict))

results = Parallel(n_jobs=n_processes)(delayed(get_metrics)(*arg) for arg in arglist)

for i, j, k, m in results:
    for key, val in m.items():
        data[key][i, j, k] = val

for key, val in data.items():
    if not isinstance(val, list):
        data[key] = val.tolist()

datastring = json.dumps(data)

with open("Data/zkc_frac_vs_beta.json", "w") as output_file:
    output_file.write(datastring)
