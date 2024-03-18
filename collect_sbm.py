import json
import os

import numpy as np
from joblib import Parallel, delayed

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

    rho = density(A)

    ps = posterior_similarity(samples, A)
    sps = samplewise_posterior_similarity(samples, A)
    fs = f_score(samples, A)
    fs_norm_random = f_score(samples, A, normalize=True, rho_guess=0.5)
    fs_norm_density = f_score(samples, A, normalize=True, rho_guess=rho)
    fc = fraction_of_correct_entries(samples, A)
    fc_norm_random = fraction_of_correct_entries(
        samples, A, normalize=True, rho_guess=0.5
    )
    fc_norm_density = fraction_of_correct_entries(
        samples, A, normalize=True, rho_guess=rho
    )
    pr = precision(samples, A)
    re = recall(samples, A)
    ar = auroc(samples, A)

    print((i, j, k, l), flush=True)

    return (
        i,
        j,
        k,
        l,
        ps,
        fs,
        fs_norm_random,
        fs_norm_density,
        fc,
        fc_norm_random,
        fc_norm_density,
        pr,
        re,
        sps,
        ar,
    )


# get number of available cores
n_processes = len(os.sched_getaffinity(0))

c_dict, b_dict, e_dict, r_dict = collect_parameters(data_dir)

n_c = len(c_dict)
n_b = len(b_dict)
n_e = len(e_dict)
n_r = len(r_dict)

ps = np.zeros((n_c, n_b, n_e, n_r))
fs = np.zeros((n_c, n_b, n_e, n_r))
fs_norm_random = np.zeros((n_c, n_b, n_e, n_r))
fs_norm_density = np.zeros((n_c, n_b, n_e, n_r))
fce = np.zeros((n_c, n_b, n_e, n_r))
fce_norm_random = np.zeros((n_c, n_b, n_e, n_r))
fce_norm_density = np.zeros((n_c, n_b, n_e, n_r))
pr = np.zeros((n_c, n_b, n_e, n_r))
re = np.zeros((n_c, n_b, n_e, n_r))
sps = np.zeros((n_c, n_b, n_e, n_r))
ar = np.zeros((n_c, n_b, n_e, n_r))

arglist = []
for f in os.listdir(data_dir):
    arglist.append((f, data_dir, c_dict, b_dict, e_dict, r_dict))

data = Parallel(n_jobs=n_processes)(delayed(get_metrics)(*arg) for arg in arglist)

for (
    i,
    j,
    k,
    l,
    metric1,
    metric2,
    metric3,
    metric4,
    metric5,
    metric6,
    metric7,
    metric8,
    metric9,
    metric10,
    metric11,
) in data:
    ps[i, j, k, l] = metric1
    fs[i, j, k, l] = metric2
    fs_norm_random[i, j, k, l] = metric3
    fs_norm_density[i, j, k, l] = metric4
    fce[i, j, k, l] = metric5
    fce_norm_random[i, j, k, l] = metric6
    fce_norm_density[i, j, k, l] = metric7
    pr[i, j, k, l] = metric8
    re[i, j, k, l] = metric9
    sps[i, j, k, l] = metric10
    ar[i, j, k, l] = metric11

data = {}
data["beta"] = list(b_dict)
data["epsilon"] = list(e_dict)
data["ps"] = ps.tolist()
data["fs"] = fs.tolist()
data["fs-norm-random"] = fs_norm_random.tolist()
data["fs-norm-density"] = fs_norm_density.tolist()
data["fce"] = fce.tolist()
data["fce-norm-random"] = fce_norm_random.tolist()
data["fce-norm-density"] = fce_norm_density.tolist()
data["precision"] = pr.tolist()
data["recall"] = re.tolist()
data["sps"] = sps.tolist()
data["auroc"] = ar.tolist()
datastring = json.dumps(data)

with open("Data/sbm.json", "w") as output_file:
    output_file.write(datastring)
