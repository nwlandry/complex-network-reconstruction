import numpy as np
import matplotlib.pyplot as plt
from src import *
import os
import json
import networkx as nx

mean_infections = []
ps = []
fraclist = set()
betalist = set()
rlist = set()
beta = []
frac = []

A = nx.adjacency_matrix(nx.karate_club_graph()).todense()

data_dir = "Data/frac_vs_beta/"
for f in os.listdir(data_dir):
    d = f.split(".json")[0].split("-")
    beta = float(d[0])
    frac = float(d[1])
    r = int(d[2])

    fraclist.add(frac)
    betalist.add(beta)
    rlist.add(r)

fraclist = sorted(fraclist)
betalist = sorted(betalist)
rlist = sorted(rlist)

frac_dict = dict(zip(fraclist, range(len(fraclist))))
beta_dict = dict(zip(betalist, range(len(betalist))))
r_dict = dict(zip(rlist, range(len(rlist))))

nf = len(fraclist)
nb = len(betalist)
nr = len(rlist)
print((nf, nb, nr), flush=True)

psmat = np.zeros((nf, nb, nr))
spsmat = np.zeros((nf, nb, nr))
ipn = np.zeros((nf, nb, nr))

it = 0
for f in os.listdir(data_dir):
    d = f.split(".json")[0].split("-")
    beta = float(d[0])
    frac = float(d[1])
    r = int(d[2])

    i = frac_dict[frac]
    j = beta_dict[beta]
    k = r_dict[r]

    fname = os.path.join(data_dir, f)

    with open(fname, "r") as file:
        data = json.loads(file.read())

    x = np.array(data["x"])
    # A = np.array(data["A"])
    samples = np.array(data["samples"])

    ipn[i, j, k] = infections_per_node(x)

    psmat[i, j, k] = posterior_similarity(A, samples)
    spsmat[i, j, k] = samplewise_posterior_similarity(A, samples)
    it += 1
    print(it, flush=True)

data = {}
data["fraction"] = fraclist
data["beta"] = beta
data["sps"] = spsmat.tolist()
data["ps"] = psmat.tolist()
data["ipn"] = ipn.tolist()

datastring = json.dumps(data)

with open("Data/frac_vs_beta.json", "w") as output_file:
    output_file.write(datastring)
