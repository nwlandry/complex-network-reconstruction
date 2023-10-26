import json
import multiprocessing as mp
import os

import networkx as nx
import numpy as np
from src import *


def single_inference(
    fname, gamma, c, rho0, A, tmax, p_c, p_rho, nsamples, burn_in, skip
):
    n = A.shape[0]
    x0 = np.zeros(n)
    x0[random.sample(range(n), int(round(rho0 * n)))] = 1

    x = contagion_process(A, gamma, c, x0, tmin=0, tmax=tmax)

    A0 = nx.adjacency_matrix(nx.fast_gnp_random_graph(n, 0.3))

    samples = infer_adjacency_matrix(
        x, A0, p_rho, p_c, nsamples=nsamples, burn_in=burn_in, skip=skip
    )
    data = {}
    data["gamma"] = gamma
    data["c"] = c.tolist()
    data["p-rho"] = p_rho.tolist()
    data["p-c"] = p_c.tolist()
    data["x"] = x.tolist()
    data["A"] = A.tolist()
    data["samples"] = samples.tolist()

    datastring = json.dumps(data)

    with open(fname, "w") as output_file:
        output_file.write(datastring)


data_dir = "Data/frac_vs_beta"
os.makedirs(data_dir, exist_ok=True)

for f in os.listdir(data_dir):
    os.remove(os.path.join(data_dir, f))

G = nx.karate_club_graph()
A = nx.adjacency_matrix(G, weight=None).todense()
n = A.shape[0]

n_processes = len(os.sched_getaffinity(0))
realizations = 10
nf = 33
nb = 33

# MCMC parameters
burn_in = 10000
nsamples = 1000
skip = 2000
p_c = np.ones((2, n))
p_rho = np.array([1, 1])

# contagion functions and parameters
sc = lambda nu, beta: 1 - (1 - beta) ** nu  # simple contagion
cc = lambda nu, tau, beta: beta * (nu >= tau)  # complex contagion

rho0 = 1.0
gamma = 0.1
tau = 2

beta = np.linspace(0, 1.0, nb)
frac = np.linspace(0, 1.0, nf)
tmax = 1000

arglist = []
for i, b in enumerate(beta):
    for j, f in enumerate(frac):
        c = f * sc(np.arange(n), b) + (1 - f) * cc(np.arange(n), tau, b)
        for k in range(realizations):
            arglist.append(
                (
                    f"Data/frac_vs_beta/{b}-{f}-{k}",
                    gamma,
                    c,
                    rho0,
                    A,
                    tmax,
                    p_c,
                    p_rho,
                    nsamples,
                    burn_in,
                    skip,
                )
            )

with mp.Pool(processes=n_processes) as pool:
    pool.starmap(single_inference, arglist)
