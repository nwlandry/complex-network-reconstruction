import json
import multiprocessing as mp
import os

import networkx as nx
import numpy as np

from src import *


def single_inference(
    fname, gamma, c, rho0, A, tmax, p_c, p_rho, nsamples, burn_in, skip
):
    n = np.size(A, axis=0)
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


data_dir = "Data/erdos-renyi_experiment"
os.makedirs(data_dir, exist_ok=True)

for f in os.listdir(data_dir):
    os.remove(os.path.join(data_dir, f))

n = 50

n_processes = len(os.sched_getaffinity(0))
realizations = 5
probabilities = np.linspace(0.0, 1.0, 33)

# MCMC parameters
burn_in = 10000
nsamples = 100
skip = 1500
p_c = np.ones((2, n))
p_rho = np.array([1, 1])

# contagion functions and parameters
cf1 = lambda nu, beta: 1 - (1 - beta) ** nu  # simple contagion
cf2 = lambda nu, beta: beta * (nu >= 2)  # complex contagion, tau=2
cf3 = lambda nu, beta: beta * (nu >= 3)  # complex contagion, tau=3

cfs = [cf1, cf2, cf3]

rho0 = 1.0
gamma = 0.1
b = 0.04

tmax = 1000


arglist = []
for i, cf in enumerate(cfs):
    for p in probabilities:
        c = cf(np.arange(n), b)
        for r in range(realizations):
            A = erdos_renyi(n, p)
            arglist.append(
                (
                    f"{data_dir}/{i}-{p}-{r}",
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
