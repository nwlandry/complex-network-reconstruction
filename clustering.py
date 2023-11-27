import json
import multiprocessing as mp
import os

import numpy as np

from lcs import *


def target_ipn(n, k, p, gamma, c, mode, rho0, tmax, realizations):
    x0 = np.zeros(n)
    x0[random.sample(range(n), int(round(rho0 * n)))] = 1
    ipn = 0
    for _ in range(realizations):
        A = watts_strogatz(n, k, p)
        x = contagion_process(A, gamma, c, x0, tmin=0, tmax=tmax)
        ipn += infections_per_node(x, mode) / realizations
    return ipn


def single_inference(
    fname, gamma, c, b, rho0, A, tmax, p_c, p_rho, nsamples, burn_in, skip
):
    n = np.size(A, axis=0)
    x0 = np.zeros(n)
    x0[random.sample(range(n), int(round(rho0 * n)))] = 1

    x = contagion_process(A, gamma, c, x0, tmin=0, tmax=tmax)
    p = beta(p_rho[0], p_rho[1]).rvs()
    A0 = erdos_renyi(n, p)
    samples = infer_adjacency_matrix(
        x, A0, p_rho, p_c, nsamples=nsamples, burn_in=burn_in, skip=skip
    )

    # json dict
    data = {}
    data["gamma"] = gamma
    data["c"] = c.tolist()
    data["b"] = b
    data["p-rho"] = p_rho.tolist()
    data["p-c"] = p_c.tolist()
    data["x"] = x.tolist()
    data["A"] = A.tolist()
    data["samples"] = samples.tolist()

    datastring = json.dumps(data)

    with open(fname, "w") as output_file:
        output_file.write(datastring)


data_dir = "Data/clustering"
os.makedirs(data_dir, exist_ok=True)

for f in os.listdir(data_dir):
    os.remove(os.path.join(data_dir, f))

n = 50
k = 6

n_processes = len(os.sched_getaffinity(0))
realizations = 10
probabilities = np.logspace(-6, 0, 49)

# MCMC parameters
burn_in = 100000
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
mode = "max"

tmax = 1000


#paramters for bipartite clustering
N = 200#the number of nodes in the network
clique_number = 2#the number ofj:w


clique_size = N//clique_number#the number of nodes per clique
clique_membership = 1#the number of cliques per node

p_dist = delta_dist(clique_number)
g_dist = delta_dist(clique_size)

#clustered_unipartite(clique_number,N,p_dist,g_dist)

edge_list,vertex_attributes = generate_hypergraph_bipartite_edge_list(10,100,p_dist,g_dist)


arglist = []
for p in probabilities:




    c = cfs[0](np.arange(n), b)
    ipn = target_ipn(n, k, p, gamma, c, mode, rho0, tmax, 1000)
    for i, cf in enumerate(cfs):
        if i != 0:
            A = watts_strogatz(n, k, p)
            bscaled = fit_ipn(0.5, ipn, cf, gamma, A, rho0, tmax, mode)
        else:
            bscaled = b
        c = cf(np.arange(n), bscaled)
        print((p, i), flush=True)

        for r in range(realizations):
            A = watts_strogatz(n, k, p)
            arglist.append(
                (
                    f"{data_dir}/{p}_{i}_{r}",
                    gamma,
                    c,
                    bscaled,
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
