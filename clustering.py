import json
import multiprocessing as mp
import os

import numpy as np

from lcs import *


def gen_clustered_clique_number(n, clique_number):
    clique_size = n // clique_number  # the number of nodes per clique
    clique_membership = 0  # the number of cliques per node

    p_dist = delta_dist(clique_size)
    g_dist = delta_dist(clique_membership)
    A = clustered_unipartite(clique_number, n, p_dist, g_dist)
    return A


data_dir = "Data/clustering"
os.makedirs(data_dir, exist_ok=True)

for f in os.listdir(data_dir):
    os.remove(os.path.join(data_dir, f))

n_processes = len(os.sched_getaffinity(0))

# contagion functions and parameters
cf1 = lambda nu, b: 1 - (1 - b) ** nu  # simple contagion
cf2 = lambda nu, b: b * (nu >= 2)  # complex contagion, tau=2
cf3 = lambda nu, b: b * (nu >= 3)  # complex contagion, tau=3

cfs = [cf1, cf2, cf3]

realizations = 10
n_c = 33
n_b = 33

n = 100
k = 6
clique_number = np.arange(1, n_c)
beta_list = np.linspace(0.0, 1.0, n_b)

rho0 = 1.0
gamma = 0.1

tmax = 1000

# MCMC parameters
burn_in = 100000
nsamples = 100
skip = 1500
p_c = np.ones((2, n))
p_rho = np.array([1, 1])


arglist = []
for i, cf in enumerate(cfs):
    for b in beta_list:
        c = cf(np.arange(n), b)
        for s in clique_number:
            for r in range(realizations):
                A = gen_clustered_clique_number(n, clique_number)
                arglist.append(
                    (
                        f"{data_dir}/{i}_{b}_{s}_{r}",
                        gamma,
                        c,
                        b,
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
