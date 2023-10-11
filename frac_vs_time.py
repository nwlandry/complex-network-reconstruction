import numpy as np
import json
from src import *
import networkx as nx
from numpy.linalg import eigh
import multiprocessing as mp
import os


def single_inference(gamma, c, A, tmax, p_c, p_rho, nsamples, burn_in, skip):
    n = np.size(A, axis=0)
    x0 = np.zeros(n)
    x0[random.randrange(n)] = 1

    x = contagion_process(A, gamma, c, x0, tmin=0, tmax=tmax)

    A0 = nx.adjacency_matrix(nx.fast_gnp_random_graph(n, 0.3))

    samples = infer_adjacency_matrix(
        x, A0, p_rho, p_c, nsamples=nsamples, burn_in=burn_in, skip=skip
    )
    return posterior_similarity(A, samples), samplewise_posterior_similarity(A, samples)


G = nx.karate_club_graph()

A = nx.adjacency_matrix(G, weight=None).todense()
n = np.size(A, axis=0)

n_processes = len(os.sched_getaffinity(0))
realizations = 10
nt = 10
nf = 17


# MCMC parameters
burn_in = 10000
nsamples = 100
skip = 1000
p_c = np.ones((2, n))
p_rho = np.array([1, 1])

# contagion functions and parameters
sc = lambda nu, beta: 1 - (1 - beta) ** nu  # simple contagion
cc = lambda nu, tau, beta: beta * (nu >= tau)  # complex contagion

gamma = 1
tau = 3
nu = eigh(A)[0][-1]
bc = gamma / nu  # quenched mean-field threshold
b = 2 * bc

tmax = np.logspace(10, 1000, nt)
frac = np.linspace(0, 1.0, nf)

ps = np.zeros((nt, nf, realizations))
sps = np.zeros((nt, nf, realizations))

arglist = []
for i, t in enumerate(tmax):
    for j, f in enumerate(frac):
        c = f * sc(np.arange(n), b) + (1 - f) * cc(np.arange(n), tau, b)
        for k in range(realizations):
            arglist.append((gamma, c, A, tmax, p_c, p_rho, nsamples, burn_in, skip))

with mp.Pool(processes=n_processes) as pool:
    similarities = pool.starmap(single_inference, arglist)

idx = 0
for i, b in enumerate(tmax):
    for j, f in enumerate(frac):
        for k in range(realizations):
            ps[i, j, k] = similarities[idx][0]
            sps[i, j, k] = similarities[idx][1]
            idx += 1

data = {}
data["gamma"] = gamma
data["beta"] = beta.tolist()
data["fraction"] = frac.tolist()
data["p-rho"] = p_rho.tolist()
data["p-c"] = p_c.tolist()
data["ps"] = ps.tolist()
data["sps"] = sps.tolist()

datastring = json.dumps(data)

with open("test.json", "w") as output_file:
    output_file.write(datastring)
