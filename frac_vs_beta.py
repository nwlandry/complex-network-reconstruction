import numpy as np
import json
from src import *
import networkx as nx

G = nx.karate_club_graph()

A = nx.adjacency_matrix(G, weight=None)
n = np.size(A, axis=0)

realizations = 5
nf = 5
nb = 5

frac = np.linspace(0, 1.0, nf)
beta = np.linspace(0, 2.0, nb)

# MCMC parameters
burn_in = 1000
nsamples = 100
skip = 100
p_c = np.ones((2, n))
p_rho = [2, 5]

# contagion functions and parameters
sc = lambda nu, beta: 1 - (1 - beta) ** nu  # simple contagion
cc = lambda nu, tau, beta: beta * (nu >= tau)  # complex contagion

gamma = 1
tau = 3
p_s = 0.1

ps = np.zeros((nf, nb, realizations))
sps = np.zeros((nf, nb, realizations))

for i, b in enumerate(beta):
    for j, f in enumerate(frac):
        c = f * sc(np.arange(n), b) + (1 - f) * cc(np.arange(n), tau, b)

        for k in range(realizations):
            s0 = np.zeros(n)
            s0[list(random.sample(range(n), int(p_s * n)))] = 1

            x = contagion_process(A, gamma, c, s0, tmin=0, tmax=100, random_seed=None)

            A0 = nx.adjacency_matrix(nx.fast_gnp_random_graph(n, 0.3))

            samples = infer_adjacency_matrix(
                x, A0, p_rho, p_c, nsamples=nsamples, burn_in=burn_in, skip=100
            )

            ps[i, j, k] = posterior_similarity(A, samples)
            sps[i, j, k] = samplewise_posterior_similarity(A, samples)

data = {}
data["ps"] = ps.tolist()
data["sps"] = sps.tolist()

datastring = json.dumps(data)

with open("test.json", "w") as output_file:
    output_file.write(datastring)