import json

import networkx as nx
import numpy as np
from scipy.stats import beta

from lcs import *

G = nx.karate_club_graph()
A = nx.adjacency_matrix(G, weight=None).todense()
n = A.shape[0]

p_gamma = [1, 1]
p_c = np.ones((2, n))
p_rho = [1, 1]


# simple contagion
rho0 = 1

x0 = np.zeros(n)
x0[list(random.sample(range(n), int(rho0 * n)))] = 1

gamma = 0.1
b1 = 0.04

contagion_function = lambda nu, b: 1 - (1 - b) ** nu
c1 = contagion_function(np.arange(n), b1)

x1 = contagion_process(A, gamma, c1, x0, tmin=0, tmax=1000, random_seed=None)

p = beta(p_rho[0], p_rho[1]).rvs()
A0 = erdos_renyi(n, p)

samples1, gamma1, cf1, l1 = infer_adjacency_matrix_and_dynamics(
    x1,
    A0,
    p_rho,
    p_gamma,
    p_c,
    nsamples=1000,
    burn_in=30000,
    skip=1000,
    nspa=10,
    return_likelihood=True,
)

print("Simple contagion complete!")

# complex contagion, tau=2
b2 = 0.2

contagion_function = lambda nu, b: b * (nu >= 2)
c2 = contagion_function(np.arange(n), b2)

x2 = contagion_process(A, gamma, c2, x0, tmin=0, tmax=1000, random_seed=None)

p_gamma = [1, 1]
p_c = np.ones((2, n))
p_rho = [1, 1]

p = beta(p_rho[0], p_rho[1]).rvs()
A0 = erdos_renyi(n, p)

samples2, gamma2, cf2, l2 = infer_adjacency_matrix_and_dynamics(
    x2,
    A0,
    p_rho,
    p_gamma,
    p_c,
    nsamples=1000,
    burn_in=30000,
    skip=1000,
    nspa=10,
    return_likelihood=True,
)

print("Threshold contagion complete!")

data = {}
data["A"] = A.tolist()
data["gamma"] = gamma
data["c1"] = c1.tolist()
data["c2"] = c1.tolist()
data["p-rho"] = p_rho.tolist()
data["p-gamma"] = p_gamma.tolist()
data["p-c"] = p_c.tolist()
data["x1"] = x1.tolist()
data["x2"] = x1.tolist()
data["samples1"] = samples1.tolist()
data["samples2"] = samples2.tolist()
data["gamma-samples1"] = gamma1.tolist()
data["gamma-samples2"] = gamma2.tolist()
data["c-samples1"] = cf1.tolist()
data["c-samples2"] = cf2.tolist()
data["l1"] = l1.tolist()
data["l2"] = l2.tolist()

datastring = json.dumps(data)

fname = "Data/infer_dynamics.json"

with open(fname, "w") as output_file:
    output_file.write(datastring)
