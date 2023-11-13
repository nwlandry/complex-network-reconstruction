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

A1_samples, gamma1_samples, c1_samples, l1 = infer_adjacency_matrix_and_dynamics(
    x1,
    A0,
    p_rho,
    p_gamma,
    p_c,
    nsamples=1000,
    burn_in=100000,
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

p_gamma = np.ones(2)
p_c = np.ones((2, n))
p_rho = np.ones(2)

p = beta(p_rho[0], p_rho[1]).rvs()
A0 = erdos_renyi(n, p)

A2_samples, gamma2_samples, c2_samples, l2 = infer_adjacency_matrix_and_dynamics(
    x2,
    A0,
    p_rho,
    p_gamma,
    p_c,
    nsamples=1000,
    burn_in=100000,
    skip=1000,
    nspa=10,
    return_likelihood=True,
)

print("Threshold contagion complete!")

data = {}
data["A"] = A.tolist()
data["gamma"] = gamma
data["c1"] = c1.tolist()
data["c2"] = c2.tolist()
data["p-rho"] = p_rho.tolist()
data["p-gamma"] = p_gamma.tolist()
data["p-c"] = p_c.tolist()
data["x1"] = x1.tolist()
data["x2"] = x2.tolist()
data["A1-samples"] = A1_samples.tolist()
data["A2-samples"] = A2_samples.tolist()
data["gamma1-samples"] = gamma1_samples.tolist()
data["gamma2-samples"] = gamma2_samples.tolist()
data["c1-samples"] = c1_samples.tolist()
data["c2-samples"] = c2_samples.tolist()
data["l1"] = l1.tolist()
data["l2"] = l2.tolist()

datastring = json.dumps(data)

fname = "Data/infer_contagion_functions.json"

with open(fname, "w") as output_file:
    output_file.write(datastring)
