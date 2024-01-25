import json

import numpy as np

from lcs import *

A = zkc()
n = A.shape[0]

p_gamma = np.ones(2)
p_c = np.ones((2, n))
p_rho = np.ones(2)

tmax = 1000
nsamples = 10000

# simple contagion
rho0 = 1

x0 = np.zeros(n)
x0[list(random.sample(range(n), int(rho0 * n)))] = 1

gamma = 0.1
b1 = 0.04

cf1 = lambda nu, b: 1 - (1 - b) ** nu
c1 = cf1(np.arange(n), b1)

x1 = contagion_process(A, gamma, c1, x0, tmin=0, tmax=tmax, random_seed=None)

gamma1_samples, c1_samples = infer_dynamics(x1, A, p_rho, p_gamma, p_c, nsamples=1000)

nu1 = np.zeros(n)
for i, val in zip(*np.unique(A @ x1.T, return_counts=True)):
    nu1[i] = val

print("Simple contagion complete!")

# complex contagion, tau=2
cf2 = lambda nu, b: b * (nu >= 2)

ipn = target_ipn(A, gamma, c1, "max", rho0, tmax, 1000)
b2 = fit_ipn(0.5, ipn, cf2, gamma, A, rho0, tmax, "max")

c2 = cf2(np.arange(n), b2)

x2 = contagion_process(A, gamma, c2, x0, tmin=0, tmax=tmax, random_seed=None)

gamma2_samples, c2_samples = infer_dynamics(x2, A, p_rho, p_gamma, p_c, nsamples=1000)

nu2 = np.zeros(n)
for i, val in zip(*np.unique(A @ x2.T, return_counts=True)):
    nu2[i] = val

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
data["gamma1-samples"] = gamma1_samples.tolist()
data["gamma2-samples"] = gamma2_samples.tolist()
data["c1-samples"] = c1_samples.tolist()
data["c2-samples"] = c2_samples.tolist()
data["nu1"] = nu1.tolist()
data["nu2"] = nu2.tolist()

datastring = json.dumps(data)

fname = "Data/infer_contagion_functions.json"

with open(fname, "w") as output_file:
    output_file.write(datastring)
