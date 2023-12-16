import os

import numpy as np
from joblib import Parallel, delayed

from lcs import *

data_dir = "Data/zkc_frac_vs_beta"
os.makedirs(data_dir, exist_ok=True)

for f in os.listdir(data_dir):
    os.remove(os.path.join(data_dir, f))

A = zkc()
n = A.shape[0]

n_processes = len(os.sched_getaffinity(0))
realizations = 10
nf = 33
nb = 33

# MCMC parameters
burn_in = 100000
nsamples = 1000
skip = 10000
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
        for k in range(realizations):
            c = f * sc(np.arange(n), b) + (1 - f) * cc(np.arange(n), tau, b)
            arglist.append(
                (
                    f"{data_dir}/{b}_{f}_{k}",
                    gamma,
                    c,
                    rho0,
                    A.copy(),
                    tmax,
                    p_c.copy(),
                    p_rho.copy(),
                    nsamples,
                    burn_in,
                    skip,
                )
            )

Parallel(n_jobs=n_processes)(delayed(single_inference)(*arg) for arg in arglist)
