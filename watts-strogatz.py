import os

import numpy as np
from joblib import Parallel, delayed

from lcs import *

data_dir = "Data/watts-strogatz"
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
n_p = 33
n_b = 33

n = 50
k = 6
p_list = np.logspace(-6, 0, 49)
beta_list = np.linspace(0.0, 1.0, n_b)
rho0 = 1.0
gamma = 0.1

tmax = 2000

# MCMC parameters
burn_in = 100000
nsamples = 100
skip = 10000
p_c = np.ones((2, n))
p_rho = np.array([1, 1])


arglist = []
for i, cf in enumerate(cfs):
    for b in beta_list:
        for p in p_list:
            for r in range(realizations):
                c = cf(np.arange(n), b)
                A = watts_strogatz(n, k, p)
                arglist.append(
                    (
                        f"{data_dir}/{i}_{b}_{p}_{r}",
                        gamma,
                        c,
                        b,
                        rho0,
                        A,
                        tmax,
                        p_c.copy(),
                        p_rho.copy(),
                        nsamples,
                        burn_in,
                        skip,
                    )
                )
Parallel(n_jobs=n_processes)(delayed(single_inference)(*arg) for arg in arglist)
