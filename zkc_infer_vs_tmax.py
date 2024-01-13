import os

import numpy as np
from joblib import Parallel, delayed

from lcs import *

data_dir = "Data/zkc_infer_vs_tmax"
os.makedirs(data_dir, exist_ok=True)

for f in os.listdir(data_dir):
    os.remove(os.path.join(data_dir, f))

n_processes = len(os.sched_getaffinity(0))

n_t = 33
tmax = np.logspace(1, 5, n_t).astype(int)
realizations = 100

A = zkc()
n = A.shape[0]

p_c = np.ones((2, n))
p_rho = [1, 1]

rho0 = 1

# MCMC parameters
burn_in = 100000
nsamples = 100
skip = 10000
p_c = np.ones((2, n))
p_rho = np.array([1, 1])

x0 = np.zeros(n)
x0[list(random.sample(range(n), int(rho0 * n)))] = 1

gamma = 0.1
b1 = 0.04

cf1 = lambda nu, b: 1 - (1 - b) ** nu
cf2 = lambda nu, b: b * (nu >= 2)

c1 = cf1(np.arange(n), b1)

ipn = target_ipn(A, gamma, c1, "max", rho0, 1000, 1000)
print(f"IPN to match is {ipn}")
b2 = fit_ipn(0.5, ipn, cf2, gamma, A, rho0, 1000, "max")

c2 = cf2(np.arange(n), b2)

cs = [c1, c2]
bs = [b1, b2]

arglist = []
for i, c in enumerate(cs):
    for tm in tmax:
        for r in range(realizations):
            arglist.append(
                (
                    f"{data_dir}/{i}_{tm}_{r}",
                    gamma,
                    c,
                    bs[i],
                    rho0,
                    A,
                    tm,
                    p_c.copy(),
                    p_rho.copy(),
                    nsamples,
                    burn_in,
                    skip,
                )
            )
Parallel(n_jobs=n_processes)(delayed(single_inference)(*arg) for arg in arglist)
