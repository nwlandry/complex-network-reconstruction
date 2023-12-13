import os
import sys
from pathlib import Path

path = Path.cwd()
parent = os.path.abspath(os.path.join(path, os.pardir))
print(parent)
src_dir = os.path.join(parent, "lcs")
print(src_dir)
sys.path.append(parent)
import warnings

warnings.filterwarnings("default")

import time

import numpy as np
from numpy.linalg import eigh
from scipy.stats import beta

from lcs import *

"""
Generate Paramters for Test
"""
A = zkc()
A = np.array(A, dtype=float)
n = np.size(A, axis=0)
x0 = np.zeros(n)
x0[random.randrange(n)] = 1

gamma = 1
nu = eigh(A)[0][-1]
b = 2 * gamma / nu

# simple contagion
nsamples = 20000

sc = lambda nu, b: 1 - (1 - b) ** nu
c = sc(np.arange(n), b)

x = contagion_process(A, gamma, c, x0, tmin=0, tmax=100)

p_c = np.ones((2, n))
p_rho = np.array([2, 5])
rho0 = beta(p_rho[0], p_rho[1]).rvs()


i = 1
start_time = time.time()
a = count_local_infection_events(i, x, A)
print("Time taken for count_local_infection_events:", time.time() - start_time)

start_time = time.time()
b = count_local_infection_events_loop(i, x, A)
print("Time taken for count_local_infection_events2:", time.time() - start_time)

start_time = time.time()
a = count_all_infection_events(x, A)
print("Time taken for count_all_infection_events:", time.time() - start_time)

start_time = time.time()
b = count_all_infection_events_loop(x, A)
print("Time taken for count_all_infection_events2:", time.time() - start_time)
