import numpy as np
import matplotlib.pyplot as plt
from src import *
import networkx as nx
from scipy.stats import beta
from numpy.linalg import eigh

#generate information
n = 50
k = 0.1
rho0 = 0.1
gamma = 1
A = erdos_renyi(n, k)
x0 = np.zeros(n)
x0[random.sample(range(n), int(rho0 * n))] = 1

nu = eigh(A)[0][-1]
b = 2 * gamma / nu

sc = lambda nu, b: 1 - (1 - b) ** nu
c = sc(np.arange(n), b)

x = contagion_process(A, gamma, c, x0, tmin=0, tmax=100)

def count_all_infection_events_loop(x, A):
    T = np.size(x, axis=0)
    n = np.size(x, axis=1)

    nl = np.zeros((n, n), dtype=int)
    ml = np.zeros((n, n), dtype=int)

    for t in range(T - 1):
        nus = A @ x[t]

        # infection events
        for i, nu in enumerate(nus):
            nu = int(round(nu))
            nl[i, nu] += x[t + 1, i] * (1 - x[t, i])
            ml[i, nu] += (1 - x[t + 1, i]) * (1 - x[t, i])
    return nl, ml



def test_count_all_infection_events():
    #count_all_infection_events(x, A)
    assert np.array_equal(count_all_infection_events(x, A),count_all_infection_events_loop(x, A))

def ensure_infer_adjacency_matrix_runs():
    samples1, l = infer_adjacency_matrix(
         x, A0, p_rho, p_c, nsamples=nsamples, burn_in=0, skip=10, return_likelihood=True
     )
    assert samples1 != 0


