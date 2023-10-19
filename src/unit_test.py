import numpy as np
import matplotlib.pyplot as plt
from src import *
import networkx as nx
from scipy.stats import beta
from numpy.linalg import eigh

"""
Generate Paramters for Test
"""
G = nx.karate_club_graph()
A = nx.adjacency_matrix(G, weight=None).todense()
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


"""
Test Functions
"""
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

def count_local_infection_events_loop(i, x, A):
    T = np.size(x, axis=0)
    n = np.size(x, axis=1)

    nl = np.zeros(n, dtype=int)
    ml = np.zeros(n, dtype=int)

    for t in range(T - 1):
        nu = A[i] @ x[t]

        nu = int(round(nu))
        nl[nu] += x[t + 1, i] * (1 - x[t, i])
        ml[nu] += (1 - x[t + 1, i]) * (1 - x[t, i])
    return nl, ml



import time
i = 1
start_time = time.time()
a = count_local_infection_events(i,x,A)
print("Time taken for count_local_infection_events:", time.time() - start_time)

start_time = time.time()
b = count_local_infection_events_loop(i,x,A)
print("Time taken for count_local_infection_events2:", time.time() - start_time)


def dynamics_log_likelihood(nl, ml, p_c):
    a = np.sum(nl, axis=0)
    b = np.sum(ml, axis=0)
    return sum(b for b in betaln(a + p_c[0], b + p_c[1]))


def test_count_all_infection_events():
    #count_all_infection_events(x, A)
    assert np.array_equal(count_all_infection_events(x, A),count_all_infection_events_loop(x, A))

def test_count_local_infection_events():
    assert np.array_equal(count_local_infection_events(1,x, A),count_local_infection_events_loop(1,x, A))

def ensure_infer_adjacency_matrix_runs():
    samples1, l = infer_adjacency_matrix(
         x, A, rho0, p_c, nsamples=nsamples, burn_in=0, skip=10, return_likelihood=True
     )
    assert samples1 != 0


