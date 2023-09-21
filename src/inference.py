import random

import numpy as np
from numpy import ndarray
from scipy.stats import beta
from scipy.special import betaln

import warnings

warnings.filterwarnings("error")


def infer_adjacency_matrix(x, A0, rho, p_c, nsamples=1, burn_in=100, skip=100):
    # This assumes a uniform prior on hypergraphs with a given rho.

    # form initial adjacency matrix
    if not isinstance(A0, ndarray):
        A0 = A0.todense()

    A = A0.copy()
    n, m = np.shape(A)

    if n != m:
        Exception("Matrix must be square!")

    nl, ml = count_all_infection_events(x, A)
    l = compute_log_likelihood(nl, ml, p_c)

    num_entries = int(np.sum(A))
    l_adjacency = adjacency_log_likelihood(num_entries, n, rho)

    samples = np.zeros((nsamples, n, n))
    accept = 0
    it = 1
    s_i = skip
    sample_num = 0

    l_vals = []

    while it <= burn_in + (nsamples - 1) * skip:

        # proposal comes from the lower triangle
        i = random.randrange(1, n)
        j = random.randrange(i)

        # alter hypergraph
        delta_entries = update_adjacency_matrix(i, j, A)

        nl_i, ml_i = count_local_infection_events(i, x, A)
        nl_j, ml_j = count_local_infection_events(j, x, A)

        new_nl = nl.copy()
        new_ml = ml.copy()

        new_nl[i] = nl_i
        new_nl[j] = nl_j
        new_ml[i] = ml_i
        new_ml[j] = ml_j

        new_l = compute_log_likelihood(new_nl, new_ml, p_c)

        # update likelihood of the incidence matrix given rho
        new_l_adjacency = adjacency_log_likelihood(num_entries + delta_entries, n, rho)

        delta = compute_delta(new_l, l) + compute_delta(new_l_adjacency, l_adjacency)

        if np.log(random.random()) <= min(delta, 0):
            nl = new_nl
            ml = new_ml
            l = new_l
            l_adjacency = new_l_adjacency
            num_entries += delta_entries
            accept += 1
        else:
            update_adjacency_matrix(i, j, A)

        l_vals.append(l)
        if it >= burn_in:
            if s_i >= skip:
                samples[sample_num, :, :] = A.copy()
                sample_num += 1
                s_i = 1
            else:
                s_i += 1

        it += 1
    print(f"Acceptance ratio is {accept/(burn_in + (nsamples - 1)*skip)}")

    return samples, l_vals


def count_all_infection_events(x, A):
    T = np.size(x, axis=0)
    n = np.size(x, axis=1)

    nl = np.zeros((n, n), dtype=int)
    ml = np.zeros((n, n), dtype=int)

    for t in range(T - 1):
        nus = A @ x[t]

        # infection events
        for i, nu in enumerate(nus):
            nu = int(round(nu))
            nl[i, nu] += x[t + 1, i] - x[t, i] == 1
            ml[i, nu] += x[t + 1, i] == x[t, i] == 0
    return nl, ml


def count_local_infection_events(i, x, A):
    T = np.size(x, axis=0)
    n = np.size(x, axis=1)

    nl = np.zeros(n, dtype=int)
    ml = np.zeros(n, dtype=int)

    for t in range(T - 1):
        nu = A[i] @ x[t]

        nu = int(round(nu))
        nl[nu] += x[t + 1, i] - x[t, i] == 1
        ml[nu] += x[t + 1, i] == x[t, i] == 0
    return nl, ml


def compute_log_likelihood(nl, ml, p_c):
    a = np.sum(nl, axis=0)
    b = np.sum(ml, axis=0)
    return sum(b for b in betaln(a + p_c[0], b + p_c[1]) if np.isfinite(b))


def compute_delta(a, b):
    if (a == -np.inf and b == -np.inf) or (a == np.inf and b == np.inf):
        return 0
    else:
        return a - b


def adjacency_log_likelihood(num_entries, n, rho):
    return num_entries * np.log(rho) + (n * (n - 1) / 2 - num_entries) * np.log(1 - rho)


def update_adjacency_matrix(i, j, A):
    if A[i, j] == 0:
        A[i, j] = A[j, i] = 1
        return 2
    else:
        A[i, j] = A[j, i] = 0
        return -2


def infer_dynamics(x, A, p_rho, p_gamma, p_c):
    # Our prior on rho is drawn from a beta distribution such that
    # the posterior is also from a beta distribution

    T = np.size(x, axis=0)
    n = np.size(x, axis=1)

    # sample rho
    e = np.sum(A)

    rho = beta(e + p_rho[0], n * (n - 1) / 2 - e + p_rho[1]).rvs()

    a = 0  # healing events
    b = 0  # non-healing events

    # sample gamma
    for t in range(T - 1):
        # healing events
        a += len(np.where(x[t + 1] - x[t] == -1)[0])
        # non-healing events
        b += len(np.where(x[t + 1] * x[t] == 1)[0])

    if a * b == 0:
        a += 1e-6
        b += 1e-6
    gamma = beta(a + p_gamma[0], b + p_gamma[1]).rvs()

    # sample c
    # These are the parameters for the beta distribution,
    # each entry corresponds to a number of infected neighbors
    a = np.zeros(n)
    b = np.zeros(n)

    for t in range(T - 1):
        nus = A @ x[t]

        # infection events
        for i, nu in enumerate(nus):
            nu = int(round(nu))
            a[nu] += x[t + 1, i] - x[t, i] == 1
            b[nu] += x[t + 1, i] == x[t, i] == 0

    c = np.zeros(n)
    for i in range(n):
        if a[i] * b[i] == 0:
            a[i] += 1e-6
            b[i] += 1e-6
        c[i] = beta(a[i] + p_c[0, i], b[i] + p_c[1, i]).rvs()

    return rho, gamma, c
