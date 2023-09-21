import random

import numpy as np
from numpy import ndarray
from scipy.stats import beta
from scipy.special import betaln


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
    l_adjacency = adjacency_log_likelihood(num_entries, n, m, rho)

    samples = np.zeros((nsamples, n, n))
    accept = 0
    it = 1
    s_i = skip
    sample_num = 0

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
        new_l_adjacency = adjacency_log_likelihood(
            num_entries + delta_entries, n, m, rho
        )
        delta = compute_delta(new_l, l) + compute_delta(
            new_l_adjacency, l_adjacency
        )

        if np.log(random.random()) <= min(delta, 0):
            nl = new_nl
            ml = new_ml
            l = new_l
            l_adjacency = new_l_adjacency
            num_entries += delta_entries
            accept += 1
        else:
            update_adjacency_matrix(i, j, A)

        if it >= burn_in:
            if s_i >= skip:
                samples[sample_num, :, :] = A.copy()
                sample_num += 1
                s_i = 1
            else:
                s_i += 1

        it += 1
    print(f"Acceptance ratio is {accept/(burn_in + (nsamples - 1)*skip)}")

    return samples
    

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
            nl[i, nu] += (x[t + 1, i] - x[t, i] == 1)
            ml[i, nu] += (x[t + 1, i] == x[t, i] == 0)
    return nl, ml


def count_local_infection_events(i, x, A):
    T = np.size(x, axis=0)
    n = np.size(x, axis=1)

    nl = np.zeros(n, dtype=int)
    ml = np.zeros(n, dtype=int)

    for t in range(T - 1):
        nu = A[i] @ x[t]
        
        nu = int(round(nu))
        nl[nu] += (x[t + 1, i] - x[t, i] == 1)
        ml[nu] += (x[t + 1, i] == x[t, i] == 0)
    return nl, ml


def compute_log_likelihood(nl, ml, p_c):
    a = np.sum(nl, axis=0)
    b = np.sum(ml, axis=0)

    return np.sum(betaln(a + p_c[0], b + p_c[1]))





def infer_adjacency_matrix_old(
    x, A0, c, rho, nsamples=1, burn_in=100, skip=100, return_likelihoods=False
):
    # This assumes a uniform prior on hypergraphs with a given rho.

    # form initial adjacency matrix
    if not isinstance(A0, ndarray):
        A0 = A0.todense()

    A = A0.copy()
    n, m = np.shape(A)

    if n != m:
        Exception("Matrix must be square!")

    samples = np.zeros((nsamples, n, n))

    l_node = np.zeros(n)

    for v in range(n):
        l_node[v] = neighborhood_log_likelihood(x, v, A, c)

    l = np.sum(l_node)
    l_vals = list()

    num_entries = int(np.sum(A))
    l_adjacency = adjacency_log_likelihood(num_entries, n, m, rho)

    accept = 0
    it = 1
    s_i = skip

    samples = np.zeros((nsamples, n, n))
    sample_num = 0

    while it <= burn_in + (nsamples - 1) * skip:

        # proposal comes from the lower triangle
        i = random.randrange(1, n)
        j = random.randrange(i)

        # alter hypergraph
        delta_entries = update_adjacency_matrix(i, j, A)

        # update dynamics likelihoods given the new I
        new_n_l = neighborhood_log_likelihood(x, i, A, c) + neighborhood_log_likelihood(x, j, A, c)

        if np.isnan(new_n_l):
            print(f"Node {i} gives a NaN")

        # update likelihood of the incidence matrix given rho
        new_l_adjacency = adjacency_log_likelihood(
            num_entries + delta_entries, n, m, rho
        )
        delta = compute_delta(new_n_l, l_node[i]) + compute_delta(
            new_l_adjacency, l_adjacency
        )

        if np.log(random.random()) <= min(delta, 0):
            l_node[i] = new_n_l

            if delta < np.inf:
                l += delta
            else:
                l = np.sum(l_node)

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

    if return_likelihoods:
        return samples, l_vals
    else:
        return samples


def compute_delta(a, b):
    if a == -np.inf and b == -np.inf:
        return 0
    else:
        return a - b


def neighborhood_log_likelihood(x, v, A, c):
    l = 0
    T = np.size(x, axis=0)
    for t in range(T - 1):
        nu = int(round(A[v] @ x[t]))

        if c[nu] > 0 and c[nu] < 1:
            lt = (1 - x[t, v]) * x[t + 1, v] * np.log(c[nu]) + (1 - x[t, v]) * (
                1 - x[t + 1, v]
            ) * np.log(1 - c[nu])
        elif c[nu] == 0 and (1 - x[t, v]) * x[t + 1, v]:
            lt = -np.inf
        elif c[nu] == 1 and (1 - x[t, v]) * (1 - x[t + 1, v]):
            lt = -np.inf
        else:
            lt = 0
        l += lt
    return l


def adjacency_log_likelihood(num_entries, n, rho):
    return num_entries * np.log(rho) + (n * (n-1)/2 - num_entries) * np.log(1 - rho)


def log_likelihood(x, A, c):
    if not isinstance(A, ndarray):
        A = A.todense()
    n = np.size(A, axis=0)

    l_node = np.zeros(n)
    for v in range(n):
        l_node[v] = neighborhood_log_likelihood(x, v, A, c)

    return np.sum(l_node)


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

    rho = beta(e + p_rho[0], n*(n - 1)/2 - e + p_rho[1]).rvs()

    n = 0 # healing events
    m = 0 # non-healing events

    # sample gamma
    for t in range(T - 1):
        # healing events
        n += len(np.where(x[t + 1] - x[t] == -1)[0])
        # non-healing events
        m += len(np.where(x[t + 1] * x[t] == 1)[0])

    if n * m == 0:
        n += 1e-6
        m += 1e-6
    gamma = beta(n + p_gamma[0], p_gamma[1]).rvs()

    # sample c
    # These are the parameters for the beta distribution,
    # each entry corresponds to a number of infected neighbors
    n = np.zeros(n)
    m = np.zeros(n)

    for t in range(T - 1):
        nus = A @ x[t]

        # infection events
        for i, nu in enumerate(nus):
            nu = int(round(nu))
            n[nu] += (x[t + 1, i] - x[t, i] == 1)
            m[nu] += (x[t + 1, i] == x[t, i] == 0)

    c = np.zeros(n)
    for i in range(n):
        if n[i] * m[i] == 0:
            n[i] += 1e-6
            m[i] += 1e-6
        c[i] = beta(n[i] + p_c[0, i], m[i] + p_c[1, i]).rvs()
    
    return rho, gamma, c

