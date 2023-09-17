import random

import numpy as np
from numpy import ndarray
from scipy.stats import beta


def infer_adjacency_matrix_and_dynamics(x, c, rho, num_iter=10, max_iter=10):
    n = np.size(x, axis=1)
    it = 0

    rho = random.random()
    A = adjacency_matrix_prior(n, rho)

    while it < max_iter:
        A, _ = infer_adjacency_matrix(x, A, c, rho, burn_in=num_iter)
        # gamma, c, rho = infer_dynamics(x, A, f)
        it += 1
    return


def infer_adjacency_matrix(
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


def adjacency_matrix_prior(n, rho):
    # n is the number of people, m is number of rooms
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(i):
            A[i, j] = A[j, i] = 1
    return A


def infer_dynamics(x, y, H, f, g):
    # Our prior on rho is drawn from a beta distribution such that
    # the posterior is also from a beta distribution
    n = np.size(x, axis=1)
    m = np.size(y, axis=1)
    T = np.size(x, axis=0)
    h_x = np.zeros(n)
    h_y = np.zeros(m)
    nh_x = np.zeros(n)
    nh_y = np.zeros(m)
    g_x = np.zeros(n)
    g_y = np.zeros(m)

    for t in range(T - 1):
        # healing events
        h_x[np.where(x[t + 1] - x[t] == -1)] += 1
        h_y[np.where(y[t + 1] - y[t] == -1)] += 1
        # non-healing events
        nh_x[np.where(x[t + 1] * x[t] == 1)] += 1
        nh_y[np.where(y[t + 1] * y[t] == 1)] += 1

    for i in range(n):
        g_x[i] = beta(h_x[i] + 1, nh_x[i] + 1).rvs()

    for i in range(m):
        g_y[i] = beta(h_y[i] + 1, nh_y[i] + 1).rvs()

    num_entries = H.edges.size.sum()
    rho = beta(num_entries, n * m - num_entries).rvs()

    b_x, b_y = sample_beta(x, y, H, f, g)

    return g_x, g_y, b_x, b_y, rho


def sample_beta(x, y, H, f, g):
    n = np.size(x, axis=1)
    m = np.size(y, axis=1)

    b_x = np.zeros(n)
    b_y = np.zeros(m)

    em = H.edges.size.max()
    nm = H.nodes.degree.max()

    fmax = f(range(nm), np.ones(nm))
    gmax = g(range(em), np.ones(em))

    for i in range(n):
        b_x[i] = sample_one_beta(i, x, y, H.nodes.memberships(), f, fmax)
    for i in range(m):
        b_y[i] = sample_one_beta(i, y, x, H.edges.members(dtype=dict), g, gmax)
    return b_x, b_y


def sample_one_beta(i, x1, x2, member_dict, f, m, max_it=100):
    it = 0
    while it < max_it:
        b = random.uniform(0, 1 / m)
        u = random.random()
        if np.log(u) < fcn(i, b, x1, x2, member_dict, f) - fcn_max(
            i, x1, x2, member_dict, f, m
        ):
            return b
        it += 1
    print(f"Index {i} did not converge!")
    return random.uniform(0, 1 / m)


def fcn(i, beta, x1, x2, member_dict, f):
    T = np.size(x1, axis=0)

    ll = 0
    for t in range(T - 1):
        p = beta * f(member_dict[i], x2[t])
        if p > 0 and p < 1:
            l = np.log(p) * x1[t + 1, i] * (1 - x1[t, i]) + np.log(1 - p) * (
                1 - x1[t + 1, i]
            ) * (1 - x1[t, i])
            if np.isnan(l):
                print(f"Probability is {p}")
            else:
                ll += l
        else:
            ll += -np.inf
    return ll


def fcn_max(i, x1, x2, member_dict, f, fmax):
    return max(
        [fcn(i, b, x1, x2, member_dict, f) for b in np.linspace(0, 1 / fmax, 10)]
    )
