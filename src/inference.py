import random

import numpy as np
import xgi
from scipy import sparse
from scipy.stats import beta


def infer_incidence_matrix_and_dynamics(
    x, beta, f, rho, num_iter=10, max_iter=10
):
    n = np.size(x, axis=1)
    it = 0

    rho = random.random()
    A = adjacency_matrix_prior(n, rho)

    while it < max_iter:
        A, _ = infer_adjacency_matrix(x, A, beta, f, rho, burn_in=num_iter)
        # gamma, beta, rho = infer_dynamics(x, A, f)
        it += 1
    return


def infer_adjacency_matrix(
    x, A0, beta, f, rho, nsamples=1, burn_in=100, skip=100, return_likelihoods=False
):
    # This assumes a uniform prior on hypergraphs with a given rho.

    # form initial contact structure. Number of patients (n) and rooms (m) is fixed.
    A = A0.copy()
    n, m = np.shape(A)

    if n != m:
        Exception("Matrix must be square!")

    samples = []

    l_node = np.zeros(n)

    for v in range(n):
        l_node[v] = node_log_likelihood(x, v, A, beta, f)

    l = np.sum(l_node)
    l_vals = list()

    num_entries = int(np.sum(A))
    l_adjacency = adjacency_log_likelihood(num_entries, n, m, rho)

    accept = 0
    it = 0
    s_i = 0
    while it < burn_in + (nsamples - 1) * skip:

        # proposal comes from the lower triangle
        i = random.randrange(n)
        j = random.randrange(i)

        # alter hypergraph
        delta_entries = update_adjacency_matrix(i, j, A)

        # update dynamics likelihoods given the new I
        new_n_l = node_log_likelihood(x, i, A, beta, f)

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

        if it + 1 >= burn_in:
            if s_i >= skip - 1:
                samples.append(A.copy())
                s_i = 0
            else:
                s_i += 1

        it += 1
    print(f"Acceptance ratio is {accept/(burn_in + (nsamples - 1)*skip)}")

    return samples, l_vals if return_likelihoods else samples


def compute_delta(a, b):
    if a == -np.inf and b == -np.inf:
        return 0
    else:
        return a - b


def node_log_likelihood(x, v, A, beta, f):
    l = 0
    T = np.size(x, axis=0)
    for t in range(T - 1):
        infected_nbrs = A[v] @ x
        p_x = beta * f(infected_nbrs)
        if p_x > 0 and p_x < 1:
            lt = (1 - x[t, v]) * x[t + 1, v] * np.log(p_x) + (1 - x[t, v]) * (
                1 - x[t + 1, v]
            ) * np.log(1 - p_x)
            if np.isnan(lt):
                print(f"Probability is {p_x}")
                lt = -np.inf
        elif p_x == 0 and (1 - x[t, v]) * x[t + 1, v]:
            lt = -np.inf
        elif p_x == 1 and (1 - x[t, v]) * (1 - x[t + 1, v]):
            lt = -np.inf
        else:
            lt = 0
        l += lt
    return l


def adjacency_log_likelihood(num_entries, n, m, rho):
    return num_entries * np.log(rho) + (n * m - num_entries) * np.log(1 - rho)


def log_likelihood(x, A, beta, f):
    n = np.size(A, axis=0)

    l_node = np.zeros(n)
    for v in range(n):
        l_node[v] = node_log_likelihood(x, v, A, beta, f)

    return np.sum(l_node)


def update_adjacency_matrix(i, j, A):
    if A[i, j] == 0:
        A[i, j] = A[j, i] = 1
        return 2
    else:
        A[i, j] = A[j, i] = 0
        return -2


def adjacency_matrix_prior(n, m, rho):
    # n is the number of people, m is number of rooms
    return np.random.choice([0, 1], size=(n, m), p=[1 - rho, rho])


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
