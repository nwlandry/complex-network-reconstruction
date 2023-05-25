import random

import numpy as np
import xgi
from scipy import sparse
from scipy.stats import beta


def infer_incidence_matrix_and_dynamics(
    x, y, H, b_x, b_y, f, g, fmax, gmax, rho, num_itations=10
):
    n = np.size(x, axis=1)
    m = np.size(y, axis=1)
    it = 0

    b_x = np.random.uniform(0, 1 / fmax, n)
    b_y = np.random.uniform(0, 1 / gmax, m)
    rho = random.random()
    H = incidence_matrix_prior(n, m, rho)

    while it < max_it:
        H, _ = infer_incidence_matrix(
            x, y, H, b_x, b_y, f, g, rho, num_itations=num_itations
        )
        g_x, g_y, b_x, b_y, rho = infer_dynamics(x, y, H, f, g)
        it += 1
    return


def infer_incidence_matrix(
    x, y, I0, b_x, b_y, f, g, rho, nsamples=1, burn_in=100, skip=100
):
    # This assumes a uniform prior on hypergraphs with a given rho.

    # form initial contact structure. Number of patients (n) and rooms (m) is fixed.
    I = I0.copy()
    n, m = np.shape(I)

    samples = []

    l_node = np.zeros(n)
    l_edge = np.zeros(m)

    for v in range(n):
        l_node[v] = node_log_likelihood(x, y, v, I, b_x, f)
    for e in range(m):
        l_edge[e] = edge_log_likelihood(x, y, e, I, b_y, g)

    l = np.sum(l_node) + np.sum(l_edge)
    l_vals = list()

    num_entries = int(np.sum(I))
    l_incidence = incidence_log_likelihood(num_entries, n, m, rho)

    accept = 0
    it = 0
    s_i = 0
    while it < burn_in + (nsamples - 1) * skip:

        # proposal
        i = random.randrange(n)
        j = random.randrange(m)

        # alter hypergraph
        delta_entries = update_incidence_matrix(i, j, I)

        # update dynamics likelihoods given the new I
        new_n_l = node_log_likelihood(x, y, i, I, b_x, f)
        new_e_l = edge_log_likelihood(x, y, j, I, b_y, g)

        if np.isnan(new_n_l):
            print(f"Node {i} gives a NaN")
        if np.isnan(new_e_l):
            print(f"Edge {j} gives a NaN")

        # update likelihood of the incidence matrix given rho
        new_l_incidence = incidence_log_likelihood(
            num_entries + delta_entries, n, m, rho
        )
        delta = (
            compute_delta(new_n_l, l_node[i])
            + compute_delta(new_e_l, l_edge[j])
            + compute_delta(new_l_incidence, l_incidence)
        )

        if np.log(random.random()) <= min(delta, 0):
            l_node[i] = new_n_l
            l_edge[j] = new_e_l

            if delta < np.inf:
                l += delta
            else:
                l = np.sum(l_node) + np.sum(l_edge)

            l_incidence = new_l_incidence

            num_entries += delta_entries
            accept += 1
        else:
            update_incidence_matrix(i, j, I)

        l_vals.append(l)

        if it + 1 >= burn_in:
            if s_i >= skip - 1:
                samples.append(I.copy())
                s_i = 0
            else:
                s_i += 1

        it += 1
    print(f"Acceptance ratio is {accept/(burn_in + (nsamples - 1)*skip)}")

    return samples, l_vals


def compute_delta(a, b):
    if a == -np.inf and b == -np.inf:
        return 0
    else:
        return a - b


def node_log_likelihood(x, y, v, I, b_x, f):
    l = 0
    T = np.size(x, axis=0)
    for t in range(T - 1):
        p_x = b_x[v] * f(I[v], y[t])
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


def edge_log_likelihood(x, y, e, I, b_y, g):
    l = 0
    T = np.size(x, axis=0)
    for t in range(T - 1):
        p_y = b_y[e] * g(I[:, e], x[t])
        if p_y != 0 and p_y != 1:
            lt = (1 - y[t, e]) * y[t + 1, e] * np.log(p_y) + (1 - y[t, e]) * (
                1 - y[t + 1, e]
            ) * np.log(1 - p_y)
            if np.isnan(lt):
                lt = -np.inf
                print(f"Probability is {p_y}")
        elif p_y == 0 and (1 - y[t, e]) * y[t + 1, e]:
            lt = -np.inf
        elif p_y == 1 and (1 - y[t, e]) * (1 - y[t + 1, e]):
            lt = -np.inf
        else:
            lt = 0
        l += lt
    return l


def incidence_log_likelihood(num_entries, n, m, rho):
    return num_entries * np.log(rho) + (n * m - num_entries) * np.log(1 - rho)


def log_likelihood(x, y, I, b_x, b_y, f, g):
    n, m = np.shape(I)

    l_node = np.zeros(n)
    l_edge = np.zeros(m)
    for v in range(n):
        l_node[v] = node_log_likelihood(x, y, v, I, b_x, f)
    for e in range(m):
        l_edge[e] = edge_log_likelihood(x, y, e, I, b_y, g)

    return np.sum(l_node) + np.sum(l_edge)


def update_incidence_matrix(i, j, I):
    if I[i, j] == 0:
        I[i, j] = 1
        return 1
    else:
        I[i, j] = 0
        return -1


def incidence_matrix_prior(n, m, rho):
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
