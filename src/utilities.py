from itertools import product

import numpy as np
import xgi
from src.contagion import *
from src.inference import *
import random


def full_incidence_matrix(H, n, m):
    I, node_dict, edge_dict = xgi.incidence_matrix(H, index=True, sparse=False)
    newI = np.zeros((n, m))

    for i, j in zip(*np.nonzero(I)):
        newI[node_dict[i], edge_dict[j]] = 1
    return newI


def posterior_similarity(I, Isamples):
    if isinstance(Isamples, list):
        meanI = np.mean(Isamples, axis=0)
        return 1 - np.sum(np.abs(I - meanI)) / np.sum(np.abs(I + meanI))
    elif isinstance(Isamples, np.ndarray):
        return 1 - np.sum(np.abs(I - Isamples)) / np.sum(np.abs(I + Isamples))


def get_all_incidence_matrices(n, m):
    if n * m > 16:
        raise Exception("Too large")
    matrices = list()
    for data in product([0, 1], repeat=n * m):
        matrices.append(np.reshape(data, (n, m)))
    return matrices


def infer_over_realizations(
    I,
    g_x,
    g_y,
    b_x,
    b_y,
    f,
    g,
    rho,
    num_realizations=10,
    tmin=0,
    tmax=20,
    dt=1,
    nsamples=100,
    burn_in=10000,
    skip=100,
):
    # returns samples and posterior similarity
    n = np.size(g_x, axis=0)
    m = np.size(g_y, axis=0)
    s_x = np.zeros(n)
    s_y = np.zeros(m)
    s_x[random.randrange(n)] = 1
    ps = list()
    samples = list()
    for sim in range(num_realizations):
        x, y = bipartite_sis_to_matrices(
            I,
            g_x,
            g_y,
            b_x,
            b_y,
            f,
            g,
            s_x,
            s_y,
            tmin=tmin,
            tmax=tmax,
            dt=dt,
            random_seed=None,
        )
        I0 = np.ones((n, m))
        data, _ = infer_incidence_matrix(
            x, y, I0, b_x, b_y, f, g, rho, nsamples=nsamples, burn_in=burn_in, skip=skip
        )
        ps.append(posterior_similarity(I, data))
        samples.extend(data)
    return samples, ps


def vary_epsilon(
    I,
    g_x,
    g_y,
    b_x,
    b_y,
    rho,
    epsilon,
    tau=0.25,
    num_realizations=10,
    tmin=0,
    tmax=20,
    dt=1,
    nsamples=100,
    burn_in=10000,
    skip=100,
):
    all_samples = []
    all_ps = []
    for eps in epsilon:
        f = lambda e, x: (e.dot(x)/max(np.sum(e), 1) >= tau)*(1 - eps) + eps
        g = lambda e, x: (e.dot(x)/max(np.sum(e), 1) >= tau)

        samples, ps = infer_over_realizations(
            I,
            g_x,
            g_y,
            b_x,
            b_y,
            f,
            g,
            rho,
            num_realizations=num_realizations,
            tmin=tmin,
            tmax=tmax,
            dt=dt,
            nsamples=nsamples,
            burn_in=burn_in,
            skip=skip,
        )
        all_samples.append(samples)
        all_ps.append(ps)

    return all_samples, all_ps


def vary_tmax(
    I,
    g_x,
    g_y,
    b_x,
    b_y,
    f,
    g,
    rho,
    num_realizations=10,
    tmin=0,
    tmax=None,
    dt=1,
    nsamples=100,
    burn_in=10000,
    skip=100,
):
    if tmax is None:
        tmax = [30, 100, 300, 1000]
    all_samples = []
    all_ps = []
    for tm in tmax:
        print()
        samples, ps = infer_over_realizations(
            I,
            g_x,
            g_y,
            b_x,
            b_y,
            f,
            g,
            rho,
            num_realizations=num_realizations,
            tmin=tmin,
            tmax=tm,
            dt=dt,
            nsamples=nsamples,
            burn_in=burn_in,
            skip=skip,
        )
        all_samples.append(samples)
        all_ps.append(ps)

    return all_samples, all_ps


def vary_beta(
    I,
    g_x,
    g_y,
    beta,
    f,
    g,
    rho,
    num_realizations=10,
    tmin=0,
    tmax=20,
    dt=1,
    nsamples=100,
    burn_in=10000,
    skip=100,
):
    n, m = np.shape(I)
    all_samples = []
    all_ps = []
    for b in beta:
        b_x = b*np.ones(n)
        b_y = b*np.ones(m)

        samples, ps = infer_over_realizations(
            I,
            g_x,
            g_y,
            b_x,
            b_y,
            f,
            g,
            rho,
            num_realizations=num_realizations,
            tmin=tmin,
            tmax=tmax,
            dt=dt,
            nsamples=nsamples,
            burn_in=burn_in,
            skip=skip,
        )
        all_samples.append(samples)
        all_ps.append(ps)

    return all_samples, all_ps