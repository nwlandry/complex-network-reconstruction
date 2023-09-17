import random

import numpy as np
import xgi

from src.contagion import *
from src.inference import *


def full_incidence_matrix(H, n, m):
    I, node_dict, edge_dict = xgi.incidence_matrix(H, index=True, sparse=False)
    newI = np.zeros((n, m))

    for i, j in zip(*np.nonzero(I)):
        newI[node_dict[i], edge_dict[j]] = 1
    return newI


def posterior_similarity(A, Asamples):
    meanA = np.mean(Asamples, axis=0)
    return 1 - np.sum(np.abs(A - meanA)) / np.sum(np.abs(A + meanA))


def samplewise_posterior_similarity(A, Asamples):
    ps = 0
    n = np.size(Asamples)
    for i in range(n):
        ps += 1 - np.sum(np.abs(A - Asamples[i])) / np.sum(np.abs(A + Asamples[i]))
    return ps


def infer_over_realizations(
    A,
    gamma,
    beta,
    f,
    rho,
    num_realizations=10,
    tmin=0,
    tmax=20,
    dt=1,
    nsamples=100,
    burn_in=1000,
    skip=100,
):
    # returns samples and posterior similarity
    n = np.size(A, axis=0)
    s = np.zeros(n)
    s[random.randrange(n)] = 1
    ps = list()
    samples = list()
    for sim in range(num_realizations):
        x = contagion_process(
            A,
            gamma,
            beta,
            f,
            s,
            tmin=tmin,
            tmax=tmax,
            dt=dt,
            random_seed=None,
        )
        A0 = np.ones((n, n))
        data = infer_adjacency_matrix(
            x, A0, beta, f, rho, nsamples=nsamples, burn_in=burn_in, skip=skip
        )
        ps.append(posterior_similarity(A, data))
        samples.extend(data)
    return samples, ps


def vary_nu(
    A,
    gamma,
    beta,
    a,
    c,
    f,
    rho,
    nu_list,
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
    for nu in nu_list:
        g = lambda I, b: 1 - (1 - b) ** I
        h = lambda I, a, c: 1 / (1 + np.exp(-(I - c) / a)) if a != 0 else I >= c

        f = lambda I: nu * g(I, beta) + (1 - nu) * h(I, a, c)

        samples, ps = infer_over_realizations(
            A,
            gamma,
            beta,
            f,
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
    A,
    gamma,
    beta,
    f,
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
        samples, ps = infer_over_realizations(
            A,
            gamma,
            beta,
            f,
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
    A,
    gamma,
    beta,
    f,
    rho,
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
    for b in beta:

        samples, ps = infer_over_realizations(
            A,
            gamma,
            b,
            f,
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
