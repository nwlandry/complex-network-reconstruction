import random

import numpy as np

from .contagion import *
from .inference import *


def to_imshow_orientation(A):
    return np.flipud(A.T)


def posterior_similarity(A, samples):
    meanA = np.mean(samples, axis=0)
    num = np.sum(np.abs(A - meanA))
    den = np.sum(np.abs(A + meanA))
    if den > 0:
        return 1 - num / den
    else:
        return 1


def samplewise_posterior_similarity(A, samples):
    ps = 0
    n = np.size(samples, axis=0)
    for i in range(n):
        num = np.sum(np.abs(A - samples[i]))
        den = np.sum(np.abs(A + samples[i]))
        if den > 0:
            ps += 1 - num / den
        else:
            ps += 1
    return ps / n


def hamming_distance(A1, A2):
    return np.sum(np.abs(A1 - A2)) / 2


def infections_per_node(x, mode="mean"):
    if mode == "mean":
        return np.mean(np.sum(x[1:] - x[:-1] > 0, axis=0))
    if mode == "median":
        return np.median(np.sum(x[1:] - x[:-1] > 0, axis=0))
    if mode == "max":
        return np.max(np.sum(x[1:] - x[:-1] > 0, axis=0))


def nu_distribution(x, A):
    k = A.sum(axis=0)
    nu = A @ x.T
    T, n = x.shape
    kmax = int(round(np.max(k)))
    mat = np.zeros((kmax + 1, kmax + 1))
    for t in range(T):
        for i in range(n):
            mat[int(k[i]), int(nu[i, t])] += 1
    return mat


def degrees(A):
    if not isinstance(A, np.ndarray):
        A = A.todense()
    return A.sum(axis=0)


def powerlaw(n, minval, maxval, r):
    u = np.random.random(n)
    return (minval ** (1 - r) + u * (maxval ** (1 - r) - minval ** (1 - r))) ** (
        1 / (1 - r)
    )


def mean_power_law(minval, maxval, r):
    if r == 1:
        num = maxval - minval
        den = np.log(maxval) - np.log(minval)
        return num / den
    elif r == 2:
        num = np.log(maxval) - np.log(minval)
        den = 1 / minval - 1 / maxval
        return num / den
    else:
        num = (minval ** (2 - r) - maxval ** (2 - r)) / (r - 2)
        den = (minval ** (1 - r) - maxval ** (1 - r)) / (r - 1)
        return num / den


def match_contagion_rates(
    cf1, cf2, gamma, b, A, tmax, realizations=100, tol=0.01, max_iter=10, mode="mean"
):
    n = A.shape[0]
    rho0 = 1

    x0 = np.zeros(n)
    x0[list(random.sample(range(n), int(rho0 * n)))] = 1

    c1 = cf1(np.arange(n), b)

    ipn_c1 = 0
    for _ in range(realizations):
        x = contagion_process(A, gamma, c1, x0, tmin=0, tmax=tmax)
        ipn_c1 += infections_per_node(x, mode) / realizations

    blo = 0
    bhi = 1
    ipn_lo = 0
    ipn_hi = 0
    c2_hi = cf2(np.arange(n), bhi)
    for _ in range(realizations):
        x = contagion_process(A, gamma, c2_hi, x0, tmin=0, tmax=tmax)
        ipn_hi += infections_per_node(x, mode) / realizations

    it = 0
    bnew = (bhi - blo) / 2
    while it < max_iter and bhi - blo > tol:
        c2_new = cf2(np.arange(n), bnew)
        ipn_new = 0
        for _ in range(realizations):
            x = contagion_process(A, gamma, c2_new, x0, tmin=0, tmax=tmax)
            ipn_new += infections_per_node(x, mode) / realizations

        if ipn_new > ipn_c1:
            bhi = bnew
        elif ipn_new < ipn_c1:
            blo = bnew
        bnew = (bhi - blo) / 2
        it += 1
        print(blo, bhi, ipn_new, ipn_c1)

    return bnew, bhi - blo
