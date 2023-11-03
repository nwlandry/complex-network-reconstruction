import random

import numpy as np
from scipy.stats import gaussian_kde

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
        return -(minval - maxval) / (np.log(maxval) - np.log(minval))
    elif r == 2:
        return (np.log(maxval) - np.log(minval)) / (1 / minval - 1 / maxval)
    else:
        return (
            (minval ** (2 - r) - maxval ** (2 - r))
            * (r - 1)
            / ((minval ** (1 - r) - maxval ** (1 - r)) * (r - 2))
        )
