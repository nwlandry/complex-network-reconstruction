import json
import random

import numpy as np
from scipy.stats import rv_discrete

from .contagion import *
from .generative import erdos_renyi
from .inference import *


def single_inference(
    fname, gamma, c, b, rho0, A, tmax, p_c, p_rho, nsamples, burn_in, skip
):
    n = np.size(A, axis=0)
    x0 = np.zeros(n)
    x0[random.sample(range(n), int(round(rho0 * n)))] = 1

    x = contagion_process(A, gamma, c, x0, tmin=0, tmax=tmax)
    p = beta(p_rho[0], p_rho[1]).rvs()
    A0 = erdos_renyi(n, p)
    samples = infer_adjacency_matrix(
        x, A0, p_rho, p_c, nsamples=nsamples, burn_in=burn_in, skip=skip
    )

    # json dict
    data = {}
    data["gamma"] = gamma
    data["c"] = c.tolist()
    data["b"] = b
    data["p-rho"] = p_rho.tolist()
    data["p-c"] = p_c.tolist()
    data["x"] = x.tolist()
    data["A"] = A.tolist()
    data["samples"] = samples.tolist()

    datastring = json.dumps(data)

    with open(fname, "w") as output_file:
        output_file.write(datastring)


def to_imshow_orientation(A):
    return np.flipud(A.T)


def posterior_similarity(samples, A):
    meanA = samples.mean(axis=0)
    num = np.sum(np.abs(A - meanA))
    den = np.sum(np.abs(A + meanA))
    if den > 0:
        return 1 - num / den
    else:
        return 1


def samplewise_posterior_similarity(samples, A):
    ps = 0
    n = samples.shape[0]
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


def fraction_of_correct_entries(samples, A):
    n = A.shape[0]
    nsamples = samples.shape[0]
    num = (np.sum(samples == A) - nsamples * n) / 2
    den = nsamples * n * (n - 1) / 2
    return num / den


def f_score(samples, A, threshold):
    Q = samples.mean(axis=0) >= threshold
    tp = np.sum(Q * A)
    fn = np.sum((1 - Q) * A)
    fp = np.sum(Q * (1 - A))

    return 2 * tp / (2 * tp + fn + fp)


def infections_per_node(x, mode="mean"):
    match mode:
        case "mean":
            return np.mean((x[1:] - x[:-1] > 0).sum(axis=0))
        case "median":
            return np.median((x[1:] - x[:-1] > 0).sum(axis=0))
        case "max":
            return np.max((x[1:] - x[:-1] > 0).sum(axis=0))
        case _:
            raise Exception("Invalid loss!")


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


def power_law(n, minval, maxval, r):
    u = np.random.random(n)
    a = minval ** (1 - r)
    b = maxval ** (1 - r)
    return np.round((a + u * (b - a)) ** (1 / (1 - r))).astype(int)


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


def delta_dist(x_prime):
    return rv_discrete(name="custom", values=([x_prime], [1.0]))
