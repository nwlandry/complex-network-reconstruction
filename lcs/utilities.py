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
    match mode:
        case "mean":
            return np.mean(np.sum(x[1:] - x[:-1] > 0, axis=0))
        case "median":
            return np.median(np.sum(x[1:] - x[:-1] > 0, axis=0))
        case "max":
            return np.max(np.sum(x[1:] - x[:-1] > 0, axis=0))
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
    return (a + u * (b - a)) ** (1 / (1 - r))


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


def ipn_func(b, ipn_target, cf, gamma, A, rho0, realizations, tmax, mode):
    n = A.shape[0]

    x0 = np.zeros(n)
    x0[list(random.sample(range(n), int(rho0 * n)))] = 1

    c = cf(np.arange(n), b)

    ipn = 0
    for _ in range(realizations):
        x = contagion_process(A, gamma, c, x0, tmin=0, tmax=tmax)
        ipn += infections_per_node(x, mode) / realizations
    return ipn - ipn_target


def robbins_monro_solve(
    f,
    x0,
    a,
    alpha,
    max_iter=100,
    tol=1e-3,
    loss="function",
    verbose=False,
    return_values=True,
):
    x = x0
    val = f(x0)

    it = 1
    xvec = [x]
    fvec = [val]
    diff = np.inf
    while diff > tol and it <= max_iter:
        a_n = a * it**-alpha
        x -= a_n * val
        x = np.clip(x, 0, 1)
        val = f(x)
        xvec.append(x)  # save results
        fvec.append(val)
        if it % 3 == 0:
            match loss:
                case "function":
                    diff = abs(x - xvec[it - 2])
                case "arg":
                    diff = abs(val)
                case _:
                    raise Exception("Invalid loss type!")

        if verbose:
            print(it, diff)
        it += 1
    if return_values:
        return x, xvec, fvec
    else:
        return x
