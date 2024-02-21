import json
import random

import numpy as np
from scipy.sparse import csr_array
from scipy.sparse.csgraph import reverse_cuthill_mckee
from scipy.stats import rv_discrete

from .contagion import *
from .generative import erdos_renyi
from .inference import *


def single_inference(
    fname, gamma, c, b, rho0, A, tmax, p_c, p_rho, nsamples, burn_in, skip
):
    n = np.size(A, axis=0)
    x0 = np.zeros(n)
    x0[random.sample(range(n), round(rho0 * n))] = 1

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


def prettify_matrix(A):
    idx = reverse_cuthill_mckee(csr_array(A), symmetric_mode=True)
    Ap = A.copy()
    Ap = Ap[idx]
    Ap = Ap[:, idx]
    return Ap


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
    kmax = round(k.max())
    mat = np.zeros((kmax + 1, kmax + 1))
    for t in range(T):
        for i in range(n):
            mat[int(k[i]), int(nu[i, t])] += 1
    return mat


def degrees(A):
    if not isinstance(A, np.ndarray):
        A = A.todense()
    return A.sum(axis=0)


def power_law(n, minval, maxval, alpha, seed=None):
    if seed is not None:
        np.random.seed(seed)
    u = np.random.random(n)
    a = minval ** (1 + alpha)
    b = maxval ** (1 + alpha)
    return np.round((a + u * (b - a)) ** (1 / (1 + alpha))).astype(int)


def mean_power_law(minval, maxval, alpha):
    if alpha == -1:
        num = maxval - minval
        den = np.log(maxval) - np.log(minval)
        return num / den
    elif alpha == -2:
        num = np.log(maxval) - np.log(minval)
        den = 1 / minval - 1 / maxval
        return num / den
    else:
        num = (minval ** (2 + alpha) - maxval ** (2 + alpha)) / (-alpha - 2)
        den = (minval ** (1 + alpha) - maxval ** (1 + alpha)) / (-alpha - 1)
        return num / den


def delta_dist(x_prime):
    return rv_discrete(name="custom", values=([x_prime], [1.0]))


def robbins_monro_solve(
    f,
    x0,
    a=0.02,
    alpha=1,
    max_iter=100,
    tol=1e-2,
    loss="function",
    verbose=False,
    return_values=False,
):
    x = x0
    val = f(x0)

    xvec = [x]
    fvec = [val]
    diff = np.inf
    it = 1
    while diff > tol and it <= max_iter:
        a_n = a * it**-alpha
        x -= a_n * val
        x = np.clip(x, 0, 1)
        val = f(x)
        xvec.append(x)  # save results
        fvec.append(val)
        match loss:
            case "arg":
                diff = abs(x - xvec[it - 1])
            case "function":
                diff = abs(val)
            case _:
                raise Exception("Invalid loss type!")

        if verbose:
            print((it, x, diff), flush=True)
        it += 1
    if return_values:
        return x, xvec, fvec
    else:
        return x


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


def fit_ipn(b0, ipn_target, cf, gamma, A, rho0, tmax, mode):
    f = lambda b: ipn_func(b, ipn_target, cf, gamma, A, rho0, 1, tmax, mode)
    bscaled = robbins_monro_solve(f, b0, verbose=True)

    f = lambda b: ipn_func(b, ipn_target, cf, gamma, A, rho0, 10, tmax, mode)
    bscaled = robbins_monro_solve(f, bscaled, verbose=True)

    f = lambda b: ipn_func(b, ipn_target, cf, gamma, A, rho0, 100, tmax, mode)
    bscaled = robbins_monro_solve(f, bscaled, verbose=True)

    return bscaled


def target_ipn(A, gamma, c, mode, rho0, tmax, realizations):
    n = A.shape[0]
    x0 = np.zeros(n)
    x0[random.sample(range(n), round(rho0 * n))] = 1
    ipn = 0
    for _ in range(realizations):
        x = contagion_process(A, gamma, c, x0, tmax=tmax)
        ipn += infections_per_node(x, mode) / realizations
    return ipn
