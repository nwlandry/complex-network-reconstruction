import numpy as np
from scipy.special import binom
from sklearn.metrics import roc_auc_score


def posterior_similarity(samples, A):
    meanA = samples.mean(axis=0)
    num = np.sum(np.abs(A - meanA))
    den = np.sum(np.abs(A + meanA))
    if den > 0:
        return 1 - num / den
    else:
        return 1


def samplewise_posterior_similarity(samples, A):
    f = 0
    n = samples.shape[0]
    for i in range(n):
        num = np.sum(np.abs(A - samples[i]))
        den = np.sum(np.abs(A + samples[i]))
        if den > 0:
            f += (1 - num / den) / n
        else:
            f += 1 / n
    return f


def f_score(samples, A, normalize=False, rho_guess=0.5):
    p = precision(samples, A)
    r = recall(samples, A)

    if np.isnan(p) or np.isnan(r):
        f = np.nan
    else:
        f = 2 * p * r / (p + r)

    if normalize:
        rho = density(A)
        # https://stats.stackexchange.com/questions/390200/what-is-the-baseline-of-the-f1-score-for-a-binary-classifier
        if rho + rho_guess > 0:
            f_random = 2 * rho * rho_guess / (rho + rho_guess)
        else:
            f_random = 0

        return f / f_random
    else:
        return f


def precision(samples, A):
    Q = samples.mean(axis=0)
    tp = np.sum(Q * A)
    fp = np.sum(Q * (1 - A))
    if tp + fp > 0:
        return tp / (tp + fp)
    else:
        return np.nan


def recall(samples, A):
    Q = samples.mean(axis=0)
    tp = np.sum(Q * A)
    fn = np.sum((1 - Q) * A)
    if tp + fn > 0:
        return tp / (tp + fn)
    else:
        return np.nan


def fraction_of_correct_entries(samples, A, normalize=False, rho_guess=0.5):
    n = A.shape[0]
    nsamples = samples.shape[0]
    num = (np.sum(samples == A) - nsamples * n) / 2
    den = nsamples * binom(n, 2)
    fce = num / den

    if normalize:
        rho = density(A)
        fce_random = rho * rho_guess + (1 - rho) * (
            1 - rho_guess
        )  # pg1 * p1 + pg0 * p0
        return fce / fce_random
    else:
        return fce


def nodal_performance(Q, A):
    return np.abs(Q - A).sum(axis=0) / A.shape[0]


def density(A):
    n = A.shape[0]
    return (A.sum() / 2) / binom(n, 2)


def hamming_distance(A1, A2):
    return np.sum(np.abs(A1 - A2)) / 2


def auroc(samples, A):
    n = A.shape[0]
    Q = samples.mean(axis=0)
    y_true = A[np.tril_indices(n, -1)]
    y_score = Q[np.tril_indices(n, -1)]
    if len(np.unique(y_true)) == 1:
        return np.nan
    return roc_auc_score(y_true, y_score)


def degrees(A):
    if not isinstance(A, np.ndarray):
        A = A.todense()
    return A.sum(axis=0)


def clustering_coefficient(A):
    T = np.diag(A @ A @ A)
    k = degrees(A)
    D = np.multiply(k, k - 1)
    C = np.divide(T, D, out=np.zeros_like(T), where=D != 0)
    return C
