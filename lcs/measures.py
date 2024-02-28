import numpy as np
from scipy.special import binom


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
    f = 2 * p * r / (p + r)
    if normalize:
        rho = density(A)
        # https://stats.stackexchange.com/questions/390200/what-is-the-baseline-of-the-f1-score-for-a-binary-classifier
        f_random = 2 * rho * rho_guess / (rho + rho_guess)
        return f / f_random
    else:
        return f


def precision(samples, A):
    Q = samples.mean(axis=0)
    tp = np.sum(Q * A)
    fp = np.sum(Q * (1 - A))
    return tp / (tp + fp)


def recall(samples, A):
    Q = samples.mean(axis=0)
    tp = np.sum(Q * A)
    fn = np.sum((1 - Q) * A)
    return tp / (tp + fn)


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


def density(A):
    n = A.shape[0]
    return (A.sum() / 2) / binom(n, 2)


def hamming_distance(A1, A2):
    return np.sum(np.abs(A1 - A2)) / 2


def auroc(samples,A):
    Q = samples.mean(axis=0)
    A = A.flatten()
    Q = Q.flatten()
    return roc_auc_score(A,Q)
