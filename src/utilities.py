import random

import numpy as np

from src.contagion import *
from src.inference import *

# import xgi


def to_imshow_orientation(A):
    return np.flipud(A.T)


def posterior_similarity(A, Asamples):
    meanA = np.mean(Asamples, axis=0)
    return 1 - np.sum(np.abs(A - meanA)) / np.sum(np.abs(A + meanA))


def samplewise_posterior_similarity(A, Asamples):
    ps = 0
    n = np.size(Asamples, axis=0)
    for i in range(n):
        ps += 1 - np.sum(np.abs(A - Asamples[i])) / np.sum(np.abs(A + Asamples[i]))
    return ps


def hamming_distance(A1, A2):
    return np.sum(np.abs(A1 - A2)) / 2


def infections_per_node(x):
    return np.mean(np.sum(x[1:] - x[:-1] > 0, axis=0))


def erdos_renyi(n, p):
    A = np.zeros((n, n))

    for i in range(n):
        for j in range(i):
            A[i, j] = A[j, i] = random.random() <= p

    return A
