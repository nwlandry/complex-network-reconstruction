import random

import numpy as np


def erdos_renyi(n, p):
    A = np.zeros((n, n))

    for i in range(n):
        for j in range(i):
            A[i, j] = A[j, i] = random.random() <= p

    return A
