import random

import numpy as np
from numpy import ndarray


def contagion_process(A, gamma, beta, f, s0, tmin=0, tmax=20, dt=1, random_seed=None):
    if not isinstance(A, ndarray):
        A = A.todense()

    if random_seed is not None:
        random.seed(random_seed)

    # infect nodes
    n, m = np.shape(A)
    if n != m:
        raise Exception("Matrix must be square!")

    T = int((tmax - tmin) / dt)
    x = np.zeros((T, n))
    x[0] = s0.copy()

    for t in range(T - 1):
        x[t + 1] = x[t].copy()

        # infect people by neighbors the rooms
        for i in range(n):
            if x[t, i] == 1 and random.random() <= gamma * dt:
                x[t + 1, i] = 0
            elif x[t, i] == 0:
                infected_nbrs = A[i] @ x[t]
                if random.random() <= beta * f(infected_nbrs) * dt:
                    x[t + 1, i] = 1
    return x
