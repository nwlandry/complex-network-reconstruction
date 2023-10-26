import random

import numpy as np
from numpy import ndarray


def contagion_process(A, gamma, c, s0, tmin=0, tmax=20, dt=1, random_seed=None):
    """A neighborhood-based contagion process on pairwise networks

    Parameters
    ----------
    A : Numpy ndarray
        The adjacency matrix
    gamma : float
        The healing rate
    c : Numpy ndarray
        A 1d vector of the contagion rates. Should be N x 1.
    s0 : Numpy ndarray
        A 1D vector of the initial nodal states, either 0 or 1.
        Should be N x 1.
    tmin : int, optional
        The time at which to start the simulation, by default 0
    tmax : float, optional
        The time at which to terminate the simulation, by default 20
    dt : float, optional
        The time step, by default 1
    random_seed : int, optional
        The seed for the random process, by default None

    Returns
    -------
    Numpy ndarray
        The time series of the contagion process. Should be T x N.

    Raises
    ------
    Exception
        If adjacency matrix isn't square
    """
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
                infected_nbrs = int(round(A[i] @ x[t]))
                if random.random() <= c[infected_nbrs] * dt:
                    x[t + 1, i] = 1
    return x
