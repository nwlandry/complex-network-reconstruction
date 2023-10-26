# %%
import cProfile
import pdb
import pstats
import time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from numpy.linalg import eigh
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs
from scipy.stats import beta
from src import *

random.seed(8)

n = 100000
k = 0.1
rho0 = 0.3
gamma = 1.0
n = 50
k = 0.01
rho0 = 0.1
gamma = 1
A = erdos_renyi(n, k)
A = csr_matrix(A)
A = erdos_renyi(n, k)
x0 = np.zeros(n)
x0[random.sample(range(n), int(rho0 * n))] = 1

nu = eigs(A)[0][-1]
b = 2 * gamma / nu

sc = lambda nu, b: 1 - (1 - b) ** nu
c = sc(np.arange(n), b)

x = contagion_process(A, gamma, c, x0, tmin=0, tmax=1000)


def count_mask_torch(array, boolean_mask, my_axis):
    """
    Count the occurrences of values in `array` that correspond to `True` values in `boolean_mask`,
    along the specified axis `my_axis`.

    Parameters
    ----------
    array : torch.Tensor
        The input tensor to count values from.
    boolean_mask : torch.Tensor
        A boolean mask with the same shape as `array`, indicating which values to count.
    my_axis : int
        The axis along which to count values.
    Returns
    -------
    torch.Tensor
        An tensor of counts, with shape `(n,)` where `n` is the number of unique values in `array`.
    """
    n = array.shape[0]
    boolean_mask = boolean_mask.to(torch.bool)
    array = array.to(torch.int)

    masked_arr = torch.where(
        boolean_mask, array, n + 1
    )  # assign all values that fail the boolean mask to n+1, these should get removed beofre returning result
    counts = torch.zeros(n + 2, dtype=torch.int)
    counts = torch.bincount(masked_arr.flatten(), minlength=n + 2)

    return counts


def count_local_infection_events_torch(i, x, A):
    n = x.shape[1]
    nl = torch.zeros((n, n), dtype=torch.int64)
    ml = torch.zeros((n, n), dtype=torch.int64)

    nus = torch.matmul(A, x[:-1].T)
    nus_i = torch.round(nus[i]).to(torch.int64)
    x_i = x[:, i]

    was_infected = x_i[1:] * (1 - x_i[:-1])
    was_not_infected = (1 - x_i[1:]) * (1 - x_i[:-1])

    ml = count_mask_torch(nus_i, was_not_infected, 0)
    nl = count_mask_torch(nus_i, was_infected, 0)

    ml = ml[:n]
    nl = nl[:n]

    return nl, ml


# %%


A_tensor = torch.tensor(A)
x_tensor = torch.tensor(x)
start_time = time.time()
b = count_local_infection_events_torch(1, x_tensor, A_tensor)
print(
    "Time taken for count_local_infection_events vectorized sparse:",
    time.time() - start_time,
)
# %%
