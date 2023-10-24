# %%
import numpy as np
import matplotlib.pyplot as plt
from src import *
import networkx as nx
from scipy.stats import beta
import time
from numpy.linalg import eigh
import cProfile
import pstats
import pdb
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs
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
print("optimize")

# %%

def count_mask(array, boolean_mask, my_axis):
    """
    Count the occurrences of values in `array` that correspond to `True` values in `boolean_mask`,
    along the specified axis `my_axis`.

    Parameters
    ----------
    array : numpy.ndarray
        The input array to count values from.
    boolean_mask : numpy.ndarray
        A boolean mask with the same shape as `array`, indicating which values to count.
    my_axis : int
        The axis along which to count values.
    Returns
    -------
    numpy.ndarray
        An array of counts, with shape `(n,)` where `n` is the number of unique values in `array`.
    """
    n = array.shape[0]
    boolean_mask = boolean_mask.astype(int)
    array = array.astype(int)
    masked_arr = np.where(boolean_mask,array.T,n+1)
    return np.apply_along_axis(np.bincount, axis=my_axis, arr=masked_arr, minlength=n+2).T




def count_all_infection_events2(x, A):
    T = x.shape[0]
    n = x.shape[1]
    nl = np.zeros((n, n), dtype=int)
    ml = np.zeros((n, n), dtype=int)

    nus = A @ x[:-1].T
    nus = np.round(nus).astype(int)

    was_infected = (x[1:]*(1-x[:-1]))#1 if node i was infected at time t, 0 otherwise
    was_not_infected = (1-x[1:])*(1-x[:-1])#1 if node i was not infected at time t, 0 otherwise

#    breakpoint()
    ml = count_mask(nus, was_not_infected, 0)
    nl = count_mask(nus, was_infected, 0)

    ml = ml[:,:n]
    nl = nl[:,:n]

    return nl, ml


def count_local_infection_events2(i,x, A):
    T = x.shape[0]
    n = x.shape[1]
    nl = np.zeros((n, n), dtype=int)
    ml = np.zeros((n, n), dtype=int)

    nus = A @ x[:-1].T
    nus_i = np.round(nus[i]).astype(int)#select infected neighbor from node i
    breakpoint()
    x_i = x[0:,i]#select node i from all time steps

    was_infected = (x_i[1:]*(1-x_i[:-1]))#1 if node i was infected at time t, 0 otherwise
    was_not_infected = (1-x_i[1:])*(1-x_i[:-1])#1 if node i was not infected at time t, 0 otherwise

#    breakpoint()
    ml = count_mask(nus_i, was_not_infected, 0)
    nl = count_mask(nus_i, was_infected, 0)

    ml = ml[:n]
    nl = nl[:n]

    return nl, ml

#%%

#a = count_all_infection_events(x ,A)
a = count_local_infection_events(1,x ,A)
b = count_local_infection_events2(1,x ,A)


#a[0].shape




#%%
a = count_all_infection_events(x ,A)
a = count_all_infection_events(x ,A)

T = x.shape[0]
n = x.shape[1]
nl = np.zeros((n, n), dtype=int)
ml = np.zeros((n, n), dtype=int)

nus = A @ x[:-1].T
nus = np.round(nus).astype(int)

was_infected = (x[1:]*(1-x[:-1]))
was_not_infected = ((1-x[1:])*(1-x[:-1]))

boolean_mask = was_not_infected
array = nus
my_axis = 0

boolean_mask = boolean_mask.astype(int)
array = array.astype(int)
masked_arr = array #np.where(boolean_mask,array.T,-1)
#np.apply_along_axis(np.bincount, axis=my_axis, arr=masked_arr, minlength=n,weights = boolean_mask[1]).T
# define a lambda function to apply np.bincount with a different set of weights
#bincount_with_weights = lambda x, w: np.bincount(x, weights=w)
# apply the lambda function to each subarray with a different set of weights
a = np.apply_along_axis(np.bincount, axis=my_axis, arr=masked_arr, minlength=n).T

#np.bincount(array[1])
subarrays = [np.bincount(x) for x, w in zip(array.T, masked_arr.T)]
# get the maximum length of the subarrays
max_length = len(subarrays)
# pad each subarray with zeros to make them the same length
padded_subarrays = [np.pad(x, (0, max_length - len(x)), mode='constant') for x in subarrays]
# stack the padded subarrays vertically
result = np.vstack(padded_subarrays)


b = np.apply_along_axis(np.bincount, axis=my_axis, arr=masked_arr, minlength=n).T


# %%

# #%%

# fig,ax = plt.subplots(nrows = 2,ncols = 2)

# b = count_all_infection_events2(x,A)


# ax[0,0].imshow(a[0])
# ax[0,0].set(title = 'nl original')

# ax[1,0].imshow(a[1])
# ax[1,0].set(title = 'ml original')

# ax[0,1].imshow(b[0])
# ax[0,1].set(title = 'nl updated')
# ax[1,1].imshow(b[1])
# ax[1,1].set(title = 'ml updated')
# plt.tight_layout()

# plt.show()




# %%
start_time = time.time()
a = count_all_infection_events(x,A)
print("Time taken for count_all_infection_events:", time.time() - start_time)

start_time = time.time()
b = count_all_infection_events2(x,A)
print("Time taken for count_all_infection_events2:", time.time() - start_time)


# %%

from numba import njit
from numba import njit, prange, types



def count_mask(array, boolean_mask, my_axis):
    """
    Count the occurrences of values in `array` that correspond to `True` values in `boolean_mask`,
    along the specified axis `my_axis`.

    Parameters
    ----------
    array : numpy.ndarray
        The input array to count values from.
    boolean_mask : numpy.ndarray
        A boolean mask with the same shape as `array`, indicating which values to count.
    my_axis : int
        The axis along which to count values.
    Returns
    -------
    numpy.ndarray
        An array of counts, with shape `(n,)` where `n` is the number of unique values in `array`.
    """
    n = array.shape[0]
    boolean_mask = boolean_mask.astype(int)
    array = array.astype(int)

    masked_arr = np.where(boolean_mask,array.T,n+1)#assign all values that fail the boolean mask to n+1, these should get removed beofre returning result
    return np.apply_along_axis(np.bincount, axis=my_axis, arr=masked_arr, minlength=n+2).T

def count_all_infection_events2(x, A):
    T = x.shape[0]
    n = x.shape[1]
    nl = np.zeros((n, n), dtype=int)
    ml = np.zeros((n, n), dtype=int)

    nus = A @ x[:-1].T
    nus = np.round(nus).astype(int)

    was_infected = (x[1:]*(1-x[:-1]))#1 if node i was infected at time t, 0 otherwise
    was_not_infected = (1-x[1:])*(1-x[:-1])#1 if node i was not infected at time t, 0 otherwise

    ml = count_mask(nus, was_not_infected, 0)
    nl = count_mask(nus, was_infected, 0)

    ml = ml[:,:n]
    nl = nl[:,:n]

    return nl, ml

def count_local_infection_events2(i, x, A):
    n = x.shape[1]
    nl = np.zeros((n, n), dtype=np.int64)
    ml = np.zeros((n, n), dtype=np.int64)

    nus = A @ x[:-1].T
    nus_i = np.round(nus[i]).astype(np.int64)
    x_i = x[0:,i]

    was_infected = (x_i[1:]*(1-x_i[:-1]))
    was_not_infected = (1-x_i[1:])*(1-x_i[:-1])

    ml = count_mask(nus_i, was_not_infected, 0)
    nl = count_mask(nus_i, was_infected, 0)

    ml = ml[:n]
    nl = nl[:n]

    return nl, ml


from numba import jit, prange

@jit(nopython=True, parallel=True)
def count_local_infection_events_multi(i, x, A):
    T = x.shape[0]
    n = x.shape[1]

    nl = np.zeros(n)
    ml = np.zeros(n)

    for t in prange(T - 1):
        nu = A[i].dot(x[t])

        nu = int(round(nu))
        nl[nu] += x[t + 1, i] * (1 - x[t, i])
        ml[nu] += (1 - x[t + 1, i]) * (1 - x[t, i])
    return nl, ml


@jit(nopython=True)
def count_local_infection_events(i, x, A):
    T = x.shape[0]
    n = x.shape[1]

    nl = np.zeros(n)
    ml = np.zeros(n)

    for t in range(T - 1):
        nu = A[i].dot(x[t])

        nu = int(round(nu))
        nl[nu] += x[t + 1, i] * (1 - x[t, i])
        ml[nu] += (1 - x[t + 1, i]) * (1 - x[t, i])
    return nl, ml


# %%
import time
i = 1
start_time = time.time()
a = count_local_infection_events(i,x,A)
print("Time taken for count_local_infection_events:", time.time() - start_time)

start_time = time.time()
b = count_local_infection_events_multi(i,x,A)
print("Time taken for count_local_infection_events multithreaded:", time.time() - start_time)


start_time = time.time()
b = count_local_infection_events2(i,x,A)
print("Time taken for count_local_infection_events vectorized:", time.time() - start_time)



A_sparse = csr_matrix(A)
start_time = time.time()
b = count_local_infection_events2(i,x,A_sparse)
print("Time taken for count_local_infection_events vectorized sparse:", time.time() - start_time)
# %%

# %%
import torch

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

    masked_arr = torch.where(boolean_mask, array.T, n+1)#assign all values that fail the boolean mask to n+1, these should get removed beofre returning result
    counts = torch.zeros(n+2, dtype=torch.int)
    counts = torch.bincount(masked_arr.flatten(), minlength=n+2)
    counts = counts.T

    return counts

def count_local_infection_events_torch(i, x, A):
    n = x.shape[1]
    nl = torch.zeros((n, n), dtype=torch.int64)
    ml = torch.zeros((n, n), dtype=torch.int64)

    nus = torch.matmul(A, x[:-1].T)
    nus_i = torch.round(nus[i]).to(torch.int64)
    x_i = x[:, i]

    was_infected = (x_i[1:]*(1-x_i[:-1]))
    was_not_infected = (1-x_i[1:])*(1-x_i[:-1])

    ml = count_mask_torch(nus_i, was_not_infected, 0)
    nl = count_mask_torch(nus_i, was_infected, 0)

    ml = ml[:n]
    nl = nl[:n]

    return nl, ml


#%%


A_tensor = to_sparse(torch.tensor(A))
x_tensor = torch.tensor(x
start_time = time.time()
b = count_local_infection_events_torch(i,x_tensor,A_tensor)
print("Time taken for count_local_infection_events vectorized sparse:", time.time() - start_time)
# %%
