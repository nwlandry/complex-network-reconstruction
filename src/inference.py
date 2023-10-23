import random

import numpy as np
from numpy import ndarray
from scipy.stats import beta
from scipy.special import betaln, binom
from scipy.sparse import csr_matrix

import warnings

warnings.filterwarnings("error")


def infer_adjacency_matrix(
    x,
    A0,
    p_rho=None,
    p_c=None,
    nsamples=1,
    burn_in=100,
    skip=100,
    return_likelihood=False,
):
    # form initial adjacency matrix
    if not isinstance(A0, ndarray):
        #A0 = A0.todense()
        pass


    A = csr_matrix(A0.copy())
    n, m = np.shape(A)

    if isinstance(p_rho, (list, tuple)):
        p_rho = np.array(p_rho)

    if isinstance(p_c, (list, tuple)):
        p_c = np.array(p_c)

    if p_c is None:
        p_c = np.ones((n, 2))
    elif np.any(np.array(p_c) <= 0):
        raise Exception("Parameters in a beta distribution must be greater than 0.")

    if p_rho is None:
        p_rho = np.ones(2)
    elif np.any(p_rho <= 0):
        raise Exception("Parameters in a beta distribution must be greater than 0.")

    if n != m:
        Exception("Matrix must be square!")

    nl, ml = count_all_infection_events(x, A)
    l_dynamics = dynamics_log_likelihood(nl, ml, p_c)

    num_entries = int(np.sum(A) / 2)
    max_entries = binom(n, 2)

    l_adjacency = adjacency_log_likelihood(num_entries, max_entries, p_rho)

    samples = np.zeros((nsamples, n, n))
    accept = 0
    it = 0
    s_i = skip
    sample_num = 0

    if return_likelihood:
        l_vals = []

    while it <= burn_in + (nsamples - 1) * skip:

        # proposal comes from the lower triangle
        i = random.randrange(1, n)
        j = random.randrange(i)

        # alter hypergraph
        delta_entries = update_adjacency_matrix(i, j, A)

        nl_i, ml_i = count_local_infection_events(i, x, A)
        nl_j, ml_j = count_local_infection_events(j, x, A)

        new_nl = nl.copy()
        new_ml = ml.copy()

        new_nl[i] = nl_i
        new_nl[j] = nl_j
        new_ml[i] = ml_i
        new_ml[j] = ml_j

        new_l_dynamics = dynamics_log_likelihood(new_nl, new_ml, p_c)

        # update likelihood of the incidence matrix given rho
        new_l_adjacency = adjacency_log_likelihood(
            num_entries + delta_entries, max_entries, p_rho
        )

        delta = compute_delta(new_l_dynamics, l_dynamics) + compute_delta(
            new_l_adjacency, l_adjacency
        )

        if np.log(random.random()) <= min(delta, 0):
            nl = new_nl
            ml = new_ml
            l_dynamics = new_l_dynamics
            l_adjacency = new_l_adjacency
            num_entries += delta_entries
            accept += 1
        else:
            update_adjacency_matrix(i, j, A)

        if return_likelihood:
            l_vals.append(l_dynamics + l_adjacency)
        if it >= burn_in:
            if s_i >= skip:
                samples[sample_num, :, :] = A.copy()
                sample_num += 1
                s_i = 1
            else:
                s_i += 1

        it += 1
    print(f"Acceptance ratio is {accept/(burn_in + (nsamples - 1)*skip)}", flush=True)

    if return_likelihood:
        return samples, l_vals
    else:
        return samples


def count_mask(array, boolean_mask, my_axis,max_val):
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

    masked_arr = np.where(boolean_mask,array.T,max_val+1)#assign all values that fail the boolean mask to n+1, these should get removed beofre returning result
    return np.apply_along_axis(np.bincount, axis=my_axis, arr=masked_arr, minlength=max_val+2).T

def count_all_infection_events(x, A):
    T = x.shape[0]
    n = x.shape[1]
    nl = np.zeros((n, n), dtype=int)
    ml = np.zeros((n, n), dtype=int)

    nus = A @ x[:-1].T
    nus = np.round(nus).astype(int)

    was_infected = (x[1:]*(1-x[:-1]))#1 if node i was infected at time t, 0 otherwise
    was_not_infected = (1-x[1:])*(1-x[:-1])#1 if node i was not infected at time t, 0 otherwise

    ml = count_mask(nus, was_not_infected, 0,n)
    nl = count_mask(nus, was_infected, 0,n)

    ml = ml[:,:n]
    nl = nl[:,:n]

    return nl, ml


def count_local_infection_events(i, x, A):
    T = x.shape[0]
    n = x.shape[1]

    nus = A @ x[:-1].T
    #nus_i = np.round(nus[i]).astype(int)#select infected neighbor from node i
    nus_i = nus[i].astype(int)#select infected neighbor from node i
    x_i = x[0:,i]#select node i from all time steps

    was_infected = (x_i[1:]*(1-x_i[:-1]))#1 if node i was infected at time t, 0 otherwise
    was_not_infected = (1-x_i[1:])*(1-x_i[:-1])#1 if node i was not infected at time t, 0 otherwise
    ml = count_mask(nus_i, was_not_infected, 0,n)
    nl = count_mask(nus_i, was_infected, 0,n)

    ml = ml[:n]
    nl = nl[:n]

    return nl, ml


def dynamics_log_likelihood(nl, ml, p_c):
    a = np.sum(nl, axis=0)
    b = np.sum(ml, axis=0)
    return sum(b for b in betaln(a + p_c[0], b + p_c[1]))


def adjacency_log_likelihood(num_entries, max_entries, p_rho):
    rho = num_entries / max_entries
    if rho == 0 or rho == 1:
        return -np.inf
    else:
        return (num_entries + p_rho[0]) * np.log(rho) + (
            max_entries - num_entries + p_rho[1]
        ) * np.log(1 - rho)


def compute_delta(a, b):
    if (a == -np.inf and b == -np.inf) or (a == np.inf and b == np.inf):
        return 0
    else:
        return a - b


def update_adjacency_matrix(i, j, A):
    if A[i, j] == 0:
        A[i, j] = A[j, i] = 1
        return 1
    else:
        A[i, j] = A[j, i] = 0
        return -1


def infer_dynamics(x, A, p_gamma, p_c):
    # Our priors are drawn from a beta distribution such that
    # the posteriors are also from a beta distribution

    if np.any(p_c <= 0) or np.any(p_gamma <= 0):
        raise Exception("Parameters in a beta distribution must be greater than 0.")

    T = np.size(x, axis=0)
    n = np.size(x, axis=1)

    # sample gamma
    a = 0  # healing events
    b = 0  # non-healing events
    for t in range(T - 1):
        # healing events
        a += len(np.where(x[t + 1] - x[t] == -1)[0])
        # non-healing events
        b += len(np.where(x[t + 1] * x[t] == 1)[0])

    gamma = beta(a + p_gamma[0], b + p_gamma[1]).rvs()

    # sample c
    # These are the parameters for the beta distribution,
    # each entry corresponds to a number of infected neighbors
    a = np.zeros(n)
    b = np.zeros(n)

    for t in range(T - 1):
        nus = A @ x[t]

        # infection events
        for i, nu in enumerate(nus):
            nu = int(round(nu))
            a[nu] += x[t + 1, i] - x[t, i] == 1
            b[nu] += x[t + 1, i] == x[t, i] == 0

    c = np.zeros(n)
    for i in range(n):
        if a[i] * b[i] == 0:
            a[i] += 1e-6
            b[i] += 1e-6
        c[i] = beta(a[i] + p_c[0, i], b[i] + p_c[1, i]).rvs()

    return gamma, c
