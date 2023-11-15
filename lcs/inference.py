import random
import warnings

import numpy as np
from numba import jit
from numpy import ndarray
from scipy.special import betaln, binom
from scipy.stats import beta


def infer_adjacency_matrix_and_dynamics(
    x,
    A0,
    p_rho=None,
    p_gamma=None,
    p_c=None,
    nsamples=1,
    nspa=1,
    burn_in=100,
    skip=100,
    return_likelihood=False,
):
    """A function to infer both the adjacency matrix and the contagion dynamics.

    Parameters
    ----------
    x : numpy ndarray
        A T x N matrix of zeros (susceptible) and ones (infected)
    A0 : numpy ndarray
        An N x N adjacency matrix to initialize the MCMC algorithm.
    p_rho : list or ndarray, optional
        A 2-array specifying the parameters of the beta distribution
        for the prior on rho, by default None. If None, it assumes a
        uniform prior.
    p_gamma : list or ndarray, optional
        A 2-array specifying the parameters of the beta distribution
        for the gamma prior, by default None. If None, it assumes a
        uniform prior.
    p_c : list of lists or ndarray, optional
        A 2 x N array of the priors on each entry in the c vector, by default None.
        If None, it assumes a uniform prior.
    nsamples : int, optional
        The number of adjacency matrix samples desired, by default 1
    nspa : int, optional
        The number of dynamics samples per adjacency matrix, by default 1
    burn_in : int, optional
        The number of iterations before storing the first sample, by default 100
    skip : int, optional
        The number of iterations between each sample, by default 100
    return_likelihood : bool, optional
        Whether to return the log posterior, by default False

    Returns
    -------
    samples, gamma, cf
        (1) `samples` is an S x N x N array where S is the number of samples
        (2) `gamma` is an 1D array of size S*D, where D is the number of dynamics
        samples per adjacency matrix.
        (3) `cf` is a 2D array of size S*D x N, where is row is a sampled contagion
        vector.
    if return_likelihood is True, also returns a list of log posterior values.

    Notes
    -----
    We assume beta priors for conjugacy.
    """
    n = x.shape[1]
    samples, l = infer_adjacency_matrix(
        x,
        A0,
        p_rho=p_rho,
        p_c=p_c,
        nsamples=nsamples,
        burn_in=burn_in,
        skip=skip,
        return_likelihood=True,
    )

    gamma = np.zeros(int(nsamples * nspa))
    cf = np.zeros((nsamples * nspa, n))
    for i in range(nsamples):
        g, c = infer_dynamics(x, samples[0], p_gamma=p_gamma, p_c=p_c, nsamples=10)

        gamma[i * nspa : (i + 1) * nspa] = g
        cf[i * nspa : (i + 1) * nspa, :] = c

    if return_likelihood:
        return samples, gamma, cf, l
    else:
        return samples, gamma, cf


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
    """A function to infer the adjacency matrix.

    Parameters
    ----------
    x : numpy ndarray
        A T x N matrix of zeros (susceptible) and ones (infected)
    A0 : numpy ndarray
        An N x N adjacency matrix to initialize the MCMC algorithm.
    p_rho : list or ndarray, optional
        A 2-array specifying the parameters of the beta distribution
        for the prior on rho, by default None. If None, it assumes a
        uniform prior.
    p_c : list of lists or ndarray, optional
        A 2 x N array of the priors on each entry in the c vector, by default None.
        If None, it assumes a uniform prior.
    nsamples : int, optional
        The number of adjacency matrix samples desired, by default 1
    burn_in : int, optional
        The number of iterations before storing the first sample, by default 100
    skip : int, optional
        The number of iterations between each sample, by default 100
    return_likelihood : bool, optional
        Whether to return the log posterior, by default False

    Returns
    -------
    samples
        An S x N x N array where S is the number of samples
    if return_likelihood is True, also returns a list of log posterior values.

    Notes
    -----
    We assume beta priors for conjugacy.
    """
    # form initial adjacency matrix
    A = A0.copy()
    if not isinstance(A, ndarray):
        A = A.todense()

    if A.dtype != "int":
        A = np.array(A, dtype=int)

    if x.dtype != "int":
        x = np.array(x, dtype=int)

    n, m = A.shape

    p_rho = _check_beta_parameters(p_rho, [2])
    p_c = _check_beta_parameters(p_c, [2, n])

    if n != m:
        Exception("Matrix must be square!")

    nl, ml = count_all_infection_events(x, A)
    l_dynamics = dynamics_log_posterior(nl, ml, p_c)

    if sum(np.diag(A)) > 0:
        raise Exception("Self-loops are not allowed.")

    num_entries = int(np.sum(A) / 2)
    max_entries = binom(n, 2)

    l_adjacency = adjacency_log_posterior(num_entries, max_entries, p_rho)

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

        new_l_dynamics = dynamics_log_posterior(new_nl, new_ml, p_c)

        # update likelihood of the incidence matrix given rho
        new_l_adjacency = adjacency_log_posterior(
            num_entries + delta_entries, max_entries, p_rho
        )

        delta = (new_l_dynamics - l_dynamics) + (new_l_adjacency - l_adjacency)

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
    print(
        f"Acceptance ratio is {accept / (burn_in + (nsamples - 1) * skip)}", flush=True
    )

    if return_likelihood:
        return samples, np.array(l_vals)
    else:
        return samples


def infer_dynamics(x, A, p_gamma=None, p_c=None, nsamples=1):
    """A function to infer the contagion dynamics.

    Parameters
    ----------
    x : numpy ndarray
        A T x N matrix of zeros (susceptible) and ones (infected)
    A : numpy ndarray
        An N x N adjacency matrix
    p_gamma : list or ndarray, optional
        A 2-array specifying the parameters of the beta distribution
        for the gamma prior, by default None. If None, it assumes a
        uniform prior.
    p_c : list of lists or ndarray, optional
        A 2 x N array of the priors on each entry in the c vector, by default None.
        If None, it assumes a uniform prior.
    nsamples : int, optional
        The number of samples of the dynamics desired, by default 1
    return_likelihood : bool, optional
        Whether to return the log posterior, by default False

    Returns
    -------
    gamma, cf
        `gamma` is an 1D array of size S*D, where D is the number of dynamics
        samples per adjacency matrix.
        `cf` is a 2D array of size S*D x N, where is row is a sampled contagion
        vector.
    if return_likelihood is True, also returns a list of log posterior values.

    Notes
    -----
    We assume beta priors for conjugacy.
    """
    if not isinstance(A, ndarray):
        A = A.todense()

    if A.dtype != "int":
        A = np.array(A, dtype=int)

    if x.dtype != "int":
        x = np.array(x, dtype=int)

    n = A.shape[0]

    p_gamma = _check_beta_parameters(p_gamma, [2])
    p_c = _check_beta_parameters(p_c, [2, n])

    # sample gamma

    # healing events
    # count the places where the indicator is equal to 1
    a = len(np.where(x[1:] * (1 - x[:-1]))[0])
    b = len(np.where(x[1:] * x[:-1])[0])

    gamma = beta(a + p_gamma[0], b + p_gamma[1]).rvs(size=nsamples)

    # sample c
    # These are the parameters for the beta distribution,
    # each entry corresponds to a number of infected neighbors
    nl, ml = count_all_infection_events(x, A)
    a = nl.sum(axis=0)
    b = ml.sum(axis=0)

    c = np.zeros((nsamples, n))
    for i in range(nsamples):
        c[i] = beta(a + p_c[0], b + p_c[1]).rvs()
    return gamma, c


def count_all_infection_events(x, A):
    """counts all the infection and non-infection events

    Parameters
    ----------
    x : numpy ndarray
        A T x N matrix of zeros (susceptible) and ones (infected)
    A : numpy ndarray
        An N x N adjacency matrix

    Returns
    -------
    nl, ml
        (1) `ml` is an N x N matrix where the rows indicate node labels
        and the columns indicate the nu value. This matrix stores non-infection
        events.
        (2) `nl` is an N x N matrix where the rows indicate node labels
        and the columns indicate the nu value. This matrix stores infection
        events.
    """
    n = x.shape[1]

    nus = A @ x[:-1].T

    # 1 if node i was infected at time t, 0 otherwise
    was_infected = x[1:] * (1 - x[:-1])

    # 1 if node i was not infected at time t, 0 otherwise
    was_not_infected = (1 - x[1:]) * (1 - x[:-1])

    ml = _count_mask(nus, was_not_infected, 0, n)[:, :n]
    nl = _count_mask(nus, was_infected, 0, n)[:, :n]

    return nl, ml


def count_local_infection_events(i, x, A):
    """counts all the infection and non-infection events

    Parameters
    ----------
    i : int
        The node index in question
    x : numpy ndarray
        A T x N matrix of zeros (susceptible) and ones (infected)
    A : numpy ndarray
        An N x N adjacency matrix

    Returns
    -------
    nl, ml
        (1) `ml` is an 1 x N matrix where the entries indicate the nu values
        for node i. This matrix stores non-infection events.
        (2) `nl` is an 1 x N matrix where the entries indicate the nu values
        for node i. This matrix stores infection events.
    """
    n = A.shape[0]
    nus_i = A[i] @ x[:-1].T
    x_i = x[:, i]  # select node i from all time steps

    # 1 if node i was infected at time t, 0 otherwise
    was_infected = x_i[1:] * (1 - x_i[:-1])

    # 1 if node i was not infected at time t, 0 otherwise
    was_not_infected = (1 - x_i[1:]) * (1 - x_i[:-1])

    ml = _count_mask(nus_i, was_not_infected, 0, n)[:n]
    nl = _count_mask(nus_i, was_infected, 0, n)[:n]

    return nl, ml


def dynamics_log_posterior(nl, ml, p_c):
    """Computes the portion of the log posterior from the dynamics

    Parameters
    ----------
    nl : numpy ndarray
        An N x N matrix of nu counts for each node
    ml : numpy ndarray
        An N x N matrix of nu counts for each node
    p_c : numpy ndarray
        2 x N array of beta parameters for the prior on the contagion
        vector.

    Returns
    -------
    float
        The log of this portion of the posterior distribution.
    """
    a = nl.sum(axis=0)
    b = ml.sum(axis=0)
    return sum(betaln(a + p_c[0], b + p_c[1]))


def adjacency_log_posterior(num_entries, max_entries, p_rho):
    """Computes the portion of the log posterior from the adjacency matrix

    Parameters
    ----------
    num_entries : int
        The number of non-zero entries in the lower triangle of the adjacency matrix
    max_entries : int
        The maximum number of entries in the lower triangle of the adjacency matrix.
    p_rho : numpy ndarray
        A 2-array specifying the parameters for the prior on rho.

    Returns
    -------
    float
        The log posterior value from the adjacency matrix portion.
    """
    return betaln(num_entries + p_rho[0], max_entries - num_entries + p_rho[1])


@jit(nopython=True)
def update_adjacency_matrix(i, j, A):
    """Flips an edge in the adjacency matrix

    Parameters
    ----------
    i : int
        row index
    j : int
        column index
    A : numpy ndarray
        The adjacency matrix

    Returns
    -------
    int
        -1 if an edge is removed, +1 if an edge is added.
    """
    if A[i, j] == 0:
        A[i, j] = A[j, i] = 1
        return 1
    else:
        A[i, j] = A[j, i] = 0
        return -1


def _count_mask(array, boolean_mask, axis, max_val):
    """
    Count the occurrences of values in `array` that correspond to `True` values in `boolean_mask`,
    along the specified axis `axis`.

    Parameters
    ----------
    array : numpy.ndarray
        The input array to count values from.
    boolean_mask : numpy.ndarray
        A boolean mask with the same shape as `array`, indicating which values to count.
    axis : int
        The axis along which to count values.
    Returns
    -------
    numpy.ndarray
        An array of counts, with shape `(n,)` where `n` is the number of unique values in `array`.
    """
    boolean_mask = boolean_mask.astype(int)
    array = array.astype(int)
    # assign all values that fail the boolean mask to n+1,
    # these should get removed before returning result
    masked_arr = np.where(boolean_mask, array.T, max_val + 1)
    return np.apply_along_axis(
        np.bincount, axis=axis, arr=masked_arr, minlength=max_val + 2
    ).T


def _check_beta_parameters(p, size):
    """Checks that the parameters for a beta distribution are in the right format.

    Parameters
    ----------
    p : numpy ndarray
        The parameters of a beta distribution. If p is None,
        this function constructs an array of ones which corresponds
        to a uniform prior.
    size : numpy ndarray
        a 2-array specifying the size that p should be.

    Returns
    -------
    p
        An array of the specified size.

    Raises
    ------
    Exception
        If the parameters are not positive.
    Exception
        If p is of the wrong shape.
    """
    if isinstance(p, (list, tuple)):
        p = np.array(p)

    if p is None:
        p = np.ones(size)
    elif np.any(np.array(p) <= 0):
        raise Exception("Parameters in a beta distribution must be greater than 0.")

    if p.shape != tuple(size):
        raise Exception("Parameters are in the wrong shape.")

    return p


############## EXTRA #####################
def infer_dyamics_loop(x, A, p_gamma, p_c):
    T, n = x.shape

    if np.any(p_c <= 0) or np.any(p_gamma <= 0):
        raise Exception("Parameters in a beta distribution must be greater than 0.")

    # sample gamma
    a = 0  # healing events
    b = 0  # non-healing events
    for t in range(T - 1):
        # healing events
        a += len(np.where(x[t + 1] * (1 - x[t]))[0])
        # non-healing events
        b += len(np.where(x[t + 1] * x[t])[0])

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
            a[nu] += x[t + 1, i] * (1 - x[t, i])
            b[nu] += (1 - x[t + 1, i]) * (1 - x[t, i])

    c = np.zeros(n)
    for i in range(n):
        c[i] = beta(a[i] + p_c[0, i], b[i] + p_c[1, i]).rvs()

    return gamma, c


def count_all_infection_events_loop(x, A):
    """
    Counts the number of infection events between all pairs of nodes in a network over time.

    Args:
        x (numpy.ndarray): A binary matrix of shape (T, n) where T is the number of time steps and n is the number of nodes.
            Each row represents the state of the nodes at a given time step, where 1 indicates an infected node and 0 indicates a susceptible node.
        A (numpy.ndarray): An adjacency matrix of shape (n, n) representing the connections between nodes in the network.

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray]: A tuple of two matrices, nl and ml, both of shape (n, n).
            nl[i, j] represents the number of times node i was infected with j infected neighbors, while ml[i, j] represents the number of times node i failed to be infected with j infected neighbors.
    """
    T = x.shape[0]
    n = x.shape[1]

    nl = np.zeros((n, n), dtype=int)
    ml = np.zeros((n, n))

    for t in range(T - 1):
        nus = A.dot(x[t])

        for i, nu in enumerate(nus):
            nu = int(np.round(nu))
            nl[i, nu] += x[t + 1, i] * (1 - x[t, i])
            ml[i, nu] += (1 - x[t + 1, i]) * (1 - x[t, i])
    return nl, ml


def count_local_infection_events_loop(i, x, A):
    T = x.shape[0]
    n = x.shape[1]

    nl = np.zeros(n)
    ml = np.zeros(n)

    for t in range(T - 1):
        nu = int(np.round(A[i].dot(x[t])))
        nl[nu] += x[t + 1, i] * (1 - x[t, i])
        ml[nu] += (1 - x[t + 1, i]) * (1 - x[t, i])
    return nl, ml
