import json
import random

import numpy as np
from scipy.sparse import csr_array
from scipy.sparse.csgraph import reverse_cuthill_mckee
from scipy.stats import rv_discrete
from sklearn.metrics import roc_auc_score

from .contagion import *
from .generative import erdos_renyi
from .inference import *


def single_inference(
    fname, gamma, c, b, rho0, A, tmax, p_c, p_rho, nsamples, burn_in, skip
):
    """
    Perform a single inference process and save the results to a file.

    Args:
        fname : str
            The file name to save the results to.
        gamma : float
            The healing rate.
        c : ndarray
            A 1d vector of the contagion rates. Should be N x 1.
        b : float
            The b parameter for the contagion process.
        rho0 : float
            The initial density of activated nodes.
        A : ndarray
            The adjacency matrix of the network.
        tmax : int
            The maximum time step for the contagion process.
        p_c : ndarray
            The parameters for the contagion probability function.
        p_rho : ndarray
            The parameters for the initial density function.
        nsamples : int
            The number of samples to generate.
        burn_in : int
            The number of burn-in steps for the MCMC algorithm.
        skip : int
            The number of steps to skip between samples.
    Returns:
        None
    """
    n = np.size(A, axis=0)
    x0 = np.zeros(n)
    x0[random.sample(range(n), round(rho0 * n))] = 1

    x = contagion_process(A, gamma, c, x0, tmin=0, tmax=tmax)
    p = beta(p_rho[0], p_rho[1]).rvs()
    A0 = erdos_renyi(n, p)
    samples = infer_adjacency_matrix(
        x, A0, p_rho, p_c, nsamples=nsamples, burn_in=burn_in, skip=skip
    )

    # json dict
    data = {}
    data["gamma"] = gamma
    data["c"] = c.tolist()
    data["b"] = b
    data["p-rho"] = p_rho.tolist()
    data["p-c"] = p_c.tolist()
    data["x"] = x.tolist()
    data["A"] = A.tolist()
    data["samples"] = samples.tolist()

    datastring = json.dumps(data)

    with open(fname, "w") as output_file:
        output_file.write(datastring)


def to_imshow_orientation(A):
    """
    Transposes and flips the input array to match the orientation expected by the `imshow` function in matplotlib.

    Parameters:
        A : ndarray
            The input array.
    Returns:
        ndarray
            The transposed and flipped array.
    """
    return np.flipud(A.T)


def prettify_matrix(A):
    """
    Rearranges the rows and columns of a matrix A to improve its visual appearance.
    
    Parameters:
    A (numpy.ndarray): The input matrix to be prettified.
    
    Returns:
    numpy.ndarray: The prettified matrix with rearranged rows and columns.
    """
    idx = reverse_cuthill_mckee(csr_array(A), symmetric_mode=True)
    Ap = A.copy()
    Ap = Ap[idx]
    Ap = Ap[:, idx]
    return Ap


def infections_per_node(x, mode="mean"):
    """
    Calculate the number of infections per node in a neighborhood-based contagion process on pairwise networks.

    Parameters
    ----------
    x : numpy.ndarray
        The time series of the contagion process. Should be T x N.
    mode : str, optional
        The mode for calculating the number of infections per node. "mean" "median" or "max". Default is "mean".

    Returns
    -------
    float
        The number of infections per node.

    Raises
    ------
    Exception
        If an invalid mode is provided.

    """
    match mode:
        case "mean":
            return np.mean((x[1:] - x[:-1] > 0).sum(axis=0))
        case "median":
            return np.median((x[1:] - x[:-1] > 0).sum(axis=0))
        case "max":
            return np.max((x[1:] - x[:-1] > 0).sum(axis=0))
        case _:
            raise Exception("Invalid mode!")


def nu_distribution(x, A):
    """

    Calculate the nu matrix, nu[i,j] counts the total number of times a node  of degree i with 
    j infected neighbors in timestep t gets infected in timestep t+1

    Parameters
    ----------
    x : numpy.ndarray
        The time series of the contagion process. Should be T x N.
    A : numpy.ndarray
        The adjacency matrix NxN

    Returns
    -------
    numpy.ndarray
        The matrix representing the distribution of infections for a node with i infected neighbors 

    Raises
    ------
    Exception
        If adjacency matrix isn't square
    """
    k = A.sum(axis=0)
    nu = A @ x.T
    T, n = x.shape
    kmax = round(k.max())
    mat = np.zeros((kmax + 1, kmax + 1))
    for t in range(T):
        for i in range(n):
            mat[int(k[i]), int(nu[i, t])] += 1
    return mat


def power_law(n, minval, maxval, alpha, seed=None):
    """
    Generates a power law distribution of random integers.

    Parameters
    ----------
    n : int
        The number of random integers to generate.
    minval : int
        The minimum value of the random integers.
    maxval : int
        The maximum value of the random integers.
    alpha : float
        The exponent of the power law distribution.
    seed : int, optional
        The seed value for the random number generator.

    Returns
    -------
    numpy.ndarray
        An array of random integers following a power law distribution.

    """
    if seed is not None:
        np.random.seed(seed)
    u = np.random.random(n)
    a = minval ** (1 + alpha)
    b = maxval ** (1 + alpha)
    return np.round((a + u * (b - a)) ** (1 / (1 + alpha))).astype(int)


def mean_power_law(minval, maxval, alpha):
    if alpha == -1:
        num = maxval - minval
        den = np.log(maxval) - np.log(minval)
        return num / den
    elif alpha == -2:
        num = np.log(maxval) - np.log(minval)
        den = 1 / minval - 1 / maxval
        return num / den
    else:
        num = (minval ** (2 + alpha) - maxval ** (2 + alpha)) / (-alpha - 2)
        den = (minval ** (1 + alpha) - maxval ** (1 + alpha)) / (-alpha - 1)
        return num / den


def delta_dist(x_prime):
    """
    A dirac delta distribution \delta(x_prime)
    """
    return rv_discrete(name="custom", values=([x_prime], [1.0]))


def robbins_monro_solve(
    f,
    x0,
    a=0.02,
    alpha=1,
    max_iter=100,
    tol=1e-2,
    loss="function",
    verbose=False,
    return_values=False,
):
    """Solves an optimization problem using the Robbins-Monro algorithm.

    Parameters
    ----------
    f : function
        The objective function to be minimized.
    x0 : float
        The initial value of the optimization variable.
    a : float, optional
        The step size parameter. Default is 0.02.
    alpha : float, optional
        The exponent parameter for the step size decay. Default is 1.
    max_iter : int, optional
        The maximum number of iterations. Default is 100.
    tol : float, optional
        The tolerance for convergence. Default is 1e-2.
    loss : str, optional
        The type of loss to be used for convergence. Default is "function".
    verbose : bool, optional
        Whether to print iteration information. Default is False.
    return_values : bool, optional
        Whether to return the optimization trajectory. Default is False.

    Returns
    -------
    float or tuple
        The optimized value of the variable if return_values is False,
        otherwise a tuple containing the optimized value, the trajectory of
        the variable, and the trajectory of the objective function.

    Raises
    ------
    Exception
        If an invalid loss type is provided.

    """
    x = x0
    val = f(x0)

    xvec = [x]
    fvec = [val]
    diff = np.inf
    it = 1
    while diff > tol and it <= max_iter:
        a_n = a * it**-alpha
        x -= a_n * val
        x = np.clip(x, 0, 1)
        val = f(x)
        xvec.append(x)  # save results
        fvec.append(val)
        match loss:
            case "arg":
                diff = abs(x - xvec[it - 1])
            case "function":
                diff = abs(val)
            case _:
                raise Exception("Invalid loss type!")

        if verbose:
            print((it, x, diff), flush=True)
        it += 1
    if return_values:
        return x, xvec, fvec
    else:
        return x


def ipn_func(b, ipn_target, cf, gamma, A, rho0, realizations, tmax, mode):
    """
    Calculates the number of infections per node for a contagion process with a given set of parameters.

    Parameters:
    ----------
    b : float
       beta, the maximum infection rate for the threshold contagion
    ipn_target : float
        The target number of infections per node.
    cf :  function
        The contagion function that determines the contagion rates, 
    gamma : float
        The healing rate.
    A : ndarray
        The adjacency matrix of the network.
    rho0 : float
        The initial density of activated nodes.
    realizations : int
        The number of realizations to perform.
    tmax : int
        The maximum time step for the contagion process.
    mode : str
        The mode for calculating infections per node.

    Returns:
    -------
    float
        The difference between the calculated number of infections per node and the target number of infections per node.
    """
    n = A.shape[0]

    x0 = np.zeros(n)
    x0[list(random.sample(range(n), int(rho0 * n)))] = 1

    c = cf(np.arange(n), b)

    ipn = 0
    for _ in range(realizations):
        x = contagion_process(A, gamma, c, x0, tmin=0, tmax=tmax)
        ipn += infections_per_node(x, mode) / realizations
    return ipn - ipn_target


def fit_ipn(b0, ipn_target, cf, gamma, A, rho0, tmax, mode):
    """
    Fit a model to result in a target number of infections per node(IPN). Adjucst the value of b to m
    
    b0 : float
       beta, the initial maximum infection rate for the threshold contagion
    ipn_target : float
        The target number of infections per node.
    cf : Numpy ndarray
        The non-parametric contagion function that determines the contagion rates, 
    gamma : float
        The healing rate.
    A : ndarray
        The adjacency matrix of the network.
    rho0 : float
        The initial density of activated nodes.
    realizations : int
        The number of realizations to perform.
    tmax : int
        The maximum time step for the contagion process.
    mode : str
        The mode for calculating infections per node.

    Returns:
    -------
    float
       The scaled b value to generate a target number of infections per node 
    """
    f = lambda b: ipn_func(b, ipn_target, cf, gamma, A, rho0, 1, tmax, mode)
    bscaled = robbins_monro_solve(f, b0, verbose=True)

    f = lambda b: ipn_func(b, ipn_target, cf, gamma, A, rho0, 10, tmax, mode)
    bscaled = robbins_monro_solve(f, bscaled, verbose=True)

    f = lambda b: ipn_func(b, ipn_target, cf, gamma, A, rho0, 100, tmax, mode)
    bscaled = robbins_monro_solve(f, bscaled, verbose=True)

    return bscaled


def target_ipn(A, gamma, c, mode, rho0, tmax, realizations):
    n = A.shape[0]
    x0 = np.zeros(n)
    x0[random.sample(range(n), round(rho0 * n))] = 1
    ipn = 0
    for _ in range(realizations):
        x = contagion_process(A, gamma, c, x0, tmax=tmax)
        ipn += infections_per_node(x, mode) / realizations
    return ipn
