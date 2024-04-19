import numpy as np
from scipy.special import binom
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)


def posterior_similarity(samples, A):
    """
    Calculate the posterior similarity between the given samples and a target matrix A.

    Parameters
    ----------
    samples : numpy.ndarray
        An array of samples.
    A : numpy.ndarray
        The target matrix.

    Returns
    -------
    float
        The posterior similarity value between 0 and 1.
    """
    meanA = samples.mean(axis=0)
    num = np.sum(np.abs(A - meanA))
    den = np.sum(np.abs(A + meanA))
    if den > 0:
        return 1 - num / den
    else:
        return 1


def samplewise_posterior_similarity(samples, A):
    """
    Calculate the posterior similarity between the given samples and a target matrix A.

    Parameters
    ----------
    samples : numpy.ndarray
        An array of sample adjacency matrices.
    A : numpy.ndarray
        The adjacency target matrix.

    Returns
    -------
    float
        The posterior similarity value between 0 and 1.
    """
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
    """
    Calculate the F-score, a measure of the accuracy of a binary classifier, given the precision and recall.

    Parameters
    ----------
    samples : numpy.ndarray
        An array of sample adjacency matrices.
    A : numpy.ndarray
        The adjacency target matrix.
    normalize : bool, optional
        Whether to normalize the F-score. Default is False.
    rho_guess : float, optional
        The guess for the density. Default is 0.5.

    Returns
    -------
    float
        The F-score.
    """
    p = precision(samples, A)
    r = recall(samples, A)

    if np.isnan(p) or np.isnan(r) or p + r == 0:
        f = np.nan
    else:
        f = 2 * p * r / (p + r)

    if normalize:
        rho = density(A)
        # https://stats.stackexchange.com/questions/390200/what-is-the-baseline-of-the-f1-score-for-a-binary-classifier
        if rho + rho_guess > 0:
            f_random = 2 * rho * rho_guess / (rho + rho_guess)
        else:
            return np.nan

        return f / f_random
    else:
        return f


def precision(samples, A):
    """
    Calculate the precision of the network reconstruction.

    Parameters
    ----------
    samples : numpy.ndarray
        An array of sample adjacency matrices.
    A : numpy.ndarray
        The adjacency target matrix.

    Returns
    -------
    float
        The precision score.
    """
    Q = samples.mean(axis=0)
    tp = np.sum(Q * A)
    fp = np.sum(Q * (1 - A))
    if tp + fp > 0:
        return tp / (tp + fp)
    else:
        return np.nan


def recall(samples, A):
    """
    Calculate the recall of the network reconstruction.

    Parameters
    ----------
    samples : numpy.ndarray
        An array of sample adjacency matrices.
    A : numpy.ndarray
        The adjacency target matrix.

    Returns
    -------
    float
        The recall score.
    """
    Q = samples.mean(axis=0)
    tp = np.sum(Q * A)
    fn = np.sum((1 - Q) * A)
    if tp + fn > 0:
        return tp / (tp + fn)
    else:
        return np.nan


def fraction_of_correct_entries(samples, A, normalize=False, rho_guess=0.5):
    """
    Calculate the fraction of correct entries in a matrix.

    Parameters
    ----------
    samples : numpy.ndarray
        An array of sample adjacency matrices.
    A : numpy.ndarray
        The adjacency target matrix.
    normalize : bool, optional
        Whether to normalize the fraction of correct entries. Default is False.
    rho_guess : float, optional
        The guess for the density of the ground truth matrix. Default is 0.5.

    Returns
    -------
    float
        The fraction of correct entries in the matrix. If `normalize` is True, the fraction is normalized.
    """
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


def nodal_performance(Q, A, norm=1):
    """
    Calculate the nodal performance of a network.

    Parameters
    ----------
    Q : ndarray
        The predicted values for each node in the network.
    A : ndarray
        The actual values for each node in the network.

    Returns
    -------
    ndarray
        The nodal performance, the average discrepancy for all edges and non edgees connecting to a given node.
    """ 
    return 1 - (np.abs((Q - A) ** norm).sum(axis=0) / A.shape[0]) ** (1.0 / norm)


def density(A):
    """
    Calculate the density of a graph represented by an adjacency matrix.

    Parameters
    ----------
    A : numpy.ndarray
        The adjacency matrix of the graph.

    Returns
    -------
    float
        The density of the graph.
    """
    n = A.shape[0]
    return (A.sum() / 2) / binom(n, 2)


def hamming_distance(A1, A2):
    """
    Calculate the Hamming distance between two adjacency matrices.

    Parameters
    ----------
    A1 : numpy.ndarray
        First matrix.
    A2 : numpy.ndarray
        Second matrix.

    Returns
    -------
    float
        The Hamming distance between the two matrices.
    """
    return np.sum(np.abs(A1 - A2)) / 2


def auroc(samples, A):
    """
    Calculate the Area Under the Receiver Operating Characteristic Curve (AUROC) for a given set of samples and adjacency matrix.

    Parameters
    ----------
    samples : numpy.ndarray
        Array of shape (m, n) representing m samples of n-dimensional feature vectors.
    A : numpy.ndarray
        Adjacency matrix of shape (n, n) representing the binary connections between nodes.

    Returns
    -------
    float
        The AUROC value.
    """
    n = A.shape[0]
    Q = samples.mean(axis=0)
    y_true = A[np.tril_indices(n, -1)]
    y_score = Q[np.tril_indices(n, -1)]
    if len(np.unique(y_true)) == 1:
        return np.nan
    return roc_auc_score(y_true, y_score)


def auprc(samples, A):
    """
    Calculate the Area Under the Precision-Recall Curve (AUPRC) for a given set of samples and an adjacency matrix.

    Parameters
    ----------
    samples : numpy.ndarray
        An array of sample adjacency matrices.
    A : numpy.ndarray
        The adjacency target matrix.

    Returns
    -------
    float
        The AUPRC score.
    """
    n = A.shape[0]
    Q = samples.mean(axis=0)
    y_true = A[np.tril_indices(n, -1)]
    y_scores = Q[np.tril_indices(n, -1)]
    # Check if y_true contains only one class
    if len(np.unique(y_true)) == 1:
        return np.nan
    # Compute precision-recall pairs for different probability thresholds
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    # Calculate the area under the precision-recall curve
    auprc_score = auc(recall, precision)

    return auprc_score


def degrees(A):
    """
    Calculate the degree of each node in a graph represented by the adjacency matrix A.

    Parameters
    ----------
    A : numpy.ndarray or scipy.sparse.csr_matrix
        The adjacency matrix of the graph.

    Returns
    -------
    numpy.ndarray
        An array containing the degree of each node in the graph.
    """
    if not isinstance(A, np.ndarray):
        A = A.todense()
    return A.sum(axis=0)


def clustering_coefficient(A):
    """
    Calculate the clustering coefficient of a graph represented by an adjacency matrix.

    Parameters
    ----------
    A : numpy.ndarray
        The adjacency matrix of the graph.

    Returns
    -------
    numpy.ndarray
        The clustering coefficient of each node in the graph.
    """
    T = np.diag(A @ A @ A)
    k = degrees(A)
    D = np.multiply(k, k - 1)
    C = np.divide(T, D, out=np.zeros_like(T), where=D != 0)
    return C
