import random
from itertools import chain

import networkx as nx
import numpy as np
import networkx as nx
from .utilities import power_law

chaini = chain.from_iterable


def zkc(format="adjacency"):
    """
    Generate the Zachary's Karate Club graph.

    Parameters
    ----------
    format : str, optional
        The format of the graph representation to return. Valid options are "adjacency" and "edgelist". Default is "adjacency".

    Returns
    -------
    numpy.ndarray or list
        The graph representation based on the specified format.

    """
    match format:
        case "adjacency":
            G = nx.karate_club_graph()
            return nx.adjacency_matrix(G, weight=None).todense()
        case "edgelist":
            G = nx.karate_club_graph()
            return [[i, j] for i, j in G.edges]


def erdos_renyi(n, p, seed=None):
    """
    Generates an Erdos-Renyi random graph.

    Parameters
    ----------
    n : int
        The number of nodes in the graph.
    p : float
        The probability of an edge between any two nodes.
    seed : int, optional
        The seed value for the random number generator.

    Returns
    -------
    numpy.ndarray
        The adjacency matrix of the generated graph.

    Raises
    ------
    None

    """
    if seed is not None:
        random.seed(seed)

    A = np.zeros((n, n), dtype=int)
    if p == 0:
        return A
    if p == 1:
        return np.ones((n, n), dtype=int) - np.eye(n, dtype=int)

    for i in range(n):
        for j in range(i):
            A[i, j] = A[j, i] = random.random() <= p
    return A



def watts_strogatz(n, k, p, seed=None):
    """
    Generates a Watts-Strogatz graph with `n` nodes, `k` nearest neighbors, and rewiring probability `p`.

    Parameters
    ----------
    n : int
        The number of nodes in the graph.
    k : int
        Each node is connected to `k` nearest neighbors in a ring topology.
    p : float
        The probability of rewiring each edge.
    seed : int, optional
        Seed for random number generator (default: None).

    Returns
    -------
    numpy.ndarray
        The adjacency matrix of the generated graph.

    Raises
    ------
    None

    """
    G = nx.watts_strogatz_graph(n, k, p, seed)
    G.add_nodes_from(range(n))
    return nx.adjacency_matrix(G).todense()


def watts_strogatz_edge_swap(n, k, p, seed=None):
    """
    Generate a Watts-Strogatz graph by performing edge swaps.

    Parameters
    ----------
    n : int
        The number of nodes in the graph.
    k : int
        Each node is connected to k nearest neighbors in a ring topology.
    p : float
        The probability of rewiring each edge.
    seed : int
        The seed value for the random number generator.

    Returns
    -------
    numpy.ndarray
        The adjacency matrix of the generated graph.

    Raises
    ------
    None

    """
    if seed is not None:
        random.seed(seed)

    A = np.zeros((n, n))
    node1 = []
    node2 = []

    nodes = list(range(n))
    for j in range(1, k // 2 + 1):
        targets = nodes[j:] + nodes[0:j]  # first j nodes are now last in list
        node1.extend(nodes)
        node2.extend(targets)

    node1 = np.array(node1)
    node2 = np.array(node2)

    m = node1.shape[0]
    idx = np.random.permutation(range(m))

    node1 = node1[idx]
    node2 = node2[idx]

    for i in range(0, m, 2):
        if random.random() <= p:
            u = node2[i]
            v = node2[i + 1]

            node2[i] = v
            node2[i + 1] = u

    for i, j in zip(node1, node2):
        A[i, j] = A[j, i] = 1
    return A


def sbm(n, k, epsilon, seed=None):
    """
    Generates a Stochastic Block Model (SBM) graph.

    Parameters
    ----------
    n : int
        The number of nodes in the graph.
    k : int
        The average degree of each node.
    epsilon : float
        The parameter controlling the ratio of inter- to intra-community edges.
    seed : int, optional
        The seed for the random number generator. Defaults to None.

    Returns
    -------
    numpy.ndarray
        The adjacency matrix of the generated SBM graph.

    Raises
    ------
    None

    """
    p = k / (n - 1)
    # ratio of inter- to intra-community edges
    p_in = (1 + epsilon) * p
    p_out = (1 - epsilon) * p
    G = nx.planted_partition_graph(2, int(n / 2), p_in, p_out, seed=seed)
    G.add_nodes_from(range(n))
    return nx.adjacency_matrix(G).todense()


def clustered_network(k1, k2, seed=None):
    """
    Generates the unipartite projection of a higher-order network. Degree sequecnes for both sets of the bipartite network are provided and the unipartite projection is generated. 

    Parameters
    ----------
    k1 : list
        The number of cliques to which each node belongs.
    k2 : list
        The clique sizes.
    seed : int, optional
        Seed for pseudorandom number generator.

    Returns
    -------
    numpy.ndarray
        An adjacency matrix with multiedges and loops removed.

    Raises
    ------
    None

    """
    if seed is not None:
        random.seed(seed)

    n1 = len(k1)
    n2 = len(k2)

    k1 = np.array(k1, dtype=int)
    k2 = np.array(k2, dtype=int)

    if k1.sum() > k2.sum():
        missing = k1.sum() - k2.sum()
        k2[random.sample(range(n2), missing)] += 1
    elif k2.sum() > k1.sum():
        missing = k2.sum() - k1.sum()
        k1[random.sample(range(n1), missing)] += 1

    stublist1 = list(chaini([i] * d for i, d in enumerate(k1)))
    stublist2 = list(chaini([i] * d for i, d in enumerate(k2)))

    # shuffle the lists
    random.shuffle(stublist1)
    random.shuffle(stublist2)

    I = np.zeros((n1, n2))
    I[stublist1, stublist2] = 1
    A = I @ I.T > 0
    np.fill_diagonal(A, 0)
    return A


def truncated_power_law_configuration(n, kmin, kmax, alpha, seed=None):
    """
    Generates a bipartite graph with a truncated power-law degree distribution.

    Parameters
    ----------
    n : int
        Number of nodes in the graph.
    kmin : int
        Minimum degree value.
    kmax : int
        Maximum degree value.
    alpha : float
        Power-law exponent.
    seed : int, optional
        Seed for the random number generator.

    Returns
    -------
    networkx.Graph
        Graph with the specified degree distribution.

    Raises
    ------
    None

    """

    if seed is not None:
        random.seed(seed)

    k = power_law(n, kmin, kmax, alpha, seed=seed)
    if np.sum(k) % 2 == 1:
        fixed = False
        while not fixed:
            i = random.randrange(n)
            if k[i] < kmax:
                k[i] += 1
                fixed = True

    stublist = list(chaini([i] * d for i, d in enumerate(k)))

    # algo copied from networkx
    half = len(stublist) // 2
    random.shuffle(stublist)

    A = np.zeros((n, n), dtype=int)
    A[stublist, stublist[half:] + stublist[:half]] = 1
    np.fill_diagonal(A, 0)
    return A
