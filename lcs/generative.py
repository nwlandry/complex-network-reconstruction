import random
from itertools import chain

import networkx as nx
import numpy as np

chaini = chain.from_iterable


def zkc(format="adjacency"):
    match format:
        case "adjacency":
            G = nx.karate_club_graph()
            return nx.adjacency_matrix(G, weight=None).todense()
        case "edgelist":
            G = nx.karate_club_graph()
            return [[i, j] for i, j in G.edges]


def erdos_renyi(n, p, seed=None):
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
    G = nx.watts_strogatz_graph(n, k, p, seed)
    G.add_nodes_from(range(n))
    return nx.adjacency_matrix(G).todense()


def watts_strogatz_edge_swap(n, k, p, seed=None):
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
    p = k / (n - 1)
    # ratio of inter- to intra-community edges
    p_in = (1 + epsilon) * p
    p_out = (1 - epsilon) * p
    G = nx.planted_partition_graph(2, int(n / 2), p_in, p_out, seed=seed)
    G.add_nodes_from(range(n))
    return nx.adjacency_matrix(G).todense()


def clustered_network(k1, k2, seed=None):
    """
    inputs:
        k1: the number of cliques to which each node belongs
        k2: the clique sizes
        seed: seed for pseudorandom number generator
    output:
        A : an adjacency matrix with multiedges and loops removed.
    """
    if seed is not None:
        random.seed(seed)

    n1 = len(k1)
    n2 = len(k2)

    k1 = np.array(k1, dtype=int)
    k2 = np.array(k2, dtype=int)

    if sum(k1) > sum(k2):
        missing = (k1 - k2).sum()
        k1[random.sample(range(n1), missing)] += 1
    if sum(k1) < sum(k2):
        missing = (k2 - k1).sum()
        k2[random.sample(range(n2), missing)] += 1

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


def truncated_power_law_configuration(n, kmin, kmax, p, seed=None):
    """
    Generates a bipartite graph with a truncated power-law degree distribution.

    Parameters:
    - n (int): Number of nodes in the graph.
    - kmin (int): Minimum degree value.
    - kmax (int): Maximum degree value.
    - p (float): Power-law exponent.
    - seed (int, optional): Seed for the random number generator.

    Returns:
    - G (networkx.Graph): Graph with the specified degree distribution.
    """
    from .utilities import power_law

    if seed is not None:
        random.seed(seed)

    k = power_law(n, kmin, kmax, p)
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
