import random

import networkx as nx
import numpy as np
import xgi


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


def sbm(n, k, epsilon, seed=None):
    p = k / (n - 1)
    # ratio of inter- to intra-community edges
    p_in = (1 + epsilon) * p
    p_out = (1 - epsilon) * p
    G = nx.planted_partition_graph(2, int(n / 2), p_in, p_out, seed=seed)
    G.add_nodes_from(range(n))
    return nx.adjacency_matrix(G).todense()


def projected_bipartite(k, s, seed=None):
    return 0
