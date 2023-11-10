import random

import networkx as nx
import numpy as np
import xgi
import math


def erdos_renyi(n, p, seed=None):
    if seed is not None:
        random.seed(seed)

    A = np.zeros((n, n))
    lp = math.log(1.0 - p)
    i = 1
    j = -1
    while i < n:
        lr = math.log(1.0 - random.random())
        i += 1 + int(lr / lp)
        while j >= i and i < n:
            j -= i
            i += 1
        if i < n:
            A[i, j] = A[j, i] = 1
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
    H = xgi.chung_lu_hypergraph(k, s, seed)
    return xgi.adjacency_matrix(H, sparse=False)
