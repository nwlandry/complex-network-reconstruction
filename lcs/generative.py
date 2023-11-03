import random

import networkx as nx
import numpy as np
import xgi


def erdos_renyi(n, p, seed=None):
    return nx.adjacency_matrix(nx.fast_gnp_random_graph(n, p, seed)).todense()


def watts_strogatz(n, k, p, seed=None):
    return nx.adjacency_matrix(nx.watts_strogatz_graph(n, k, p, seed)).todense()


def sbm(n, k, epsilon, seed=None):
    p = k / (n - 1)
    # ratio of inter- to intra-community edges
    p_in = (1 + epsilon) * p
    p_out = (1 - epsilon) * p
    return nx.adjacency_matrix(
        nx.planted_partition_graph(2, int(n / 2), p_in, p_out, seed=seed)
    ).todense()


def projected_bipartite(k, s, seed=None):
    H = xgi.chung_lu_hypergraph(k, s, seed)
    return xgi.adjacency_matrix(H, sparse=False)
