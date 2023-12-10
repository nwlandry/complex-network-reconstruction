import random

import networkx as nx
import numpy as np


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


def generate_bipartite_edge_list(n_groups, n_inds, p_dist, g_dist, seed=None):
    """
    generate_bipartite_edge_list(): generates a hypergraph in the style of Newman's model in "Community Structure in social and biological networks"

    inputs:
        n_groups: the number of groups or cliques to create
        n_inds: the number of individuals to create(may be less than this total)
        p_dist: The distribution of clique sizes, must be from numpy.random
        g_dist: The distribution of number of cliques belonged to per individual
        seed: seed for pseudorandom number generator
    output:
        edge_list: the edge list for a bi-partite graph. The first n-indices represent the clique edges and the rest represent individuals
    """

    # generate rng with seed
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    chairs = []
    butts = []

    # generate chairs

    for i in range(1, n_inds + 1):
        g_m = g_dist.rvs() + 1  # select the number of butts in clique i
        butts.extend([i] * g_m)  # add g_m butts to individuals

    for i in range(1, n_groups + 1):
        p_n = p_dist.rvs()  # select the number of chairs in clique i
        p_n = int(
            p_n if i < n_groups else len(butts) - len(chairs)
        )  # pull a random length or select a length to make the two lists equal if we are bout to go over
        print(p_n)
        chairs.extend([i] * p_n)  # add p_n chairs belonging to clique i
    chairs = [chair + n_inds for chair in chairs]

    # shuffle the lists
    rng.shuffle(chairs)
    rng.shuffle(butts)

    # generate edge_list
    edge_list = list(zip(chairs, butts))
    edge_list = [(int(edge[0]), int(edge[1])) for edge in edge_list]

    # create vertex meta_data, if the index is a clique, give it a 0, if the vertex is in individual give it a 1
    vertex_attributes = {i: 1 if i <= max(butts) else 2 for i in set(chairs + butts)}

    return edge_list, vertex_attributes


def bipartite_graph(edge_list):
    B = nx.Graph()
    a = np.vstack(edge_list)
    node_list1, node_list2 = np.unique(a[:, 1]), np.unique(a[:, 0])
    B.add_nodes_from(node_list1, bipartite=0)
    B.add_nodes_from(node_list2, bipartite=1)
    B.add_edges_from(edge_list)
    return B


def clustered_unipartite(n_groups, n_ind, my_p_dist, my_g_dist, **kwargs):
    edge_list, vertex_attributes = generate_bipartite_edge_list(
        n_groups, n_ind, my_p_dist, my_g_dist
    )
    projected_nodes = [
        k for k, v in vertex_attributes.items() if v == 1
    ]  # identify ndes to project graph onto
    B = bipartite_graph(edge_list)
    U = nx.projected_graph(B, projected_nodes)  # create unipartite projection
    return nx.adjacency_matrix(U).todense()


def truncated_power_law_configuration(n, x_min, x_max, r, seed=None):
    """
    Generates a bipartite graph with a truncated power-law degree distribution.

    Parameters:
    - n (int): Number of nodes in the graph.
    - x_min (int): Minimum degree value.
    - x_max (int): Maximum degree value.
    - r (float): Power-law exponent.
    - seed (int, optional): Seed for the random number generator.

    Returns:
    - G (networkx.Graph): Graph with the specified degree distribution.
    """
    from .utilities import power_law
    
    if seed is not None:
        random.seed(seed)

    degree_sequence = np.round(power_law(n, x_min, x_max, r)).astype(int)
    if np.sum(degree_sequence) % 2 == 1:
        degree_sequence = np.append(degree_sequence, 1)

    G = nx.configuration_model(degree_sequence)
    G = nx.Graph(G)
    G.remove_edges_from(nx.selfloop_edges(G))
    return G
