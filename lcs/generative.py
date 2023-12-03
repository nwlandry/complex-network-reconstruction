import random

import networkx as nx
import numpy as np
import xgi
from scipy.stats import rv_discrete


def zkc():
    G = nx.karate_club_graph()
    return nx.adjacency_matrix(G).todense()


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


def delta_dist(x_prime):
    return rv_discrete(name = 'custom',values = ([x_prime],[1.]))


def generate_bipartite_edge_list(N_groups, N_inds, p_dist, g_dist,seed = None):
    """
    generate_bipartite_edge_list(): generates a hypergraph in the style of Newman's model in "Community Structure in social and biological networks"
    inputs:
        N_groups: the number of groups or cliques to create
        N_inds: the number of individuals to create(may be less than this total)
        p_dist: The distribution of clique sizes, must be from numpy.random
        g_dist: The distribution of number of cliques belonged to per individual
        seed: seed for pseudorandom number generator
    output:
        edge_list: the edge list for a bi-partite graph. The first n-indices represent the clique edges and the rest represent individuals
    """

    #generate rng with seed
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()


    chairs = []
    butts = []

    # generate chairs

    for i in range(1, N_inds + 1):
        g_m = g_dist.rvs() + 1  # select the number of butts in clique i
        butts.extend([i for _ in range(g_m)])  # add g_m butts to individuals

    for i in range(1, N_groups + 1):
        p_n = p_dist.rvs()  # select the number of chairs in clique i
        #p_n = int(p_n if len(chairs) + p_n <= len(butts) else len(butts) - len(chairs))  # pull a random length or select a length to make the two lists equal if we are bout to go over
        p_n = int(p_n if i < N_groups else len(butts) - len(chairs))  # pull a random length or select a length to make the two lists equal if we are bout to go over
        print(p_n)
        chairs.extend([i for _ in range(int(p_n))])  # add p_n chairs belonging to clique i
        #chairs.extend([chairs[-1] for i in range(len(butts) - len(chairs))])
    chairs = [chair + N_inds for chair in chairs]

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
    node_list1,node_list2 = np.unique(a[:,1]),np.unique(a[:,0])
    B.add_nodes_from(node_list1,bipartite=0)
    B.add_nodes_from(node_list2,bipartite=1)
    B.add_edges_from(edge_list)
    return B


def clustered_unipartite(n_groups,n_ind,my_p_dist,my_g_dist,**kwargs):
    edge_list,vertex_attributes = generate_bipartite_edge_list(n_groups,n_ind,my_p_dist,my_g_dist)
    projected_nodes = [k for k,v in vertex_attributes.items() if v == 1]#identify ndes to project graph onto
    B = bipartite_graph(edge_list)
    U = nx.projected_graph(B,projected_nodes)#create unipartite projection
    return nx.adjacency_matrix(U).todense()
