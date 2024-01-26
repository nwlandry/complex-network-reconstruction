from lcs import *


def test_erdos_renyi():
    A = erdos_renyi(10, 0)
    assert A.shape == (10, 10)
    assert np.all(A == np.zeros((10, 10)))

    A = erdos_renyi(10, 1)

    assert np.all(A == np.ones((10, 10)) - np.eye(10))

    A = erdos_renyi(10, 0.3, seed=0)
    assert A.sum() == 16



def test_unipartite_clustering():
    n = 21
    clique_numbers = np.arange(1, 20)
    clique_number = 5
    clique_size = n // clique_number  # the number of nodes per clique
    clique_membership = 1  # the number of cliques per node
    my_p_dist = delta_dist(clique_size)
    my_g_dist = delta_dist(clique_membership)

    
    k1 = [clique_membership]*n
    k2 = [clique_size]*clique_number

    #A = clustered_unipartite(clique_number, n, my_p_dist, my_g_dist)
    A = clustered_network(k1,k2)

    G = nx.from_numpy_array(A)
    # Calculate the Laplacian matrix
    L = nx.laplacian_matrix(G).toarray()
    # Calculate the eigenvalues of the Laplacian matrix
    eigenvalues = np.linalg.eigvals(L)
    num_zeros = np.count_nonzero(
        np.isclose(eigenvalues, 0)
    )  # this should be equal to the clique number
    breakpoint()
    assert num_zeros == clique_number


def test_truncated_power_law_configuration_model():
    n = 500
    x_min = 1
    x_max = n
    r = 2
    A = truncated_power_law_configuration(n, x_min, x_max, r)
    G = nx.from_numpy_array(A)
    max_degree = max(dict(G.degree()).values())
    assert max_degree <= x_max
