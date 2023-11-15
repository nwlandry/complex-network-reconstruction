from lcs import *


def test_erdos_renyi():
    A = erdos_renyi(10, 0)
    assert A.shape == (10, 10)
    assert np.all(A == np.zeros((10, 10)))

    A = erdos_renyi(10, 1)

    assert np.all(A == np.ones((10, 10)) - np.eye(10))

    A = erdos_renyi(10, 0.3, seed=0)
    assert A.sum() == 16
