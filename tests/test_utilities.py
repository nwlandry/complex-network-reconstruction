from lcs import *


def test_infections_per_node(x1):
    assert infections_per_node(x1) == 0.3
    assert infections_per_node(x1, "max") == 1
    assert infections_per_node(x1, "median") == 0


def test_degrees(A1):
    d1 = degrees(A1)
    true_d1 = np.array([0, 2, 2, 1, 1])
    assert len(d1.shape) == 1
    assert np.all(d1 == true_d1)
