from lcs import *


def test_posterior_similarity(A1, A2, A3, samples1):
    assert np.isclose(posterior_similarity(samples1, A1), 0.857142857142857)
    assert np.isclose(posterior_similarity(samples1, A2), 0.916666666666666)
    assert np.isclose(posterior_similarity(samples1, A3), 0.888888888888888)


def test_samplewise_posterior_similarity(A1, A2, A3, samples1):
    assert np.isclose(f_score(samples1, A1), 0.869047619047619)
    assert np.isclose(f_score(samples1, A2), 0.915343915343915)
    assert np.isclose(f_score(samples1, A3), 0.879629629629629)


def test_hamming_distance(A1, A2, A3):
    assert hamming_distance(A1, A2) == 1
    assert hamming_distance(A1, A1) == 0
    assert hamming_distance(A1, A3) == 2
    assert hamming_distance(A2, A3) == 1


def test_infections_per_node(x1):
    assert infections_per_node(x1) == 0.3
    assert infections_per_node(x1, "max") == 1
    assert infections_per_node(x1, "median") == 0


def test_degrees(A1):
    d1 = degrees(A1)
    true_d1 = np.array([0, 2, 2, 1, 1])
    assert len(d1.shape) == 1
    assert np.all(d1 == true_d1)
