from lcs import *


def test_posterior_similarity(A1, A2, A3, samples1):
    assert np.isclose(posterior_similarity(A1, samples1), 0.857142857142857)
    assert np.isclose(posterior_similarity(A2, samples1), 0.916666666666666)
    assert np.isclose(posterior_similarity(A3, samples1), 0.888888888888888)


def test_samplewise_posterior_similarity(A1, A2, A3, samples1):
    assert np.isclose(samplewise_posterior_similarity(A1, samples1), 0.869047619047619)
    assert np.isclose(samplewise_posterior_similarity(A2, samples1), 0.915343915343915)
    assert np.isclose(samplewise_posterior_similarity(A3, samples1), 0.879629629629629)


def test_hamming_distance(A1, A2, A3):
    assert hamming_distance(A1, A2) == 1
    assert hamming_distance(A1, A1) == 0
    assert hamming_distance(A1, A3) == 2
    assert hamming_distance(A2, A3) == 1


def test_infections_per_node(x1):
    assert infections_per_node(x1) == 0.3
