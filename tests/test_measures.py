from lcs import *


def test_posterior_similarity(A1, A2, A3, samples1):
    assert np.isclose(posterior_similarity(samples1, A1), 0.857142857142857)
    assert np.isclose(posterior_similarity(samples1, A2), 0.916666666666666)
    assert np.isclose(posterior_similarity(samples1, A3), 0.888888888888888)


def test_samplewise_posterior_similarity(A1, A2, A3, samples1):
    assert np.isclose(samplewise_posterior_similarity(samples1, A1), 0.869047619047619)
    assert np.isclose(samplewise_posterior_similarity(samples1, A2), 0.915343915343915)
    assert np.isclose(samplewise_posterior_similarity(samples1, A3), 0.879629629629629)


def test_hamming_distance(A1, A2, A3):
    assert hamming_distance(A1, A2) == 1
    assert hamming_distance(A1, A1) == 0
    assert hamming_distance(A1, A3) == 2
    assert hamming_distance(A2, A3) == 1


def test_precision(A1, A2, A3, samples1):
    complete_network_samples = np.array([erdos_renyi(10, 1.0) for i in range(20)])
    empty_network_samples = np.zeros([20, 10, 10])
    complete_network = erdos_renyi(10, 1.0)
    empty_network = erdos_renyi(10, 0.0)

    pr = precision(complete_network_samples, empty_network)
    assert pr == 0.0

    pr = precision(complete_network_samples, complete_network)
    assert pr == 1.0

    pr = precision(empty_network_samples, empty_network)
    assert np.isnan(pr)

    pr = precision(empty_network_samples, complete_network)
    assert np.isnan(pr)


def test_recall(A1, A2, A3, samples1):
    complete_network_samples = np.array([erdos_renyi(10, 1.0) for i in range(20)])
    empty_network_samples = np.zeros([20, 10, 10])
    complete_network = erdos_renyi(10, 1.0)
    empty_network = erdos_renyi(10, 0.0)

    pr = recall(complete_network_samples, empty_network)
    assert np.isnan(pr)

    pr = recall(complete_network_samples, complete_network)
    assert pr == 1.0

    pr = recall(empty_network_samples, empty_network)
    assert np.isnan(pr)

    pr = recall(empty_network_samples, complete_network)
    assert pr == 0.0
