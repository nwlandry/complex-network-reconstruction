from lcs import *


def test_infer_adjacency_matrix(x4, A4):
    samples = infer_adjacency_matrix(
        x4,
        np.zeros_like(A4),
        p_rho=None,
        p_c=None,
        nsamples=5,
        burn_in=100,
        skip=100,
        return_likelihood=False,
    )
    assert samples.shape == (5, 10, 10)


def test_count_all_infection_events(x4, A4):
    # count_all_infection_events(x, A)
    assert np.array_equal(
        count_all_infection_events(x4, A4), count_all_infection_events_loop(x4, A4)
    )


def test_count_local_infection_events(x4, A4):
    assert np.array_equal(
        count_local_infection_events(1, x4, A4),
        count_local_infection_events_loop(1, x4, A4),
    )
