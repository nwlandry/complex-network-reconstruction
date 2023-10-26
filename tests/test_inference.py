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
