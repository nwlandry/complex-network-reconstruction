from lcs import *


def test_contagion_process(A4):
    n = A4.shape[0]

    x = contagion_process(
        A4, 0.1, np.zeros(n), np.ones(n), tmin=0, tmax=5, dt=1, random_seed=None
    )
    assert infections_per_node(x) == 0

    x = contagion_process(
        A4, 1, np.zeros(n), np.ones(n), tmin=0, tmax=5, dt=1, random_seed=None
    )
    assert np.all(x[1] == np.zeros(n))

    x0 = np.zeros(n)
    x0[0] = 1
    x = contagion_process(A4, 0, np.ones(n), x0, tmin=0, tmax=5, dt=1, random_seed=None)
    assert infections_per_node(x) == 0.9
