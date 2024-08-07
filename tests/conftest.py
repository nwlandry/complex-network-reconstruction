import numpy as np
import pytest


# @pytest.fixture
def mat_A1():
    A = np.zeros((5, 5))
    A[1, 2] = A[2, 1] = 1
    A[1, 4] = A[4, 1] = 1
    A[2, 3] = A[3, 2] = 1
    return A


# @pytest.fixture
def mat_A2():
    A = np.zeros((5, 5))
    A[1, 2] = A[2, 1] = 1
    A[1, 4] = A[4, 1] = 1
    A[2, 3] = A[3, 2] = 1
    A[0, 4] = A[4, 0] = 1
    return A


# @pytest.fixture
def mat_A3():
    A = np.zeros((5, 5))
    A[1, 2] = A[2, 1] = 1
    A[1, 4] = A[4, 1] = 1
    A[2, 3] = A[3, 2] = 1
    A[0, 4] = A[4, 0] = 1
    A[2, 4] = A[4, 2] = 1
    return A


@pytest.fixture
def A1():
    return mat_A1()


@pytest.fixture
def A2():
    return mat_A2()


@pytest.fixture
def A3():
    return mat_A3()


@pytest.fixture
def samples1():
    samples = np.empty((3, 5, 5))
    samples[0] = mat_A1()
    samples[1] = mat_A2()
    samples[2] = mat_A3()

    return samples


@pytest.fixture
def x1():
    x = np.array(
        [
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0],
            [1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
        ]
    )
    return x


@pytest.fixture
def x4():
    x = np.array(
        [
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0],
            [0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0],
            [0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0],
        ]
    )
    return x


@pytest.fixture
def A4():
    A = np.array(
        [
            [0, 1, 1, 1, 0, 0, 0, 1, 0, 0],
            [1, 0, 0, 0, 1, 0, 1, 0, 0, 1],
            [1, 0, 0, 1, 1, 1, 0, 0, 0, 0],
            [1, 0, 1, 0, 0, 0, 1, 1, 0, 1],
            [0, 1, 1, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 1, 0, 0, 0, 1, 1, 1, 1],
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 0],
            [1, 0, 0, 1, 0, 1, 1, 0, 0, 1],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
            [0, 1, 0, 1, 1, 1, 0, 1, 1, 0],
        ]
    )
    return A
