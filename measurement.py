import numpy as np
import scipy
from numpy import imag


def create_measurement_matrix(m, N):
    A = np.zeros((m, N), dtype=np.complex_)

    # Create a diagonal matrix of 1s
    diag = np.zeros((N, N))
    for i in range(0, N):
        diag[i][i] = 1

    for i in range(0, int(m / N)):
        A[i * N: (i * N) + N] = np.matmul(scipy.linalg.dft(N), diag * (np.random.rand(N) + 1J * np.random.rand(N)))

    return A

