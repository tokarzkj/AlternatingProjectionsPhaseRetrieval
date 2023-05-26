import sys
sys.path.append("../AlternatingProjectionsPhaseRetrieval")

import numpy as np
import scipy

import measurement


def calculate_measurements_using_python():
    m = 100
    N = 1000
    A = np.zeros((m, N), dtype=np.complex_)

    diag = np.zeros((N, N))
    for i in range(0, N):
        diag[i][i] = 1

    vec = np.random.rand(N) + 1J * np.random.rand(N)
    for i in range(0, int(m / N)):
        shift = int(i * np.round(N / (m / N)))
        shifted_mask = np.roll(vec, shift)
        A[i * N: (i * N) + N] = np.matmul(scipy.linalg.dft(N), diag * shifted_mask)


def calculate_measurements_using_numpy():
    m = 100
    N = 1000
    A_fft = np.zeros((m, N), dtype=np.complex_)

    diag = np.zeros((N, N))
    for i in range(0, N):
        diag[i][i] = 1

    vec = np.random.rand(N) + 1J * np.random.rand(N)

    for i in range(0, int(m / N)):
        shift = int(i * np.round(N / (m / N)))
        shifted_mask = np.roll(vec, shift)

        diag_mask = np.multiply(diag, shifted_mask)
        A_fft[i * N: (i * N) + N] = np.matmul(scipy.linalg.dft(N), diag_mask)

def calculate_measurements_using_dftmtx():
    N = 100
    m = 8 * N
    A = np.zeros((m, N), dtype=np.complex_)

    diag = np.zeros((N, N))
    for i in range(0, N):
        diag[i][i] = 1

    for _ in range(0, 600):
        vec = np.random.rand(N) + 1J * np.random.rand(N)
        for i in range(0, int(m / N)):
            shift = int(i * np.round(N / (m / N)))
            shifted_mask = np.roll(vec, shift)
            A[i * N: (i * N) + N] = np.matmul(scipy.linalg.dft(N), diag * shifted_mask)


def calculate_measurements_using_fft():
        N = 100
        m = 8 * N
        A = np.zeros((m, N), dtype=np.complex_)
        A_fft = np.zeros((m, N), dtype=np.complex_)

        diag = np.zeros((N, N))
        for i in range(0, N):
            diag[i][i] = 1

        for _ in range(0, 600):
            vec = np.random.rand(N) + 1J * np.random.rand(N)
            for i in range(0, int(m / N)):
                shift = int(i * np.round(N / (m / N)))
                shifted_mask = np.roll(vec, shift)

                # Confirm this is equivalent to python's diag * shifted_mask
                diag_mask = np.multiply(diag, shifted_mask)
                temp = np.zeros((N, N), dtype=np.complex_)
                for c in range(0, N):
                    col = scipy.fft.fft(diag_mask[c, : N])
                    temp[c] = col

                A_fft[i * N: (i * N) + N] = temp.T



__benchmarks__ = [
    (calculate_measurements_using_python, calculate_measurements_using_numpy, "Using python vs numpy to create our diagonal mask"),
    (calculate_measurements_using_dftmtx, calculate_measurements_using_fft, "Using dft matrix vs fft"),
]