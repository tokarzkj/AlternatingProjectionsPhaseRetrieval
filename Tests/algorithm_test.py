import numpy as np
import scipy


def test_numpy_multiply_for_diag_mask_creation():
    N = 25
    m = 4 * N
    A = np.zeros((m, N), dtype=np.complex_)
    A_fft = np.zeros((m, N), dtype=np.complex_)

    diag = np.zeros((N, N))
    for i in range(0, N):
        diag[i][i] = 1

    vec = np.random.rand(N) + 1J * np.random.rand(N)
    for i in range(0, int(m / N)):
        shift = int(i * np.round(N / (m / N)))
        shifted_mask = np.roll(vec, shift)
        A[i * N: (i * N) + N] = np.matmul(scipy.linalg.dft(N), diag * shifted_mask)

    for i in range(0, int(m / N)):
        shift = int(i * np.round(N / (m / N)))
        shifted_mask = np.roll(vec, shift)

        #Confirm this is equivalent to python's diag * shifted_mask
        diag_mask = np.multiply(diag, shifted_mask)
        A_fft[i * N: (i * N) + N] = np.matmul(scipy.linalg.dft(N), diag_mask)

    assert np.array_equiv(A, A_fft)


def test_measurement_matrix_creation_using_fft():
    N = 25
    m = 4 * N
    A = np.zeros((m, N), dtype=np.complex_)
    A_fft = np.zeros((m, N), dtype=np.complex_)

    diag = np.zeros((N, N))
    for i in range(0, N):
        diag[i][i] = 1

    vec = np.random.rand(N) + 1J * np.random.rand(N)
    for i in range(0, int(m / N)):
        shift = int(i * np.round(N / (m / N)))
        shifted_mask = np.roll(vec, shift)
        A[i * N: (i * N) + N] = np.matmul(scipy.linalg.dft(N), diag * shifted_mask)

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

    foundInequality = False
    for idx1 in range(0, m):
        for idx2 in range(0, N):
            if A[idx1, idx2] != A_fft[idx1, idx2]:
                foundInequality = True
                break
        if foundInequality:
            break

    assert foundInequality