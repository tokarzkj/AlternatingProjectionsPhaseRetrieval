import numpy as np
import scipy
from numpy import imag


def signum(value):
    # np.sign's complex implementation is different from matlab's. Changing to accommodate that difference.
    if imag(value) == 0J:
        return np.sign(value)
    else:
        return value / np.abs(value)


def create_measurement_matrix(m, N):
    A = np.zeros((m, N), dtype=np.complex_)

    # Create a diagonal matrix of 1s
    diag = np.zeros((N, N))
    for i in range(0, N):
        diag[i][i] = 1

    for i in range(0, int(m / N)):
        A[i * N: (i * N) + N] = np.matmul(scipy.linalg.dft(N), diag * (np.random.rand(N) + 1J * np.random.rand(N)))

    return A


def alternate_phase_projection(N, m, number_iterations, seed, do_add_noise):
    if len(seed) > 0:
        seed = int(seed)
        np.random.seed(seed)

    x = np.random.rand(N) + 1J * np.random.rand(N)

    A = create_measurement_matrix(m, N)
    inverse_A = scipy.linalg.pinv(A)

    # Measurements (magnitude of masked DFT coefficients)
    b = np.abs(np.matmul(A, x))

    if do_add_noise:
        snr = 40    # 40db of noise
        signal_power = np.square(np.linalg.norm(b)) / len(b)
        noise_power = signal_power / np.power(10, snr/10)
        noise = np.sqrt(noise_power) * np.random.rand(len(b))
        b = np.add(b, noise)

    x_recon = np.random.rand(N) + 1J * np.random.rand(N)

    for i in range(0, number_iterations):
        temp = np.array(list(map(signum, np.matmul(A, x_recon))), dtype=np.complex_)
        x_recon = np.matmul(inverse_A, np.multiply(b, temp))

    phasefac = np.matmul(np.conjugate(x_recon).T, x) / np.matmul(np.conjugate(x).T, x)
    x_recon = np.multiply(x_recon, signum(phasefac))

    error = np.linalg.norm(x - x_recon) / np.linalg.norm(x)

    return x, x_recon, phasefac, error