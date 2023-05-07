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

    mask = np.random.rand(N) + 1J * np.random.rand(N)
    for i in range(0, int(m / N)):
        shifted_mask = np.roll(mask, int(i*np.round(N/(m/N))))
        A[i * N: (i * N) + N] = np.matmul(scipy.linalg.dft(N), diag * shifted_mask)

    return A


def alternate_phase_projection(N, m, number_iterations, seed, do_add_noise):
    """
    This is the basic algorithm for taking a signal with specified parameters and attempting to
    reconstruct using our simulated measurements
    :param N: Length of signal
    :param m: Number of masks
    :param number_iterations: Number of iterations for reconstruction process
    :param seed: seed for the random number generator
    :param do_add_noise: Add noise to the phase-less measurement vector
    :return:
    """
    if len(seed) > 0:
        seed = int(seed)
        np.random.seed(seed)

    x = np.random.rand(N) + 1J * np.random.rand(N)

    A = create_measurement_matrix(m, N)
    inverse_A = scipy.linalg.pinv(A)

    # Measurements (magnitude of masked DFT coefficients)
    b = np.abs(np.matmul(A, x))

    if do_add_noise:
        b = simulate_noise_in_measurement(b)

    x_recon = initial_x_reconstruction_signal(A, N, b, inverse_A, number_iterations)

    phasefac = np.matmul(np.conjugate(x_recon).T, x) / np.matmul(np.conjugate(x).T, x)
    x_recon = np.multiply(x_recon, signum(phasefac))

    error = np.linalg.norm(x - x_recon) / np.linalg.norm(x)

    return x, x_recon, phasefac, error


def modified_alternate_phase_projection(N, m, number_iterations, seed, do_add_noise):
    if len(seed) > 0:
        seed = int(seed)
        np.random.seed(seed)

    x = np.random.rand(N) + 1J * np.random.rand(N)

    A = create_measurement_matrix(m, N)


    # Measurements (magnitude of masked DFT coefficients)
    b = np.abs(np.matmul(A, x))

    if do_add_noise:
        b = simulate_noise_in_measurement(b)

    perturbation = np.random.rand(m, N) + 1J * np.random.rand(m, N)
    perturbation = np.multiply(perturbation, 1/np.power(10, 4))

    perturbed_A = np.subtract(A, perturbation)
    inverse_perturbed_A = scipy.linalg.pinv(A)
    x_recon = initial_x_reconstruction_signal(perturbed_A, N, b, inverse_perturbed_A, number_iterations)

    phasefac = np.matmul(np.conjugate(x_recon).T, x) / np.matmul(np.conjugate(x).T, x)
    x_recon = np.multiply(x_recon, signum(phasefac))

    error = np.linalg.norm(x - x_recon) / np.linalg.norm(x)

    return x, x_recon, phasefac, error


def initial_x_reconstruction_signal(A, N, b, inverse_A, number_iterations):
    """
    The core method for reconstructing x
    :param A:
    :param N:
    :param b:
    :param inverse_A:
    :param number_iterations:
    :return:
    """
    x_recon = np.random.rand(N) + 1J * np.random.rand(N)
    for i in range(0, number_iterations):
        temp = np.array(list(map(signum, np.matmul(A, x_recon))), dtype=np.complex_)
        x_recon = np.matmul(inverse_A, np.multiply(b, temp))

    return x_recon


def simulate_noise_in_measurement(b):
    """
    This method simulates real-world noise in measurements
    :param b: The phaseless measurement
    :return: The phaseless measurement with noise
    """
    snr = 40  # 40db of noise
    signal_power = np.square(np.linalg.norm(b)) / len(b)
    noise_power = signal_power / np.power(10, snr / 10)
    noise = np.sqrt(noise_power) * np.random.rand(len(b))
    b = np.add(b, noise)

    return b
