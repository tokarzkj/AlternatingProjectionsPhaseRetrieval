import numpy as np
import scipy
from numpy import imag

from utilities import perturb_vec, simulate_noise_in_measurement


def alternating_phase_projection_recovery(N, m, number_iterations, seed, do_add_noise: bool,
                                          x=None, mask=None):
    """
    This is the basic algorithm for taking a signal with specified parameters and attempting to
    reconstruct using our simulated measurements
    :param N: Length of signal
    :param m: Number of masks
    :param number_iterations: Number of iterations for reconstruction process
    :param seed: seed for the random number generator
    :param do_add_noise: Add noise to the phase-less measurement vector
    :param x: The signal to use for the recovery. Default value is None and random signal of length N is constructed
    :param mask: The mask to use for the recovery. Default value is None and random mask of length N is constructed
    :return: Returns a tuple with the signal, reconstructed signal, phase factors, and the error
    """
    if (isinstance(seed, str) and len(seed) > 0) or seed > 0:
        seed = int(seed)
        np.random.seed(seed)

    if x is None:
        x = np.random.rand(N) + 1J * np.random.rand(N)

    if mask is None:
        mask = np.random.rand(N) + 1J * np.random.rand(N)

    A = create_measurement_matrix(m, N, mask)
    inverse_A = scipy.linalg.pinv(A)

    # Measurements (magnitude of masked DFT coefficients)
    b = np.abs(np.matmul(A, x))

    if do_add_noise:
        b = simulate_noise_in_measurement(b)

    x_recon, x_recon_iterations = initial_x_reconstruction_signal(A, N, b, inverse_A, number_iterations)

    phasefac = np.matmul(np.conjugate(x_recon).T, x) / np.matmul(np.conjugate(x).T, x)
    x_recon = np.multiply(x_recon, signum(phasefac))

    error = np.linalg.norm(x - x_recon) / np.linalg.norm(x)

    return x, x_recon, phasefac, error, x_recon_iterations


def modified_alternating_phase_projection_recovery(N, m, number_iterations, seed, do_add_noise: bool,
                                                   x=None, mask=None):
    """
    This is similar to the basic algorithm for taking a signal with specified parameters and attempting to
    reconstruct using our simulated measurements. The major difference is the matrix A is perturbed before
    beginning the recovery process to understand a real-world scenario of only approximately knowing A.
    :param N: Length of signal
    :param m: Number of masks
    :param number_iterations: Number of iterations for reconstruction process
    :param seed: seed for the random number generator
    :param do_add_noise: Add noise to the phase-less measurement vector
    :param x: The signal to use for the recovery. Default value is None and random signal of length N is constructed
    :param mask: The mask to use for the recovery. Default value is None and random mask of length N is constructed
    :return: Returns a tuple with the signal, reconstructed signal, phase factors, and the error
    """
    if (isinstance(seed, str) and len(seed) > 0) or seed > 0:
        seed = int(seed)
        np.random.seed(seed)

    if x is None:
        x = np.random.rand(N) + 1J * np.random.rand(N)

    if mask is None:
        mask = np.random.rand(N) + 1J * np.random.rand(N)

    A = create_measurement_matrix(m, N, mask)

    # Measurements (magnitude of masked DFT coefficients)
    b = np.abs(np.matmul(A, x))

    if do_add_noise:
        b = simulate_noise_in_measurement(b)

    perturbation = np.random.rand(m, N) + 1J * np.random.rand(m, N)
    perturbation = np.multiply(perturbation, 1 / np.power(10, 6))

    perturbed_A = np.subtract(A, perturbation)
    inverse_perturbed_A = scipy.linalg.pinv(perturbed_A)
    x_recon, x_recon_iterations = initial_x_reconstruction_signal(perturbed_A, N, b, inverse_perturbed_A, number_iterations)

    phasefac = np.matmul(np.conjugate(x_recon).T, x) / np.matmul(np.conjugate(x).T, x)
    x_recon = np.multiply(x_recon, signum(phasefac))

    error = np.linalg.norm(x - x_recon) / np.linalg.norm(x)

    return x, x_recon, phasefac, error, x_recon_iterations


def alternating_phase_projection_recovery_with_error_reduction(N, m, number_iterations, seed, do_add_noise: bool,
                                                               x=None, mask=None):
    """
    This is similar to the basic algorithm for taking a signal with specified parameters and attempting to
    reconstruct using our simulated measurements. The major difference is the matrix A is perturbed before
    beginning the recovery process to understand a real-world scenario of only approximately knowing A and
    it assumes the mask and signal are unknown. The error reduction uses a targeted optimization process
    to try to recover the original mask and signal.
    :param N: Length of signal
    :param m: Number of masks
    :param number_iterations: Number of iterations for reconstruction process
    :param seed: seed for the random number generator
    :param do_add_noise: Add noise to the phase-less measurement vector
    :param x: The signal to use for the recovery. Default value is None and random signal of length N is constructed
    :param mask: The mask to use for the recovery. Default value is None and random mask of length N is constructed
    :return: Returns a tuple with the signal, reconstructed signal, phase factors, and the error
    """
    if (isinstance(seed, str) and len(seed) > 0) or seed > 0:
        seed = int(seed)
        np.random.seed(seed)

    if x is None:
        x = np.random.rand(N) + 1J * np.random.rand(N)

    if mask is None:
        mask = np.random.rand(N) + 1J * np.random.rand(N)

    A = create_measurement_matrix(m, N, mask)

    # Measurements (magnitude of masked DFT coefficients)
    b = np.abs(np.matmul(A, x))

    if do_add_noise:
        b = simulate_noise_in_measurement(b)

    x_recon = np.random.rand(N) + 1J * np.random.rand(N)
    mask_approx = perturb_vec(mask)
    progressive_errors = dict()
    for idx in range(0, 100):
        A_approx = create_measurement_matrix(m, N, mask_approx)
        A_pinv = scipy.linalg.pinv(A_approx)
        x_recon, _ = reconstructed_signal(x_recon, A_approx, b, A_pinv, number_iterations)

        M_approx = create_measurement_matrix(m, N, x_recon, True)
        M_pinv = scipy.linalg.pinv(M_approx)

        m_recon = mask_approx
        mask_approx, _ = reconstructed_signal(m_recon, M_approx, b, M_pinv, number_iterations)

        if idx % 10 == 0:
            progressive_errors[idx] = np.linalg.norm(x - x_recon) / np.linalg.norm(x)

    phasefac = np.matmul(np.conjugate(x_recon).T, x) / np.matmul(np.conjugate(x).T, x)
    x_recon = np.multiply(x_recon, signum(phasefac))

    error = np.linalg.norm(x - x_recon) / np.linalg.norm(x)

    return x, x_recon, mask, mask_approx, phasefac, error, progressive_errors


def signum(value):
    # np.sign's complex implementation is different from matlab's. Changing to accommodate that difference.
    if imag(value) == 0J:
        return np.sign(value)
    else:
        return value / np.abs(value)


def create_measurement_matrix(m, N, vec, do_shift_left=False):
    A = np.zeros((m, N), dtype=np.complex_)

    # Create a diagonal matrix of 1s
    diag = np.zeros((N, N))
    for i in range(0, N):
        diag[i][i] = 1

    for i in range(0, int(m / N)):
        shift = int(i * np.round(N / (m / N)))
        if do_shift_left:
            shift = shift * -1

        shifted_mask = np.roll(vec, shift)
        diag_shift = np.multiply(diag, shifted_mask)
        A[i * N: (i * N) + N] = np.matmul(scipy.linalg.dft(N), diag_shift)

    return A


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
    return reconstructed_signal(x_recon, A, b, inverse_A, number_iterations)


def reconstructed_signal(signal, A, b, inverse_A, number_iterations):
    reconstructed_signal_iterations = dict()
    for i in range(0, number_iterations):
        temp = np.array(list(map(signum, np.matmul(A, signal))), dtype=np.complex_)
        signal = np.matmul(inverse_A, np.multiply(b, temp))

        if i % 50 == 0:
            reconstructed_signal_iterations[i] = signal

    return signal, reconstructed_signal_iterations
