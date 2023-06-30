import numpy as np
import scipy
from collections import OrderedDict

from utilities import perturb_vec, simulate_noise_in_measurement, signum


def alternating_phase_projection_recovery(N, m, number_iterations, do_add_noise: bool,
                                          x, mask):
    """
    This is the basic algorithm for taking a signal with specified parameters and attempting to
    reconstruct using our simulated measurements
    :param N: Length of signal
    :param m: Number of masks
    :param number_iterations: Number of iterations for reconstruction process
    :param do_add_noise: Add noise to the phase-less measurement vector
    :param x: The signal to use for the recovery. Default value is None and random signal of length N is constructed
    :param mask: The mask to use for the recovery. Default value is None and random mask of length N is constructed
    :return: Returns a tuple with the reconstructed signal, phase factors, and the error
    """

    A = create_measurement_matrix(m, N, mask)
    inverse_A = scipy.linalg.pinv(A)

    # Measurements (magnitude of masked DFT coefficients)
    b = np.abs(np.matmul(A, x))

    if do_add_noise:
        b = simulate_noise_in_measurement(b)

    x_recon, x_recon_iterations = iterative_signal_reconstruction(A, N, b, inverse_A, number_iterations)

    phasefac = np.matmul(np.conjugate(x_recon).T, x) / np.matmul(np.conjugate(x).T, x)
    x_recon = np.multiply(x_recon, signum(phasefac))

    error = np.linalg.norm(x - x_recon) / np.linalg.norm(x)

    return x_recon, phasefac, error, x_recon_iterations


def modified_alternating_phase_projection_recovery(N, m, number_iterations, do_add_noise: bool,
                                                   x, mask):
    """
    This is similar to the basic algorithm for taking a signal with specified parameters and attempting to
    reconstruct using our simulated measurements. The major difference is the matrix A is perturbed before
    beginning the recovery process to understand a real-world scenario of only approximately knowing A.
    :param N: Length of signal
    :param m: Number of masks
    :param number_iterations: Number of iterations for reconstruction process
    :param do_add_noise: Add noise to the phase-less measurement vector
    :param x: The signal to use for the recovery. Default value is None and random signal of length N is constructed
    :param mask: The mask to use for the recovery. Default value is None and random mask of length N is constructed
    :return: Returns a tuple with the signal, reconstructed signal, phase factors, and the error
    """
    A = create_measurement_matrix(m, N, mask)

    # Measurements (magnitude of masked DFT coefficients)
    b = np.abs(np.matmul(A, x))

    if do_add_noise:
        b = simulate_noise_in_measurement(b)

    perturbation = np.random.rand(m, N) + 1J * np.random.rand(m, N)
    perturbation = np.multiply(perturbation, 1 / np.power(10, 6))

    perturbed_A = np.subtract(A, perturbation)
    inverse_perturbed_A = scipy.linalg.pinv(perturbed_A)
    x_recon, x_recon_iterations = iterative_signal_reconstruction(perturbed_A, N, b, inverse_perturbed_A, number_iterations)

    phasefac = np.matmul(np.conjugate(x_recon).T, x) / np.matmul(np.conjugate(x).T, x)
    x_recon = np.multiply(x_recon, signum(phasefac))

    error = np.linalg.norm(x - x_recon) / np.linalg.norm(x)

    return x, x_recon, phasefac, error, x_recon_iterations


def alternating_phase_projection_recovery_with_error_reduction(N, m, number_iterations, do_add_noise: bool,
                                                               x, mask):
    """
    This is similar to the basic algorithm for taking a signal with specified parameters and attempting to
    reconstruct using our simulated measurements. The major difference is the matrix A is perturbed before
    beginning the recovery process to understand a real-world scenario of only approximately knowing A and
    it assumes the mask and signal are unknown. The error reduction uses a targeted optimization process
    to try to recover the original mask and signal.
    :param N: Length of signal
    :param m: Number of masks
    :param number_iterations: Number of iterations for reconstruction process
    :param do_add_noise: Add noise to the phase-less measurement vector
    :param x: The signal to use for the recovery. Default value is None and random signal of length N is constructed
    :param mask: The mask to use for the recovery. Default value is None and random mask of length N is constructed
    :return: Returns a tuple with the reconstructed signal, phase factors, the final error, and x_recon after iterations
    """

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

    return x_recon, mask, mask_approx, phasefac, error, progressive_errors


def modified_alternating_phase_projection_recovery_for_mask(mask, x, m, number_iterations, do_add_noise: bool):
    """
    The purpose of this method is to prove that this method is equivalent to the modified_alternating_phase_projection_recovery.
    Instead of computing x_n * m_(n-l) this method solves for the mask and computes x_(n+l) * m_n
    :param mask: An N length vector representing the mask
    :param x: An N length signal
    :param m: The number of shifted masks to generated from the mask param
    :param number_iterations: The number of iterations to use in the recovery process
    :param do_add_noise: Add noise to simulate the real world
    :return: Returns final x_recon, final error, and x_recon after every 50 iterations
    """
    N = len(x)
    B = create_measurement_matrix(m, N, x, do_shift_left=True)

    # Measurements are multiplied by mask instead of x
    b = np.abs(np.matmul(B, mask))

    if do_add_noise:
        b = simulate_noise_in_measurement(b)

    perturbation = np.random.rand(m, N) + 1J * np.random.rand(m, N)
    perturbation = np.multiply(perturbation, 1 / np.power(10, 6))

    perturbed_B = np.subtract(B, perturbation)
    inverse_perturbed_B = scipy.linalg.pinv(perturbed_B)
    mask_recon, mask_recon_iterations = iterative_signal_reconstruction(perturbed_B, N, b, inverse_perturbed_B,
                                                                        number_iterations)

    phasefac = np.matmul(np.conjugate(mask_recon).T, mask) / np.matmul(np.conjugate(mask).T, mask)
    mask_recon = np.multiply(mask_recon, signum(phasefac))

    error = np.linalg.norm(mask - mask_recon) / np.linalg.norm(mask)

    return mask_recon, error, mask_recon_iterations


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


def iterative_signal_reconstruction(measurement_matrix, N, b, inverse_A, number_iterations):
    """
    The core method for reconstructing x
    :param measurement_matrix:
    :param N: Number of samples
    :param b: N length vector
    :param inverse_A:
    :param number_iterations:
    :return:
    """
    signal_recon = np.random.rand(N) + 1J * np.random.rand(N)
    return reconstructed_signal(signal_recon, measurement_matrix, b, inverse_A, number_iterations)


def reconstructed_signal(signal, A, b, inverse_A, number_iterations):
    reconstructed_signal_iterations = OrderedDict()
    for i in range(0, number_iterations):
        temp = np.array(list(map(signum, np.matmul(A, signal))), dtype=np.complex_)
        signal = np.matmul(inverse_A, np.multiply(b, temp))

        if i % 50 == 0:
            reconstructed_signal_iterations[i] = signal

    return signal, reconstructed_signal_iterations
