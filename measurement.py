import numpy as np
import scipy
from numpy import imag

from utilities import perturb_vec


def alternating_phase_projection_recovery(N, m, number_iterations, seed, do_add_noise: bool,
                                          x=None, mask=None, do_time_shift_signal: bool = False):
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
    :param do_time_shift_signal: When True, the mask is shifted to the right during the construction of our recovery matrix A
    :return: Returns a tuple with the signal, reconstructed signal, phase factors, and the error
    """
    if (isinstance(seed, str) and len(seed) > 0) or seed > 0:
        seed = int(seed)
        np.random.seed(seed)

    if x is None:
        x = np.random.rand(N) + 1J * np.random.rand(N)

    if mask is None:
        mask = np.random.rand(N) + 1J * np.random.rand(N)

    A = create_measurement_matrix(m, N, mask, do_time_shift_signal)
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


def modified_alternating_phase_projection_recovery(N, m, number_iterations, seed, do_add_noise: bool,
                                                   x=None, mask=None, do_time_shift_signal: bool = False):
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
    :param do_time_shift_signal: When True, the mask is shifted to the right during the construction of our recovery matrix A
    :return: Returns a tuple with the signal, reconstructed signal, phase factors, and the error
    """
    if (isinstance(seed, str) and len(seed) > 0) or seed > 0:
        seed = int(seed)
        np.random.seed(seed)

    if x is None:
        x = np.random.rand(N) + 1J * np.random.rand(N)

    if mask is None:
        mask = np.random.rand(N) + 1J * np.random.rand(N)

    A = create_measurement_matrix(m, N, mask, do_time_shift_signal)

    # Measurements (magnitude of masked DFT coefficients)
    b = np.abs(np.matmul(A, x))

    if do_add_noise:
        b = simulate_noise_in_measurement(b)

    perturbation = np.random.rand(m, N) + 1J * np.random.rand(m, N)
    perturbation = np.multiply(perturbation, 1 / np.power(10, 6))

    perturbed_A = np.subtract(A, perturbation)
    inverse_perturbed_A = scipy.linalg.pinv(perturbed_A)
    x_recon = initial_x_reconstruction_signal(perturbed_A, N, b, inverse_perturbed_A, number_iterations)

    phasefac = np.matmul(np.conjugate(x_recon).T, x) / np.matmul(np.conjugate(x).T, x)
    x_recon = np.multiply(x_recon, signum(phasefac))

    error = np.linalg.norm(x - x_recon) / np.linalg.norm(x)

    return x, x_recon, phasefac, error

def alternating_phase_projection_recovery_with_error_reduction(N, m, number_iterations, seed, do_add_noise: bool,
                                                   x=None, mask=None, do_time_shift_signal: bool = False):
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
    :param do_time_shift_signal: When True, the mask is shifted to the right during the construction of our recovery matrix A
    :return: Returns a tuple with the signal, reconstructed signal, phase factors, and the error
    """
    if (isinstance(seed, str) and len(seed) > 0) or seed > 0:
        seed = int(seed)
        np.random.seed(seed)

    if x is None:
        x = np.random.rand(N) + 1J * np.random.rand(N)

    if mask is None:
        mask = np.random.rand(N) + 1J * np.random.rand(N)

    A = create_measurement_matrix(m, N, mask, do_time_shift_signal)

    # Measurements (magnitude of masked DFT coefficients)
    b = np.abs(np.matmul(A, x))

    if do_add_noise:
        b = simulate_noise_in_measurement(b)

    x_recon = np.random.rand(N) + 1J * np.random.rand(N)
    mask_approx = perturb_vec(mask)
    for idx in range(0, 100):
        A_approx = create_measurement_matrix(m, N, mask_approx)
        A_pinv = scipy.linalg.pinv(A_approx)
        x_recon = reconstructed_signal(x_recon, A_approx, b, A_pinv, number_iterations)

        M_approx = create_measurement_matrix(m, N, x_recon, True)
        M_pinv = scipy.linalg.pinv(M_approx)

        m_recon = mask_approx
        mask_approx = reconstructed_signal(m_recon, M_approx, b, M_pinv, number_iterations)

    phasefac = np.matmul(np.conjugate(x_recon).T, x) / np.matmul(np.conjugate(x).T, x)
    x_recon = np.multiply(x_recon, signum(phasefac))

    error = np.linalg.norm(x - x_recon) / np.linalg.norm(x)

    return x, x_recon, mask, mask_approx, phasefac, error

def signum(value):
    # np.sign's complex implementation is different from matlab's. Changing to accommodate that difference.
    if imag(value) == 0J:
        return np.sign(value)
    else:
        return value / np.abs(value)


def create_measurement_matrix(m, N, vec, do_time_shift_signal=False):
    A = np.zeros((m, N), dtype=np.complex_)

    # Create a diagonal matrix of 1s
    diag = np.zeros((N, N))
    for i in range(0, N):
        diag[i][i] = 1

    for i in range(0, int(m / N)):
        shift = int(i * np.round(N / (m / N)))
        if do_time_shift_signal:
            shift = shift * -1

        shifted_mask = np.roll(vec, shift)
        A[i * N: (i * N) + N] = np.matmul(scipy.linalg.dft(N), diag * shifted_mask)

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
    for i in range(0, number_iterations):
        temp = np.array(list(map(signum, np.matmul(A, signal))), dtype=np.complex_)
        signal = np.matmul(inverse_A, np.multiply(b, temp))

    return signal

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
