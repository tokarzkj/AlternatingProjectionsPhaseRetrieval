import numpy as np


def perturb_vec(vec: np.array):
    n = vec.shape[0]
    perturbation = np.random.rand(n) + 1J * np.random.rand(n)
    perturbation = np.multiply(perturbation, 1 / np.power(10, 2))

    return np.subtract(vec, perturbation)


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