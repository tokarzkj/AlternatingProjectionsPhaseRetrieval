import numpy as np


def perturb_vec(vec: np.array):
    n = vec.shape[0]
    perturbation = np.random.rand(n) + 1J * np.random.rand(n)
    perturbation = np.multiply(perturbation, 1 / np.power(10, 2))

    return np.subtract(vec, perturbation)