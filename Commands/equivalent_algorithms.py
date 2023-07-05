import numpy as np

import measurement
from utilities import perturb_vec, signum


def display_forward_and_backward_time_shift_equivalence():
    N = 25
    m = 8 * N
    number_iterations = 600

    x = np.random.rand(N) + 1J * np.random.rand(N)
    mask = np.random.rand(N) + 1J * np.random.rand(N)

    (_, x_recon, _, signal_error, signal_iterations) = \
        measurement.modified_alternating_phase_projection_recovery(N, m,
                                                                   number_iterations,
                                                                   x,
                                                                   mask)

    print('x, x_recon', 'x - x_recon')
    for idx in range(0, len(x_recon)):
        x_sample = x[idx]
        x_recon_sample = x_recon[idx]

        print('{:e}, {:e}, {:e}'.format(x_sample, x_recon_sample, x_sample - x_recon_sample))
    print('The overall error for the signal recovery is {:e}'.format(signal_error))

    (mask_recon, mask_error, mask_iterations) = \
        measurement.modified_alternating_phase_projection_recovery_for_mask(mask, x, m, number_iterations)

    print("#############################################################################################")
    print("#############################################################################################")

    print('mask, mask_recon', 'mask - mask_recon')
    for idx in range(0, len(x_recon)):
        mask_sample = mask[idx]
        mask_recon_sample = mask_recon[idx]

        print('{:e}, {:e}, {:e}'.format(mask_sample, mask_recon_sample, mask_sample - mask_recon_sample))
    print('The overall error for the mask recovery is {:e}'.format(mask_error))

    print("#############################################################################################")
    print("#############################################################################################")
    print("Iteration Number, Signal Iteration Err, Mask Iteration Err")
    for k in signal_iterations.keys():
        signal_recon_iter = signal_iterations[k]
        phasefac = np.matmul(np.conjugate(signal_recon_iter).T, x) / np.matmul(np.conjugate(x).T, x)
        signal_recon_iter = np.multiply(signal_recon_iter, signum(phasefac))
        signal_iter_err = np.linalg.norm(x - signal_recon_iter) / np.linalg.norm(x)

        mask_recon_iter = mask_iterations[k]
        phasefac = np.matmul(np.conjugate(mask_recon_iter).T, mask) / np.matmul(np.conjugate(mask).T, mask)
        mask_recon_iter = np.multiply(mask_recon_iter, signum(phasefac))
        mask_iter_err = np.linalg.norm(mask - mask_recon_iter) / np.linalg.norm(mask)
        print('{:d}, {:e}, {:e}'.format(k, signal_iter_err, mask_iter_err))
