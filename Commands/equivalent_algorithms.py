import numpy as np

import measurement
from utilities import perturb_vec


def display_forward_and_backward_time_shift_equivalence():
    N = 25
    m = 8 * N
    number_iterations = 600

    x = np.random.rand(N) + 1J * np.random.rand(N)
    mask = np.random.rand(N) + 1J * np.random.rand(N)
    mask_estimate = perturb_vec(mask)
    x_estimate = perturb_vec(x)

    (_, x_recon, _, signal_error, signal_recon_iterations) = \
        measurement.modified_alternating_phase_projection_recovery(N, m,
                                                                   number_iterations,
                                                                   0,
                                                                   False,
                                                                   x=x,
                                                                   mask=mask)

    (mask_recon, _, _) = \
        measurement.modified_alternating_phase_projection_recovery_for_mask(mask, x, m, number_iterations,
                                                                            False)

    print('mask, mask_recon', 'mask - mask_recon')
    for idx in range(0, len(x_recon)):
        mask_sample = mask[idx]
        mask_recon_sample = mask_recon[idx]

        print('{:e}, {:e}, {:e}'.format(mask_sample, mask_recon_sample, mask_sample - mask_recon_sample))