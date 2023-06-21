import numpy as np
import sys

sys.path.append("../AlternatingProjectionsPhaseRetrieval")
import measurement
from utilities import perturb_vec


def test_forward_and_backward_time_shift_equivalence():
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

    assert np.array_equiv(x_recon, mask_recon)
