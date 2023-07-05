import numpy as np
import tabulate
from matplotlib import pyplot as plt

import measurement
import utilities


def unknown_mask_accuracy_vs_noise():
    """
    Generate results for the average error of the recovery process, when the signal is known, and
    """
    trial_count = 25
    noises = [20, 40, 60, 80, 100]

    N = 25
    m = 8 * N
    results = []
    for noise in noises:
        print("Calculating with noise " + str(noise))
        errors = np.zeros(trial_count)

        i = 0
        while i < trial_count:
            signal, mask = utilities.create_signal_and_mask(0, N)
            (_, _, _, error, _) = \
                measurement.modified_alternating_phase_projection_recovery(N, m, 250, signal, mask, noise)
            errors[i] = error
            i += 1

        avg_err = np.average(errors)
        results.append([noise, avg_err])

    table = tabulate.tabulate(results, headers=["Noise (db)", "Avg Error"], tablefmt="latex_raw")

    print("###############################################")
    print("Table for unknown mask recovery")
    print(table)
    print("###############################################")

    fig, ax = plt.subplots(1, 1, num='Accuracy vs Error')

    y = list(map(lambda r: r[1], results))
    ax.plot(noises, y)
    plt.show()


def unknown_signal_and_mask_accuracy_vs_noise():
    """
    Generate results for the average error of the recovery process, when the signal is known, and
    """
    trial_count = 5
    noises = [20, 40, 60, 80, 100]

    N = 25
    m = 8 * N
    results = []
    for noise in noises:
        print("Calculating with noise " + str(noise))
        errors = np.zeros(trial_count)

        i = 0
        while i < trial_count:
            signal, mask = utilities.create_signal_and_mask(0, N)
            (_, _, _, _, error, _) = \
                measurement.alternating_phase_projection_recovery_with_error_reduction(N, m, 250, signal, mask, noise)
            errors[i] = error
            i += 1

        avg_err = np.average(errors)
        results.append([noise, avg_err])

    table = tabulate.tabulate(results, headers=["Noise (db)", "Avg Error"], tablefmt="latex_raw")
    print("###############################################")
    print("Table for unknown signal and mask recovery")
    print(table)
    print("###############################################")

    fig, ax = plt.subplots(1, 1, num='Accuracy vs Error')

    y = list(map(lambda r: r[1], results))
    ax.plot(noises, y)
    plt.show()
