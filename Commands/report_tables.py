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
    snrs = [20, 40, 60, 80, 100]

    N = 25
    m = 8 * N
    results = []
    for snr in snrs:
        print("Calculating with noise " + str(snr))
        errors = np.zeros(trial_count)

        for i in range(0, trial_count):
            signal, mask = utilities.create_signal_and_mask(0, N)
            (_, _, _, error, _) = \
                measurement.modified_alternating_phase_projection_recovery(N, m, 250, signal, mask, snr)
            errors[i] = error

        avg_err = np.average(errors)
        results.append([snr, avg_err])

    table = tabulate.tabulate(results, headers=["Noise (db)", "Avg Error"], tablefmt="latex_raw")

    print("###############################################")
    print("Table for unknown mask recovery")
    print(table)
    print("###############################################")

    fig, ax = plt.subplots(1, 1, num='Accuracy vs Error')

    y = list(map(lambda r: r[1], results))
    ax.plot(snrs, y)
    plt.show()


def unknown_signal_and_mask_accuracy_vs_noise():
    """
    Generate results for the average error of the recovery process, when the signal is known, and
    """
    trial_count = 25
    snrs = [20, 40, 60, 80, 100]

    N = 25
    m = 8 * N
    results = []
    for snr in snrs:
        print("Calculating with noise " + str(snr))
        errors = np.zeros(trial_count)

        for i in range(0, trial_count):
            signal, mask = utilities.create_signal_and_mask(0, N)
            (_, _, error, _, _) = \
                measurement.alternating_phase_projection_recovery_with_error_reduction(N, m, 250, signal, mask, snr)
            errors[i] = error

        avg_err = np.average(errors)
        results.append([snr, avg_err])

    table = tabulate.tabulate(results, headers=["Noise (db)", "Avg Error"], tablefmt="latex_raw")
    print("###############################################")
    print("Table for unknown signal and mask recovery")
    print(table)
    print("###############################################")

    fig, ax = plt.subplots(1, 1, num='Accuracy vs Error')

    y = list(map(lambda r: r[1], results))
    ax.plot(snrs, y)
    plt.show()


def unknown_mask_iteration_vs_error():
    trial_count = 2
    N = 25
    m = 8 * N
    iterations = 600

    modified_results = dict()
    modified_results[iterations] = np.empty(trial_count)

    error_recovery_results = dict()
    error_recovery_results[iterations] = np.empty(trial_count)
    for i in range(0, trial_count):
        signal, mask = utilities.create_signal_and_mask(0, N)

        (_, _, _, final_error, signal_recon_iter) = \
            measurement.modified_alternating_phase_projection_recovery(N, m, iterations, signal, mask)

        for k in signal_recon_iter.keys():
            signal_recon = signal_recon_iter[k]
            phasefac = np.matmul(np.conjugate(signal_recon).T, signal) / np.matmul(np.conjugate(signal).T, signal)
            signal_recon = np.multiply(signal_recon, utilities.signum(phasefac))
            signal_iter_err = np.linalg.norm(signal - signal_recon) / np.linalg.norm(signal)

            if k in modified_results:
                modified_results[k][i] = signal_iter_err
            else:
                modified_results[k] = np.empty(trial_count)
                modified_results[k][i] = signal_iter_err

        modified_results[iterations][i] = final_error

    avg_modified_results = dict()
    for k in modified_results.keys():
        iteration_errors = modified_results[k]
        iteration_avg_error = np.average(iteration_errors)
        avg_modified_results[k] = iteration_avg_error

    avg_modified_results = [(k, v) for k, v in avg_modified_results.items()]

    table = tabulate.tabulate(avg_modified_results, headers=["Iteration", "Avg Error"], tablefmt="latex_raw")
    print(table)


def unknown_signal_and_mask_iteration_vs_error():
    trial_count = 2
    N = 25
    m = 8 * N
    iterations = 600

    error_recovery_results = dict()
    error_recovery_results[iterations] = np.empty(trial_count)
    for i in range(0, trial_count):
        signal, mask = utilities.create_signal_and_mask(0, N)

        (_, _, final_error, _, signal_recon_iter) = \
            measurement.alternating_phase_projection_recovery_with_error_reduction(N, m, iterations, signal, mask)

        for k in signal_recon_iter.keys():
            signal_recon = signal_recon_iter[k]
            phasefac = np.matmul(np.conjugate(signal_recon).T, signal) / np.matmul(np.conjugate(signal).T, signal)
            signal_recon = np.multiply(signal_recon, utilities.signum(phasefac))
            signal_iter_err = np.linalg.norm(signal - signal_recon) / np.linalg.norm(signal)

            if k in error_recovery_results:
                error_recovery_results[k][i] = signal_iter_err
            else:
                error_recovery_results[k] = np.empty(trial_count)
                error_recovery_results[k][i] = signal_iter_err

        error_recovery_results[iterations][i] = final_error

    avg_error_reduced_results = dict()
    for k in error_recovery_results.keys():
        iteration_errors = error_recovery_results[k]
        iteration_avg_error = np.average(iteration_errors)
        avg_error_reduced_results[k] = iteration_avg_error

    avg_error_reduced_results = [(k, v) for k, v in avg_error_reduced_results.items()]

    table = tabulate.tabulate(avg_error_reduced_results, headers=["Iteration", "Avg Error"], tablefmt="latex_raw")
    print(table)
