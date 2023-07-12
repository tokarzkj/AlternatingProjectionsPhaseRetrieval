import time

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
        print("Beginning snr " + str(snr))
        for i in range(0, trial_count):
            print("Beginning trial " + str(i))
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
    trial_count = 25
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
    trial_count = 25
    N = 25
    m = 8 * N

    error_recovery_results = dict()
    for iter_count in range(100, 700, 100):
        error_recovery_results[iter_count] = np.empty(trial_count)
        print("Iteration Count " + str(iter_count))
        for i in range(0, trial_count):
            print("Beginning trial " + str(i))
            signal, mask = utilities.create_signal_and_mask(0, N)

            (_, _, final_error, _, _) = \
                measurement.alternating_phase_projection_recovery_with_error_reduction(N, m, iter_count, signal, mask)

            error_recovery_results[iter_count][i] = final_error

    avg_error_reduced_results = dict()
    for k in error_recovery_results.keys():
        iteration_errors = error_recovery_results[k]
        iteration_avg_error = np.average(iteration_errors)
        avg_error_reduced_results[k] = iteration_avg_error

    avg_error_reduced_results = [(k, v) for k, v in avg_error_reduced_results.items()]

    table = tabulate.tabulate(avg_error_reduced_results, headers=["Iteration", "Avg Error"], tablefmt="latex_raw")
    print(table)


def unknown_mask_sample_size_vs_time():
    trials_count = 25
    N = [10, 20, 30, 40, 50]

    timing_results = dict()
    for n in N:
        timing_results[n] = np.empty(trials_count)
        m = 8 * n
        for i in range(0, trials_count):
            signal, mask = utilities.create_signal_and_mask(0, n)

            # start watch
            start = time.time()
            measurement.modified_alternating_phase_projection_recovery(n, m, 600, signal, mask)
            end = time.time()

            timing_results[n][i] = end - start
            # end watch

    avg_iter_time_results = []
    for k in timing_results.keys():
        iter_times = timing_results[k]
        iter_avg_time = np.average(iter_times)
        avg_iter_time_results.append((k, iter_avg_time))

    table = tabulate.tabulate(avg_iter_time_results, headers=["Sample Size", "Avg Time"], tablefmt="latex_raw")
    print(table)


def unknown_signal_and_unknown_mask_sample_size_vs_time():
    trials_count = 25
    N = [10, 20, 30, 40, 50]

    timing_results = dict()
    for n in N:
        timing_results[n] = np.empty(trials_count)
        m = 8 * n
        print("Samples: " + str(n))
        for i in range(0, trials_count):
            print("Beginning trial: " + str(i))
            signal, mask = utilities.create_signal_and_mask(0, n)

            # start watch
            start = time.time()
            measurement.alternating_phase_projection_recovery_with_error_reduction(n, m, 600, signal, mask)
            end = time.time()

            run_time = end - start
            timing_results[n][i] = run_time
            print("Trial: " + str(i) + " took " + str(run_time))
            # end watch

    avg_iter_time_results = []
    for k in timing_results.keys():
        iter_times = timing_results[k]
        iter_avg_time = np.average(iter_times)
        avg_iter_time_results.append((k, iter_avg_time))

    table = tabulate.tabulate(avg_iter_time_results, headers=["Sample Size", "Avg Time"], tablefmt="latex_raw")
    print(table)