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
    trial_count = 50
    snrs = [20, 40, 60, 80, 100]

    N = 25
    m = 8 * N
    results = []

    signal, mask = utilities.create_signal_and_mask(0, N)
    for snr in snrs:
        print("Calculating with noise " + str(snr))
        errors = np.zeros(trial_count)

        for i in range(0, trial_count):
            (_, _, _, error, _) = \
                measurement.modified_alternating_phase_projection_recovery(N, m, 600, signal, mask, snr)
            errors[i] = error

        avg_err = np.average(errors)
        results.append([snr, avg_err])

    table = tabulate.tabulate(results, headers=["Noise (db)", "Avg Error"], tablefmt="latex_raw")

    print("###############################################")
    print("Table for unknown mask recovery")
    print(table)
    print("###############################################")

    with open('figures\\unknown_mask_noise_plot.txt', 'w') as f:
        f.write(table)

    fig, ax = plt.subplots(1, 1, num='Accuracy vs Avg Error')
    ax.set_xlabel("Signal-to-Noise Ratio")
    ax.set_ylabel("Avg Error")
    y = list(map(lambda r: r[1], results))
    ax.plot(snrs, y)
    plt.xticks(snrs)
    plt.savefig("figures\\unknown_mask_noise_plot.png")


def unknown_signal_and_mask_accuracy_vs_noise():
    """
    Generate results for the average error of the recovery process, when the signal is known, and
    """
    trial_count = 25
    snrs = [20, 40, 60, 80, 100]

    N = 25
    m = 8 * N
    results = []

    signal, mask = utilities.create_signal_and_mask(0, N)
    for snr in snrs:
        print("Calculating with noise " + str(snr))
        errors = np.zeros(trial_count)
        for i in range(0, trial_count):
            print("Trial " + str(i))
            (_, _, error, _, _) = \
                measurement.alternating_phase_projection_recovery_with_error_reduction(N, m, 600, signal, mask, snr)
            errors[i] = error

        avg_err = np.average(errors)
        results.append([snr, avg_err])

    table = tabulate.tabulate(results, headers=["Noise (db)", "Avg Error"], tablefmt="latex_raw")
    print("###############################################")
    print("Table for unknown signal and mask recovery")
    print(table)
    print("###############################################")

    with open('figures\\unknown_signal_and_mask_noise_plot.txt', 'w') as f:
        f.write(table)

    fig, ax = plt.subplots(1, 1, num='Noise vs Avg Error')
    ax.set_xlabel("Signal-to-Noise Ratio")
    ax.set_ylabel("Avg Error")
    y = list(map(lambda r: r[1], results))
    ax.plot(snrs, y)
    plt.xticks(snrs)
    plt.savefig("figures\\unknown_signal_and_mask_noise_plot.png")


def unknown_mask_iteration_vs_error():
    trial_count = 25
    N = 25
    m = 8 * N
    iterations = 600

    modified_results = dict()

    error_recovery_results = dict()
    error_recovery_results[iterations] = np.empty(trial_count)

    signal, mask = utilities.create_signal_and_mask(0, N)
    for i in range(0, trial_count):
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

        if 600 in modified_results:
            modified_results[iterations][i] = final_error
        else:
            modified_results[iterations] = np.empty(trial_count)
            modified_results[iterations][i] = final_error



    avg_modified_results = dict()
    for k in modified_results.keys():
        iteration_errors = modified_results[k]
        iteration_avg_error = np.average(iteration_errors)
        avg_modified_results[k] = iteration_avg_error

    avg_modified_results.pop(0)
    avg_modified_results = [(k, v) for k, v in avg_modified_results.items()]

    table = tabulate.tabulate(avg_modified_results, headers=["Iteration", "Avg Error"], tablefmt="latex_raw")
    print(table)

    with open('figures\\unknown_mask_iterations_plot.txt', 'w') as f:
        f.write(table)

    fig, ax = plt.subplots(1, 1, num='Iteration Count vs Avg Error')
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Avg Error")
    x = list(map(lambda r: r[0], avg_modified_results))
    y = list(map(lambda r: r[1], avg_modified_results))
    ax.plot(x, y)
    plt.xticks(x)
    plt.savefig("figures\\unknown_mask_iterations_plot.png")


def unknown_signal_and_mask_iteration_vs_error():
    trial_count = 25
    N = 25
    m = 8 * N

    error_recovery_results = dict()
    signal, mask = utilities.create_signal_and_mask(0, N)
    for iter_count in range(100, 700, 100):
        error_recovery_results[iter_count] = np.empty(trial_count)
        print("Iteration Count " + str(iter_count))
        for i in range(0, trial_count):
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

    with open('figures\\unknown_signal_and_mask_iterations_plot.txt', 'w') as f:
        f.write(table)

    fig, ax = plt.subplots(1, 1, num='Iteration Count vs Avg Error')
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Avg Error")
    x = list(map(lambda r: r[0], avg_error_reduced_results))
    y = list(map(lambda r: r[1], avg_error_reduced_results))
    ax.plot(x, y)
    plt.xticks(x)
    plt.savefig("figures\\unknown_signal_and_mask_iterations_plot.png")

def unknown_mask_sample_size_vs_time():
    trials_count = 25
    N = [10, 20, 30, 40, 50]

    timing_results = dict()
    for n in N:
        timing_results[n] = np.empty(trials_count)
        m = 8 * n
        signal, mask = utilities.create_signal_and_mask(0, n)
        for i in range(0, trials_count):
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

    with open('figures\\unknown_mask_sample_size_plot.txt', 'w') as f:
        f.write(table)

    plt.clf()
    fig, ax = plt.subplots(1, 1, num='Sample Size vs Execution Time')
    ax.set_xlabel("Sample Size")
    ax.set_ylabel("Execution Time (seconds)")
    x = list(map(lambda r: r[0], avg_iter_time_results))
    y = list(map(lambda r: r[1], avg_iter_time_results))
    ax.plot(x, y)
    plt.xticks(x)
    plt.savefig("figures\\unknown_mask_sample_size_plot.png")
    plt.close()


def unknown_signal_and_unknown_mask_sample_size_vs_time():
    trials_count = 5
    N = [10, 20, 30, 40, 50]

    timing_results = dict()
    for n in N:
        timing_results[n] = np.empty(trials_count)
        m = 8 * n
        print("Samples: " + str(n))
        signal, mask = utilities.create_signal_and_mask(0, n)
        for i in range(0, trials_count):
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

    with open('figures\\unknown_signal_and_mask_sample_size_plot.txt', 'w') as f:
        f.write(table)

    plt.clf()
    fig, ax = plt.subplots(1, 1, num='Sample Size vs Execution Time')
    ax.set_xlabel("Sample Size")
    ax.set_ylabel("Execution Time (seconds)")
    x = list(map(lambda r: r[0], avg_iter_time_results))
    y = list(map(lambda r: r[1], avg_iter_time_results))
    ax.plot(x, y)
    plt.xticks(x)
    plt.savefig("figures\\unknown_signal_and_mask_sample_size_plot.png")
