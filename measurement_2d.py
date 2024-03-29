import time

import numpy as np
import scipy.fft
from matplotlib import pyplot as plt

import utilities
from utilities import signum
from PIL import Image


def alternating_projection_recovery_2d(n1, n2, number_iterations: int = 500):
    element_count = n1 * n2

    # ToDo: Read in an actual image
    x1 = Image.open('./Images/IMG_2103.JPG').convert("L")
    x1 = x1.resize((n1, n2))
    x1 = np.asarray(x1)

    x2 = Image.open('./Images/IMG_2068.JPG').convert("L")
    x2 = x2.resize((n1, n2))
    x2 = np.asarray(x2)

    ovr1 = 3
    ovr2 = 3
    m = (ovr1 * ovr2) * n1 * n2

    x = x1 + 1J * x2

    dft2d_matrix = create_dft2d_matrix(element_count, n1, n2)
    mask = np.random.rand(n1, n2) + 1J * np.random.rand(n1, n2)
    measurement_matrix = create_measurement_matrix(dft2d_matrix, element_count, m, mask, n1, n2)

    inv_measurement_matrix = scipy.linalg.pinv(measurement_matrix)
    b = np.abs(np.matmul(measurement_matrix, x.reshape(element_count)))

    x_recon = np.array(np.random.rand(element_count) + 1J * np.random.rand(element_count))
    for i in range(0, number_iterations):
        temp = np.array(list(map(signum, np.matmul(measurement_matrix, x_recon))), dtype=np.complex_)
        x_recon = np.matmul(inv_measurement_matrix, np.multiply(b, temp))

    vec_x = x.reshape(element_count)
    phasefac = np.matmul(np.conjugate(x_recon).T, vec_x) / np.matmul(np.conjugate(vec_x).T, vec_x)
    x_recon = np.multiply(x_recon, signum(phasefac))

    x_recon = x_recon.reshape(n2, n1)
    sig_error = np.linalg.norm(x - x_recon) / np.linalg.norm(x)
    print("The signal error is {:e}".format(sig_error))
    fig , (ax1, ax2) = plt.subplots(2, 2)

    ax1[0].set_xticks([])
    ax1[0].set_yticks([])
    ax1[0].set_title("Real(Original)")
    ax1[1].set_xticks([])
    ax1[1].set_yticks([])
    ax1[1].set_title("Imaginary(Original)")

    ax1[0].imshow(np.real(x))
    ax1[1].imshow(np.imag(x))

    ax2[0].set_xticks([])
    ax2[0].set_yticks([])
    ax2[0].set_title("Real(Recovered)")
    ax2[1].set_xticks([])
    ax2[1].set_yticks([])
    ax2[1].set_title("Imaginary(Recovered)")

    ax2[0].imshow(np.real(x_recon))
    ax2[1].imshow(np.imag(x_recon))

    plt.show()


def alternating_projection_recovery_2d_with_error_reduction(n1, n2, number_iterations: int = 500):
    element_count = n1 * n2

    # ToDo: Read in an actual image
    x1 = Image.open('./Images/IMG_2103.JPG').convert("L")
    x1 = x1.resize((n1, n2))
    x1 = np.asarray(x1)

    x2 = Image.open('./Images/IMG_2068.JPG').convert("L")
    x2 = x2.resize((n1, n2))
    x2 = np.asarray(x2)

    ovr1 = 3
    ovr2 = 3
    m = (ovr1 * ovr2) * n1 * n2

    x = x1 + 1J * x2

    dft2d_matrix = create_dft2d_matrix(element_count, n1, n2)
    mask = np.random.rand(n1, n2) + 1J * np.random.rand(n1, n2)
    measurement_matrix = create_measurement_matrix(dft2d_matrix, element_count, m, mask, n1, n2)
    b = np.abs(np.matmul(measurement_matrix, x.reshape(element_count)))

    mask_approx = utilities.perturb_matrix(mask)
    x_recon = np.array(np.random.rand(element_count) + 1J * np.random.rand(element_count))
    for idx in range(0, 100):
        time_start = time.time()
        A_approx = create_measurement_matrix(dft2d_matrix, element_count, m, mask_approx, n1, n2)
        A_approx_pinv = scipy.linalg.pinv(A_approx)

        for i in range(0, number_iterations):
            temp = np.array(list(map(signum, np.matmul(A_approx, x_recon))), dtype=np.complex_)
            x_recon = np.matmul(A_approx_pinv, np.multiply(b, temp))

        M_approx = create_measurement_matrix(dft2d_matrix, element_count, m, x_recon.reshape(n1, n2), n1, n2)
        M_approx_pinv = scipy.linalg.pinv(M_approx)

        vec_mask_approx = mask_approx.reshape(element_count)
        for i in range(0, number_iterations):
            temp = np.array(list(map(signum, np.matmul(M_approx, vec_mask_approx))), dtype=np.complex_)
            vec_mask_approx = np.matmul(M_approx_pinv, np.multiply(b, temp))

        mask_approx = vec_mask_approx.reshape(n1, n2)
        time_end = time.time()
        print("Iteration " + str(idx) + ' took ' + str(time_end - time_start))

    vec_x = x.reshape(element_count)
    phasefac = np.matmul(np.conjugate(x_recon).T, vec_x) / np.matmul(np.conjugate(vec_x).T, vec_x)
    x_recon = np.multiply(x_recon, signum(phasefac))

    x_recon = x_recon.reshape(n2, n1)
    sig_error = np.linalg.norm(x - x_recon) / np.linalg.norm(x)
    print("The signal error is {:e}".format(sig_error))

    fig, (ax1, ax2) = plt.subplots(2, 2)

    ax1[0].set_xticks([])
    ax1[0].set_yticks([])
    ax1[0].set_title("Real(Original)")
    ax1[1].set_xticks([])
    ax1[1].set_yticks([])
    ax1[1].set_title("Imaginary(Original)")

    ax1[0].imshow(np.real(x))
    ax1[1].imshow(np.imag(x))

    ax2[0].set_xticks([])
    ax2[0].set_yticks([])
    ax2[0].set_title("Real(Recovered)")
    ax2[1].set_xticks([])
    ax2[1].set_yticks([])
    ax2[1].set_title("Imaginary(Recovered)")

    ax2[0].imshow(np.real(x_recon))
    ax2[1].imshow(np.imag(x_recon))

    plt.show()


def create_measurement_matrix(dft2d_matrix, element_count, m, mask, n1, n2):
    measurement_matrix = np.zeros((m, element_count), dtype=np.complex_)
    for i in range(0, int(m / element_count)):
        (row_idx, col_idx) = np.unravel_index(i, (n1, n2))
        shifted_mask = np.roll(mask, (row_idx, col_idx))
        vec_shifted_mask = shifted_mask.reshape(element_count)
        measurement_matrix[i * element_count: i * element_count + element_count] = np.matmul(dft2d_matrix,
                                                                                             np.diag(vec_shifted_mask))
    return measurement_matrix


def create_dft2d_matrix(element_count, n1, n2):
    dft2d_matrix = np.zeros((element_count, element_count), dtype=np.complex_)
    for i in range(0, element_count):
        std_basis = np.zeros((element_count, 1))
        std_basis[i] = 1
        std_basis = np.reshape(std_basis, (n1, n2))
        fft_output = scipy.fft.fft2(std_basis)
        dft2d_matrix[:, i] = np.reshape(fft_output, element_count)

    return dft2d_matrix
