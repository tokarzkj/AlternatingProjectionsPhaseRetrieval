import numpy as np
import scipy.fft
from matplotlib import pyplot as plt

from utilities import signum
from PIL import Image


def alternating_projection_recovery_2d(n1, n2, number_iterations: int = 500):
    np.random.seed(3140)
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

    measurement_matrix = np.zeros((m, element_count), dtype=np.complex_)
    dft2d_matrix = np.zeros((element_count, element_count), dtype=np.complex_)

    for i in range(0, element_count):
        std_basis = np.zeros((element_count, 1))
        std_basis[i] = 1
        std_basis = np.reshape(std_basis, (n1, n2))
        fft_output = scipy.fft.fft2(std_basis)
        dft2d_matrix[:, i] = np.reshape(fft_output, element_count)

    mask = np.random.rand(element_count) + 1J * np.random.rand(element_count)

    for i in range(0, int(m / element_count)):
        (row_idx, col_idx) = np.unravel_index(i, (n1, n2))
        shifted_mask = np.roll(mask, (row_idx, col_idx))
        measurement_matrix[i * element_count: i * element_count + element_count] = np.multiply(shifted_mask, np.diag(shifted_mask))

    inv_measurement_matrix = scipy.linalg.pinv(measurement_matrix)
    b = np.abs(np.matmul(measurement_matrix, x.reshape(element_count)))

    x_recon = np.array(np.random.rand(element_count) + 1J * np.random.rand(element_count))
    for i in range(0, number_iterations):
        temp = np.array(list(map(signum, np.matmul(measurement_matrix, x_recon))), dtype=np.complex_)
        x_recon = np.matmul(inv_measurement_matrix, np.multiply(b, temp))

    # ToDo: Once we read in an image as a matrix we will need to vectorize x here

    vec_x = x.reshape(element_count)
    phasefac = np.matmul(np.conjugate(x_recon).T, vec_x) / np.matmul(np.conjugate(vec_x).T, vec_x)
    x_recon = np.multiply(x_recon, signum(phasefac))



    x_recon = x_recon.reshape(n1, n2)

    fig , (ax1, ax2) = plt.subplots(2, 2)

    ax1[0].imshow(np.real(x))
    ax1[1].imshow(np.imag(x))

    ax2[0].imshow(np.real(x_recon))
    ax2[1].imshow(np.imag(x_recon))

    plt.show()