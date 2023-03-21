import numpy as np
import scipy
import matplotlib.pyplot as plt
from numpy import real, imag


def signum(value):
    # np.sign's complex implementation is different than matlab's. Changing to accommodate that difference.
    if imag(value) == 0J:
        return np.sign(value)
    else:
        return value / np.abs(value)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    N = 100        # N Samples
    m = 4*N
    numberIterations = 600

    x = np.random.rand(N) + 1J*np.random.rand(N)

    A = np.zeros((m, N), dtype=np.complex_)

    # Create a diagonal matrix of 1s
    diag = np.zeros((N, N))
    for i in range(0, N):
        diag[i][i] = 1

    for i in range(0, int(m/N)):
        A[i * N: (i * N) + N] = np.matmul(scipy.linalg.dft(N), diag * (np.random.rand(N) + 1J * np.random.rand(N)))

    inverse_A = scipy.linalg.pinv(A)

    # Measurements (magnitude of masked DFT coefficients)
    b = np.abs(np.matmul(A, x))

    x_recon = np.random.rand(N) + 1J*np.random.rand(N)

    for i in range(0, numberIterations):
        temp = np.array(list(map(signum, np.matmul(A, x_recon))), dtype=np.complex_)
        x_recon = np.matmul(inverse_A, np.multiply(b, temp))

    phasefac = np.matmul(np.transpose(x_recon), x) / np.matmul(np.transpose(x), x)
    x_recon = np.multiply(x_recon, signum(phasefac))

    print(np.linalg.norm(x - x_recon)/np.linalg.norm(x))

    fig, ax1 = plt.subplots(1, 2)

    ax1[0].set_title('Real Part')
    ax1[0].stem([real(e) for e in x], markerfmt='o')
    ax1[0].stem([real(e) for e in x_recon], linefmt='g--', markerfmt='+')

    ax1[1].set_title('Real Part')
    ax1[1].stem([imag(e) for e in x], markerfmt='o')
    ax1[1].stem([imag(e) for e in x_recon], linefmt='g--', markerfmt='+')

    plt.show()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
