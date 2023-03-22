import numpy as np
import scipy
import matplotlib.pyplot as plt
from numpy import real, imag


def signum(value):
    # np.sign's complex implementation is different from matlab's. Changing to accommodate that difference.
    if imag(value) == 0J:
        return np.sign(value)
    else:
        return value / np.abs(value)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    N = 100        # N Samples
    m = 4*N
    numberIterations = 600

    # Need to set particular seed or the recovery values won't always align as expected
    np.random.seed(3140)
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

    phasefac = np.matmul(np.conjugate(x_recon).T, x) / np.matmul(np.conjugate(x).T, x)
    x_recon = np.multiply(x_recon, signum(phasefac))

    print(np.linalg.norm(x - x_recon)/np.linalg.norm(x))

    fig, ax1 = plt.subplots(1, 2)
    fig.set_figheight(7)
    fig.set_figwidth(15)
    fig.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.1, hspace=0.75)

    ax1[0].set_title('Real Part')
    ax1[0].stem([real(e) for e in x], markerfmt='x', label='True' )
    ax1[0].stem([real(e) for e in x_recon], linefmt='g--', markerfmt='+', label='Recovered')

    ax1[1].set_title('Real Part')
    true2 = ax1[1].stem([imag(e) for e in x], markerfmt='x', label='True')
    recovered2 = ax1[1].stem([imag(e) for e in x_recon], linefmt='g--', markerfmt='+', label='Recovered')

    fig.legend(handles=[true2, recovered2])

    plt.show()

