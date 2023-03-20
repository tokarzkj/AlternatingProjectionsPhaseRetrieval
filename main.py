import numpy as np
import scipy

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    N = 5        # N Samples
    m = 4*N
    numberIterations = 600

    x = np.random.rand(N) + 1J*np.random.rand(N)

    A = np.zeros((m, N), dtype=np.complex_)

    diag = np.zeros((N, N))
    for i in range(0, N):
        diag[i][i] = 1

    for i in range(0, int(m/N)):

        A[i * N: (i * N) + N] = np.matmul(scipy.linalg.dft(N), diag * (np.random.rand(N) + 1J * np.random.rand(N)))

    inverse_A = scipy.linalg.pinv(A)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
