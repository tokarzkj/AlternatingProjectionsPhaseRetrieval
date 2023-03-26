from random import random

import numpy as np
from PySide6 import QtCore, QtWidgets, QtGui
import scipy
from numpy import imag, real
import matplotlib.pyplot as plt

import measurement

class MainPage(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__()

        self.seed_label = QtWidgets.QLabel('Seed')
        self.seed_value = QtWidgets.QLineEdit()
        self.seed_value.setText('3140')

        self.graph_random_projection_button = QtWidgets.QPushButton('Graph Recovery')
        self.graph_random_projection_button.clicked.connect(self.graph)

        self.layout = QtWidgets.QGridLayout(self)
        self.layout.addWidget(self.seed_label, 0, 0)
        self.layout.addWidget(self.seed_value, 0, 1)
        self.layout.addWidget(self.graph_random_projection_button)

    @QtCore.Slot()
    def graph(self):
        N = 100  # N Samples
        m = 4 * N
        numberIterations = 600

        # Need to set particular seed or the recovery values won't always align as expected
        # If you leave it blank the odds of success will be dependent on number of masks
        seed = self.seed_value.text()
        if len(seed) > 0:
            seed = int(seed)
            np.random.seed(seed)


        x = np.random.rand(N) + 1J * np.random.rand(N)

        A = measurement.create_measurement_matrix(m, N)
        inverse_A = scipy.linalg.pinv(A)

        # Measurements (magnitude of masked DFT coefficients)
        b = np.abs(np.matmul(A, x))

        x_recon = np.random.rand(N) + 1J * np.random.rand(N)

        for i in range(0, numberIterations):
            temp = np.array(list(map(signum, np.matmul(A, x_recon))), dtype=np.complex_)
            x_recon = np.matmul(inverse_A, np.multiply(b, temp))

        phasefac = np.matmul(np.conjugate(x_recon).T, x) / np.matmul(np.conjugate(x).T, x)
        x_recon = np.multiply(x_recon, signum(phasefac))

        print(np.linalg.norm(x - x_recon) / np.linalg.norm(x))

        fig, ax1 = plt.subplots(1, 2)
        fig.set_figheight(7)
        fig.set_figwidth(15)
        fig.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.1, hspace=0.75)

        ax1[0].set_title('Real Part')
        ax1[0].stem([real(e) for e in x], markerfmt='x', label='True')
        ax1[0].stem([real(e) for e in x_recon], linefmt='g--', markerfmt='+', label='Recovered')

        ax1[1].set_title('Real Part')
        true2 = ax1[1].stem([imag(e) for e in x], markerfmt='x', label='True')
        recovered2 = ax1[1].stem([imag(e) for e in x_recon], linefmt='g--', markerfmt='+', label='Recovered')

        fig.legend(handles=[true2, recovered2])

        plt.show()


def signum(value):
    # np.sign's complex implementation is different from matlab's. Changing to accommodate that difference.
    if imag(value) == 0J:
        return np.sign(value)
    else:
        return value / np.abs(value)

