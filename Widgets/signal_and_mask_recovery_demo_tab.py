import matplotlib.table
import numpy as np
from PySide6 import QtWidgets, QtCore
from PySide6.QtWidgets import QWidget
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from numpy import real, imag

import measurement
from utilities import perturb_vec


class SignalMaskRecoveryDemo(QWidget):
    def __init__(self, parent=None):
        super().__init__()

        self.N = 25
        self.run_recovery_example_button = QtWidgets.QPushButton('Run example')
        self.run_recovery_example_button.setToolTip(
            'Run the recovery process and show details about signal and results')
        self.run_recovery_example_button.clicked.connect(self.run_recovery_example)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.run_recovery_example_button)

        self.setLayout(layout)

    @QtCore.Slot()
    def run_recovery_example(self):
        m = 8 * self.N
        number_iterations = 600

        x = np.random.rand(self.N) + 1J * np.random.rand(self.N)
        mask = np.random.rand(self.N) + 1J * np.random.rand(self.N)
        mask_estimate = perturb_vec(mask)
        x_estimate = perturb_vec(x)

        (_, x_recon, _, _, _, signal_error, signal_iterative_error) = measurement.alternating_phase_projection_recovery_with_error_reduction(self.N, m,
                                                                                                   number_iterations,
                                                                                                   0,
                                                                                                   False,
                                                                                                   x=x_estimate,
                                                                                                   mask=mask_estimate)

        (_, mask_recon, _, _, _, mask_error, mask_iterative_error) = measurement.alternating_phase_projection_recovery_with_error_reduction(self.N, m,
                                                                                                    number_iterations,
                                                                                                    0,
                                                                                                    False,
                                                                                                    x=mask_estimate,
                                                                                                    mask=x_estimate,
                                                                                                    do_time_shift_signal=True)

        # This isn't ideal, but prevents overwhelming the screen with graphs and allows user to save individual plots
        fig, ax = plt.subplots(1, 2, num='Signal Graphs')
        ax[0].set_title('Real Part of Signal')
        ax[0].stem([real(e) for e in x])

        ax[1].set_title('Imaginary Part of Signal')
        ax[1].stem([imag(e) for e in x])

        plt.show()

        fig, ax = plt.subplots(1, 2, num='Mask Graphs')
        ax[0].set_title('Real Part of Mask')
        ax[0].stem([real(e) for e in mask])

        ax[1].set_title('Imaginary Part of Mask')
        ax[1].stem([imag(e) for e in mask])

        plt.show()

        fig, ax = plt.subplots(1, 2, num='Signal Recovery Graphs')
        ax[0].set_title('Recovery vs Original Signal (Real)')
        ax[0].stem([real(e) for e in x], markerfmt='x', label='True')
        ax[0].stem([real(e) for e in x_recon], linefmt='g--', markerfmt='+', label='Recovered')

        ax[1].set_title('Recovery vs Original Signal (Imag)')
        ax[1].stem([imag(e) for e in x], markerfmt='x', label='True')
        ax[1].stem([imag(e) for e in x_recon], linefmt='g--', markerfmt='+', label='Recovered')

        plt.show()

        fig, ax = plt.subplots(1, 2, num='Mask Recovery Graphs')
        ax[0].set_title('Recovery vs Original Mask (Real)')
        ax[0].stem([real(e) for e in mask], markerfmt='x', label='True')
        ax[0].stem([real(e) for e in mask_recon], linefmt='g--', markerfmt='+', label='Recovered')

        ax[1].set_title('Recovery vs Original Mask (Imag)')
        ax[1].stem([imag(e) for e in mask], markerfmt='x', label='True')
        ax[1].stem([imag(e) for e in mask_recon], linefmt='g--', markerfmt='+', label='Recovered')

        plt.show()

        fig, ax = plt.subplots(1, 2, num='Mask Recovery Graphs')
        column_labels = ('Signal Error', '8 Masks')
        # This only works as long as the signal and mask use the same iteration count
        row_labels = ['After %d iterations' % k for k in signal_iterative_error.keys()].append('Final Error')

        cell_text = []
        for i in signal_iterative_error.keys():
            sig_err = signal_iterative_error[i]
            mask_err = mask_iterative_error[i]
            cell_text.append('{:e}'.format(sig_err) + ' ' + '{:e}'.format(mask_err))

        cell_text.append('{:e}'.format(signal_error) + ' ' + '{:e}'.format(mask_error))
        table = plt.table(cellText=cell_text, rowLabels=row_labels, loc='bottom')

        plt.show()