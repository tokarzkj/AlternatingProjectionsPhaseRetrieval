import matplotlib.table
import numpy as np
from PySide6 import QtWidgets, QtCore
from PySide6.QtGui import QStandardItemModel, QStandardItem
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
        self.error_results_table = ErrorResultsTableView()

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.error_results_table)
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

        (_, x_recon, _, _, _, signal_error,
         signal_iterative_error) = measurement.alternating_phase_projection_recovery_with_error_reduction(self.N, m,
                                                                                                          number_iterations,
                                                                                                          0,
                                                                                                          False,
                                                                                                          x=x_estimate,
                                                                                                          mask=mask_estimate)

        (_, mask_recon, _, _, _, mask_error,
         mask_iterative_error) = measurement.alternating_phase_projection_recovery_with_error_reduction(self.N, m,
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

        error_results = dict()
        for k in signal_iterative_error.keys():
            sig_err = signal_iterative_error[k]
            mask_err = mask_iterative_error[k]
            error_results[k] = (sig_err, mask_err)

        error_results['final'] = (signal_error, mask_error)

        self.error_results_table.update_table(error_results)
        self.layout().update()


class ErrorResultsTableView(QtWidgets.QTableView):
    def __init__(self, results = None):
        super().__init__()

        self.model = QStandardItemModel()
        self.update_table(results)
        self.setModel(self.model)
    def update_table(self, results):
        self.model.clear()
        self.model.setHorizontalHeaderLabels(['# of Iterations', 'Signal Error', 'Mask Error'])
        if results is not None:
            for k in results.keys():
                (sig_err, mask_err) = results[k]
                item1 = QStandardItem(str(k))
                item2 = QStandardItem('{:e}'.format(sig_err))
                item3 = QStandardItem('{:e}'.format(mask_err))
                self.model.appendRow([item1, item2, item3])

            self.update()

