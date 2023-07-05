import numpy as np
from PySide6 import QtWidgets, QtCore
from PySide6.QtGui import QStandardItemModel, QStandardItem
from PySide6.QtWidgets import QWidget
from matplotlib import pyplot as plt
from numpy import real, imag

import measurement
from utilities import perturb_vec


class SignalMaskRecoveryDemo(QWidget):
    def __init__(self, parent=None):
        super().__init__()

        self.alternating_projection_algorithm = 'Alternating Projection Algorithm'
        self.alternating_projection_algorithm_error_reduction = 'Error Reduced Alternating Projection Algorithm'

        self.N = 25
        self.run_recovery_example_button = QtWidgets.QPushButton('Run example')
        self.run_recovery_example_button.setToolTip(
            'Run the recovery process and show details about signal and results')
        self.run_recovery_example_button.clicked.connect(self.run_recovery_example)
        self.error_results_table = ErrorResultsTableView()

        self.algorithm_selection_box = QtWidgets.QComboBox()
        self.algorithm_selection_box.addItem(self.alternating_projection_algorithm)
        self.algorithm_selection_box.addItem(self.alternating_projection_algorithm_error_reduction)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.algorithm_selection_box)
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

        mask_error, mask_iterative_error, mask_recon, signal_error, signal_iterative_error, x_recon = self.get_table_results(
            x, mask, m, mask_estimate, number_iterations, x_estimate)

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

    def get_table_results(self, x, mask, m, mask_estimate, number_iterations, x_estimate):
        if self.algorithm_selection_box.currentText() == self.alternating_projection_algorithm:
            (_, x_recon, _, signal_error, signal_recon_iterations) = \
                measurement.modified_alternating_phase_projection_recovery(self.N, m,
                                                                           number_iterations,
                                                                           x_estimate,
                                                                           mask)

            (_, mask_recon, _, mask_error, mask_recon_iterations) = \
                measurement.modified_alternating_phase_projection_recovery(self.N, m,
                                                                           number_iterations,
                                                                           mask_estimate,
                                                                           x)

            signal_iterative_error = dict()
            for k in signal_recon_iterations.keys():
                sig_recon = signal_recon_iterations[k]
                signal_iterative_error[k] = np.linalg.norm(x - sig_recon) / np.linalg.norm(x)

            mask_iterative_error = dict()
            for k in mask_recon_iterations.keys():
                m_recon = mask_recon_iterations[k]
                mask_iterative_error[k] = np.linalg.norm(mask - m_recon) / np.linalg.norm(mask)
        elif self.algorithm_selection_box.currentText() == self.alternating_projection_algorithm_error_reduction:
            (x_recon, _, _, _, signal_error,
             signal_iterative_error) = measurement.alternating_phase_projection_recovery_with_error_reduction(self.N, m,
                                                                                                              number_iterations,
                                                                                                              x_estimate,
                                                                                                              mask_estimate)

            (mask_recon, _, _, _, mask_error,
             mask_iterative_error) = measurement.alternating_phase_projection_recovery_with_error_reduction(self.N, m,
                                                                                                            number_iterations,
                                                                                                            mask_estimate,
                                                                                                            x_estimate)

        return mask_error, mask_iterative_error, mask_recon, signal_error, signal_iterative_error, x_recon


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
