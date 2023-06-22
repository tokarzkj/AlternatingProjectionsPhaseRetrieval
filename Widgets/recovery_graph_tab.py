import numpy as np
from PySide6 import QtWidgets, QtCore
from PySide6.QtWidgets import QWidget, QTableWidgetItem, QTableWidget
from matplotlib import pyplot as plt
from numpy import real, imag

import measurement
import utilities


class RecoveryGraphTab(QWidget):
    def __init__(self, parent=None):
        super().__init__()

        self.trials_window = None
        self.samples_label = QtWidgets.QLabel('Sample Count')
        self.samples_value = QtWidgets.QLineEdit()
        self.samples_value.setText('100')

        self.seed_label = QtWidgets.QLabel('Seed')
        self.seed_value = QtWidgets.QLineEdit()
        self.seed_value.setText('3140')

        self.mask_label = QtWidgets.QLabel('# of Masks')
        self.mask_combo_box = QtWidgets.QComboBox()
        self.mask_combo_box.addItem('2')
        self.mask_combo_box.addItem('3')
        self.mask_combo_box.addItem('4')
        self.mask_combo_box.addItem('5')
        self.mask_combo_box.addItem('6')
        self.mask_combo_box.addItem('7')
        self.mask_combo_box.addItem('8')
        self.mask_combo_box.addItem('9')
        self.mask_combo_box.addItem('10')
        self.mask_combo_box.setCurrentIndex(2)

        self.trials_label = QtWidgets.QLabel('# of Trials')
        self.trials_value = QtWidgets.QLineEdit()
        self.trials_value.setText('50')

        self.snr_checkbox_label = QtWidgets.QLabel('Add Noise?')
        self.snr_checkbox = QtWidgets.QCheckBox()

        self.graph_random_projection_button = QtWidgets.QPushButton('Graph Recovery')
        self.graph_random_projection_button.clicked.connect(self.graph)

        self.modified_graph_random_projection_button = QtWidgets.QPushButton('Modified Graph Recovery')
        self.modified_graph_random_projection_button.clicked.connect(self.modified_recovery_graph)

        self.error_reduced_graph_random_projection_button = QtWidgets.QPushButton('Error Reduced Graph Recovery')
        self.error_reduced_graph_random_projection_button.clicked.connect(self.error_reduced_recovery_graph)

        self.layout = QtWidgets.QGridLayout(self)
        self.layout.addWidget(self.samples_label, 0, 0)
        self.layout.addWidget(self.samples_value, 0, 1)
        self.layout.addWidget(self.seed_label, 1, 0)
        self.layout.addWidget(self.seed_value, 1, 1)
        self.layout.addWidget(self.mask_label, 2, 0)
        self.layout.addWidget(self.mask_combo_box, 2, 1)
        self.layout.addWidget(self.trials_label, 3, 0)
        self.layout.addWidget(self.trials_value, 3, 1)
        self.layout.addWidget(self.snr_checkbox_label, 4, 0)
        self.layout.addWidget(self.snr_checkbox, 4, 1)
        self.layout.addWidget(self.graph_random_projection_button, 5, 0)
        self.layout.addWidget(self.modified_graph_random_projection_button, 6, 0)
        self.layout.addWidget(self.error_reduced_graph_random_projection_button, 7, 0)

    @QtCore.Slot()
    def graph(self):
        N = int(self.samples_value.text())  # N Samples
        mask_count = int(self.mask_combo_box.currentText())
        m = mask_count * N
        number_iterations = 600

        # Need to set particular seed or the recovery values won't always align as expected
        # If you leave it blank the odds of success will be dependent on number of masks
        seed = self.seed_value.text()
        do_add_noise = self.snr_checkbox.isChecked()

        (x, x_recon, phasefac, error, _) = measurement.alternating_phase_projection_recovery(N, m, number_iterations, seed,
                                                                                          do_add_noise)

        print(error)

        self.graph_recovery(x, x_recon, 'Alternating Projection Recovery')

    @QtCore.Slot()
    def modified_recovery_graph(self):
        N = int(self.samples_value.text())  # N Samples
        mask_count = int(self.mask_combo_box.currentText())
        m = mask_count * N
        number_iterations = 600

        # Need to set particular seed or the recovery values won't always align as expected
        # If you leave it blank the odds of success will be dependent on number of masks
        seed = self.seed_value.text()
        do_add_noise = self.snr_checkbox.isChecked()

        (x, x_recon, phasefac, error, _) = measurement.modified_alternating_phase_projection_recovery(N, m, number_iterations, seed,
                                                                                                   do_add_noise)

        print(error)

        self.graph_recovery(x, x_recon, 'Modified Recovery')

    @QtCore.Slot()
    def error_reduced_recovery_graph(self):
        N = int(self.samples_value.text())  # N Samples
        mask_count = int(self.mask_combo_box.currentText())
        m = mask_count * N
        number_iterations = 600

        # Need to set particular seed or the recovery values won't always align as expected
        # If you leave it blank the odds of success will be dependent on number of masks
        seed = self.seed_value.text()
        do_add_noise = self.snr_checkbox.isChecked()

        (x, mask) = utilities.create_signal_and_mask(seed, N)

        (x, x_recon, _, _, phasefac, error, _) = measurement.alternating_phase_projection_recovery_with_error_reduction(
                                                                                                   N, m,
                                                                                                   number_iterations,
                                                                                                   do_add_noise,
                                                                                                   x,
                                                                                                   mask)

        print(error)

        self.graph_recovery(x, x_recon, 'Error Reduced Recovery')

    @staticmethod
    def graph_recovery(x, x_recon, name):
        fig, ax1 = plt.subplots(1, 2, num=name)

        fig.set_figheight(7)
        fig.set_figwidth(15)
        fig.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.1, hspace=0.75)
        ax1[0].set_title('Real Part')
        ax1[0].stem([real(e) for e in x], markerfmt='x', label='True')
        ax1[0].stem([real(e) for e in x_recon], linefmt='g--', markerfmt='+', label='Recovered')
        ax1[1].set_title('Imaginary Part')
        true2 = ax1[1].stem([imag(e) for e in x], markerfmt='x', label='True')
        recovered2 = ax1[1].stem([imag(e) for e in x_recon], linefmt='g--', markerfmt='+', label='Recovered')
        fig.legend(handles=[true2, recovered2])
        plt.show()
