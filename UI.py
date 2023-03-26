from random import random

import numpy as np
from PySide6 import QtCore, QtWidgets
import scipy
from PySide6.QtWidgets import QTableWidget, QTableWidgetItem
from numpy import imag, real
import matplotlib.pyplot as plt

import measurement


class MainPage(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__()

        self.trials_window = None
        self.samples_label = QtWidgets.QLabel('Sample Count')
        self.samples_value = QtWidgets.QLineEdit()
        self.samples_value.setText('100')

        self.seed_label = QtWidgets.QLabel('Seed')
        self.seed_value = QtWidgets.QLineEdit()
        self.seed_value.setText('3140')

        self.mask_label = QtWidgets.QLabel('Seed')
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

        self.graph_random_projection_button = QtWidgets.QPushButton('Graph Recovery')
        self.graph_random_projection_button.clicked.connect(self.graph)

        self.trials_button = QtWidgets.QPushButton('Calc Trials')
        self.trials_button.clicked.connect(self.trials)

        self.layout = QtWidgets.QGridLayout(self)
        self.layout.addWidget(self.samples_label, 0, 0)
        self.layout.addWidget(self.samples_value, 0, 1)
        self.layout.addWidget(self.seed_label, 1, 0)
        self.layout.addWidget(self.seed_value, 1, 1)
        self.layout.addWidget(self.mask_label, 2, 0)
        self.layout.addWidget(self.mask_combo_box, 2, 1)
        self.layout.addWidget(self.trials_label, 3, 0)
        self.layout.addWidget(self.trials_value, 3, 1)
        self.layout.addWidget(self.graph_random_projection_button, 4, 0)
        self.layout.addWidget(self.trials_button, 5, 0)

    @QtCore.Slot()
    def graph(self):
        N = int(self.samples_value.text())  # N Samples
        mask_count = int(self.mask_combo_box.currentText())
        m = mask_count * N
        number_iterations = 600

        # Need to set particular seed or the recovery values won't always align as expected
        # If you leave it blank the odds of success will be dependent on number of masks
        seed = self.seed_value.text()

        (x, x_recon, phasefac, error) = measurement.alternate_phase_projection(N, m, number_iterations, seed)

        print(error)

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

    @QtCore.Slot()
    def trials(self):
        N = int(self.samples_value.text())  # N Samples
        mask_count = int(self.mask_combo_box.currentText())
        number_iterations = 600
        seed = self.seed_value.text()
        trials_count = int(self.trials_value.text())

        self.trials_window = TrialsWindow(N, mask_count,number_iterations, trials_count)
        self.trials_window.resize(600, 800)
        self.trials_window.show()


class TrialsWindow(QtWidgets.QWidget):
    def __init__(self, N, mask_count, number_iterations, trials_count):
        super().__init__()

        lowest_mask_count = 1
        self.table = QTableWidget(self)
        self.table.setRowCount(trials_count + 2)

        headers = []
        for i in range(lowest_mask_count, mask_count + 1):
            headers.append('Trial #' + str(i))

        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)

        trial_errors = np.zeros((mask_count, trials_count), dtype=np.float_)
        for i in range(0, trials_count):
            for j in range(lowest_mask_count, mask_count + 1):
                m = j * N
                (_, _, _, error) = measurement.alternate_phase_projection(N, m, number_iterations, '')
                trial_errors[j - 1, i] = error

                item = QTableWidgetItem()
                item.setData(0, '{:e}'.format(error))
                self.table.setItem(i, j - 1, item)

        avg_errors = np.zeros(mask_count)
        for row in range(0, len(trial_errors)):
            avg_errors[row] = np.average(trial_errors[row, :])

        for i in range(0, mask_count):
            row = trials_count

            item = QTableWidgetItem()
            item.setData(0, 'Avg Error' + str(i + 1))
            self.table.setItem(row, i, item)

            item = QTableWidgetItem()
            item.setData(0, '{:e}'.format(avg_errors[i]))
            self.table.setItem(row + 1, i, item)

        self.table.resize(600, 800)
        self.table.show()

