import numpy as np
from PySide6 import QtWidgets, QtCore
from PySide6.QtWidgets import QWidget, QTableWidgetItem, QTableWidget
from matplotlib import pyplot as plt
from numpy import real, imag

import measurement


class RecoveryTrialsTab(QWidget):
    def __init__(self, parent=None):
        super().__init__()

        self.trials_window = None
        self.number_iterations = None
        self.N = [10, 25, 50]
        self.mask_count = [4, 6, 8]
        self.number_iterations = 600
        self.trials_count = 25

        self.trials_button = QtWidgets.QPushButton('Calc Trials')
        self.trials_button.clicked.connect(self.trials)

        self.modified_recovery_trials_button = QtWidgets.QPushButton('Modified Recovery Trials')
        self.modified_recovery_trials_button.clicked.connect(self.modified_recovery_trials)

        self.layout = QtWidgets.QGridLayout(self)
        self.layout.addWidget(self.trials_button, 0, 0)
        self.layout.addWidget(self.modified_recovery_trials_button, 1, 0)

    @QtCore.Slot()
    def trials(self):
        results_by_sample_size = dict()
        trials_count = 25
        for n in self.N:
            results = []
            for mc in self.mask_count:
                trial_errors = np.zeros(trials_count, dtype=np.float_)
                m = mc * n
                for i in range(0, trials_count):
                    (_, _, _, error) = measurement.alternate_phase_projection(n, m, self.number_iterations,
                                                                              3140, False)
                    trial_errors[i] = error
                results.append(np.average(trial_errors))

            results_by_sample_size[n] = results

        self.trials_window = TrialsWindow(results_by_sample_size, self.N, self.mask_count)
        self.trials_window.resize(600, 800)
        self.trials_window.show()

    @QtCore.Slot()
    def modified_recovery_trials(self):
        results_by_sample_size = dict()
        trials_count = 25
        for n in self.N:
            results = []
            for mc in self.mask_count:
                trial_errors = np.zeros(trials_count, dtype=np.float_)
                m = mc * n
                for i in range(0, trials_count):
                    (_, _, _, error) = measurement.modified_alternate_phase_projection(n, m, self.number_iterations,
                                                                                       3140, False)
                    trial_errors[i] = error
                results.append(np.average(trial_errors))

            results_by_sample_size[n] = results

        self.trials_window = TrialsWindow(results_by_sample_size, self.N, self.mask_count)
        self.trials_window.resize(600, 800)
        self.trials_window.show()


class TrialsWindow(QtWidgets.QWidget):
    def __init__(self, results_by_sample_size, N, mask_count):
        super().__init__()

        self.table = QTableWidget(self)
        self.table.setRowCount(3)

        headers = ['Sample #']
        for mc in mask_count:
            headers.append('Mask Count ' + str(mc))

        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)

        row = 0
        for n in N:
            results = results_by_sample_size[n]

            item = QTableWidgetItem()
            item.setData(0, 'Sample # ' + str(n))
            self.table.setItem(row, 0, item)

            for idx, value in enumerate(results):
                item = QTableWidgetItem()
                item.setData(0, '{:e}'.format(value))
                self.table.setItem(row, idx + 1, item)

            row += 1


        self.table.resize(600, 800)
        self.table.show()
