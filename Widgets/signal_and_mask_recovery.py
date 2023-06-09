import numpy as np
from PySide6 import QtWidgets, QtCore
from PySide6.QtGui import QStandardItemModel, QStandardItem
from PySide6.QtWidgets import QWidget

import measurement
from utilities import perturb_vec


class SignalMaskRecovery(QWidget):
    def __init__(self, parent=None):
        super().__init__()

        self.N = 100
        self.trial_count = 25

        self.run_button = QtWidgets.QPushButton('Run average')
        self.run_button.setToolTip(f'Run the recovery {self.trial_count} times and calculate '
                                   f'average error for the recovered signal and mask')
        self.run_button.clicked.connect(self.run_average)

        self.layout = QtWidgets.QGridLayout(self)
        self.layout.addWidget(self.run_button, 1, 0)

        x = np.random.rand(self.N) + 1J * np.random.rand(self.N)
        mask = np.random.rand(self.N) + 1J * np.random.rand(self.N)

        self.table_layout = QtWidgets.QGridLayout(self)
        self.signal_table = SignalTableView(x, mask)
        self.table_layout.addWidget(self.signal_table, 0, 0)

        self.mask_table = MaskTableView(x, mask)
        self.table_layout.addWidget(self.mask_table, 0, 1)

        self.layout.addLayout(self.table_layout, 0, 0, 1, -1)

        self.avg_signal_label = QtWidgets.QLabel('Avg error for Signal')
        self.layout.addWidget(self.avg_signal_label, 1, 1)

        self.avg_signal_value = QtWidgets.QLineEdit()
        self.avg_signal_value.setReadOnly(True)
        self.avg_signal_value.setText('N/A')
        self.layout.addWidget(self.avg_signal_value, 2, 1)

        self.avg_mask_label = QtWidgets.QLabel('Avg error for Mask')
        self.layout.addWidget(self.avg_mask_label, 1, 2)

        self.avg_mask_value = QtWidgets.QLineEdit()
        self.avg_mask_value.setReadOnly(True)
        self.avg_mask_value.setText('N/A')
        self.layout.addWidget(self.avg_mask_value, 2, 2)

        self.run_average_unknown_recovery_button = QtWidgets.QPushButton('Run average for unknowns')
        self.run_average_unknown_recovery_button.setToolTip(f'Run the recovery {self.trial_count} times and calculate '
                                   f'average error for the recovered signal and mask from an unknown signal and mask')
        self.run_average_unknown_recovery_button.clicked.connect(self.run_average_unknown_recovery)

        self.layout.addWidget(self.run_average_unknown_recovery_button, 3, 0)

        self.avg_unknown_signal_label = QtWidgets.QLabel('Avg error for unknown Signal')
        self.layout.addWidget(self.avg_unknown_signal_label, 3, 1)

        self.avg_unknown_signal_value = QtWidgets.QLineEdit()
        self.avg_unknown_signal_value.setReadOnly(True)
        self.avg_unknown_signal_value.setText('N/A')
        self.layout.addWidget(self.avg_unknown_signal_value, 4, 1)

        self.avg_unknown_mask_label = QtWidgets.QLabel('Avg error for unknown Mask')
        self.layout.addWidget(self.avg_unknown_mask_label, 3, 2)

        self.avg_unknown_mask_value = QtWidgets.QLineEdit()
        self.avg_unknown_mask_value.setReadOnly(True)
        self.avg_unknown_mask_value.setText('N/A')
        self.layout.addWidget(self.avg_unknown_mask_value, 4, 2)


    @QtCore.Slot()
    def run_average(self):
        """
        Runs a specified number of trials to calculate the average error when recovering the signal and mask
        """
        m = 8 * self.N
        number_iterations = 600
        seed = 3140

        signal_trial_errors = np.zeros(self.trial_count, dtype=np.float_)
        mask_trial_errors = np.zeros(self.trial_count, dtype=np.float_)

        for i in range(0, self.trial_count):
            x = np.random.rand(self.N) + 1J * np.random.rand(self.N)
            mask = np.random.rand(self.N) + 1J * np.random.rand(self.N)
            mask_estimate = perturb_vec(mask)
            x_estimate = perturb_vec(x)
            (_, x_recon, _, signal_error, _) = measurement.modified_alternating_phase_projection_recovery(self.N, m,
                                                                                                       number_iterations,
                                                                                                       x, mask_estimate)
            signal_trial_errors[i] = signal_error

            (_, mask_recon, _, mask_error, _) = measurement.modified_alternating_phase_projection_recovery(self.N, m,
                                                                                                        number_iterations,
                                                                                                        mask, x_estimate)
            mask_trial_errors[i] = mask_error

        avg_trial_error = np.average(signal_trial_errors)
        avg_mask_error = np.average(mask_trial_errors)

        self.avg_signal_value.setText('{:e}'.format(avg_trial_error))
        self.avg_mask_value.setText('{:e}'.format(avg_mask_error))

    @QtCore.Slot()
    def run_average_unknown_recovery(self):
        """
        Runs a specified number of trials to calculate the average error when recovering the signal and mask
        """
        m = 8 * self.N
        number_iterations = 600

        signal_trial_errors = np.zeros(self.trial_count, dtype=np.float_)
        mask_trial_errors = np.zeros(self.trial_count, dtype=np.float_)

        for i in range(0, self.trial_count):
            x = np.random.rand(self.N) + 1J * np.random.rand(self.N)
            mask = np.random.rand(self.N) + 1J * np.random.rand(self.N)
            mask_estimate = perturb_vec(mask)
            x_estimate = perturb_vec(x)
            (_, x_recon, _, signal_error, _) = measurement.modified_alternating_phase_projection_recovery(self.N, m,
                                                                                                       number_iterations,
                                                                                                       x_estimate,
                                                                                                       mask_estimate)
            signal_trial_errors[i] = signal_error

            (_, mask_recon, _, mask_error, _) = measurement.modified_alternating_phase_projection_recovery(self.N, m,
                                                                                                        number_iterations,
                                                                                                        mask_estimate,
                                                                                                        x_estimate)
            mask_trial_errors[i] = mask_error

        avg_trial_error = np.average(signal_trial_errors)
        avg_mask_error = np.average(mask_trial_errors)

        self.avg_unknown_signal_value.setText('{:e}'.format(avg_trial_error))
        self.avg_unknown_mask_value.setText('{:e}'.format(avg_mask_error))


class SignalTableView(QtWidgets.QTableView):
    def __init__(self, x, mask):
        super().__init__()
        self.N = 100
        m = 8 * self.N
        number_iterations = 600
        mask_estimate = perturb_vec(mask)

        (_, x_recon, _, error, _) = measurement.modified_alternating_phase_projection_recovery(self.N, m, number_iterations,
                                                                                               x, mask_estimate)
        model = QStandardItemModel()
        model.setHorizontalHeaderLabels(['x', 'recon x'])
        for idx, value in enumerate(x_recon):
            item1 = QStandardItem(str(x[idx]))
            item2 = QStandardItem(str(value))
            model.appendRow([item1, item2])

        self.setModel(model)


class MaskTableView(QtWidgets.QTableView):
    def __init__(self, x, mask):
        super().__init__()
        self.N = 100
        m = 8 * self.N
        number_iterations = 600
        x_estimate = perturb_vec(x)

        (_, mask_recon, _, error, _) = measurement.modified_alternating_phase_projection_recovery(self.N, m, number_iterations,
                                                                                                  mask, x_estimate)

        model = QStandardItemModel()
        model.setHorizontalHeaderLabels(['mask', 'recon mask'])

        for idx, value in enumerate(mask_recon):
            item1 = QStandardItem(str(mask[idx]))
            item2 = QStandardItem(str(value))
            model.appendRow([item1, item2])

        self.setModel(model)
