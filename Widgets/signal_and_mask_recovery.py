import numpy as np
from PySide6 import QtWidgets, QtCore
from PySide6.QtGui import QStandardItemModel, QStandardItem
from PySide6.QtWidgets import QWidget, QTableWidgetItem, QTableWidget

import measurement


class SignalMaskRecovery(QWidget):
    def __init__(self, parent=None):
        super().__init__()

        self.N = 100


        #self.run_button = QtWidgets.QPushButton('Run')
        #self.run_button.clicked.connect(self.update_tables)

        self.layout = QtWidgets.QGridLayout(self)
        #self.layout.addWidget(self.run_button, 1, 0)

        x = np.random.rand(self.N) + 1J * np.random.rand(self.N)
        mask = np.random.rand(self.N) + 1J * np.random.rand(self.N)

        self.signal_table = SignalTableView(x, mask)
        self.layout.addWidget(self.signal_table, 0, 0)

        self.mask_table = MaskTableView(x, mask)
        self.layout.addWidget(self.mask_table, 0, 1)


class SignalTableView(QtWidgets.QTableView):
    def __init__(self, x, mask):
        super().__init__()
        self.N = 100
        m = 8 * self.N
        number_iterations = 600
        seed = 3140
        mask_estimate = perturb_vec(mask)

        (_, x_recon, _, error) = measurement.modified_alternate_phase_projection(self.N, m, number_iterations, seed,
                                                                                 False,
                                                                                 x=x, mask=mask_estimate)
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
        seed = 3140
        x_estimate = perturb_vec(x)

        (_, mask_recon, _, error) = measurement.modified_alternate_phase_projection(self.N, m, number_iterations, seed,
                                                                                    False,
                                                                                    x=mask, mask=x_estimate,
                                                                                    do_time_shift_signal=True)

        model = QStandardItemModel()
        model.setHorizontalHeaderLabels(['mask', 'recon mask'])
        for idx, value in enumerate(mask_recon):
            item1 = QStandardItem(str(mask[idx]))
            item2 = QStandardItem(str(value))
            model.appendRow([item1, item2])

        self.setModel(model)


def perturb_vec(vec: np.array):
    n = vec.shape[0]
    perturbation = np.random.rand(n) + 1J * np.random.rand(n)
    perturbation = np.multiply(perturbation, 1 / np.power(10, 4))

    return np.subtract(vec, perturbation)
