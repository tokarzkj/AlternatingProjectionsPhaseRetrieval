import numpy as np
from PySide6 import QtWidgets, QtCore
from PySide6.QtWidgets import QWidget, QTableWidgetItem, QTableWidget
from matplotlib import pyplot as plt
from numpy import real, imag

import measurement


class ModifiedAlternatingProjectTab(QWidget):
    def __init__(self, parent=None):
        super().__init__()

        self.trials_window = None
        self.number_iterations = None
        self.N = [10, 25, 50]
        self.mask_count = [4, 6, 8]
        self.number_iterations = 600
        self.trials_count = 25
        
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

        self.modified_graph_random_projection_button = QtWidgets.QPushButton('Modified Graph Recovery')
        self.modified_graph_random_projection_button.clicked.connect(self.modified_recovery_graph)

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
        self.layout.addWidget(self.snr_checkbox_label, 4, 0)
        self.layout.addWidget(self.snr_checkbox, 4, 1)
        self.layout.addWidget(self.modified_graph_random_projection_button, 5, 0)
        self.layout.addWidget(self.trials_button, 6, 0)

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

        (x, x_recon, phasefac, error) = measurement.modified_alternate_phase_projection(N, m, number_iterations, seed,
                                                                                        do_add_noise)

        print(error)

        self.graph_recovery(x, x_recon)

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
                    (_, _, _, error) = measurement.modified_alternate_phase_projection(n, m, self.number_iterations, "", False)
                    trial_errors[i] = error
                results.append(np.average(trial_errors))

            results_by_sample_size[n] = results

        self.trials_window = TrialsWindow(results_by_sample_size, self.N, self.mask_count)
        self.trials_window.resize(600, 800)
        self.trials_window.show()

    @staticmethod
    def graph_recovery(x, x_recon):
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
