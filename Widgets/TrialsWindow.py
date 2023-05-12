import numpy as np
from PySide6 import QtWidgets
from PySide6.QtWidgets import QTableWidgetItem, QTableWidget

import measurement


class TrialsWindow(QtWidgets.QWidget):
    def __init__(self, N, mask_count, number_iterations, trials_count, do_add_noise):
        super().__init__()

        lowest_mask_count = 1
        self.table = QTableWidget(self)
        self.table.setRowCount(trials_count + 2)

        headers = []
        for i in range(lowest_mask_count, mask_count + 1):
            headers.append('Trial #' + str(i))

        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)

        # Begin the alternating projection process. Store the errors for averaging later
        # and assign them to table cells as they are calculated.
        trial_errors = np.zeros((mask_count, trials_count), dtype=np.float_)
        for i in range(0, trials_count):
            for j in range(lowest_mask_count, mask_count + 1):
                # Calculate the new mask length required for the mask we are currently working against.
                m = j * N
                # We don't want to seed our random data.
                (_, _, _, error) = measurement.alternate_phase_projection(N, m, number_iterations, '', do_add_noise)
                trial_errors[j - 1, i] = error

                item = QTableWidgetItem()
                item.setData(0, '{:e}'.format(error))
                self.table.setItem(i, j - 1, item)

        # Calculate the average error for a given mask against all trial results.
        avg_errors = np.zeros(mask_count)
        for row in range(0, len(trial_errors)):
            avg_errors[row] = np.average(trial_errors[row, :])

        # Add two rows. The first is a header saying avg error and the second is the average itself.
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