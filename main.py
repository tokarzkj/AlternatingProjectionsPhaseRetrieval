import sys

from PySide6.QtWidgets import QApplication

import UI
import measurement_2d
from Commands import report_tables
from Commands.equivalent_algorithms import display_forward_and_backward_time_shift_equivalence

if __name__ == '__main__':
    print("Enter GUI or CLI?")
    cmd = input()

    if cmd.lower() == 'gui':
        app = QApplication(sys.argv)
        # Create and show the form
        mainpage = UI.MainPage()
        mainpage.resize(800, 800)
        mainpage.show()
        # Run the main Qt loop
        sys.exit(app.exec())
    elif cmd.lower() == 'cli':
        print("Enter command")
        cmd = input()



        if cmd.lower() == 'time shift equivalence':
            display_forward_and_backward_time_shift_equivalence()
        elif cmd.lower() == "noise reports":
            report_tables.unknown_mask_accuracy_vs_noise()
            report_tables.unknown_signal_and_mask_accuracy_vs_noise()
        elif cmd.lower() == "iteration accuracy":
            report_tables.unknown_mask_iteration_vs_error()
            report_tables.unknown_signal_and_mask_iteration_vs_error()
        elif cmd.lower() == "sample timing":
            report_tables.unknown_mask_sample_size_vs_time()
            report_tables.unknown_signal_and_unknown_mask_sample_size_vs_time()
        elif cmd.lower() == "2d":
            measurement_2d.alternating_projection_recovery_2d(40, 42)



