import sys

from PySide6.QtWidgets import QApplication

import UI
from Commands import report_tables
from Commands.equivalent_algorithms import display_forward_and_backward_time_shift_equivalence

if __name__ == '__main__':
    print("Enter GUI or CLI?")
    cmd = input()

    if cmd == 'GUI':
        app = QApplication(sys.argv)
        # Create and show the form
        mainpage = UI.MainPage()
        mainpage.resize(800, 800)
        mainpage.show()
        # Run the main Qt loop
        sys.exit(app.exec())
    elif cmd == 'CLI':
        print("Enter command")
        cmd = input()

        if cmd == 'time shift equivalence':
            display_forward_and_backward_time_shift_equivalence()
        elif cmd == "Noise Reports":
            report_tables.unknown_mask_accuracy_vs_noise()
            report_tables.unknown_signal_and_mask_accuracy_vs_noise()



