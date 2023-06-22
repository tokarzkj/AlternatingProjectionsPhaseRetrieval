import sys

from PySide6.QtWidgets import QApplication

import UI
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
        display_forward_and_backward_time_shift_equivalence()



