import sys

from PySide6.QtWidgets import QApplication

import UI


if __name__ == '__main__':
    app = QApplication(sys.argv)
    # Create and show the form
    mainpage = UI.MainPage()
    mainpage.resize(600, 800)
    mainpage.show()
    # Run the main Qt loop
    sys.exit(app.exec())



