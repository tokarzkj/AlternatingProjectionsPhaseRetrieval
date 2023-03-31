import sys

import numpy as np
import scipy
import matplotlib.pyplot as plt
from PySide6 import QtWidgets
from PySide6.QtWidgets import QDialog, QApplication, QLineEdit, QPushButton, QVBoxLayout
from numpy import real, imag

import measurement
import UI


if __name__ == '__main__':
    app = QApplication(sys.argv)
    # Create and show the form
    mainpage = UI.MainPage()
    mainpage.resize(600, 800)
    mainpage.show()
    # Run the main Qt loop
    sys.exit(app.exec())



