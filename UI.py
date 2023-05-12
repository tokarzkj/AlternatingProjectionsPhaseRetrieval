from random import random

import numpy as np
from PySide6 import QtCore, QtWidgets
import scipy
from PySide6.QtWidgets import QTableWidget, QTableWidgetItem, QTabWidget, QWidget
from numpy import imag, real
import matplotlib.pyplot as plt

import measurement
from Widgets.TrialsWindow import TrialsWindow
from Widgets.alternating_projection import AlternatingProjectTab, ModifiedAlternatingProjectTab


class MainPage(QtWidgets.QTabWidget):
    def __init__(self, parent=None):
        super().__init__()

        self.alternating_project_phase_retrieval_tab = AlternatingProjectTab()
        self.addTab(self.alternating_project_phase_retrieval_tab, "Alternating Projection")

        self.modified_alternating_project_phase_retrieval_tab = ModifiedAlternatingProjectTab()
        self.addTab(self.modified_alternating_project_phase_retrieval_tab, "Modified Alternating Projection")
