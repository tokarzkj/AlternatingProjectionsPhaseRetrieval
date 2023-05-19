from PySide6 import QtCore, QtWidgets

from Widgets.alternating_projection import AlternatingProjectTab
from Widgets.modified_alternating_projection import ModifiedAlternatingProjectTab
from Widgets.signal_and_mask_recovery import SignalMaskRecovery


class MainPage(QtWidgets.QTabWidget):
    def __init__(self, parent=None):
        super().__init__()

        self.alternating_project_phase_retrieval_tab = AlternatingProjectTab()
        self.addTab(self.alternating_project_phase_retrieval_tab, "Alternating Projection")

        self.modified_alternating_project_phase_retrieval_tab = ModifiedAlternatingProjectTab()
        self.addTab(self.modified_alternating_project_phase_retrieval_tab, "Modified Alternating Projection")

        self.signal_and_mask_recovery = SignalMaskRecovery()
        self.addTab(self.signal_and_mask_recovery, "Signal and Mask Recovery")
