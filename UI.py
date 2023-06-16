from PySide6 import QtCore, QtWidgets

from Widgets.recovery_graph_tab import RecoveryGraphTab
from Widgets.recovery_trials_tab import RecoveryTrialsTab
from Widgets.signal_and_mask_recovery import SignalMaskRecovery
from Widgets.signal_and_mask_recovery_demo_tab import SignalMaskRecoveryDemo


class MainPage(QtWidgets.QTabWidget):
    def __init__(self, parent=None):
        super().__init__()

        self.alternating_project_phase_retrieval_tab = RecoveryGraphTab()
        self.addTab(self.alternating_project_phase_retrieval_tab, "Recovery Graph")

        self.modified_alternating_project_phase_retrieval_tab = RecoveryTrialsTab()
        self.addTab(self.modified_alternating_project_phase_retrieval_tab, "Trials")

        self.signal_and_mask_recovery = SignalMaskRecovery()
        self.addTab(self.signal_and_mask_recovery, "Signal and Mask Recovery")

        self.signal_and_mask_recovery_demo = SignalMaskRecoveryDemo()
        self.addTab(self.signal_and_mask_recovery_demo, "Signal and Mask Recovery Demo")
