from ScopeFoundry import Measurement
from ScopeFoundry.helper_funcs import sibling_path, load_qt_ui_file
from ScopeFoundry import h5_io
import pyqtgraph as pg
import numpy as np
from datetime import datetime
import os
import time

class HexSimMeasurement(Measurement):

    name = 'hexsim_measurement'

    def setup(self):
        # load ui file
        self.ui_filename = sibling_path(__file__,"hexsim_measurement.ui")
        self.ui = load_qt_ui_file(self.ui_filename)

        # camera settings
        # self.settings.New('record', dtype=bool, initial=False, hardware_set_func=self.setRecord, hardware_read_func=self.getRecord, reread_from_hardware_after_write=True)
        # self.settings.New('save_h5', dtype=bool, initial=False, hardware_set_func=self.setSaveH5, hardware_read_func=self.getSaveH5)
        # self.settings.New('refresh_period', dtype=float, unit='s', spinbox_decimals = 4, initial=0.02 , hardware_set_func=self.setRefresh, vmin = 0)
        # self.settings.New('autoRange', dtype=bool, initial=True, hardware_set_func=self.setautoRange)
        # self.settings.New('autoLevels', dtype=bool, initial=True, hardware_set_func=self.setautoLevels)
        # self.settings.New('level_min', dtype=int, initial=60, hardware_set_func=self.setminLevel, hardware_read_func = self.getminLevel)
        # self.settings.New('level_max', dtype=int, initial=150, hardware_set_func=self.setmaxLevel, hardware_read_func = self.getmaxLevel)
        # self.settings.New('threshold', dtype=int, initial=500, hardware_set_func=self.setThreshold)

        self.camera = self.app.hardware['HamamatsuHardware']
        self.screen = self.app.hardware['screenHardware']

        # self.autoRange = self.settings.autoRange.val
        # self.display_update_period = self.settings.refresh_period.val
        # self.autoLevels = self.settings.autoLevels.val
        # self.level_min = self.settings.level_min.val
        # self.level_max = self.settings.level_max.val

        # self.display_update_period = 0.02

    def setup_figure(self):
        self.ui.camButton.clicked.connect(self.camButtonPressed)
        self.ui.slmButton.clicked.connect(self.slmButtonPressed)
        self.ui.previousPatternButton.clicked.connect(self.screen.previousPattern)
        self.ui.nextPatternButton.clicked.connect(self.screen.nextPattern)

    def update_display(self):
        print('display here')

    def run(self):
        print('run here')

    # Actions
    def camButtonPressed(self):
        if self.ui.camButton.text()=='ON':
            # self.screen.slm_dev
            self.ui.camButton.setText('OFF')
            print('OFF')
        elif self.ui.camButton.text()=='OFF':
            self.ui.camButton.setText('ON')
            print('ON')

    def slmButtonPressed(self):
        if self.ui.slmButton.text()=='ON':
            self.screen.openSLM()
            self.screen.manualDisplay()
            self.ui.slmButton.setText('OFF')
            print('OFF')
        elif self.ui.slmButton.text()=='OFF':
            self.screen.closeSLM()
            self.ui.slmButton.setText('ON')
            print('ON')