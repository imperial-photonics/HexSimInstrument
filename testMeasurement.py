from ScopeFoundry import Measurement
from ScopeFoundry.helper_funcs import sibling_path, load_qt_ui_file
from ScopeFoundry import h5_io
import pyqtgraph as pg
import numpy as np
from datetime import datetime
import os
import time

class testMeasurement(Measurement):

    name = 'test_measurement'

    def setup(self):
        # load ui file
        self.ui_filename = sibling_path(__file__,"test.ui")

        self.ui = load_qt_ui_file(self.ui_filename)
        self.display_update_period = 0.02

    def setup_figure(self):
        self.ui.camButton.clicked.connect(self.camButtonPressed)

    def update_display(self):
        print('display here')

    def run(self):
        print('run here')

    # Actions
    def camButtonPressed(self):
        if self.ui.camButton.text()=='ON':
            self.ui.camButton.setText('OFF')
            print('OFF')
        elif self.ui.camButton.text()=='OFF':
            self.ui.camButton.setText('ON')
            print('ON')