__author__ = "Meizhu Liang @Imperial College London"

"""
Python code to control the NI USB 6351. Functions are tailored for HexSIM.
"""

import nidaqmx
from ScopeFoundry import HardwareComponent
from nidaqmx.constants import LineGrouping

class NI_hw(HardwareComponent):
    name = 'NI_hw'
    def initiate(self):
        '''Adds channels and create a task'''
        self.task = nidaqmx.Task()
        self.task.do_channels.add_do_chan(
            "Dev1/port0/line0:3", line_grouping=LineGrouping.CHAN_FOR_ALL_LINES)
        self.task.do_channels.add_do_chan(lines='Dev1/port1/line0')