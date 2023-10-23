__author__ = "Meizhu Liang @Imperial College London"

"""
Python code to control the NI USB 6351. Functions are tailored for HexSIM.
"""

import nidaqmx
from ScopeFoundry import HardwareComponent
from nidaqmx.constants import LineGrouping

class NI_device(object):
    def initiate_h(self):
        '''Adds channels and create a task for HexSIM'''
        self.task = nidaqmx.Task()
        self.task.do_channels.add_do_chan(
            "Dev1/port0/line0:3", line_grouping=LineGrouping.CHAN_FOR_ALL_LINES)
        self.task.do_channels.add_do_chan(lines='Dev1/port1/line0')

    def write_h(self, bl, read, yel):
        '''writes signal in two channels for HexSIM
        bl: blue high time; read: low time; also the readout time; yel:  yellow high time'''
        data1 = [0]  # trigger of camera/s exposure
        for f in range(14):
            for i in range(bl):
                data1.append(1)
            for i in range(read):
                data1.append(0)
            for i in range(yel):
                data1.append(1)
            for i in range(read):
                data1.append(0)
        data2 = []
        for i in range(14 * (bl + read * 2 + yel)):
            data2.append(1)
        data2.append(0)
        print(len(data1), len(data2))
        data = [data1, data2]
        print(self.task.write(data, auto_start=True))

    def initiate_p(self):
        '''Adds channels and create a task for phase recovery'''
        self.task = nidaqmx.Task()
        self.task.do_channels.add_do_chan(lines='Dev1/port1/line0')

    def write_p(self, v):
        '''writes value for phase recovery'''
        self.task.write(v)

    def close(self):
        self.task.close()
