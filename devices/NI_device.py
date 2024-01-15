__author__ = "Meizhu Liang @Imperial College London"

"""
Python code to control the NI USB 6351. Functions are tailored for HexSIM.
"""

import nidaqmx
import time
from nidaqmx.constants import LineGrouping

class NI_device(object):
    def initiate_h(self):
        '''Adds channels and create a task for HexSIM'''
        if hasattr(self, 'task'):
            self.close()
            del self.task
        self.task = nidaqmx.Task()
        self.task.do_channels.add_do_chan("Dev1/port0/line0:3", line_grouping=LineGrouping.CHAN_FOR_ALL_LINES)

    def write_h(self, bl, low, yel, n, frame_wait=0):
        '''writes signal in two channels for HexSIM
        bl: blue high time; low: low time; yel:  yellow high time; n: number of frames'''
        data1 = []  # trigger of camera/s exposure
        for f in range(n):
            data1.append(1)
            for w in range(bl + yel + 2 * low):
                data1.append(0)
            for k in range(7):
                for i in range(bl):
                    data1.append(1)
                for i in range(low):
                    data1.append(0)
                for i in range(yel):
                    data1.append(1)
                for i in range(low):
                    data1.append(0)
            # add wait time at the end for the z stage movement
            for i in range(frame_wait):
                data1.append(0)
            # data2 = [0]
            # for i in range(14 * (bl + low * 2 + yel)):
            #     data2.append(1)
            # for i in range(frame_wait):
            #     data2.append(0)
        self.task.timing.cfg_samp_clk_timing(1000, samps_per_chan=len(data1))
        print(f'HexSIM signal length: {len(data1)}')
        self.task.write(data1, auto_start=True)
        time.sleep(len(data1) / 1000)

    def initiate_p(self):
        '''Adds channels and create a task for phase recovery'''
        if hasattr(self, 'task'):
            self.close()
            del self.task
        self.task = nidaqmx.Task()
        self.task.do_channels.add_do_chan(lines='Dev1/port1/line1')

    def write_p(self, v):
        '''writes value for phase recovery'''
        self.task.write(v)

    def close(self):
        if hasattr(self, 'task'):
            self.task.close()
