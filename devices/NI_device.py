__author__ = "Meizhu Liang @Imperial College London"

"""
Python code to control the NI USB 6351. Functions are tailored for HexSIM.
"""

import nidaqmx
from ScopeFoundry import HardwareComponent
from nidaqmx.constants import LineGrouping

class NI_CO_hw(HardwareComponent):
    name = 'NI_CO_hw'

    def setup(self):
        # create logged quantities, that are related to the graphical interface
        board, terminals, trig = self.update_channels()

        self.devices = self.add_logged_quantity('device', dtype=str, initial=board)
        self.channel1 = self.add_logged_quantity('Ext_exposure', dtype=str, choices=terminals, initial=terminals[0])
        self.high_time1 = self.add_logged_quantity('high_time(exp)', dtype=float, initial=0.450,
                                                   vmin=0.0001,spinbox_decimals=3, unit='s')
        self.low_time1 = self.add_logged_quantity('low_time(exp)', dtype=float, initial=0.050, vmin=0.0001,
                                                  spinbox_decimals=3, unit='s')
        self.channel2 = self.add_logged_quantity('Ext_run', dtype=str, choices=terminals, initial=terminals[1])
        self.high_time2 = self.add_logged_quantity('high_time', dtype=float, initial=1.2, vmin=0.0001,
                                                   spinbox_decimals=3, unit='s')
        self.period2 = self.add_logged_quantity('period', dtype=float, initial=1.250, vmin=0.0001,
                                                spinbox_decimals=3, unit='s')
        self.initial_delay_chan1 = self.add_logged_quantity('initial_delay_chan1', dtype=float, initial=0, vmin=0,
                                                            spinbox_decimals=6, unit='s')
        self.initial_delay_chan2 = self.add_logged_quantity('initial_delay_chan2', dtype=float, initial=0, vmin=0,
                                                            spinbox_decimals=6, unit='s')

        self.trigger = self.add_logged_quantity('trigger', dtype=bool, initial=True)
        self.trigger_source = self.add_logged_quantity('trigger_source', dtype=str, choices=trig, initial=trig[1])
        self.trigger_edge = self.add_logged_quantity('trigger_edge', dtype=str, choices=['rising', 'falling'],
                                                     initial='rising')

        self.add_operation("start_task", self.start)
        self.add_operation("stop_task", self.stop)



"""Example for writing digital signal."""
import nidaqmx
from nidaqmx.constants import LineGrouping

import nidaqmx.system as ni
system = ni.System.local()
device = system.devices[0]
board = device.product_type + ' : ' + device.name
terminals = []
trig = []
for line in device.co_physical_chans:
    terminals.append(line.name)
for j in device.terminals:
    if 'PFI' in j:
        trig.append(j)



with nidaqmx.Task() as task:
    task.do_channels.add_do_chan(
        "Dev1/port0/line0:3", line_grouping=LineGrouping.CHAN_FOR_ALL_LINES)

    # try:
    #     print("N Lines 1 Sample Boolean Write (Error Expected): ")
    #     print(task.write([True, False, True, False]))
    # except nidaqmx.DaqError as e:
    #     print(e)

    # print("1 Channel N Lines 1 Sample Unsigned Integer Write: ")
    # print(task.write(8))

    print("1 Channel N Lines N Samples Unsigned Integer Write: ")
    data = [0]
    bl = 50  # blue high time
    l_t = 50  # low time; also the readout time
    yel = 30  # yellow high time

    for f in range(14):
        for i in range(bl):
            data.append(1)
        for i in range(l_t):
            data.append(0)
        for i in range(yel):
            data.append(1)
        for i in range(l_t):
            data.append(0)

    print(task.write(data, auto_start=True))