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


"""Example for writing digital signal."""
import nidaqmx
from nidaqmx.constants import LineGrouping

# import nidaqmx.system as ni
# system = ni.System.local()
# device = system.devices[0]
# board = device.product_type + ' : ' + device.name
# terminals = []
# trig = []
# for line in device.co_physical_chans:
#     terminals.append(line.name)
# for j in device.terminals:
#     if 'PFI' in j:
#         trig.append(j)



# with nidaqmx.Task() as task:
task = nidaqmx.Task()
task.do_channels.add_do_chan(
    "Dev1/port0/line0:3", line_grouping=LineGrouping.CHAN_FOR_ALL_LINES)
task.do_channels.add_do_chan(lines='Dev1/port1/line0')

# try:
#     print("N Lines 1 Sample Boolean Write (Error Expected): ")
#     print(task.write([True, False, True, False]))
# except nidaqmx.DaqError as e:
#     print(e)

# print("1 Channel N Lines 1 Sample Unsigned Integer Write: ")
# print(task.write(8))

print("1 Channel N Lines N Samples Unsigned Integer Write: ")
data1 = [0]
bl = 50  # blue high time
l_t = 50  # low time; also the readout time
yel = 30  # yellow high time

for f in range(14):
    for i in range(bl):
        data1.append(1)
    for i in range(l_t):
        data1.append(0)
    for i in range(yel):
        data1.append(1)
    for i in range(l_t):
        data1.append(0)
data2 = []
for i in range(14 * (bl+l_t*2+yel)):
    data2.append(1)
data2.append(0)
print(len(data1), len(data2))
data = [data1, data2]

print(task.write(data, auto_start=True))
task.close()