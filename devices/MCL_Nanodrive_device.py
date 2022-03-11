"""
Python code to control the Mad City Labs controller (MCL, Nano-Drive® C, 00-55-550-0800) of the piezo stage.
"""

import ctypes as ct
import time


class MCLPiezo(object):
    def __init__(self):
        self.piezo = ct.windll.LoadLibrary('C:/Program Files/Mad City Labs/NanoDrive/Madlib.dll')
        # release existing handles
        self.piezo.MCL_ReleaseAllHandles()
        # create a new handle
        self.handle = self.piezo.MCL_InitHandle()
        # print the device information
        self.piezo.MCL_PrintDeviceInfo(self.handle)
        # connect to the instrument
        if self.handle != 0:
            print("Connected to the device with handle: " + str(self.handle))
        else:
            print("Failure to connect the instrument. Make sure the piezo stage is turned on.")

    def singleReadZ(self):
        self.piezo.MCL_SingleReadZ.restype = ct.c_double
        re = self.piezo.MCL_SingleReadZ(self.handle)
        # re: return value
        print('Current position: ' + str(re))

    def monitorZ(self, po):
        position = ct.c_double(po)
        self.piezo.MCL_MonitorZ.restype = ct.c_double
        re = self.piezo.MCL_MonitorZ(position, self.handle)
        print('Position is moved from ' + str(re))

    def WfSetup(self):
        """
        This function sets up and trigger a waveform on z axis.
        """
        pyarray = [0, 1, 2, 3, 4, 5]
        array = (ct.c_double * len(pyarray))(* pyarray)
        # x = axis[0]
        # y = axis[0]
        # z = axis[1]
        # *x, = axis
        # *y, = axis
        # *z, = axis
        self.piezo.MCL_Setup_LoadWaveFormN(3, 1000, ct.c_double(2), ct.pointer(array), self.handle)
        self.piezo.MCL_Setup_ReadWaveFormN(3, 1000, ct.c_double(2), self.handle)
        re = self.piezo.MCL_TriggerWaveformAcquisition(3, 1000, ct.pointer(array), self.handle)
        if re == 0:
            print("Waveform is prepared")
        else:
            print('fail!' + str(re))


if __name__ == "__main__":
    import ctypes as ct

    mcl = MCLPiezo()
    mcl.singleReadZ()
    mcl.monitorZ(0.000000)
    time.sleep(1)
    mcl.singleReadZ()
    # mcl.WfSetup()
