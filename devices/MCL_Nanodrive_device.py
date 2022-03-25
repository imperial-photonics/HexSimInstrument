"""
Python code to control the Mad City Labs controller (MCL, Nano-Drive® C, 00-55-550-0800) of the piezo stage.
"""

import ctypes as ct
import numpy as np
import time


class MCLPiezo(object):
    def __init__(self):
        self.mcl = ct.windll.LoadLibrary('C:/Program Files/Mad City Labs/NanoDrive/Madlib.dll')
        num_check = self.mcl.MCL_NumberOfCurrentHandles()
        if num_check == 0:
            # create a new handle or get the existing one if the handle is not released properly.
            self.handle = self.mcl.MCL_InitHandleOrGetExisting()
        else:
            print('The current handle number is not zero!\n Handle number: ' + str(num_check))

        # print the device information
        self.mcl.MCL_PrintDeviceInfo(self.handle)


        # connect to the instrument
        if self.handle:
            print("Connected to the device with handle: " + str(self.handle))
        else:
            print("Failure to connect the instrument. Make sure the piezo stage is turned on.")

    def singleReadZ(self):
        self.mcl.MCL_SingleReadZ.restype = ct.c_double
        re = self.mcl.MCL_SingleReadZ(self.handle)
        # re: return value
        print('Current position: ' + str(re))

    def monitorZ(self, position):
        if position < 0:
            self.ceaseOperation()
            print('The command position is too low! Operation ceased.')
        elif position > 300:
            self.ceaseOperation()
            print('The command voltage is too high! Operation ceased.')
        else:
            self.mcl.MCL_MonitorZ.restype = ct.c_double
            re = self.mcl.MCL_MonitorZ(ct.c_double(position), self.handle)
            print('Position is moved from ' + str(re))

    def WfLoad(self):
        """
        This function sets up and trigger a waveform load on z axis.
        """
        pyarray = np.linspace(0, 150, 251)
        array = (ct.c_double * len(pyarray))(* pyarray)


        start = time.time()
        print("hello")

        self.mcl.MCL_Setup_LoadWaveFormN(3, len(pyarray),  ct.c_double(2), array, self.handle)
        self.mcl.MCL_Setup_ReadWaveFormN(3, len(pyarray), ct.c_double(2), self.handle)
        re = self.mcl.MCL_TriggerWaveformAcquisition(3, len(pyarray), array,  self.handle)

        # self.mcl.MCL_Trigger_LoadWaveFormN(3, self.handle)

        # re = self.mcl.MCL_LoadWaveFormN(3, len(pyarray), ct.c_double(2), array, self.handle)

        if re == 0:
            print('Waveform loading started.')
        else:
            print(re)
        # print(r2)
        end = time.time()
        print(end - start)
        return array

    def WfRead(self):
        points = 200
        waveform_type = points * ct.c_double
        waveform = waveform_type()
        pyarray = np.zeros(200)
        # pyarray = []
        array = (ct.c_double * len(pyarray))(*pyarray)
        re = self.mcl.MCL_ReadWaveFormN(3, len(pyarray), ct.c_double(2), waveform, self.handle)
        if re == 0:
            print('Waveform reading started.')
        else:
            print(re)
        return waveform
        # print(type(waveform))
        # ct.cast(waveform, ct.c_void_p).value
        # print(str(waveform[0]))
        # self.mcl.MCL_LineClock(self.handle)


    def bindClock(self):
        r1 = self.mcl.MCL_IssSetClock(2, 1, self.handle)
        r2 = self.mcl.MCL_IssBindClockToAxis(2, 3, 3, self.handle)
        print(r1)
        print(r2)

    def ceaseOperation(self):
        """
        This function moves the stage to 0 and release all handles.
        """
        self.mcl.MCL_MonitorZ(ct.c_double(0), self.handle)
        self.mcl.MCL_ReleaseAllHandles()

    def shutDown(self):
        mcl.monitorZ(0)
        time.sleep(3)
        self.mcl.MCL_ReleaseHandle(self.handle)
        num = self.mcl.MCL_NumberOfCurrentHandles()
        print(num)
        if num != 0:
            print('The handle still exist!')




if __name__ == "__main__":
    import ctypes as ct

    mcl = MCLPiezo()
    mcl.singleReadZ()
    mcl.monitorZ(0)
    time.sleep(2)
    mcl.singleReadZ()
    # mcl.bindClock()
    # mcl.WfLoad()
    # mcl.monitorZ(0)
    # mcl.WfRead()
    # data = [mcl.WfRead()]
    # time.sleep(2)
    data = [mcl.WfLoad()]
    fp = open("stage_data.txt", "w")
    for i in range(250):
        for datum in data:
            fp.write(str(datum[i]) + ",")
        fp.write("\n")
    # time.sleep(1)
    mcl.singleReadZ()
    mcl.shutDown()

