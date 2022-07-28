"""
Python code to control the Mad City Labs controller (MCL, Nano-Drive® C, 00-55-550-0800) of the piezo stage.
"""

import ctypes as ct
import numpy as np
import time
import matplotlib.pyplot as plt


class MCLPiezo(object):
    def __init__(self):
        self.mcl = ct.windll.LoadLibrary('C:/Program Files/Mad City Labs/NanoDrive/Madlib.dll')
        num_check = self.mcl.MCL_NumberOfCurrentHandles()
        if num_check == 0:
            # create a new handle or get the existing one if the handle is not released properly.
            self.handle = self.mcl.MCL_InitHandleOrGetExisting()
        else:
            print('The current handle number is not zero!\n Handle number: ' + str(num_check))
        self.mcl.MCL_PrintDeviceInfo(self.handle)  #print the device information
        # connect to the instrument
        if self.handle:
            print("Connected to the device with handle: " + str(self.handle))
        else:
            raise Exception("Failure to connect the instrument. Make sure the piezo stage is turned on.")

    def singleReadZ(self):
        self.mcl.MCL_SingleReadZ.restype = ct.c_double
        return self.mcl.MCL_SingleReadZ(self.handle)
            # 'Current position: ' + str(re) + 'μm'

    def monitorZ(self, position):
        if (position < 0) | (position > 300):
            raise ValueError('The command position should be from 0 to 300 inclusive.')
        else:
            self.mcl.MCL_MonitorZ.restype = ct.c_double
            re1 = self.mcl.MCL_MonitorZ(ct.c_double(position), self.handle)
            print(f'Position is moved from {re1} to {self.singleReadZ()}μm')

    def WfAcquisition(self, sign):
        """
        This function sets up load and read waveform functions and then triggers them simultaneously.
        """

        self.t = np.linspace(0, 9999, 10000)
        self.pyarray = 140 + sign * 140 * np.cos(self.t * 2 * np.pi / 600)
        array = (ct.c_double * len(self.pyarray))(* self.pyarray)
        pyarray_out = np.zeros_like(self.pyarray)
        self.array_out = (ct.c_double * len(self.pyarray))(* pyarray_out)

        start = time.time()

        re1 = self.mcl.MCL_Setup_LoadWaveFormN(3, len(self.pyarray), ct.c_double(0.1), array, self.handle)
        time.sleep(0.01)
        re2 = self.mcl.MCL_Setup_ReadWaveFormN(3, len(self.pyarray), ct.c_double(0.1), self.handle)
        time.sleep(0.01)
        re3 = self.mcl.MCL_TriggerWaveformAcquisition(3, len(self.pyarray), self.array_out,  self.handle)

        # plt.figure()
        # plt.plot(self.t, np.array(self.array_out))
        # plt.plot(self.t, self.pyarray)
        # plt.figure()
        # plt.plot(self.pyarray, np.array(self.array_out))

        # self.mcl.MCL_Trigger_LoadWaveFormN(3, self.handle)

        # re = self.mcl.MCL_LoadWaveFormN(3, len(pyarray), ct.c_double(2), array, self.handle)

        if re1 + re2 + re3 == 0:
            print('Waveform loading started.')
        else:
            print(re1, re2, re3)
        # print(r2)
        end = time.time()
        print(end - start)
        return self.array_out

    def WfRead(self):
        pyarray = 140 + 140 * np.cos(self.t * 2 * np.pi / 600)
        points = len(self.pyarray)
        waveform_type = points * ct.c_double
        waveform = waveform_type()
        re = self.mcl.MCL_ReadWaveFormN(3, points, ct.c_double(2), waveform, self.handle)
        if re == 0:
            print('Waveform reading started.')
        else:
            print(re)
        return waveform

    def bindClock(self):
        r1 = self.mcl.MCL_FrameClock(self.handle)
        # r1 = self.mcl.MCL_IssSetClock(2, 1, self.handle)
        r2 = self.mcl.MCL_IssBindClockToAxis(3, 2, 6, self.handle)
        print(r1)
        print(r2)

    def clock(self):
        print(self.mcl.MCL_IssSetClock(2, 1, self.handle))

    def characterisation(self, sign):
        """
        This function characterises the stage by running the waveform acquisition function and plotting figures.
        """
        self.WfAcquisition(sign)
        plt.figure()
        # Plot loaded and resultant waveforms (input and output voltages)
        plt.plot(self.t, np.array(self.array_out))
        plt.plot(self.t + 244, self.pyarray)
        plt.figure()
        # Plot relationship between the input and output voltages
        plt.plot(self.pyarray, np.array(self.array_out))
        plt.show()

    def shutDown(self):
        self.monitorZ(0)
        # time.sleep(1)
        self.singleReadZ()
        self.mcl.MCL_ReleaseHandle(self.handle)
        num = self.mcl.MCL_NumberOfCurrentHandles()
        if num != 0:
            raise Exception('Fail to release the handle')

if __name__ == "__main__":
    import ctypes as ct

    stage = MCLPiezo()
    stage.singleReadZ()
    # print(stage.mcl.MCL_PrintDeviceInfo(stage.handle))

    stage.monitorZ(20)
    stage.monitorZ(0)
    time.sleep(1)

    #

    # for x in range(2):
    #     # stage.WfAcquisition(1 - 2 * x)
    #     stage.characterisation(2 * x - 1)
    # #     stage.monitorZ(140)
    # #     time.sleep(1)
    # #     stage.singleReadZ()
    # #     # stage.bindClock()
    # #     # mcl.WfLoad()
    # #     # mcl.monitorZ(0)
    # #     stage.WfRead()
    # #     data = [mcl.WfRead()]
    # #     time.sleep(2)
    # #     data = [stage.WfAcquisition(1 - 2 * x)]
    # #     stage.monitorZ(0)
    # #     fp = open(f"stage_data{x}.txt", "w")
    # #     for i in range(len(stage.pyarray)):
    # #         for datum in data:
    # #             fp.write(str(datum[i]) + ",")
    # #         fp.write("\n")
    #     time.sleep(1)
    #     stage.singleReadZ()
    stage.shutDown()



