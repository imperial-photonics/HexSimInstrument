"""
Python code to control the Mad City Labs controller (MCL, Nano-Drive® C, 00-55-550-0800) of the piezo stage.
"""
__author__ = "Meizhu Liang @Imperial College London"

import ctypes as ct
import numpy as np
import time
import matplotlib.pyplot as plt

err_dict = {
    0: 'Task has been completed successfully.',
    -1: 'These errors generally occur due to an internal sanity check failing.',
    -2: 'A problem occurred when transferring data to the Nano-Drive. It is likely that the Nano-Drive will have to be '
        'power cycled to correct these errors.',
    -3: 'The Nano-Drive cannot complete the task because it is not attached.',
    -4: 'Using a function from the library which the Nano-Drive does not support causes these errors.',
    -5: 'The Nano-Drive is currently completing or waiting to complete another task.',
    -6: 'An argument is out of range or a required pointer is equal to NULL.',
    -7: 'Attempting an operation on an axis that does not exist in the Nano-Drive.',
    -8: 'The handle is not valid.  Or at least is not valid in this instance of the DLL.'
}


class MCLPiezo(object):
    """
    Device driver of the MCL piezo z-stage.
    """

    def __init__(self):
        self.mcl = ct.windll.LoadLibrary('C:/Program Files/Mad City Labs/NanoDrive/Madlib.dll')
        num_check = self.mcl.MCL_NumberOfCurrentHandles()
        if num_check == 0:
            # create a new handle or get the existing one if the handle is not released properly.
            self.handle = self.mcl.MCL_InitHandleOrGetExisting()
        else:
            print('The current handle number is not zero!\n Handle number: ' + str(num_check))
        self.mcl.MCL_PrintDeviceInfo(self.handle)  # print the device information
        # connect to the instrument
        if self.handle:
            print("Connected to the device with handle: " + str(self.handle))
        else:
            raise Exception("Failure to connect the instrument. Make sure the piezo stage is turned on.")

    def singleReadZ(self):
        """
        Returns the current z-position.
        """
        if not hasattr(self, 'handle'):
            self.handle = self.mcl.MCL_InitHandleOrGetExisting()
        self.mcl.MCL_SingleReadZ.restype = ct.c_double
        re = self.mcl.MCL_SingleReadZ(self.handle)
        print(f're: {re}')
        self.checkError(re)
        return re

    def singleWriteZ(self, position):
        """
        Moves the stage to the absolute position in μm.
        """
        if not hasattr(self, 'handle'):
            self.handle = self.mcl.MCL_InitHandleOrGetExisting()
        if (position < 0) | (position > 300):
            raise ValueError('The command position should be from 0 to 300 inclusive.')
        else:
            re = self.mcl.MCL_SingleWriteZ(ct.c_double(position), self.handle)
            time.sleep(0.1)
            self.checkError(re)

    def wfLoad(self, axis, DataPoints, ms, waveform):
        """
        Sets up and triggers a waveform load on z axis.
        axis(int): Which axis to move.  (X=1,Y=2,Z=3,AUX=4)
        DataPoints(int):	Number of data points to read. 16 bit: Range: 1 – 10000, 20 bit: Range: 1 – 6666
        milliseconds(int): Rate at which to write data. 16 bit: Range: 5ms – 1/30ms, 20 bit: Range: 5ms – 1/6ms
        waveform(int): Pointer to an array of commanded positions.
        handle		[IN]	Specifies which Nano-Drive to communicate with.
        """
        self.handle = self.mcl.MCL_InitHandleOrGetExisting()
        re = self.mcl.MCL_LoadWaveFormN(axis, DataPoints, ct.c_double(ms), waveform, self.handle)
        del self.handle
        self.checkError(re)

    def zScan(self, pos, nStep, stepSize, t=260):
        """
        Scans from the current position to the relative highest position. The waveform is a stairs-shape square wave.
        t: exp1 + exp2 + low time * 2
        pos(μm): the current position
        nStep: the number of steps along z axis.
        stepSize(μm): the size of the step.
        """
        stage_offset = nStep * stepSize
        pos0 = pos - stage_offset / 2.0
        self.singleWriteZ(pos0)

        rate = 5  # ms
        n0 = int(t / rate)  # number of the square wave
        data = np.zeros(n0) + pos0
        for i in range(1, nStep):
            data = np.append(data, pos0 + np.ones(n0) * i * stepSize)
        print(len(data))
        print(data)
        ctData = (ct.c_double * len(data))(*data)
        self.wfLoad(3, len(data), rate, ctData)

    def WfAcquisition(self, sign):
    # def WfAcquisition(self):
        """
        Sets up load and read waveform functions and then triggers them simultaneously.
        """
        self.t = np.linspace(0, 999, 1000)
        self.pyarray = 140 + sign * 140 * np.cos(self.t * 2 * np.pi / 600)
        # self.pyarray_r = np.linspace(0, 9999, 1591)
        array = (ct.c_double * len(self.pyarray))(*self.pyarray)
        pyarray_out = np.zeros_like(self.pyarray)
        self.array_out = (ct.c_double * len(self.pyarray))(*pyarray_out)

        start = time.time()

        re1 = self.mcl.MCL_Setup_LoadWaveFormN(3, len(self.pyarray), ct.c_double(5), array, self.handle)
        re4 = self.mcl.MCL_Trigger_LoadWaveFormN(3, self.handle)
        time.sleep(0.01)
        re2 = self.mcl.MCL_Setup_ReadWaveFormN(3, len(self.pyarray), ct.c_double(5), self.handle)
        time.sleep(0.01)
        re3 = self.mcl.MCL_TriggerWaveformAcquisition(3, len(self.pyarray), self.array_out, self.handle)



        # plt.figure()
        # plt.plot(self.t, np.array(self.array_out))
        # plt.plot(self.t, self.pyarray)
        # plt.figure()
        # plt.plot(self.pyarray, np.array(self.array_out))

        # self.mcl.MCL_Trigger_LoadWaveFormN(3, self.handle)

        # re = self.mcl.MCL_LoadWaveFormN(3, len(pyarray), ct.c_double(2), array, self.handle)

        if re2 + re3 == 0:
            print('Waveform loading started.')
        else:
            print( re2, re3)
        # print(re1, re2, re3, re4)
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

    def characterisation(self, sign):
        """
        Characterises the stage by running the waveform acquisition function and plotting figures.
        """
        self.WfAcquisition(sign)
        plt.figure()
        # Plot loaded and resultant waveforms (input and output voltages)
        plt.plot(self.t, np.array(self.array_out))
        # plt.plot(self.t + 244, self.pyarray)
        plt.plot(self.t, self.pyarray)
        plt.figure()
        # Plot relationship between the input and output voltages
        plt.plot(self.pyarray, np.array(self.array_out))
        plt.show()

    def binClock(self):
        """
        Bound an external clock pulse to the read of a particular axis.
        """
        # re1 = self.mcl.MCL_IssSetClock(1, 1, self.handle)
        re = self.mcl.MCL_IssBindClockToAxis(4, 3, 5, self.handle)
        # re0 = self.mcl.MCL_PixelClock(self.handle)
        self.checkError(re)
        # self.checkError(re0)
        # self.checkError(re1)

    def shutDown(self):
        """
        Moves to zero position and releases the handle.
        """
        self.singleWriteZ(0)
        self.mcl.MCL_ReleaseHandle(self.handle)
        num = self.mcl.MCL_NumberOfCurrentHandles()
        if num != 0:
            raise Exception('Fail to release the handle')

    @staticmethod
    def checkError(code):
        for n, e in err_dict.items():
            if (code == n) & (code != 0):
                print(n, e)
                raise Exception(f'Error code: {n, e}')


if __name__ == "__main__":
    import ctypes as ct
    import time

    stage = MCLPiezo()
    # stage.binClock()
    # stage.WfAcquisition()

    # stage.singleWriteZ(30)
    # time.sleep(1)
    print(stage.singleReadZ())
    stage.singleWriteZ(50)
    # stage.zScan(0, 10, 20)


    # time.sleep(8)
    print(stage.singleReadZ())

    #
    #
    # for x in range(2):
    #     stage.WfAcquisition(1 - 2 * x)
        # stage.characterisation(1 - 2 * x)
    # for x in range(2):
    #     # stage.WfAcquisition(1 - 2 * x)
    #     stage.characterisation(1 - 2 * x)

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
