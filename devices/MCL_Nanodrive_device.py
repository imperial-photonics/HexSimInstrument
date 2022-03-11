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
        # pyadd = []
        # add = (ct.c_int * len(pyadd))(*pyadd)
        # self.piezo.MCL_GetProductInfo.argtypes = [ct.c_char, ct.c_short, ct.c_short, ct.c_short, ct.c_short, ct.c_short]
        # re1 = self.piezo.MCL_GetProductInfo(ct.c_char_p(), ct.c_short(), ct.c_short(), ct.c_short(), ct.c_short(), ct.c_short(), self.handle)
        # print(re1)
        # print(add.value)

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

    def monitorZ(self, position):
        if position < 0:
            self.ceaseOperation()
            print('The command position is too low! Operation ceased.')
        elif position > 300:
            self.ceaseOperation()
            print('The command voltage is too high! Operation ceased.')
        else:
            self.piezo.MCL_MonitorZ.restype = ct.c_double
            re = self.piezo.MCL_MonitorZ(ct.c_double(position), self.handle)
            print('Position is moved from ' + str(re))

    def WfSetup(self):
        """
        This function sets up and trigger a waveform on z axis.
        """
        pyarray = [x for x in range(0, 18)]
        for x in pyarray:
            if x > 20:
                self.ceaseOperation()
        print('errrorrrrrrrrrrrrr')
        array = (ct.c_double * len(pyarray))(* pyarray)
        # x = axis[0]
        # y = axis[0]
        # z = axis[1]
        # *x, = axis
        # *y, = axis
        # *z, = axis

        start = time.time()
        print("hello")
        self.piezo.MCL_LoadWaveFormN(3, len(pyarray), ct.c_double(4), ct.pointer(array), self.handle)
        end = time.time()
        print(end - start)
        # self.piezo.MCL_ReadWaveFormN(3, 9000, ct.c_double(4), self.handle)

        # if self.handle:
        #     if points < 1000:
        #         wave_form_data_type = ct.c_double * points
        #         wave_form_data = wave_form_data_type()
        #         self.piezo.MCL_ReadWaveFormN(ct.c_ulong(3), ct.c_ulong(points), ct.c_double(4.0), wave_form_data, self.handle)
        #         return wave_form_data
        #
        #     else:
        #         print
        #         "MCL stage can only acquire a maximum of 999 points"


        # re = self.piezo.MCL_TriggerWaveformAcquisition(3, 1000, ct.pointer(array), self.handle)
        # if re == 0:
        #     print("Waveform is prepared")
        # else:
        #     print('fail!' + str(re))

    def ceaseOperation(self):
        """
        This function moves the stage to 0 and release all handles.
        """
        self.piezo.MCL_MonitorZ(ct.c_double(0), self.handle)
        self.piezo.MCL_ReleaseAllHandles()




if __name__ == "__main__":
    import ctypes as ct

    mcl = MCLPiezo()
    mcl.singleReadZ()
    mcl.monitorZ(0.000)
    time.sleep(1)
    mcl.singleReadZ()
    mcl.WfSetup()
    # points = 500
    # data = [mcl.WfSetup(points)]
    # fp = open("stage_data.txt", "w")
    # for i in range(points):
    #     for datum in data:
    #         fp.write(str(datum[i]) + ",")
    #     fp.write("\n")

    time.sleep(1)
    mcl.singleReadZ()
    mcl.piezo.MCL_ReleaseAllHandles()
