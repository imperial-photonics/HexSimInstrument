__author__ = "Meizhu Liang @Imperial College London"

import time

"""
Python code to control the FDD 2K SLM with R11 system (Forth Dimension Displays).
Some parameters can be modified to work with a QXGA SLM.
Inspired by:
Written by Ruizhe Lin and Peter Kner, University of Georgia, 2019 for controlling the FDD QXGA SLM
"""

import ctypes as ct
import numpy as np
# import repfile
# from repfile import rep

class Dev(ct.Structure):
    pass
Dev._fields_ = [("id", ct.c_char_p), ("next", ct.POINTER(Dev))]

class SLMDev(object):
    """
    This is the low level dummy device object.
    Typically when instantiated it will connect to the real-world
    Methods allow for device read and write functions
    """

    def __init__(self):

        self.NULL = ct.POINTER(ct.c_int)()
        self.RS485_DEV_TIMEOUT = ct.c_uint16(1000)
        self.r11 = ct.windll.LoadLibrary('C:/Program Files/MetroCon-4.1/R11CommLib-1.8-x64.dll')
        self.xpix = 2048
        self.ypix = 2048
    def initiate(self):
        ver = ct.create_string_buffer(8)
        maxlen = ct.c_uint8(10)
        res = self.r11.R11_LibGetVersion(ver, maxlen)
        if (res == 0):
            print('Software version: %s' % ver.value)
        else:
            raise Exception('Libarary not loaded')
        guid = ct.c_char_p(b"54ED7AC9-CC23-4165-BE32-79016BAFB950")
        devcount = ct.c_uint16(0)
        devlist = ct.POINTER(Dev)()
        res = self.r11.FDD_DevEnumerateWinUSB(guid, ct.pointer(devlist), ct.byref(devcount))
        if res == 0:
            port = devlist.contents.id.decode()
            print('Dev port: %s' % port)
        else:
            raise Exception('Cannot find the port')

    def open_usb_port(self):
        port = ct.c_char_p(b'\\\\?\\usb#vid_19ec&pid_0503#0175000881#{54ed7ac9-cc23-4165-be32-79016bafb950}')
        re = self.r11.FDD_DevOpenWinUSB(port, self.RS485_DEV_TIMEOUT)
        if re == 0:
            print('Open Dev port successfully')
            dispTemp = ct.c_uint16(0)
            self.r11.R11_RpcSysGetDisplayTemp(ct.byref(dispTemp))
            print('Display temperature: %s' % dispTemp.value)
        else:
            raise Exception(' Fail to open the port ')

    def activate(self,):
        res = self.r11.R11_RpcRoActivate(ct.c_void_p())
        if res != 0:
            raise Exception(f'Fail to deactivate. Error code: {res}')

    def deactivate(self):
        res = self.r11.R11_RpcRoDeactivate(ct.c_void_p())
        if res != 0:
            raise Exception(f'Fail to deactivate. Error code: {res}')

    def getordernum(self):
        rocount = ct.c_uint16(0)
        res = self.r11.R11_RpcRoGetCount(ct.byref(rocount))
        if res == 0:
            num = rocount.value
            print('order number: %s' % num)
        else:
            raise Exception('Fail to get the order number')

    def getActivationType(self):
        """Retrieve the activation typr of the currently loaded running order"""
        actType = ct.c_uint8(0)
        res = self.r11.R11_RpcRoGetActivationType(ct.byref(actType))
        if res == 0:
            return actType.value
        else:
            raise Exception('Failed to get activation type')

    def getRO(self):
        roindex = ct.c_uint16(0)
        res_index = self.r11.R11_RpcRoGetSelected(ct.byref(roindex))
        roName = ct.create_string_buffer(255)
        res_name = self.r11.R11_RpcRoGetName(roindex, roName, len(roName))
        if (res_index == 0) & (res_name == 0):
            return roindex.value, roName.value
        elif res_index != 0:
            raise Exception('Failed to get RO index')
        elif res_name != 0:
            raise Exception('Failed to get RO name')

    def getRepName(self):
        """Retrieve the name of the repertoire loaded on the board."""
        repName = ct.create_string_buffer(255)
        re = self.r11.R11_RpcSysGetRepertoireName(repName, len(repName))
        if re == 0:
            return repName.value
        else:
            raise Exception('Failed to get repertoire name')

    def setRO(self, n):
        roindex = ct.c_uint16(n)
        res = self.r11.R11_RpcRoSetSelected(roindex)
        if res == 0:
            num = roindex.value
            print(f'Order is set to {num}')
        else:
            raise Exception('Failed to set the order')


    def getState(self):
        actState = ct.c_uint8(0)
        self.r11.R11_RpcRoGetActivationState(ct.byref(actState))
        return actState.value

    def close(self):
        res = self.r11.FDD_DevClose()
        if res == 0:
            print('Port closed successfully')
        else:
            raise Exception('Fail to closE R11 SLM')

    def interleaving(self):
        x0 = np.zeros(self.xpix)
        y0 = np.arange(self.ypix)

        nr = np.arange(8)
        for i in range(64):
            x0[(i * 32):(i * 32 + 8)] = nr + i * 8
            x0[(i * 32 + 8):(i * 32 + 16)] = nr + i * 8 + 512
            x0[(i * 32 + 16):(i * 32 + 24)] = nr + i * 8 + 1024
            x0[(i * 32 + 24):(i * 32 + 32)] = nr + i * 8 + 1536
        # x0 is now an array of interleaved x values in the correct places for sending to the SLM

        x, y = np.meshgrid(x0, y0)
        return x, y

    def sendBitplane(self, data, frameno):
        print(f'sending frame number {frameno}')
        t0 = time.time()
        for block in range(4):
            block_address = 0x01000000 + block * 64 + frameno * 256
            # Flash blocks per bitplane: 4
            # Flash pages per bitplane: 256
            res = self.r11.R11_RpcFlashEraseBlock(ct.c_uint32(block_address))
            if res != 0:
                raise Exception(f'Fail to erase block {block}')
            for page in range(64):
                buf = np.uint8(data[(block * 64 + page) * 2048:(block * 64 + page) * 2048 + 2048])
                res = self.r11.R11_FlashWrite(buf.ctypes.data_as(ct.POINTER(ct.c_uint8)), ct.c_uint16(0),
                                              ct.c_uint16(2048))
                if res != 0:
                    raise Exception(f'Fail write block {block}: page {page}')
                page_address = ct.c_uint32(block_address + page)
                res = self.r11.R11_FlashBurn(page_address)
                if res != 0:
                    raise Exception(f'Fail burn block {block}: page {page}')
        print(f'bitplane sent in {time.time() - t0}')

    def eraseBitplane(self, frameno):
        for block in range(4):
            block_address = 0x01000000 + block * 64 + frameno * 256

            res = self.r11.R11_RpcFlashEraseBlock(ct.c_uint32(block_address))
        if res == 0:
            print(f'Frame number {frameno} erased')
        else:
            raise Exception(f'Fail to erase block {block}')

    def repReload(self):
        res = self.r11.R11_RpcSysReloadRepertoire()
        t0 = time.time()
        while self.getProgress() != 100:
            self.getProgress()
        t = time.time() - t0
        print(f'Elapsed time of reloading: {t}')
        if res != 0:
            raise Exception('Fail reload repertoire')

    def getProgress(self):
        """Get the progress of the current board operation."""
        p = ct.c_uint8(0)
        res = self.r11.R11_DevGetProgress(ct.byref(p))
        if res != 0:
            raise Exception(f'Fail to get progress. Error code: {res}')
        return p.value




# if __name__ == '__main__':
#
#     print(Dev)
#     try:
#         slm = SLMDev()
#         print('slm started')
#         slm.initiate()
#         slm.open_usb_port()
#         slm.activate()
#
#     except Exception as err:
#         print(err)

    # finally:
    #     slm.close()