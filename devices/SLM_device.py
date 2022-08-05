__author__ = "Meizhu Liang @Imperial College London"
"""
Python code to control the QXGA SLM with R11 system (Forth Dimension Displays).
Inspired by:
Written by Ruizhe Lin and Peter Kner, University of Georgia, 2019 for controlling the QXGA SLM
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
        # RS485_BAUDRATE = ct.c_uint32(256000)
        # RS232_BAUDRATE = ct.c_uint32(115200)
        self.r11 = ct.windll.LoadLibrary('C:/Program Files/MetroCon-4.1/R11CommLib-1.8-x64.dll')

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
        if res == 0:
            print('Activate QXGA successfully')
        else:
            raise Exception('Fail to activate QXGA')

    def deactivate(self):
        res = self.r11.R11_RpcRoDeactivate(ct.c_void_p())
        if res == 0:
            print('Deactivate QXGA successfully')
        else:
            raise Exception('Fail to deactivate')

    def getordernum(self):
        rocount = ct.c_uint16(0)
        res = self.r11.R11_RpcRoGetCount(ct.byref(rocount))
        if res == 0:
            num = rocount.value
            print('order number: %s' % num)
        else:
            raise Exception('Fail to get the order number')

    def getActivationType(self):
        actType = ct.c_uint8(0)
        res = self.r11.R11_RpcRoGetActivationType(ct.byref(actType))
        if res == 0:
            num = actType.value
            if num == 1:
                print(f'activation type: Immediate')
            elif num == 2:
                print(f'activation type: Software')
            elif num == 4:
                print(f'activation type: Hardware')
        else:
            raise Exception('Failed to get activation type')

    def getRO(self):
        roindex = ct.c_uint16(0)
        res_index = self.r11.R11_RpcRoGetSelected(ct.byref(roindex))

        roName = ct.create_string_buffer(b'xxxxxxxx')
        print(f'Running order index: {roindex.value}\nRO name:{roName.value} {len(roName)} ')

        maxlen = ct.c_uint8(10)
        res_name = self.r11.R11_RpcRoGetName(ct.byref(roindex), roName, len(roName))
        # res_name = self.r11. R11_RpcSysGetRepertoireName(roName, len(roName))

        if (res_index == 0) & (res_name == 0):
            print(f'Running order index: {roindex.value}\nRO name:{roName.value} {len(roName)}')
        elif (res_index != 0):
            raise Exception('Failed to get RO index')
        elif (res_name != 0):
            raise Exception('Failed to get RO name')


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
        re = self.r11.R11_RpcRoGetActivationState(ct.byref(actState))
        self.state = actState.value

    def close(self):
        res = self.r11.FDD_DevClose()
        if res == 0:
            print('Port closed successfully')
        else:
            raise Exception('Fail to close QXGA')

    def sendBitplane(self,data, frameno):
        print(f'sending frame number {frameno}')
        for block in range(3):
            block_address = 0x01000000 + block * 64 + frameno * 192
            # Flash blocks per bitplane: 3
            # Flash pages per bitplane: 192

            res = self.r11.R11_RpcFlashEraseBlock(ct.c_uint32(block_address))
            if res != 0:
                raise Exception(f'Fail to erase block {block}')
            for page in range(64):
                buf = np.uint8(data[(block * 64 + page) * 2048:(block * 64 + page) * 2048 + 2048])
                res = self.r11.R11_FlashWrite(buf.ctypes.data_as(ct.POINTER(ct.c_uint8)), ct.c_uint16(0), ct.c_uint16(2048))
                # buf.ctypes.data_as(ct.POINTER(ct.c_uint8)): ctypes accepts an array of c_uint8????
                if res != 0:
                    raise Exception(f'Fail write block {block}: page {page}')
                page_address = ct.c_uint32(block_address + page)
                res = self.r11.R11_FlashBurn(page_address)
                if (res != 0):
                    raise Exception(f'Fail burn block {block}: page {page}')

    def eraseBitplane(self,frameno):
        for block in range(3):
            block_address = 0x01000000 + block * 64 + frameno * 192

            res = self.r11.R11_RpcFlashEraseBlock(ct.c_uint32(block_address))

        if res == 0:
            print(f'Frame number {frameno} erased')
        else:
            raise Exception(f'Fail to erase block {block}')

    def repreload(self):
        res = self.r11.R11_RpcSysReloadRepertoire()
        if res == 0:
            print('Repertoire reloaded')
        else:
            raise Exception('Fail reload repertoire')

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