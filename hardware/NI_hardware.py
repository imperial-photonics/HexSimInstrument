__author__="MeizhuLiang@ImperialCollegeLondon"
from ScopeFoundry import HardwareComponent
from devices.NI_device import NI_device


class NI_hw(HardwareComponent):
    name='NI_hw'

    def setup(self):
        #createloggedquantities,thatarerelatedtothegraphicalinterface
        self.high_time1=self.add_logged_quantity('high_time1',dtype=int,initial=50,vmin=0, unit='*1.2ms')
        self.low_time=self.add_logged_quantity('low_time', dtype=int,initial=17,  vmin=0, unit='*1.2ms')
        self.high_time2=self.add_logged_quantity('high_time2',dtype=int,initial=30,vmin=0,unit='*1.2ms')
        self.settings.b_exp = self.add_logged_quantity('blue_exp',dtype=float,initial=0.000, vmin=0.000, spinbox_decimals=3,
                                            unit='ms')
        self.settings.y_exp = self.add_logged_quantity('yel_exp', dtype=float, initial=0.000, vmin=0.000, spinbox_decimals=3,
                                              unit='ms')
        self.add_operation(name='HexSIM signal', op_func=self.hex_wrap)

    def connect(self):
        self.ni_device=NI_device()
        self.settings.b_exp.connect_to_hardware(read_func=self.get_bExp)
        self.settings.y_exp.connect_to_hardware(read_func=self.get_yExp)

    def start_h(self):
        '''startsHexSIM'''
        self.ni_device.initiate_h()

    def hex_write(self):
        if hasattr(self, 'ni_device'):
            print(self.high_time1.val, self.low_time.val, self.high_time2.val)
            self.ni_device.write_h(self.high_time1.val, self.low_time.val, self.high_time2.val)
            self.updateHardware()

    def hex_wrap(self):
        self.start_h()
        self.hex_write()

    def start_p(self):
        '''startsphaserecovery'''
        self.ni_device.initiate_p()

    def ph_write(self,value):
        self.ni_device.write_p(bool(value))

    def close_task(self):
        try:
            self.ni_device.close()
            print('Task is closed.')
        except Exception as e:
            print(e)

    def get_bExp(self):
        vn = 1024  # for a 1024 * 1024 frame
        readout = (vn + 5) * 18.64706e-3
        re = (self.high_time1.val + self.low_time.val) * 1.2 - readout
        # self.settings.y_exp.val = (self.high_time2.val + self.low_time.val) * 1.2 - readout
        return re

    def get_yExp(self):
        vn = 1024  # for a 1024 * 1024 frame
        readout = (vn + 5) * 18.64706e-3
        re = (self.high_time2.val + self.low_time.val) * 1.2 - readout
        return re
    def disconnect(self):
        if hasattr(self,'ni_device'):
            del self.ni_device

    def updateHardware(self):
        if hasattr(self, 'ni_device'):
            self.settings.b_exp.read_from_hardware()
            self.settings.y_exp.read_from_hardware()


