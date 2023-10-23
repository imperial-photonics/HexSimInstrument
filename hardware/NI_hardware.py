__author__="MeizhuLiang@ImperialCollegeLondon"
from ScopeFoundry import HardwareComponent
from devices.NI_device import NI_device


class NI_hw(HardwareComponent):
    name='NI_hw'

    def setup(self):
        #createloggedquantities,thatarerelatedtothegraphicalinterface
        self.high_time1=self.add_logged_quantity('blue_exp',dtype=int,initial=50,vmin=0,unit='*1.2ms')
        self.low_time=self.add_logged_quantity('readout',dtype=int,initial=40,vmin=0,unit='*1.2ms')
        self.high_time2=self.add_logged_quantity('yellow_exp',dtype=int,initial=50,vmin=0,unit='*1.2ms')

    def connect(self):
        self.ni_device=NI_device()

    def start_h(self):
        '''startsHexSIM'''
        self.ni_device.initiate_p()

    def hex_write(self):
     self.ni_device.write_h(self.high_time1.val, self.low_time.val, self.high_time2.val)

    def start_p(self):
        '''startsphaserecovery'''
        self.ni_device.initiate_p()

    def ph_write(self,value):
     self.ni_device.write_p(value)

    def close_task(self):
        self.ni_device.close()

    def disconnect(self):
        if hasattr(self,'ni_device'):
            del self.ni_device