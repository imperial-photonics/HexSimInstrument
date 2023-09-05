__author__ = "Meizhu Liang @Imperial College London"

from ScopeFoundry import HardwareComponent
from pylablib.devices import Thorlabs

"""
Python code to exploit the Thorlabs camera library as a device in Scopefoundry frame to control a CS2100M-USB 
camera (Thorlabs). Functions are tailored for the phase recovery in HexSIM.
"""
class ThorCamHW(HardwareComponent):
    name = 'ThorCam_hardware'

    def setup(self):
        self.settings.trigger_mode = self.add_logged_quantity(name='trigger mode', dtype=str,
                                                            choices=['int', 'ext', 'bulb'], initial='int', ro=False)
        self.settings.exposure = self.add_logged_quantity(name='exposure', spinbox_step=0.001, spinbox_decimals=3,
                                                          dtype=float, unit='s', initial=0.001, ro=False)
        self.settings.xcent = self.add_logged_quantity(name='roi x center', dtype=int, spinbox_step=1, ro=False, initial=577)
        self.settings.ycent = self.add_logged_quantity(name='roi y center', dtype=int, spinbox_step=1, ro=False, initial=750)

    def connect(self):
        # create an instance of the Device
        print(f'Thorlabs cameras attached: {Thorlabs.list_cameras_tlcam()}')
        self.thorCam = Thorlabs.ThorlabsTLCamera()
        self.thorCam.open()

        # Connect settings to hardware:
        self.settings.trigger_mode.connect_to_hardware(write_func=self.set_tm, read_func=self.get_tm)
        self.settings.exposure.connect_to_hardware(write_func=self.set_exp, read_func=self.get_exp)
        self.read_from_hardware()

    def disconnect(self):
        if hasattr(self, 'thorCam'):
            self.close()
            del self.thorCam

    # define operations
    def get_exp(self):
        return self.thorCam.get_exposure()

    def set_exp(self, e):
        self.thorCam.set_exposure(e)

    def get_tm(self):
        return self.thorCam.get_trigger_mode()

    def set_tm(self, t):
        self.thorCam.set_trigger_mode(t)

    def fullScreenCheck(self):
        self.set_tm('int')
        self.thorCam.set_roi(hstart=0, hend=None, vstart=0, vend=None, hbin=1, vbin=1)
        return self.thorCam.snap()

    def close(self):
        self.thorCam.close()

    def updateHardware(self):
        if hasattr(self, 'thorCam'):
            self.settings.trigger_mode.read_from_hardware()
            self.settings.exposure.read_from_hardware()