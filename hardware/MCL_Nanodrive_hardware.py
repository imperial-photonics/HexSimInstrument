__author__ = "Meizhu Liang @Imperial College London"

from ScopeFoundry import HardwareComponent
from devices.MCL_Nanodrive_device import MCLPiezo

class NanoDriveHW(HardwareComponent):
    name = 'MCLNanoDriveHardware'

    def setup(self):
        self.settings.absolute_position = self.add_logged_quantity(name='Absolute position', dtype=float, unit='μm',
                                                                   vmin=0, vmax=300, initial=150, ro=False)
        self.stepsize = self.settings.New(name='Step size', dtype=float, unit='μm', vmin=0, vmax=50, initial=0.05,
                                                          ro=False)
        self.add_operation(name='Z scan', op_func=self.zScanHW)

    def connect(self):
        self.nanoscanz = MCLPiezo()
        self.settings.absolute_position.connect_to_hardware(
            read_func=self.nanoscanz.singleReadZ,
            write_func=self.movePositionHW
        )
        self.movePositionHW(150)
        self.read_from_hardware()

    def disconnect(self):
        if hasattr(self, 'nanoscanz'):
            self.nanoscanz.shutDown()
            del self.nanoscanz
            self.settings.disconnect_all_from_hardware()

    def movePositionHW(self, value):
        if hasattr(self, 'nanoscanz'):
            self.nanoscanz.singleWriteZ(value)
            self.updateHardware()

    def updateHardware(self):
        if hasattr(self, 'nanoscanz'):
            # self.settings.relative_position.read_from_hardware()
            self.settings.absolute_position.read_from_hardware()
            # print('REL position: ',self.settings.relative_position.val)
            print('ABS position: ', self.settings.absolute_position.val, 'μm')

    def moveUpHW(self):
        if hasattr(self, 'nanoscanz'):
            self.nanoscanz.singleWriteZ(self.settings.absolute_position.val + self.stepsize.val)
            self.updateHardware()

    def moveDownHW(self):
        if hasattr(self, 'nanoscanz'):
            self.nanoscanz.WfAcquisition()
            self.updateHardware()

    def zScanHW(self):
        if hasattr(self, 'nanoscanz'):
            self.nanoscanz.zScan(self.settings.absolute_position.val, 200, self.settings["Step size"])
            self.updateHardware()
if __name__ == '__main__':

    d = NanoDriveHW(HardwareComponent)
    d.setup()
    print(d.step_size.value)



