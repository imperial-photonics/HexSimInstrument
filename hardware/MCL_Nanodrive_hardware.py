from ScopeFoundry import HardwareComponent
from devices.MCL_Nanodrive_device import MCLPiezo

class NanoDriveHW(HardwareComponent):
    name = 'NanoDriveHardware'

    def setup(self):
        self.settings.absolute_position = self.add_logged_quantity(name='Absolute position', dtype=float, unit='μm',
                                                                   vmin=0, vmax=300, ro=False)
        # self.add_operation(name='Return to Zero', op_func=self.moveZeroPositionHW)

    def connect(self):
        self.nanoscanz = MCLPiezo()
        self.settings.absolute_position.connect_to_hardware(
            read_func=self.nanoscanz.singleReadZ,
            write_func=self.nanoscanz.monitorZ
        )
        self.read_from_hardware()

    def disconnect(self):
        if hasattr(self, 'nanoscanz'):
            self.nanoscanz.shutDown()
            del self.nanoscanz
            self.settings.disconnect_all_from_hardware()

    def moveZeroPositionHW(self):
        pass

    def setPositionHW(self,value):
        if hasattr(self, 'nanoscanz'):
            self.nanoscanz.monitorZ(value)
            self.updateHardware()

    def updateHardware(self):
        if hasattr(self, 'nanoscanz'):
            # self.settings.relative_position.read_from_hardware()
            self.settings.absolute_position.read_from_hardware()
            # print('REL position: ',self.settings.relative_position.val)
            print('ABS position: ',self.settings.absolute_position.val,'um')


