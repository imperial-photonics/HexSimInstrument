"""
Intro:  Hardware interface for stage control with serial commands.
        Stage model: Prior NanoScanZ
Author: Hai Gong
Email:  h.gong@imperial.ac.uk
Time:   Oct 2020
Address:Imperial College London
"""

from ScopeFoundry import HardwareComponent
from NanoScanDevice import NanoScan
import time

class NanoScanHW(HardwareComponent):
    name = 'NanoScanHardware'

    def setup(self):
        #self.settings.alarm = self.add_logged_quantity(name='Alarm', dtype=str, initial='None')
        self.settings.com_port = self.add_logged_quantity(name='COM port', dtype=str, initial='COM8')

        self.settings.relative_position = self.add_logged_quantity(name='Relative position', dtype=str, ro=True)
        self.settings.absolute_position = self.add_logged_quantity(name='Absolute position', dtype=float, unit='um', vmin=0, vmax=50, ro=False)

        self.add_operation(name='Set to Zero', op_func=self.setPostionZeroHW)
        self.add_operation(name='Return to Zero', op_func=self.moveZeroPositionHW)

        self.settings.move_distance = self.add_logged_quantity(name='Move distance',dtype=float,initial=0.0,ro=False)
        self.settings.stepsize = self.add_logged_quantity(name='Step size',dtype=float,vmin=0,vmax=50,initial=0.05,ro=False,
                                                          reread_from_hardware_after_write=True)

        self.add_operation(name='UP', op_func=self.moveUpHW)
        self.add_operation(name='DOWN',op_func=self.moveDownHW)

        self.add_operation(name='Reset',op_func=self.resetStageHW)

        self.settings.travel_range = self.add_logged_quantity(name='Travel range', dtype=str, ro=True)
        # self.settings.info = self.add_logged_quantity(name='Stage Information', dtype=str)
        # self.stage_absolute_position = 0.0
        # self.settings.set_current_position = self.add_logged_quantity('Set current position',dtype=float,unit='um',
        #                   ro=False)
        # self.settings.unit = self.add_logged_quantity(name='Unit', dtype=str, choices=["microns", "steps"], initial="microns",
        #                                               reread_from_hardware_after_write=True)

    def connect(self):
        # print(self.settings.com_port.val)
        self.nanoscanz = NanoScan(port=self.settings.com_port.val,debug=self.settings['debug_mode'])

        self.settings.travel_range.connect_to_hardware(
            read_func=self.getTravelRangeHW
        )
        # self.settings.info.connect_to_hardware(
        #     read_func=self.nanoscanz.getInfo
        # )

        self.settings.relative_position.connect_to_hardware(
             read_func=self.getPositionHW
        )

        self.settings.absolute_position.connect_to_hardware(
            read_func=self.nanoscanz.getPositionAbs,
            write_func=self.moveAbsolutePositionHW
        )

        self.settings.move_distance.connect_to_hardware(
            write_func=self.moveRelativePositionHW
        )
        # self.settings.unit.connect_to_hardware(
        #     read_func=self.getUnitHW,
        #     write_func=self.setUnitHW
        # )
        self.settings.stepsize.connect_to_hardware(
            read_func=self.nanoscanz.getStep,
            write_func=self.nanoscanz.setStep
        )
        self.read_from_hardware()

    def disconnect(self):

        if hasattr(self, 'nanoscanz'):
            self.nanoscanz.moveAbsolutePosition(0.0)
            self.nanoscanz.setPosition(0)
            self.nanoscanz.closePort()
            self.settings.disconnect_all_from_hardware()

# Customized functions =================================================================================================
#     def setUnitHW(self,string):
#         if hasattr(self, 'nanoscanz'):
#             self.nanoscanz._units = string
#
#     def getUnitHW(self):
#         if hasattr(self, 'nanoscanz'):
#             return self.nanoscanz._units

    def moveAbsolutePositionHW(self,value):
        if hasattr(self, 'nanoscanz'):
            self.nanoscanz.moveAbsolutePosition(value)
            self.updateHardware()

    def setPositionHW(self,value):
        if hasattr(self, 'nanoscanz'):
            self.nanoscanz.setPosition(value)
            self.updateHardware()

    def getPositionHW(self):
        if hasattr(self, 'nanoscanz'):
            return str(self.nanoscanz.getPositionRel())+' um'

    def getPositionABS(self):
        if hasattr(self, 'nanoscanz'):
            return float(self.nanoscanz.getPositionAbs())

    def setPostionZeroHW(self):
        if hasattr(self, 'nanoscanz'):
            self.nanoscanz.setPosition(0)
            self.updateHardware()

    def getTravelRangeHW(self):
        if hasattr(self, 'nanoscanz'):
            return str(self.nanoscanz.getTravelRange()) + ' um'

    def moveZeroPositionHW(self):
        if hasattr(self, 'nanoscanz'):
            self.nanoscanz.moveZeroPosition()
            self.updateHardware()

    def moveRelativePositionHW(self,value):
        if hasattr(self, 'nanoscanz'):
            self.nanoscanz.moveRelativePosition(value)
            self.updateHardware()

    def moveUpHW(self):
        if hasattr(self, 'nanoscanz'):
            self.nanoscanz.moveUp()
            self.updateHardware()

    def moveDownHW(self):
        if hasattr(self, 'nanoscanz'):
            self.nanoscanz.moveDown()
            self.updateHardware()

    def resetStageHW(self):
        if hasattr(self, 'nanoscanz'):
            self.nanoscanz.resetStage()
            self.updateHardware()

    def updateHardware(self):
        if hasattr(self, 'nanoscanz'):
            self.settings.relative_position.read_from_hardware()
            # time.sleep(0.5)
            self.settings.absolute_position.read_from_hardware()
            # time.sleep(0.5)
            # self.stage_absolute_position = self.nanoscanz.getPositionAbs()
            # time.sleep(0.5)
            print('REL position: ',self.settings.relative_position.val)
            # time.sleep(0.5)
            print('ABS position: ',self.settings.absolute_position.val,'um')




