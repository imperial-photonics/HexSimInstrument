"""
Intro:  Device driver for stage control with serial commands.
        Stage model: Prior NanoScanZ
Author: Hai Gong
Email:  h.gong@imperial.ac.uk
Time:   Oct 2020
Address:Imperial College London
"""

# from __future__ import division, absolute_import, print_function
import serial
import logging
#import numpy as np
#import time
import sys

logger = logging.getLogger(__name__)

class NanoScan(object):

    name = "nanoscan_control"

    def __init__(self, port="COM8", debug=False):
        self.port = port
        self.debug = debug
        if self.debug:
            logger.debug("NanoScan.__init__, port={}".format(self.port))

        self.ser = serial.Serial()
        self._is_port_open = False
        self._resp_buffer = None
        self._units = 'micrometer' # or 'steps'
        self._zeroposition_abs = 0.0
        self._position_rel = 0.0
        self._position_abs = 0.0
        self.openPort()
        self.resetStage()
    # Low level functions==============================================================================================

    def isPortOpen(self):
        return self.ser.isOpen()

    def openPort(self):
        if not self.isPortOpen():
            try:
                self.ser = serial.Serial(port=self.port, baudrate=9600, bytesize=8,
                                         parity='N', stopbits=1, timeout=0.05)

                print("NanoScanZ is connected to:",self.port)
            except:
                print("Error at opening COM port: ", sys.exc_info()[0])

    def closePort(self):
        if self.isPortOpen():
            try:
                self.ser.close()
                print("NanoScanZ is disconnected.")
            except:
                print("Close error at closing COM port: ", sys.exc_info()[0])

    def sendCmd(self,command):
        if self.isPortOpen():
            try:
                command += '\r'
                self.ser.write(command.encode('ascii'))
                self._resp_buffer = self.ser.read_until('\r')
                # self._resp_buffer = self.ser.readlines()
                # self._resp_buffer = self.ser.read_until('\r0\r')
            except:
                print("Error sending message: ", sys.exc_info()[0], command)
            else:
                return True
        else:
            return False

    def getCmd(self, command):
        if self.isPortOpen():
            command = '<' + command+'\r'
            self.sendCmd(command)
            resp = self._resp_buffer.decode('ascii')
            newresp = resp.splitlines()
            #newresp = resp.replace('\r','')
            #newresp = resp.replace('\r0\r','')
            return newresp[0].replace('<', '')
        else:
            return None

    def setCmd(self, command, value):
        if self.isPortOpen():
            command = command + ' ' + value
            self.sendCmd(command.strip())
        else:
            return None

    # High level functions==============================================================================================

    def getMovingState(self):
        """Returns the movement status, 0 stationary, 4 moving
        """
        return self.getCmd('$')
    
    def getTravelRange(self):
        """ Range of travel
        """
        return self.getCmd('PIEZORANGE')
        
    def setTravelRange(self,value):
        self.setCmd('PIEZORANGE','{:0.3f}'.format(value))    
        
    def getInfo(self):
        """Identification of the device
        """
        return str(self.getCmd('SERIAL')) +'\n'+str(self.getCmd('DATE')) # + ' ' + +# str(self.getCmd('PIEZORANGE'))

    def getPositionRel(self):
        """Report current relative position: unit 'micrometer'
        """
        self._position_rel = float(self.getCmd('PZ'))
        return self._position_rel

    def getPositionAbs(self):
        """Report current absolute position: unit 'micrometer'
        """
        self._position_abs = self.getPositionRel()+self._zeroposition_abs
        return self._position_abs

    def setPosition(self,value):
        """Set current position to @value. Unit 'micrometer'
        Sets to REL display mode (unless position same as ABS). e.g., if at absolute
        position 50, setting position to zero "PZ,0", position then reported range of -50 to +50 microns.
        To return to ABS mode, use "V,0" "PZ,0", this will return the stage to 0 microns and set the screen
        display to ABS mode.
        """
        self.setCmd('PZ','{:0.3f}'.format(value))
        self._zeroposition_abs = self._position_abs-value

    def moveZeroPosition(self):
        """Move to zero including any oofset added by PZ command, moves to REL 0 (relative display mode) if
        in REL mode and ABS 0 if in ABS mode.
        """
        self.getCmd('M')
        self._position_abs = self._zeroposition_abs

    def moveAbsolutePosition(self, value):
        """Move to absolute position n, range (0,100): unit 'micrometer'.
        This is a "real" absolute position and is independent of any relative offset added by the PZ command.
        """
        self.setCmd('V', '{:0.3f}'.format(value))
        self._position_abs = value

    def moveRelativePosition(self, value):
        if value >= 0:
           self.setCmd('U', '{:0.3f}'.format(value))
        elif value < 0:
           self.setCmd('D', '{:0.3f}'.format(-value))

    # def moveRelativePosition(self, value):
    #     """
    #     Units in either 'micrometer' or 'steps'
    #     Move the stage position relative to the current position by an amount determined by 'value'.
    #     If value is given in micrometer, thats the amount the stage is going to move, in microns.
    #     If value is given in steps, the stage will move a distance  value.magnitude * step. The step is defined by the step
    #     """
    #     try:
    #         if self._units == 'micrometer':
    #             if value >= 0:
    #                 self.setCmd('U', '{:0.3f}'.format(value))
    #             elif value < 0:
    #                 self.setCmd('D', '{:0.3f}'.format(-value))
    #         elif self._units == 'steps':
    #             if value >= 0:
    #                 for x in range(0, value):
    #                     self.setCmd('U','')
    #             elif value < 0:
    #                 for x in range(0, -value):
    #                     self.setCmd('D','')
    #     except:
    #         raise ValueError('Specify the translation distance in micrometers or steps')

    def moveUp(self):
        self.setCmd('U', '')
        # if self._units =='steps':
        #     self.setCmd('U', '')
        # else:
        #     raise ValueError('The unit is not steps.')

    def moveDown(self):
        self.setCmd('D', '')
        # if self._units =='steps':
        #     self.setCmd('D', '')
        # else:
        #     raise ValueError('The unit is not steps.')

    def getStep(self):
        return self.getCmd('C')
         
    def setStep(self,value):
        self.setCmd('C','{:0.3f}'.format(value))

    def getVersion(self):
        return self.getCmd('VER')

    def resetStage(self):
        self.moveAbsolutePosition(0)
        self.setPosition(0)
        self._zeroposition_abs = 0
        self._position_rel = 0
        self._position_abs = 0

if __name__=='__main__':
    nsz = NanoScan()
    nsz.openPort()
    #nsz.getMovingState()
   # b = nsz.getStep()
    nsz.moveAbsolutePosition(50)
    nsz.setPosition(0)
    nsz.moveAbsolutePosition(40)
    
    
    print(nsz.getPositionRel())
    print(nsz.getPositionAbs())
    
    #nsz.setPosition(0)
    # print(nsz.getTravelRange())
    
    #nsz.moveZeroPosition()

    #
    #nsz.closePort()
    # nsz.setPosition(50)
    # print(b)
    # print(nsz.getMovingState())
    # print(nsz.getVersion())
    # print(nsz.getPosition())
    # print(nsz.getStep())
    # nsz.closePort()
    















