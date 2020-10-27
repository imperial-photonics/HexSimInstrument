import serial
import sys
from time import sleep
cmd_lbx = {'analog_control_mode': 'ACC',
           'analog_modulation_state': 'AM',
           'base_plate_temp': 'BT',
           'laser_diode_current': 'C',
           'cdrh_state': 'CDRH',
           'modulation_state': 'CW',
           'diode_temp_set_point': 'DST',
           'measured_diode_temp': 'DT',
           'fault_number': 'F',
           'cum_time_of_operation': 'HH',
           'serial_num_and_wavelength': 'HID',
           'interlock_state': 'INT',
           'input_voltage': 'IV',
           'laser_emission_activation': 'L',
           'max_laser_current': 'MAXLC',
           'max_laser_power': 'MAXLP',
           'laser_output_power': 'P',
           'processor_temp': 'PST',
           'current_set_point': 'SC',
           'power_set_point': 'SP',
           'operating_status': 'STA',
           'software_version': 'SV',
           'alarm_reset': 'RST',
           'tec_enable': 'T',
           'digital_modulation_state': 'TTL'}

cmd_lcx = {'base_plate_temp': 'BT',
           'cdrh_state': 'CDRH',
           'fault_number': 'F',
           'cum_time_of_operation': 'HH',
           'serial_num_and_wavelength': 'HID',
           'interlock_state': 'INT',
           'input_voltage': 'IV',
           'key_state': 'KEY',
           'laser_emission_activation': 'L',
           'laser_output_power': 'P',
           'laser_output_power_percent': 'IP',
           'processor_temp': 'PST',
           'power_set_point': 'SP',
           'operating_status': 'STA',
           'software_version': 'SV',
           'contoller_reset': 'RST',
           'tec_enable': 'T',
           'port_configuration': 'CDC'}

laser_mode = {0: 'Constant power',
              1: 'Constant current',
              10:'Error in detection'}

fault = {0: 'No alarm',
         1: 'Diode current',
         2: 'Laser power',
         3: 'Power supply',
         4: 'Diode/Module temperature',
         5: 'Base temperature',
         6: 'Warning end of Life',
         7: 'Not interlocked',
         8: 'User-generated alarm',
         10:'Error in detection'}

operating_status = {1: 'Warm up',
                    2: 'Standby',
                    3: 'Laser ON',
                    4: 'Error',
                    5: 'Alarm',
                    6: 'Sleep',
                    7: 'Searching SLM point',
                    10:'Error in detection'}


class OxxiusController(object):

    def __init__(self):
        self.comportName = ""
        self.baudrate = 192000
        self.ReceiveCallback = None
        self.isopen = False
        self.receivedMessage = None
        self.serialport = serial.Serial()
        self.command = dict()
        self.laser_mode_dict = laser_mode
        self.operating_status_dict = operating_status
        self.fault_dict = fault
        self.mode = 0  # initial mode constant power
        self.fault = 0
        self.operating_status = 0
        self.islaseron = False

    def isOpen(self):
        return self.isopen

    def Open(self, model_name):
        if model_name == '488':
            portname = 'COM5'
            self.command = cmd_lbx
        elif model_name == '561':
            portname = 'COM6'
            self.command = cmd_lcx
        if not self.isopen:
            try:
                self.serialport = serial.Serial(port=portname,
                                                baudrate=self.baudrate,
                                                parity=serial.PARITY_NONE,
                                                stopbits=serial.STOPBITS_ONE,
                                                bytesize=serial.EIGHTBITS,
                                                timeout=0.5,
                                                write_timeout=0.5,
                                                inter_byte_timeout=None,
                                                exclusive=True)
                self.isopen = self.serialport.isOpen()
                print("Laser is connected to:", portname)
            except:
                print("Error at opening COM port: ", sys.exc_info()[0])

    def Disconnect(self):
        if self.isopen:
            try:
                self.serialport.close()
                self.isopen = self.serialport.isOpen()
                print("Laser is disconnected")
            except:
                print("Close error at closing COM port: ", sys.exc_info()[0])

    def Send(self, message):
        if self.isopen:
            try:
                newmessage = message.strip()
                newmessage += '\n'
                self.serialport.write(newmessage.encode('utf-8'))
                sleep(0.005)
            except:
                print("Error sending message: ", sys.exc_info()[0], message)
            else:
                return True
        else:
            return False

    def Read(self, query_name):
        if self.isopen:
            message = '?' + self.command[query_name]
            self.Send(message)
            res = self.serialport.read_until('\r\n')
            sleep(0.005)
            res = res.decode('utf-8')
            return res.replace('\r\n', '')
        else:
            return False

    def Set(self, cmd_name, value):
        if self.isopen:
            message = self.command[cmd_name] + '=' + str(value)
            self.Send(message)
            # return self.Read(cmd_name)
        else:
            return False

    def LaserOn(self):
        if self.isopen:
            # if self.Read('laser_output_power')
            try:
                self.Set('laser_emission_activation', 1)
                return print("Laser ON")
            except:
                print("Error: ", sys.exc_info()[0])
        else:
            return print("The laser is not connected.")

    def LaserOFF(self):
        if self.isopen:
            try:
                self.Set('laser_emission_activation', 0)
                return print("Laser OFF")
            except:
                print("Error: ", sys.exc_info()[0])
        else:
            return print("The laser is not connected.")

    def activateLaser(self,value):
        if self.isopen:
            try:
                self.Set('laser_emission_activation', int(value))
                if int(value)==1:
                    print("Laser ON.")
                    self.islaseron = True
                elif int(value)==2:
                    print("Laser on at low power")
                    self.islaseron = True
                elif int(value==0):
                    print("Laser OFF.")
                    self.islaseron = False
            except:
                print("Error: ", sys.exc_info()[0])
        else:
            return print("The laser is not connected.")
    # '''Set/Read laser control mode'''
    def setMode(self, value):

        inv_dict = {v: k for k, v in self.laser_mode_dict.items()}
        self.Set('analog_control_mode', inv_dict[value])

    def getMode(self):
        # get the mode: constant power/ constant current
        # print(self.Read('analog_control_mode'))
        ret = self.Read('analog_control_mode')
        try:
            self.mode = int(ret)
            return self.laser_mode_dict[self.mode]
        except:
            print('Error in reading')
            return self.laser_mode_dict[10]
    # '''Set/Read laser current'''
    def setCurrent(self, value):
        if self.mode == 1 and self.fault == 0:
            self.Set('laser_diode_current', value)

    def getCurrent(self):

        return self.Read('laser_diode_current')

    def getCurrentSetPoint(self):

        return self.Read('current_set_point')

    # '''Set/Read laser power'''
    def setPower(self, value):
        if self.mode == 0 and self.fault == 0:
            self.Set('laser_output_power', value)
            # print('Laser switched')

    def getPower(self):

        return self.Read('laser_output_power')

    def getPowerSetPoint(self):

        return self.Read('power_set_point')

    # '''Set/Read analog modulation state'''
    def setAnalogModulation(self, value):
        self.Set('analog_modulation_state', int(value))

    def getAnalogModulation(self):

        return self.Read('analog_modulation_state')

    # '''Set/Read digital modulation state'''
    def setDigitalModulation(self, value):
        self.Set('digital_modulation_state', int(value))
        # print('SET DM')

    def getDigitalModulation(self):

        return self.Read('digital_modulation_state')

    # '''Read temperatures'''
    def getBaseTemperature(self):

        return self.Read('base_plate_temp')

    def getDiodeTemperature(self):

        return self.Read('measured_diode_temp')

    # '''Read laser operating status'''
    def getOperatingStatus(self):
        ret = self.Read('operating_status')
        try:
            self.operating_status = int(ret)
            return self.operating_status_dict[self.operating_status]
        except:
            print('Error in reading:',ret)
            return self.operating_status_dict[10]

    # '''Read fault'''
    def getFault(self):
        ret = self.Read('fault_number')
        try:
            self.fault = int(ret)
            return self.fault_dict[self.fault]
        except:
            print('Error in reading:',ret)
            return self.fault_dict[10]
    # '''Read interlock state'''
    def getInterlock(self):
        return self.Read('interlock_state')

    # '''Read serial number and wavelength'''
    def getLaserInfo(self):
        return self.Read('serial_num_and_wavelength')
