from ScopeFoundry import HardwareComponent
from LaserDevice import OxxiusController
from threading import Thread,currentThread


class Laser488HW(HardwareComponent):
    name = 'Laser 488nm'

    def setup(self):
        # Laser status check
        self.laser_info = self.add_logged_quantity('Laser information', dtype=str, si=False, ro=1)
        self.operating_status = self.add_logged_quantity('Operating status', dtype=str, si=False, ro=1)
        self.laser_alarm = self.add_logged_quantity('Laser alarm', dtype=str, si=False, ro=1)

        # Laser mode setting
        self.mode = self.add_logged_quantity('Mode', dtype=str, si=False, ro=0,
                                             choices=["Constant current", "Constant power"], initial='Constant power',
                                             reread_from_hardware_after_write=True)
        self.laser_on = self.add_logged_quantity('Laser ON', dtype=bool, si=False, ro=0, initial=False)
        # Laser control operating
        self.power_set_point = self.add_logged_quantity('Power set point', dtype=float, unit='mW', ro=0, vmin=0,
                                                        vmax=220, reread_from_hardware_after_write=False)
        self.power = self.add_logged_quantity('Power', dtype=str, ro=1, initial=0, unit='mW')
        self.current_set_point = self.add_logged_quantity('Current set point', dtype=float, unit='mA', ro=0, vmin=0,
                                                          vmax=320, reread_from_hardware_after_write=False)
        self.current = self.add_logged_quantity('Current', dtype=str, ro=0, initial=0, unit='mA')

        self.analog_modulation = self.add_logged_quantity('Analog Modulation', dtype=bool, si=False, ro=0,
                                                          initial=False,reread_from_hardware_after_write=True)
        self.digital_modulation = self.add_logged_quantity('Digital Modulation', dtype=bool, si=False, ro=0,
                                                           initial=False,reread_from_hardware_after_write=True)
        # Temperature check
        self.interlock_state = self.add_logged_quantity('Interlock state', dtype=bool, si=False, ro=1)
        self.diode_temperature = self.add_logged_quantity('Diode temperature ' + chr(176) + 'C', dtype=str, si=False,
                                                          ro=1)
        self.base_temperature = self.add_logged_quantity('Base temperature ' + chr(176) + 'C', dtype=str, si=False,
                                                         ro=1)

    def connect(self):
        self.laser488 = OxxiusController()
        # Open connection to the laser
        self.laser488.Open('488')

        # Read laser info functions
        self.diode_temperature.hardware_read_func = self.laser488.getDiodeTemperature
        self.base_temperature.hardware_read_func = self.laser488.getBaseTemperature

        self.power_set_point.hardware_read_func = self.laser488.getPowerSetPoint
        self.current_set_point.hardware_read_func = self.laser488.getCurrentSetPoint

        self.power.hardware_read_func = self.laser488.getPower
        self.current.hardware_read_func = self.laser488.getCurrent

        self.interlock_state.hardware_read_func = self.laser488.getInterlock
        self.laser_alarm.hardware_read_func = self.laser488.getFault
        self.operating_status.hardware_read_func = self.laser488.getOperatingStatus
        self.laser_info.hardware_read_func = self.laser488.getLaserInfo
        self.analog_modulation.hardware_read_func = self.laser488.getAnalogModulation
        self.digital_modulation.hardware_read_func = self.laser488.getDigitalModulation
        self.mode.hardware_read_func = self.laser488.getMode

        # Write settings to the laser
        self.power_set_point.hardware_set_func = self.laser488.setPower
        self.current_set_point.hardware_set_func = self.laser488.setCurrent

        self.analog_modulation.hardware_set_func = self.laser488.setAnalogModulation
        self.digital_modulation.hardware_set_func = self.laser488.setDigitalModulation
        self.mode.hardware_set_func = self.laser488.setMode
        # Switch on/off laser
        self.laser_on.hardware_set_func = self.laser488.activateLaser
        self.read_from_hardware()
        # Real time measurement
        self.measureLaser = Thread(target=self.measureHardware)
        self.measureLaser.start()

    def disconnect(self):
        if hasattr(self, 'laser488'):
            self.laser488.LaserOFF()
            self.measureLaser.do_run = False
            self.measureLaser.join()
            self.laser488.Disconnect()
            self.settings.disconnect_all_from_hardware()

    def measureHardware(self):
        # Read laser info
        t = currentThread()
        while getattr(t, "do_run", True):
            self.diode_temperature.read_from_hardware()
            self.base_temperature.read_from_hardware()
            self.power.read_from_hardware()
            self.current.read_from_hardware()
            self.laser_alarm.read_from_hardware()
            self.operating_status.read_from_hardware()
            # print('Thread activate')


class Laser561HW(HardwareComponent):
    name = 'Laser 561nm'

    def setup(self):
        # Laser status check
        self.laser_info = self.add_logged_quantity('Laser information', dtype=str, si=False, ro=1)
        self.operating_status = self.add_logged_quantity('Operating status', dtype=str, si=False, ro=1)
        self.laser_alarm = self.add_logged_quantity('Laser alarm', dtype=str, si=False, ro=1)

        # Laser mode setting
        self.laser_on = self.add_logged_quantity('Laser ON', dtype=bool, si=False, ro=0, initial=False)
        # Laser control operating
        self.power_set_point = self.add_logged_quantity('Power set point', dtype=float, unit='mW', ro=0, vmin=0,
                                                        vmax=220, reread_from_hardware_after_write=False)
        self.power = self.add_logged_quantity('Power', dtype=str, ro=1, initial=0, unit='mW')

        # Temperature check
        self.interlock_state = self.add_logged_quantity('Interlock state', dtype=bool, si=False, ro=1)
        self.base_temperature = self.add_logged_quantity('Base temperature ' + chr(176) + 'C', dtype=str, si=False,
                                                         ro=1)
    def connect(self):
        self.laser561 = OxxiusController()
        # Open connection to the laser
        self.laser561.Open('561')

        # Read laser info
        self.base_temperature.hardware_read_func = self.laser561.getBaseTemperature
        self.power_set_point.hardware_read_func = self.laser561.getPowerSetPoint
        self.power.hardware_read_func = self.laser561.getPower
        self.interlock_state.hardware_read_func = self.laser561.getInterlock
        self.laser_alarm.hardware_read_func = self.laser561.getFault
        self.operating_status.hardware_read_func = self.laser561.getOperatingStatus
        self.laser_info.hardware_read_func = self.laser561.getLaserInfo

        # Write settings to the laser
        self.power_set_point.hardware_set_func = self.laser561.setPower
        # Switch on/off laser
        self.laser_on.hardware_set_func = self.laser561.activateLaser
        self.read_from_hardware()
        # Real time measurement
        self.measureLaser = Thread(target=self.measureHardware)
        self.measureLaser.start()

    def disconnect(self):
        if hasattr(self, 'laser561'):
            self.laser561.LaserOFF()
            self.measureLaser.do_run = False
            self.measureLaser.join()
            self.laser561.Disconnect()
            self.settings.disconnect_all_from_hardware()

    def measureHardware(self):
        # Read laser info
        t = currentThread()
        while getattr(t, "do_run", True):
            self.base_temperature.read_from_hardware()
            self.power.read_from_hardware()
            self.laser_alarm.read_from_hardware()
            self.operating_status.read_from_hardware()
            # print('Thread activate')
