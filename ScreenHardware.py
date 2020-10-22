from ScopeFoundry import HardwareComponent
from numpy import pi

from ScreenDevice import ScreenDisplay

class ScreenHW(HardwareComponent):
    name = 'screen_display'

    def setup(self):
        self.binning = self.add_logged_quantity('monitor_number', dtype=int, ro=0,
                                                choices=[0, 1, 2], initial=1)
        self.settings.New(name='update_time', dtype=float,ro=0,initial=100,unit = 'ms', vmin = 0)
                                                   # reread_from_hardware_after_write = True, vmin = 0)
        # self.settings.New = self.add_logged_quantity('update_time', dtype=float,ro=0,initial=100,unit = 'ms',
        #                                            reread_from_hardware_after_write = True, vmin = 0)
        # self.settings.New(name='monitor_number', initial=1, dtype=int, ro=False)
        self.settings.New(name='shift_orientation', initial=pi/9.0, dtype=float, ro=False)
        self.settings.New(name='scale', initial=pi/16, dtype=float, ro=False)

        self.screen_width = self.add_logged_quantity('Width', dtype=str, si=False, ro=1,unit = 'px')
        self.screen_height = self.add_logged_quantity('Height', dtype=str, si=False, ro=1, unit = 'px')

    def connect(self):
        # open connection to the SLM
        self.slm_dev = ScreenDisplay(monitor_number = self.binning.val,
                                  shift_orientation = self.settings['shift_orientation'],
                                  scale = self.settings['scale'], update_time=self.settings['update_time'])

        self.slm_dev.showFullScreen()
        print('slm started')

        # connect settings to the SLM
        self.slm_dev.enableTimer()

        self.settings.update_time.connect_to_hardware(write_func=self.slm_dev.writeUpdateTime)
        # self.updatetime.hardware_read_func=self.slm_dev.writeUpdateTime()
        self.slm_dev.timer.start()
        self.screen_width.hardware_read_func=self.slm_dev.getScreenWidth
        self.screen_height.hardware_read_func=self.slm_dev.getScreenHeight

        self.settings.update_time.hardware_set_func = self.slm_dev.changeTimer

        self.read_from_hardware()
        # )
    def disconnect(self):
        if hasattr(self, 'slm_dev'):
            self.slm_dev.close()
            self.slm_dev.disableTimer()
            self.settings.disconnect_all_from_hardware()
            print('run disconnect')
