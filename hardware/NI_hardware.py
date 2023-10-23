from ScopeFoundry import HardwareComponent
from devices.ni_co_device import NI_CO_device



class NI_CO_hw(HardwareComponent):
    name = 'NI_hw'

    def setup(self):
        # create logged quantities, that are related to the graphical interface
        self.high_time1 = self.add_logged_quantity('blue_exp', dtype=int, initial=50, vmin=0, unit='*1.2ms')
        self.low_time = self.add_logged_quantity('readout', dtype=int, initial=40, vmin=0, unit='*1.2ms')
        self.high_time2 = self.add_logged_quantity('yellow_exp', dtype=int, initial=50, vmin=0, unit='*1.2ms')

        self.add_operation("start_task", self.start)
        self.add_operation("stop_task", self.stop)

    def connect(self):
        # open connection to hardware
        self.channel1.change_readonly(True)
        self.channel2.change_readonly(True)

        self.CO_device = NI_CO_device(channel1=self.channel1.val, channel2=self.channel2.val,
                                      initial_delay_chan1=self.initial_delay_chan1.val,
                                      initial_delay_chan2=self.initial_delay_chan2.val,
                                      high_time1=self.high_time1.val, low_time1=self.low_time1.val,
                                      high_time2=self.high_time2.val, period2=self.period2.val,
                                      trigger=self.trigger.val, trigger_source=self.trigger_source.val,
                                      trigger_edge=self.trigger_edge.val, debug=self.debug_mode.val)

        # connect logged quantities
        self.initial_delay_chan1.hardware_set_func = self.CO_device.set_initial_delay_chan1
        self.initial_delay_chan2.hardware_set_func = self.CO_device.set_initial_delay_chan2
        self.high_time1.hardware_set_func = self.CO_device.set_high_time1
        self.low_time1.hardware_set_func = self.CO_device.set_low_time1
        self.high_time2.hardware_set_func = self.CO_device.set_high_time2
        self.period2.hardware_set_func = self.CO_device.set_period2
        self.trigger.hardware_set_func = self.CO_device.set_trigger
        self.trigger_source.hardware_set_func = self.CO_device.set_trigger_source
        self.trigger_edge.hardware_set_func = self.CO_device.set_trigger_edge

        self.initial_delay_chan1.hardware_read_func = self.get_initial_delay_chan1
        self.initial_delay_chan2.hardware_read_func = self.get_initial_delay_chan2
        self.high_time1.hardware_read_func = self.get_high_time1
        self.low_time1.hardware_read_func = self.get_low_time1
        self.high_time2.hardware_read_func = self.get_high_time2
        self.period2.hardware_read_func = self.get_period2
        self.trigger.hardware_read_func = self.get_trigger
        self.trigger_source.hardware_read_func = self.get_trigger_source
        self.trigger_edge.hardware_read_func = self.get_trigger_edge

    def disconnect(self):
        self.channel1.change_readonly(False)
        self.channel2.change_readonly(False)
        # disconnect hardware
        if hasattr(self, 'CO_device'):
            if self.CO_device.task_ni._handle:
                self.CO_device.close()
            del self.CO_device


        for lq in self.settings.as_list():
            lq.hardware_read_func = None
            lq.hardware_set_func = None