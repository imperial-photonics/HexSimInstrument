from ScopeFoundry import HardwareComponent

from devices.ni_co_device import NI_CO_device

import nidaqmx.system as ni


class NI_CO_hw(HardwareComponent):
    name = 'NI_CO_hw'

    def setup(self):
        # create logged quantities, that are related to the graphical interface
        board, terminals, trig = self.update_channels()

        self.devices = self.add_logged_quantity('device', dtype=str, initial=board)
        self.channel1 = self.add_logged_quantity('Ext_exposure', dtype=str, choices=terminals, initial=terminals[0])
        self.high_time1 = self.add_logged_quantity('high_time(exp)', dtype=float, initial=0.450,
                                                   vmin=0.0001,spinbox_decimals=3, unit='s')
        self.low_time1 = self.add_logged_quantity('low_time(exp)', dtype=float, initial=0.050, vmin=0.0001,
                                                  spinbox_decimals=3, unit='s')
        self.channel2 = self.add_logged_quantity('Ext_run', dtype=str, choices=terminals, initial=terminals[1])
        self.high_time2 = self.add_logged_quantity('high_time', dtype=float, initial=1.2, vmin=0.0001,
                                                   spinbox_decimals=3, unit='s')
        self.period2 = self.add_logged_quantity('period', dtype=float, initial=1.250, vmin=0.0001,
                                                spinbox_decimals=3, unit='s')
        self.initial_delay_chan1 = self.add_logged_quantity('initial_delay_chan1', dtype=float, initial=0, vmin=0,
                                                            spinbox_decimals=6, unit='s')
        self.initial_delay_chan2 = self.add_logged_quantity('initial_delay_chan2', dtype=float, initial=0, vmin=0,
                                                            spinbox_decimals=6, unit='s')

        self.trigger = self.add_logged_quantity('trigger', dtype=bool, initial=True)
        self.trigger_source = self.add_logged_quantity('trigger_source', dtype=str, choices=trig, initial=trig[1])
        self.trigger_edge = self.add_logged_quantity('trigger_edge', dtype=str, choices=['rising', 'falling'],
                                                     initial='rising')

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

    def start(self):
        # if hasattr(self.CO_device, 'task_ni'):
        #     if self.CO_device.task_ni._handle != None:
        #         print('Current task is not closed yet')
        # else:
        self.CO_device.start_task()

    def stop(self):
        print(self.CO_device.task_ni.is_task_done())
        self.CO_device.stop_task()
        if hasattr(self.CO_device.task_ni, '_handle'):
            print(self.CO_device.task_ni._handle)
        else:
            print('no')

    def update_channels(self):
        ''' Find a NI device and return board + do_terminals + trigger terminals'''
        system = ni.System.local()
        device = system.devices[0]
        board = device.product_type + ' : ' + device.name
        terminals = []
        trig = []
        for line in device.co_physical_chans:
            terminals.append(line.name)
        for j in device.terminals:
            if 'PFI' in j:
                trig.append(j)

        return board, terminals, trig

    def get_initial_delay_chan1(self):

        return float("{0:.6f}".format(self.initial_delay_chan1.val))

    def get_initial_delay_chan2(self):

        return float("{0:.6f}".format(self.initial_delay_chan2.val))

    def get_high_time1(self):

        return float("{0:.4f}".format(self.high_time1.val))

    def get_low_time1(self):

        return float("{0:.4f}".format(self.low_time1.val))

    def get_high_time2(self):

        return float("{0:.4f}".format(self.high_time2.val))

    def get_period2(self):

        return float("{0:.4f}".format(self.period2.val))

    def get_trigger(self):

        return self.trigger.val

    def get_trigger_source(self):

        return self.trigger_source.val

    def get_trigger_edge(self):

        return self.trigger_edge.val

