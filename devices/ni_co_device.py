import nidaqmx
# import warnings
# import math
import numpy as np



class NI_CO_device(object):

    def __init__(self, channel1, channel2, initial_delay_chan1, initial_delay_chan2, high_time1, low_time1,
                 high_time2, period2, trigger, trigger_source,trigger_edge, debug=False):
        self.dict = {"rising": nidaqmx.constants.Edge.RISING,
                     "falling": nidaqmx.constants.Edge.FALLING}
        self.debug = debug
        self.channel1 = channel1
        self.channel2 = channel2
        self.initial_delay_chan1 = initial_delay_chan1
        self.initial_delay_chan2 = initial_delay_chan2
        self.high_time1 = high_time1
        self.low_time1 = low_time1
        self.high_time2 = high_time2
        self.period2 = period2
        self.trigger = trigger
        self.trigger_source = trigger_source
        self.trigger_edge = trigger_edge


    def create_task(self):
        self.task_ni = nidaqmx.Task()
        self.task_ni.co_channels.add_co_pulse_chan_time(counter=self.channel1,
                                                        initial_delay=self.initial_delay_chan1,
                                                        high_time=self.high_time1, low_time=self.low_time1)
        self.task_ni.co_channels.add_co_pulse_chan_time(counter=self.channel2, high_time=self.high_time2,
                                                        initial_delay=self.initial_delay_chan2,
                                                        low_time=self.period2 - self.high_time2)
        # self.task.co_channels.add_co_pulse_chan_time(counter=self.channel, units=nidaqmx.constants.TimeUnits.SECONDS,
        # idle_state=nidaqmx.constants.Level.LOW, initial_delay=0, low_time=0.01, high_time=0.01)
        self.task_ni.timing.cfg_implicit_timing(sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS,
                                                samps_per_chan=100)
        # samps_per_chan(*or/)freq must be an integer! (maybe)
        if self.trigger:
            self.task_ni.triggers.start_trigger.trig_type = nidaqmx.constants.TriggerType.DIGITAL_EDGE
            self.task_ni.triggers.start_trigger.cfg_dig_edge_start_trig(trigger_source=self.trigger_source,
                                                                        trigger_edge=self.dict.get(self.trigger_edge))


    def create_stream(self):
        self.stream_ni = nidaqmx.stream_writers.CounterWriter.write_many_sample_pulse_time(np.array([20, 50]), np.array([30, 30]))
        with nidaqmx.Task() as self.task_ni:
            self.task_ni


    def start_task(self):
        print('Task started')
        self.create_task()
        # print(self.task_ni.name)
        self.task_ni.start()

    def update_task(self):
        pass

    def set_initial_delay_chan1(self, initial_delay_chan1):
        self.initial_delay_chan1 = initial_delay_chan1

    def set_initial_delay_chan2(self, initial_delay_chan2):
        self.initial_delay_chan2 = initial_delay_chan2

    def set_high_time1(self, high_time1):
        self.high_time1 = high_time1

    def set_low_time1(self, low_time1):
        self.low_time1 = low_time1

    def set_high_time2(self, high_time2):
        self.high_time2 = high_time2

    def set_period2(self, period2):
        self.period2 = period2

    def set_trigger(self, trigger):
        self.trigger = trigger

    def set_trigger_source(self, trigger_source):
        self.trigger_source = trigger_source

    def set_trigger_edge(self, trigger_edge):
        self.trigger_edge = trigger_edge

    def stop_task(self):
        print('Task stop')
        # suppress warning that might occur when task i stopped during acquisition
        # warnings.filterwarnings('ignore', category=nidaqmx.DaqWarning)
        self.task_ni.stop()  # stop the task(different from the closing of the task, I suppose)
        # warnings.filterwarnings('default',category=nidaqmx.DaqWarning)
        self.task_ni.close()

    def close(self):
        self.task_ni.close()  # close the task
        # self.task_co_ni.close()
