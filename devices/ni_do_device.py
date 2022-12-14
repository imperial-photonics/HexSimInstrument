import nidaqmx
import warnings
import math
import numpy as np

from nidaqmx import stream_writers

class NI_DO_device(object):
    
    def __init__(self, channel, debug=False):
    
        self.debug = debug 
        self.channel = channel
                 
        self.Task()
    
    def Task(self):
        if hasattr(self, 'task'):
            self.close()
            
        self.task = nidaqmx.Task()
        self.task.do_channels.add_do_chan(lines=self.channel)
        
        
    def write(self, value):
        
        self.task.write(bool(value))

        
    def stop_task(self):
        #suppress warning that might occurr when task i stopped during acquisition
        #warnings.filterwarnings('ignore', category=nidaqmx.DaqWarning)
        self.task.stop() #stop the task(different from the closing of the task, I suppose)
        #warnings.filterwarnings('default',category=nidaqmx.DaqWarning)
        
            
    def close(self):
        
        self.task.close() #close the task
        