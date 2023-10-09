from ScopeFoundry import HardwareComponent

from devices.ni_do_device import NI_DO_device

import nidaqmx.system as ni

class NI_DO_hw(HardwareComponent):
    
    name = 'NI_DO_hw' 
    
    def setup(self):
        #create logged quantities, that are related to the graphical interface
        
        board, terminals=self.update_channels()
        
        self.devices = self.add_logged_quantity('device',  dtype=str, initial=board)        
        self.channel = self.add_logged_quantity('channel', dtype=str, choices=terminals, initial='Dev1/port1/line1')
        self.value = self.add_logged_quantity('value', dtype=int, initial='0', vmax=1, vmin=0)
        self.add_operation("write_value", self.write_value)
      
        
    def connect(self):
        #continuous_to_constant = {False:10178, True:10123} #create a dictionary for mapping the mode with the corresponding constants in nidaqmx
            
        #open connection to hardware
        self.channel.change_readonly(True)
        self.DO_device = NI_DO_device(channel=self.channel.val, debug=self.debug_mode.val)
        #connect logged quantities
        
        
    def disconnect(self):
        self.channel.change_readonly(False)
        #disconnect hardware
        if hasattr(self, 'DO_device'):
            self.DO_device.close()
            del self.DO_device
        
        for lq in self.settings.as_list():
            lq.hardware_read_func = None
            lq.hardware_set_func = None
            
    def write_value(self):
        self.DO_device.write(self.value.val)
        print(f'write {self.value.val}')
        
              
    def update_channels(self):
        ''' Find a NI device and return board + do_terminals'''
        system = ni.System.local()
        device=system.devices[0]
        board=device.product_type + ' : ' + device.name
        terminals=[]
        for line in device.do_lines :
            terminals.append(line.name)
              
        return board, terminals

if __name__ == '__main__':

    import time
    d = NI_DO_hw(HardwareComponent)
    d.setup()
    d.connect()
    time.sleep(3)
    d.value.val = 1
    d.disconnect()

