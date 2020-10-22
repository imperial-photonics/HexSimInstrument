from CameraMeasurement import HamamatsuMeasurement
from ScopeFoundry.helper_funcs import sibling_path, load_qt_ui_file
from ScopeFoundry import h5_io
import pyqtgraph as pg
import numpy as np

class HamamatsuPostSavingMeasurement(HamamatsuMeasurement):
    ''' SubClass of HamamatsuMeasurement but with new hardwares to control. 
    The specific settings to make a synchronized measurement are set before and at 
    the end of the acquisition.'''
    
    name = "HamamatsuPostSavingMeasurement"
    

    def setup(self):
        
        super().setup()
        self.debug=False

        self.settings.New('Acq_freq', dtype=float, unit='Hz', initial=50)
        self.settings.New('xsampling', dtype=float, unit='um', initial=0.11)
        self.settings.New('ysampling', dtype=float, unit='um', initial=0.11)
        self.settings.New('zsampling', dtype=float, unit='um', initial=1.0)
        
    def run(self):
        
        self.eff_subarrayh = int(self.camera.subarrayh.val/self.camera.binning.val)
        self.eff_subarrayv = int(self.camera.subarrayv.val/self.camera.binning.val)
        
        self.image = np.zeros((self.eff_subarrayv,self.eff_subarrayh),dtype=np.uint16)
        
        self.image[0,0] = 1 #Otherwise we get the "all zero pixels" error (we should modify pyqtgraph...)

        try:
            
            self.camera.read_from_hardware()

            self.camera.hamamatsu.startAcquisition()
            
            index = 0
            
            if self.camera.acquisition_mode.val == "fixed_length":
            
                if self.settings['save_h5']:
                    self.initH5()
                    print("\n \n ******* \n \n Saving :D !\n \n *******")
                    
                while index < self.camera.hamamatsu.number_image_buffers:
        
                    # Get frames.
                    #The camera stops acquiring once the buffer is terminated (in snapshot mode)
                    [frames, dims] = self.camera.hamamatsu.getFrames()
                    
                    # Save frames.
                    for aframe in frames:
                        
                        self.np_data = aframe.getData()  
                        self.image = np.reshape(self.np_data,(self.eff_subarrayv, self.eff_subarrayh)) 
                        if self.settings['save_h5']:
                            self.image_h5[index,:,:] = self.image # saving to the h5 dataset
                            self.h5file.flush() # maybe is not necessary
                                            
                        if self.interrupt_measurement_called:
                            break
                        index+=1
                        print(index)
                    
                    if self.interrupt_measurement_called:
                        break    
                    #index = index + len(frames)
                    #np_data.tofile(bin_fp)
                    self.settings['progress'] = index*100./self.camera.hamamatsu.number_image_buffers
                    
            elif self.camera.acquisition_mode.val == "run_till_abort":
                
                #save = True
                
                while not self.interrupt_measurement_called:

                    [frame, dims] = self.camera.hamamatsu.getLastFrame()        
                    self.np_data = frame.getData()
                    self.image = np.reshape(self.np_data,(self.eff_subarrayv, self.eff_subarrayh))
                    
                    if self.settings['save_h5']:
                        self.camera.hamamatsu.stopAcquisitionNotReleasing()
                        self.initH5()
                        print("\n \n ******* \n \n Saving :D !\n \n *******")
                        [frames, dims] = self.camera.hamamatsu.getLastTotFrames()
                        for aframe in frames:
                            self.np_data = aframe.getData()
                            self.image_on_the_run = np.reshape(self.np_data, (self.eff_subarrayv, self.eff_subarrayh))
                            self.image_h5[index, :, :] = self.image_on_the_run  # saving to the h5 dataset
                            self.h5file.flush()  # maybe is not necessary
                            self.settings['progress'] = index*100./self.camera.hamamatsu.number_image_buffers
                            index+=1
                            
                        
                        self.h5file.close()
                        self.settings['save_h5'] = False
                        index = 0
                        self.settings['progress'] = index*100./self.camera.hamamatsu.number_image_buffers
                        self.camera.hamamatsu.startAcquisitionWithoutAlloc()
        finally:
            
            self.camera.hamamatsu.stopAcquisition()

            if self.settings['save_h5']:
                self.h5file.close() # close h5 file  
    
    def update_display(self):
        """
        Displays the numpy array called self.image.  
        This function runs repeatedly and automatically during the measurement run,
        its update frequency is defined by self.display_update_period.
        """
        #self.optimize_plot_line.setData(self.buffer) 

        #self.imv.setImage(np.reshape(self.np_data,(self.camera.subarrayh.val, self.camera.subarrayv.val)).T)
        #self.imv.setImage(self.image, autoLevels=False, levels=(100,340))
        if self.autoLevels == False:  
            self.imv.setImage((self.image).T, autoLevels=self.settings.autoLevels.val, autoRange=self.settings.autoRange.val, levels=(self.level_min, self.level_max))
        else: #levels should not be sent when autoLevels is True, otherwise the image is displayed with them
            self.imv.setImage((self.image).T, autoLevels=self.settings.autoLevels.val, autoRange=self.settings.autoRange.val)
            self.settings.level_min.read_from_hardware()
            self.settings.level_max.read_from_hardware()
