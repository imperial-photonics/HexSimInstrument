""" 
   Written by Michele Castriotta, Alessandro Zecchi, Andrea Bassi (Polimi).
   Code for creating the measurement class of ScopeFoundry for the Orca Flash 4V3
   11/18
"""

from ScopeFoundry import Measurement
from ScopeFoundry.helper_funcs import sibling_path, load_qt_ui_file
from ScopeFoundry import h5_io
import pyqtgraph as pg
import numpy as np
from datetime import datetime
import os
import time

class HamamatsuMeasurement(Measurement):
    
    name = "hamamatsu_image"
        
    def setup(self):
        
        "..."

        self.ui_filename = sibling_path(__file__, "form.ui")
        
    
        self.ui = load_qt_ui_file(self.ui_filename)
        self.settings.New('record', dtype=bool, initial=False, hardware_set_func=self.setRecord, hardware_read_func=self.getRecord, reread_from_hardware_after_write=True)
        self.settings.New('save_h5', dtype=bool, initial=False, hardware_set_func=self.setSaveH5, hardware_read_func=self.getSaveH5)
        self.settings.New('refresh_period', dtype=float, unit='s', spinbox_decimals = 4, initial=0.02 , hardware_set_func=self.setRefresh, vmin = 0)
        self.settings.New('autoRange', dtype=bool, initial=True, hardware_set_func=self.setautoRange)
        self.settings.New('autoLevels', dtype=bool, initial=True, hardware_set_func=self.setautoLevels)
        self.settings.New('level_min', dtype=int, initial=60, hardware_set_func=self.setminLevel, hardware_read_func = self.getminLevel)
        self.settings.New('level_max', dtype=int, initial=150, hardware_set_func=self.setmaxLevel, hardware_read_func = self.getmaxLevel)
        self.settings.New('threshold', dtype=int, initial=500, hardware_set_func=self.setThreshold)
        self.camera = self.app.hardware['HamamatsuHardware']
        
        self.autoRange = self.settings.autoRange.val
        self.display_update_period = self.settings.refresh_period.val
        self.autoLevels = self.settings.autoLevels.val
        self.level_min = self.settings.level_min.val
        self.level_max = self.settings.level_max.val
        
        #self.img = pg.gaussianFilter(np.random.normal(size=(400, 600)), (5, 5)) * 20 + 100
        
    def setup_figure(self):
        """
        Runs once during App initialization, after setup()
        This is the place to make all graphical interface initializations,
        build plots, etc.
        """
                
        # connect ui widgets to measurement/hardware settings or functions
        self.ui.start_pushButton.clicked.connect(self.start)
        self.ui.interrupt_pushButton.clicked.connect(self.interrupt)
        self.settings.save_h5.connect_to_widget(self.ui.save_h5_checkBox)
        
        # connect ui widgets of live settings
        self.settings.autoLevels.connect_to_widget(self.ui.autoLevels_checkBox)
        self.settings.autoRange.connect_to_widget(self.ui.autoRange_checkBox)
        self.settings.level_min.connect_to_widget(self.ui.min_doubleSpinBox) #spinBox doesn't work nut it would be better
        self.settings.level_max.connect_to_widget(self.ui.max_doubleSpinBox) #spinBox doesn't work nut it would be better
        
        # Set up pyqtgraph graph_layout in the UI
        self.imv = pg.ImageView()
        self.ui.plot_groupBox.layout().addWidget(self.imv)
        
        # Image initialization
        self.image = np.zeros((int(self.camera.subarrayv.val),int(self.camera.subarrayh.val)),dtype=np.uint16)
        
        # Create PlotItem object (a set of axes)  
        
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
                
                save = True
                
                while not self.interrupt_measurement_called:
                    
                    [frame, dims] = self.camera.hamamatsu.getLastFrame()        
                    self.np_data = frame.getData()
                    self.image = np.reshape(self.np_data,(self.eff_subarrayv, self.eff_subarrayh))
                    
                    if self.settings['record']:
                        self.camera.hamamatsu.stopAcquisition()
                        self.camera.hamamatsu.startRecording()
                        self.camera.hamamatsu.stopRecording()
                        self.interrupt()    
                                        
                    if self.settings['save_h5']:
                        
                        if save:
                            self.initH5()
                            save = False #at next cycle, we don't do initH5 again (we have already created the file)
                        
                        mean_value = np.mean(self.np_data)
                        last_frame_index = self.camera.hamamatsu.buffer_index
                        #print(self.camera.hamamatsu.last_frame_number)
                        if self.debug:
                            print("The mean is: ", mean_value)
#===============================================================================
#                         if mean_value > self.settings['threshold']:
#                             print("\n \n ******* \n \n Saving :D !\n \n *******")
#                             j = 0
#                             
#                             #while(len(frames)) < self.camera.number_frames.val: #we want 200 frames
#                             while j < self.camera.number_frames.val: 
#                                 upgraded_last_frame_index, upgraded_frame_number = self.camera.hamamatsu.getTransferInfo() #we upgrade the transfer information
#                                 
#                                 if last_frame_index < upgraded_last_frame_index: #acquisition has not reached yet the end of the buffer    
#                                     j = self.getThresholdH5(last_frame_index,  upgraded_last_frame_index + 1, j)
# #                                     for i in range(last_frame_index, upgraded_last_frame_index + 1):
# #                                         
# #                                         if j < self.camera.number_frames.val:
# #                                             frame = self.camera.hamamatsu.getRequiredFrame(i)[0]
# #                                             self.np_data = frame.getData() #-1 takes the last element
# #                                             self.image = np.reshape(self.np_data,(int(self.camera.subarrayv.val), int(self.camera.subarrayh.val))).T
# #                                             self.image_h5[j,:,:] = self.image # saving to the h5 dataset
# #                                             j+=1
# #                                             self.settings['progress'] = j*100./self.camera.hamamatsu.number_image_buffers
#                                 
#                                 else: #acquisition has reached the end of the buffer
#                                     j = self.getThresholdH5(last_frame_index+1, 3*self.camera.hamamatsu.number_image_buffers + 1, j)
#                                     j = self.getThresholdH5(0, upgraded_last_frame_index, j)
#                                 
#                                 last_frame_index = upgraded_last_frame_index
#                                 
#                                 
#                                 
#                                 
#                             self.interrupt()
#                             print(self.camera.hamamatsu.last_frame_number)
#===============================================================================
                     
                        if mean_value > self.settings['threshold']:
                            
                            print("\n \n ******* \n \n Saving :D !\n \n *******")
                            j = 0
                            #starting_index=last_frame_index
                            stalking_number = 0
                            remaining = False
                            while j < self.camera.number_frames.val: 
                                
                                self.get_and_save_Frame(j,last_frame_index)
                                last_frame_index = self.updateIndex(last_frame_index)
                                
                                if self.debug:
                                    print("The last_frame_index is: ", last_frame_index)
                                    
                                j+=1
                                
                                if not remaining:
                                    upgraded_last_frame_index = self.camera.hamamatsu.getTransferInfo()[0] # upgrades the transfer information
                                    #The stalking_number represents the relative steps the camera has made in acquisition with respect to the saving.
                                    stalking_number = stalking_number + self.camera.hamamatsu.backlog - 1
                                    
                                    if self.debug:
                                        print('upgraded_last_frame_index: ' , upgraded_last_frame_index)
                                        print('stalking_number: ' , stalking_number)
                                        print('The camera is at {} passes from you'.format(self.camera.hamamatsu.number_image_buffers - stalking_number))
                                    
                                    if stalking_number + self.camera.hamamatsu.backlog > self.camera.hamamatsu.number_image_buffers: 
                                        self.camera.hamamatsu.stopAcquisitionNotReleasing() #stop acquisition when we know that at next iteration, some images may be rewritten
                                        remaining = True #if the buffer reach us, we execute the "while" without the "if not remaining" block.
                                   
                            self.interrupt()
                            self.camera.hamamatsu.stopAcquisition()
                            if self.debug:
                                print("The last_frame_number is: ", self.camera.hamamatsu.last_frame_number)
                         
        finally:
            
            self.camera.hamamatsu.stopAcquisition()

            if self.settings['save_h5']:
                self.h5file.close() # close h5 file  
            self.settings.save_h5.update_value(new_val = False)

    def setRefresh(self, refresh_period):  
        self.display_update_period = refresh_period
    
    def setautoRange(self, autoRange):
        self.autoRange = autoRange

    def setautoLevels(self, autoLevels):
        self.autoLevels = autoLevels
            
    def setminLevel(self, level_min):
        self.level_min = level_min
        
    def setmaxLevel(self, level_max):
        self.level_max = level_max
    
    def getminLevel(self):
        return self.imv.levelMin
        
    def getmaxLevel(self):
        return self.imv.levelMax
            
    def setThreshold(self, threshold):
        self.threshold = threshold
    
    def setSaveH5(self, save_h5):
        self.settings.save_h5.val = save_h5
        
    def getSaveH5(self):
        if self.settings['record']:
            self.settings.save_h5.val = False
        return self.settings.save_h5.val 
        
    def setRecord(self, record):
        self.settings.record = record
    
        
    def getRecord(self):
        if self.settings['save_h5']:
            self.settings.record = False
        return self.settings.record 
    
    def initH5(self):
        """
        Initialization operations for the h5 file.
        """
        
        self.h5file = h5_io.h5_base_file(app=self.app, measurement=self)
        self.h5_group = h5_io.h5_create_measurement_group(measurement=self, h5group=self.h5file)
        img_size=self.image.shape
        length=self.camera.hamamatsu.number_image_buffers
        self.image_h5 = self.h5_group.create_dataset( name  = 't0/c0/image', 
                                                      shape = ( length, img_size[0], img_size[1]),
                                                      dtype = self.image.dtype, chunks = (1, self.eff_subarrayv, self.eff_subarrayh)
                                                      )
        """
        THESE NAMES MUST BE CHANGED
        """
        self.image_h5.dims[0].label = "z"
        self.image_h5.dims[1].label = "y"
        self.image_h5.dims[2].label = "x"
        
        #self.image_h5.attrs['element_size_um'] =  [self.settings['zsampling'], self.settings['ysampling'], self.settings['xsampling']]
        self.image_h5.attrs['element_size_um'] =  [1,1,1]
        
    def initH5_temp(self):
        """
        Initialization operations for the h5 file.
        """
        t0 = time.time()
        f = self.app.settings['data_fname_format'].format(
            app=self.app,
            measurement=self,
            timestamp=datetime.fromtimestamp(t0),
            sample=self.app.settings["sample"],
            ext='h5')
        fname = os.path.join(self.app.settings['save_dir'], f)
        
        self.h5file = h5_io.h5_base_file(app=self.app, measurement=self, fname = fname)
        self.h5_group = h5_io.h5_create_measurement_group(measurement=self, h5group=self.h5file)
        img_size=self.image.shape
        length=self.camera.hamamatsu.number_image_buffers
        self.image_h5 = self.h5_group.create_dataset( name  = 't0/c1/image', 
                                                      shape = ( length, img_size[0], img_size[1]),
                                                      dtype = self.image.dtype, chunks = (1, self.eff_subarrayv, self.eff_subarrayh)
                                                      )
        self.image_h5_2 = self.h5_group.create_dataset( name  = 't0/c2/image', 
                                                      shape = ( length, img_size[0], img_size[1]),
                                                      dtype = self.image.dtype, chunks = (1, self.eff_subarrayv, self.eff_subarrayh)
                                                      )
        """
        THESE NAMES MUST BE CHANGED
        """
        self.image_h5.dims[0].label = "z"
        self.image_h5.dims[1].label = "y"
        self.image_h5.dims[2].label = "x"
        
        #self.image_h5.attrs['element_size_um'] =  [self.settings['zsampling'], self.settings['ysampling'], self.settings['xsampling']]
        self.image_h5.attrs['element_size_um'] =  [1,1,1]
        
        self.image_h5_2.dims[0].label = "z"
        self.image_h5_2.dims[1].label = "y"
        self.image_h5_2.dims[2].label = "x"
        
        #self.image_h5.attrs['element_size_um'] =  [self.settings['zsampling'], self.settings['ysampling'], self.settings['xsampling']]
        self.image_h5_2.attrs['element_size_um'] =  [1,1,1]
                
    def initH5_temp2(self):
        """
        Initialization operations for the h5 file.
        """
        t0 = time.time()
        f = self.app.settings['data_fname_format'].format(
            app=self.app,
            measurement=self,
            timestamp=datetime.fromtimestamp(t0),
            sample=self.app.settings["sample"],
            ext='h5')
        fname = os.path.join(self.app.settings['save_dir'], f)
        
        self.h5file = h5_io.h5_base_file(app=self.app, measurement=self, fname = fname)
        self.h5_group = h5_io.h5_create_measurement_group(measurement=self, h5group=self.h5file)
        img_size=self.image.shape
        length=self.camera.hamamatsu.number_image_buffers
        self.image_h5 = self.h5_group.create_dataset( name  = 't0/c0/image', 
                                                      shape = ( length, img_size[0], img_size[1]),
                                                      dtype = self.image.dtype, chunks = (1, self.eff_subarrayv, self.eff_subarrayh)
                                                      )
        """
        THESE NAMES MUST BE CHANGED
        """
        self.image_h5.dims[0].label = "z"
        self.image_h5.dims[1].label = "y"
        self.image_h5.dims[2].label = "x"
        
        #self.image_h5.attrs['element_size_um'] =  [self.settings['zsampling'], self.settings['ysampling'], self.settings['xsampling']]
        self.image_h5.attrs['element_size_um'] =  [1,1,1]
        
    #===========================================================================
    # def getThresholdH5(self, start, end, j):
    #     """
    #     Get the data at the i-th frame (from start to end-1), and 
    #     save the reshaped data into an h5 file.
    #     
    #     j is a variable that gets updated every time. It represents
    #     the number of saved images. If this number gets bigger than
    #     the wanted number of frames, the below operation is not
    #     executed (we dont want to save other frames).
    #     
    #     Upload the progress bar.
    #     """
    #     for i in range(start, end):
    #         #put elements in new_frames until the end of buffer
    #         if j < self.camera.number_frames.val:
    #             frame = self.camera.hamamatsu.getRequiredFrame(i)[0]
    #             self.np_data = frame.getData()
    #             self.image = np.reshape(self.np_data,(self.eff_subarrayv, self.eff_subarrayh)).T
    #             self.image_h5[j,:,:] = self.image # saving to the h5 dataset
    #             j+=1
    #             self.settings['progress'] = j*100./self.camera.hamamatsu.number_image_buffers
    #             
    #     return j
    #===========================================================================
    
    def get_and_save_Frame(self, saveindex, lastframeindex):
        """
        Get the data at the lastframeindex, and 
        save the reshaped data into an h5 file.
        saveindex is an index representing the position of the saved image
        in the h5 file. 
        Update the progress bar.
        """
           
        frame = self.camera.hamamatsu.getRequiredFrame(lastframeindex)[0]
        self.np_data = frame.getData()
        self.image = np.reshape(self.np_data,(self.eff_subarrayv, self.eff_subarrayh))
        self.image_h5[saveindex,:,:] = self.image # saving to the h5 dataset
        self.h5file.flush() # maybe is not necessary
        self.settings['progress'] = saveindex*100./self.camera.hamamatsu.number_image_buffers
    
    def updateIndex(self, last_frame_index):
        """
        Update the index of the image to fetch from buffer. 
        If we reach the end of the buffer, we reset the index.
        """
        last_frame_index+=1
        
        if last_frame_index > self.camera.hamamatsu.number_image_buffers - 1: #if we reach the end of the buffer
            last_frame_index = 0 #reset
        
        return last_frame_index