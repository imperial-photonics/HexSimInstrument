import os
import time
from datetime import datetime
from threading import Thread

import numpy as np
import pyqtgraph as pg
import tifffile as tif
from ScopeFoundry import Measurement
from ScopeFoundry import h5_io
from ScopeFoundry.helper_funcs import sibling_path, load_qt_ui_file
from hexSimProcessor import *

from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QLineEdit, QFileDialog

class HexSimMeasurement(Measurement):
    name = 'hexsim_measurement'

    def setup(self):

        # load ui file
        self.ui_filename = sibling_path(__file__, "hexsim_measurement.ui")
        self.ui = load_qt_ui_file(self.ui_filename)

        # create reconstructor
        self.h = hexSimProcessor()

        # camera settings
        self.settings.New('record', dtype=bool, initial=False, hardware_set_func=self.setRecord,
                          hardware_read_func=self.getRecord, reread_from_hardware_after_write=True)
        self.settings.New('save_h5', dtype=bool, initial=False, hardware_set_func=self.setSaveH5,
                          hardware_read_func=self.getSaveH5)
        self.settings.New('refresh_period', dtype=float, unit='s', spinbox_decimals=4, initial=0.02,
                          hardware_set_func=self.setRefresh, vmin=0)
        self.settings.New('autoRange', dtype=bool, initial=True, hardware_set_func=self.setautoRange)
        self.settings.New('autoLevels', dtype=bool, initial=True, hardware_set_func=self.setautoLevels)
        self.settings.New('level_min', dtype=int, initial=60, hardware_set_func=self.setminLevel,
                          hardware_read_func=self.getminLevel)
        self.settings.New('level_max', dtype=int, initial=150, hardware_set_func=self.setmaxLevel,
                          hardware_read_func=self.getmaxLevel)
        self.settings.New('threshold', dtype=int, initial=500, hardware_set_func=self.setThreshold)

        self.camera = self.app.hardware['HamamatsuHardware']
        self.screen = self.app.hardware['screenHardware']

        self.autoRange = self.settings.autoRange.val
        self.display_update_period = self.settings.refresh_period.val
        self.autoLevels = self.settings.autoLevels.val
        self.level_min = self.settings.level_min.val
        self.level_max = self.settings.level_max.val

        self.isStreamSimuRun = False
        self.isUpdateImageViewer = False
        self.isCameraRun = False

    def setup_figure(self):

        # Camera
        # connect ui widgets to measurement/hardware settings or functionss
        self.ui.camButton.clicked.connect(self.camButtonPressed)

        # Set up pyqtgraph graph_layout in the UI
        self.imv = pg.ImageView()
        self.imvRaw = pg.ImageView()
        self.imvSIM = pg.ImageView()

        self.ui.imgStreamLayout.addWidget(self.imv)
        self.ui.rawImageLayout.addWidget(self.imvRaw)
        self.ui.simImageLayout.addWidget(self.imvSIM)

        # Image initialization
        self.image = np.zeros((int(self.camera.subarrayv.val), int(self.camera.subarrayh.val)), dtype=np.uint16)
        self.imageRaw = np.zeros((1,int(self.camera.subarrayv.val), int(self.camera.subarrayh.val)), dtype=np.uint16)
        self.imageSIM = np.zeros((1,2 * int(self.camera.subarrayv.val), 2 * int(self.camera.subarrayh.val)),
                                 dtype=np.uint16)

        self.imvRaw.setImage((self.imageRaw[0,:,:]).T, autoRange=True, autoLevels=True,  autoHistogramRange=True)
        self.imvSIM.setImage((self.imageSIM[0, :, :]).T, autoRange=True, autoLevels=True,  autoHistogramRange=True)
        # print(self.imageRaw.shape())

        # Control
        self.ui.slmButton.clicked.connect(self.slmButtonPressed)
        self.ui.previousPatternButton.clicked.connect(self.screen.previousPattern)
        self.ui.nextPatternButton.clicked.connect(self.screen.nextPattern)

        # Measure
        self.ui.captureStandardButton.clicked.connect(self.captureStandardButtonPressed)
        self.ui.rawImageSlider.valueChanged.connect(self.rawImageSliderChanged)
        self.ui.simImageSlider.valueChanged.connect(self.simImageSliderChanged)

        # Test
        self.ui.snapshotButton.clicked.connect(self.snapshotButtonPressed)
        self.ui.snapshotSeqButton.clicked.connect(self.snapshotSeqButtonPressed)

        self.ui.standardSimuButton.clicked.connect(self.standardSimuButtonPressed)
        self.ui.standardSimuUpdate.clicked.connect(self.standardReconstructionUpdate)

        self.ui.streamSimuButton.clicked.connect(self.streamSimuButtonPressed)
        self.ui.streamSimuStop.clicked.connect(self.streamSimuStopPressed)

        self.ui.batchSimuButton.clicked.connect(self.batchSimuButtonPressed)
        self.ui.batchSimuUpdate.clicked.connect(self.batchReconstructionUpdate)

    def update_display(self):
        """
        Displays the numpy array called self.image.
        This function runs repeatedly and automatically during the measurement run,
        its update frequency is defined by self.display_update_period.
        """

        # update camera viewer
        if self.isCameraRun:
            if self.autoLevels == False:
                self.imv.setImage((self.image).T, autoLevels=self.settings.autoLevels.val,
                                  autoRange=self.settings.autoRange.val, levels=(self.level_min, self.level_max))

            else:  # levels should not be sent when autoLevels is True, otherwise the image is displayed with them
                self.imv.setImage((self.image).T, autoLevels=self.settings.autoLevels.val,
                                  autoRange=self.settings.autoRange.val)

                self.settings.level_min.read_from_hardware()
                self.settings.level_max.read_from_hardware()

        # update hexsim viwer
        if self.isStreamSimuRun or self.isUpdateImageViewer:
            self.updateImageViewer()
            self.isUpdateImageViewer = False

        # print('Display')

    def run(self):

        self.eff_subarrayh = int(self.camera.subarrayh.val / self.camera.binning.val)
        self.eff_subarrayv = int(self.camera.subarrayv.val / self.camera.binning.val)

        self.image = np.zeros((self.eff_subarrayv, self.eff_subarrayh), dtype=np.uint16)
        self.image[0, 0] = 1  # Otherwise we get the "all zero pixels" error (we should modify pyqtgraph...)
        # self.imageRaw = np.zeros((self.eff_subarrayv, self.eff_subarrayh), dtype=np.uint16)
        # self.imageRaw[0, 0] = 1

        while not self.interrupt_measurement_called:
            self.cameraRunTest()

            # print('run here')
            # time.sleep(5)
        # self.cameraRun()

    ############ Actions ########################
    def camButtonPressed(self):
        if self.ui.camButton.text() == 'ON':
            # self.start()
            self.cameraStart()
            self.ui.camButton.setText('OFF')
            print('Camera ON')
        elif self.ui.camButton.text() == 'OFF':
            # self.interrupt()
            self.cameraInterrupt()
            self.ui.camButton.setText('ON')
            print('Camera OFF')

    def slmButtonPressed(self):
        if self.ui.slmButton.text() == 'ON':
            self.screen.openSLM()
            self.screen.manualDisplay()
            self.ui.slmButton.setText('OFF')
            print('OFF')
        elif self.ui.slmButton.text() == 'OFF':
            self.screen.closeSLM()
            self.ui.slmButton.setText('ON')
            print('ON')

    def captureStandardButtonPressed(self):
        # initialize the raw image array
        self.imageRaw = np.zeros((7, self.eff_subarrayv, self.eff_subarrayh), dtype=np.uint16)
        self.imageRaw[0, 0, 0] = 1
        # acqusition
        self.standardAcquistion()
        # display the raw images
        self.ui.rawImageSlider.setMinimum(0)
        self.ui.rawImageSlider.setMaximum(self.imageRaw.shape[0] - 1)
        self.ui.rawImageSlider.setValue(0)

        self.imvRaw.setImage((self.imageRaw[0, :, :]).T)


    def rawImageSliderChanged(self):
        self.ui.rawImageSlider.setMinimum(0)
        self.ui.rawImageSlider.setMaximum(self.imageRaw.shape[0] - 1)

        self.imvRaw.setImage((self.imageRaw[int(self.ui.rawImageSlider.value()), :, :]).T)

        self.ui.rawImageNth.setText(str(self.ui.rawImageSlider.value()+1))
        self.ui.rawImageNtotal.setText(str(len(self.imageRaw)))

    def simImageSliderChanged(self):
        self.ui.simImageSlider.setMinimum(0)
        self.ui.simImageSlider.setMaximum(self.imageSIM.shape[0] - 1)
        temp = (self.imageSIM[int(self.ui.simImageSlider.value()), :, :]).T
        self.imvSIM.setImage(temp,levels = (0,0.7*np.amax(temp)))

        self.ui.simImageNth.setText(str(self.ui.simImageSlider.value()+1))
        self.ui.simImageNtotal.setText(str(len(self.imageSIM)))

    def updateImageViewer(self):
        self.rawImageSliderChanged()
        self.simImageSliderChanged()

    def standardSimuButtonPressed(self):
        # read data
        # filename = "./data/standardData.tif"
        filename, _ = QFileDialog.getOpenFileName(directory="./data")
        self.imageRaw = np.single(tif.imread(filename))
        self.standardReconstructionUpdate()


    def standardReconstructionUpdate(self):
        self.standardSimuThread = Thread(target=self.standardReconstruction)
        self.standardSimuThread.start()
        # self.standardSimuThread.join()

    def standardReconstruction(self):
        # create reconstruction object
        self.setReconstructor(self.imageRaw)

        # calibrate & reconstruction
        if self.h.gpuenable:
            self.h.calibrate_fast(self.imageRaw)
            self.imageSIM = self.h.reconstruct_cupy(self.imageRaw).get()

        elif not self.h.gpuenable:
            self.h.calibrate(self.imageRaw)
            self.imageSIM = self.h.reconstruct_rfftw(self.imageRaw)

        self.imageSIM = self.imageSIM[np.newaxis, :, :]
        # update display
        self.isUpdateImageViewer = True
        print('One SIM image is processed.')

    def streamSimuButtonPressed(self):
        self.virtualRecording()
        self.isStreamSimuRun = True
        # create reconstruction object
        [_,width,height] = self.imageRawStack.shape
        self.imageRaw = np.zeros((7, width, height), dtype=np.uint16)
        # calibration from the first 7 images
        self.streamIndex = 0

        for i in range(7):
            self.imageRaw[i,:,:] = self.imageRawStack[i, :, :]
            self.streamIndex +=1
            time.sleep(0.5)
            print(self.streamIndex)

        self.standardReconstruction()
        self.updateImageViewer()

        self.streamSimuThread = Thread(target=self.streamReconstructionLoop)
        self.streamSimuThread.start()

    def streamReconstructionLoop(self):
        while self.isStreamSimuRun and self.streamIndex<=280:
            self.imageRawFrame = self.imageRawStack[self.streamIndex,:,:]

            self.streamReconstruction(self.imageRawFrame,(self.streamIndex) % 7)
            time.sleep(0.5)
            self.streamIndex +=1
            print(self.streamIndex)

    def streamReconstruction(self,newFrame,index):
        if self.h.gpuenable:
            self.imageSIM = (self.h.reconstructframe_cupy(newFrame,index)).get()
        elif not self.h.gpuenable:
            self.imageSIM = self.h.reconstructframe_rfftw(newFrame, index)

        self.imageSIM = self.imageSIM[np.newaxis, :, :]

    def streamSimuStopPressed(self):
        self.isStreamSimuRun = False

    def batchSimuButtonPressed(self):
        # read data
        # filename = "./data/stackData.tif"
        filename, _ = QFileDialog.getOpenFileName(directory="./data")
        self.imageRaw = np.single(tif.imread(filename))
        self.batchReconstructionUpdate()

    def batchReconstructionUpdate(self):
        self.batchSimuThread = Thread(target=self.batchReconstruction)
        self.batchSimuThread.start()
        # self.batchSimuThread.join()
        # self.updateImageViewer()
        # print('One SIM stack is processed.')

    def batchReconstruction(self):
        # create reconstruction object
        self.setReconstructor(self.imageRaw)
        nStack = len(self.imageRaw)

        # calibrate & reconstruction
        if self.h.gpuenable:
            self.h.calibrate_fast(self.imageRaw[int(nStack / 2):int(nStack / 2 + 7), :, :])
            if self.h.compact:
                self.imageSIM = self.h.batchreconstructcompact_cupy(self.imageRaw).get()
            elif not self.h.compact:
                self.imageSIM = self.h.batchreconstruct_cupy(self.imageRaw).get()

        elif not self.h.gpuenable:
            self.h.calibrate(self.imageRaw[int(nStack / 2):int(nStack / 2 + 7), :, :])
            if self.h.compact:
                self.imageSIM = self.h.batchreconstructcompact(self.imageRaw)
            elif not self.h.compact:
                self.imageSIM = self.h.batchreconstruct(self.imageRaw)

        ## update display
        self.isUpdateImageViewer = True
        # print('One SIM stack is processed.')
        # self.updateImageViewer()
    # def captureBatchButtonPressed(self):
    #
    # def startStreamingButton(self):
    #

    def virtualRecording(self):
        filename, _ = QFileDialog.getOpenFileName(directory="./data")
        self.imageRawStack = np.single(tif.imread(filename))



    def snapshotButtonPressed(self):
        self.cameraInterrupt()
        # print('camera stopped')
        self.imageRaw = self.getOneFrame()
        # self.imageRaw = np.vstack((self.imageRaw ,self.imageRaw ))
        self.camera.updateCameraSettings()
        self.start()

        self.imageRaw = self.imageRaw[np.newaxis, :, :]
        self.rawImageSliderChanged()
        # self.ui.rawImageSlider.setMinimum(0)
        # self.ui.rawImageSlider.setMaximum(self.imageRaw.shape[0] - 1)
        # self.ui.rawImageSlider.setValue(0)
        # self.imvRaw.setImage((self.imageRaw[0, :, :]).T)

    def snapshotSeqButtonPressed(self):
        self.interrupt()
        # print('camera stopped')
        self.imageRaw = np.zeros((7, self.eff_subarrayv, self.eff_subarrayh), dtype=np.uint16)
        for i in range(7):
            time.sleep(0.1)
            self.imageRaw[i, :, :] = self.getOneFrame()
        self.camera.updateCameraSettings()
        self.start()

        self.rawImageSliderChanged()
        # self.ui.rawImageSlider.setMinimum(0)
        # self.ui.rawImageSlider.setMaximum(self.imageRaw.shape[0] - 1)
        # self.ui.rawImageSlider.setValue(0)
        # self.imvRaw.setImage((self.imageRaw[0, :, :]).T)

    ############ Methods ############
    def setReconstructor(self,imgStack):

        self.h.debug = self.ui.debugCheck.isChecked()
        self.h.cleanup = self.ui.cleanupCheck.isChecked()
        self.h.gpuenable = self.ui.gpuCheck.isChecked()
        self.h.axial = self.ui.axialCheck.isChecked()
        self.h.usemodulation = self.ui.usemodulationCheck.isChecked()
        self.h.compact = self.ui.compactCheck.isChecked()

        self.h.N = len(imgStack[0,:,:])
        self.h.magnification =self.ui.magnificationValue.value()
        self.h.NA = self.ui.naValue.value()
        self.h.n = self.ui.nValue.value()
        self.h.wavelength = self.ui.wavelengthValue.value()
        self.h.pixelsize = self.ui.pixelsizeValue.value()

        self.h.alpha = self.ui.alphaValue.value()
        self.h.beta = self.ui.betaValue.value()
        self.h.w = self.ui.wValue.value()
        self.h.eta = self.ui.etaValue.value()

    ## Acqusition
    def standardAcquistion(self):
        self.standardMeasureThread = Thread(target=self.standardMeasure)
        self.standardMeasureThread.start()

    def standardMeasure(self):
        self.interrupt()
        # # initialize the raw image array
        # self.imageRaw = np.zeros((7, self.eff_subarrayv, self.eff_subarrayh), dtype=np.uint16)
        # project the patterns and acquire raw images
        for i in range(7):
            self.screen.slm_dev.displayFrameN(i)
            # time.sleep(0.1)
            self.imageRaw[i, :, :] = self.getOneFrame()
            print('Take frame:', i)
        # recover camera streaming
        self.camera.updateCameraSettings()
        self.start()

    def getOneFrame(self):
        self.camera.hamamatsu.setACQMode("fixed_length", number_frames=1)
        self.camera.hamamatsu.startAcquisition()
        [frames, dims] = self.camera.hamamatsu.getFrames()
        self.camera.hamamatsu.stopAcquisition()
        if len(frames) > 0:
            self.last_image = np.reshape(frames[0].getData().astype(np.uint16), dims)
        else:
            self.last_image = np.zeros([self.eff_subarrayv, self.eff_subarrayh])
            print("Camera buffer empty")

        return self.last_image[np.newaxis, :, :]

    # Settings
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
        img_size = self.image.shape
        length = self.camera.hamamatsu.number_image_buffers
        self.image_h5 = self.h5_group.create_dataset(name='t0/c0/image',
                                                     shape=(length, img_size[0], img_size[1]),
                                                     dtype=self.image.dtype,
                                                     chunks=(1, self.eff_subarrayv, self.eff_subarrayh)
                                                     )
        """
        THESE NAMES MUST BE CHANGED
        """
        self.image_h5.dims[0].label = "z"
        self.image_h5.dims[1].label = "y"
        self.image_h5.dims[2].label = "x"

        # self.image_h5.attrs['element_size_um'] =  [self.settings['zsampling'], self.settings['ysampling'], self.settings['xsampling']]
        self.image_h5.attrs['element_size_um'] = [1, 1, 1]

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

        self.h5file = h5_io.h5_base_file(app=self.app, measurement=self, fname=fname)
        self.h5_group = h5_io.h5_create_measurement_group(measurement=self, h5group=self.h5file)
        img_size = self.image.shape
        length = self.camera.hamamatsu.number_image_buffers
        self.image_h5 = self.h5_group.create_dataset(name='t0/c1/image',
                                                     shape=(length, img_size[0], img_size[1]),
                                                     dtype=self.image.dtype,
                                                     chunks=(1, self.eff_subarrayv, self.eff_subarrayh)
                                                     )
        self.image_h5_2 = self.h5_group.create_dataset(name='t0/c2/image',
                                                       shape=(length, img_size[0], img_size[1]),
                                                       dtype=self.image.dtype,
                                                       chunks=(1, self.eff_subarrayv, self.eff_subarrayh)
                                                       )
        """
        THESE NAMES MUST BE CHANGED
        """
        self.image_h5.dims[0].label = "z"
        self.image_h5.dims[1].label = "y"
        self.image_h5.dims[2].label = "x"

        # self.image_h5.attrs['element_size_um'] =  [self.settings['zsampling'], self.settings['ysampling'], self.settings['xsampling']]
        self.image_h5.attrs['element_size_um'] = [1, 1, 1]

        self.image_h5_2.dims[0].label = "z"
        self.image_h5_2.dims[1].label = "y"
        self.image_h5_2.dims[2].label = "x"

        # self.image_h5.attrs['element_size_um'] =  [self.settings['zsampling'], self.settings['ysampling'], self.settings['xsampling']]
        self.image_h5_2.attrs['element_size_um'] = [1, 1, 1]

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

        self.h5file = h5_io.h5_base_file(app=self.app, measurement=self, fname=fname)
        self.h5_group = h5_io.h5_create_measurement_group(measurement=self, h5group=self.h5file)
        img_size = self.image.shape
        length = self.camera.hamamatsu.number_image_buffers
        self.image_h5 = self.h5_group.create_dataset(name='t0/c0/image',
                                                     shape=(length, img_size[0], img_size[1]),
                                                     dtype=self.image.dtype,
                                                     chunks=(1, self.eff_subarrayv, self.eff_subarrayh)
                                                     )
        """
        THESE NAMES MUST BE CHANGED
        """
        self.image_h5.dims[0].label = "z"
        self.image_h5.dims[1].label = "y"
        self.image_h5.dims[2].label = "x"

        # self.image_h5.attrs['element_size_um'] =  [self.settings['zsampling'], self.settings['ysampling'], self.settings['xsampling']]
        self.image_h5.attrs['element_size_um'] = [1, 1, 1]

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
        self.image = np.reshape(self.np_data, (self.eff_subarrayv, self.eff_subarrayh))
        self.image_h5[saveindex, :, :] = self.image  # saving to the h5 dataset
        self.h5file.flush()  # maybe is not necessary
        self.settings['progress'] = saveindex * 100. / self.camera.hamamatsu.number_image_buffers

    def updateIndex(self, last_frame_index):
        """
        Update the index of the image to fetch from buffer.
        If we reach the end of the buffer, we reset the index.
        """
        last_frame_index += 1

        if last_frame_index > self.camera.hamamatsu.number_image_buffers - 1:  # if we reach the end of the buffer
            last_frame_index = 0  # reset

        return last_frame_index

    def cameraRun(self):
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
                    # The camera stops acquiring once the buffer is terminated (in snapshot mode)
                    [frames, dims] = self.camera.hamamatsu.getFrames()

                    # Save frames.
                    for aframe in frames:

                        self.np_data = aframe.getData()
                        self.image = np.reshape(self.np_data, (self.eff_subarrayv, self.eff_subarrayh))
                        if self.settings['save_h5']:
                            self.image_h5[index, :, :] = self.image  # saving to the h5 dataset
                            self.h5file.flush()  # maybe is not necessary

                        if self.interrupt_measurement_called:
                            break
                        index += 1
                        print(index)

                    if self.interrupt_measurement_called:
                        break
                        # index = index + len(frames)
                    # np_data.tofile(bin_fp)
                    self.settings['progress'] = index * 100. / self.camera.hamamatsu.number_image_buffers

            elif self.camera.acquisition_mode.val == "run_till_abort":

                save = True

                while not self.interrupt_measurement_called:
                    # print('runing loop')
                    [frame, dims] = self.camera.hamamatsu.getLastFrame()
                    self.np_data = frame.getData()
                    self.image = np.reshape(self.np_data, (self.eff_subarrayv, self.eff_subarrayh))

                    if self.settings['record']:
                        self.camera.hamamatsu.stopAcquisition()
                        self.camera.hamamatsu.startRecording()
                        self.camera.hamamatsu.stopRecording()
                        self.interrupt()

                    if self.settings['save_h5']:

                        if save:
                            self.initH5()
                            save = False  # at next cycle, we don't do initH5 again (we have already created the file)

                        mean_value = np.mean(self.np_data)
                        last_frame_index = self.camera.hamamatsu.buffer_index
                        # print(self.camera.hamamatsu.last_frame_number)
                        if self.debug:
                            print("The mean is: ", mean_value)

                        if mean_value > self.settings['threshold']:

                            print("\n \n ******* \n \n Saving :D !\n \n *******")
                            j = 0
                            # starting_index=last_frame_index
                            stalking_number = 0
                            remaining = False
                            while j < self.camera.number_frames.val:

                                self.get_and_save_Frame(j, last_frame_index)
                                last_frame_index = self.updateIndex(last_frame_index)

                                if self.debug:
                                    print("The last_frame_index is: ", last_frame_index)

                                j += 1

                                if not remaining:
                                    upgraded_last_frame_index = self.camera.hamamatsu.getTransferInfo()[
                                        0]  # upgrades the transfer information
                                    # The stalking_number represents the relative steps the camera has made in acquisition with respect to the saving.
                                    stalking_number = stalking_number + self.camera.hamamatsu.backlog - 1

                                    if self.debug:
                                        print('upgraded_last_frame_index: ', upgraded_last_frame_index)
                                        print('stalking_number: ', stalking_number)
                                        print('The camera is at {} passes from you'.format(
                                            self.camera.hamamatsu.number_image_buffers - stalking_number))

                                    if stalking_number + self.camera.hamamatsu.backlog > self.camera.hamamatsu.number_image_buffers:
                                        self.camera.hamamatsu.stopAcquisitionNotReleasing()  # stop acquisition when we know that at next iteration, some images may be rewritten
                                        remaining = True  # if the buffer reach us, we execute the "while" without the "if not remaining" block.

                            self.interrupt()
                            self.camera.hamamatsu.stopAcquisition()
                            if self.debug:
                                print("The last_frame_number is: ", self.camera.hamamatsu.last_frame_number)

        finally:
            self.camera.hamamatsu.stopAcquisition()

    def cameraRunTest(self):
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
                    # The camera stops acquiring once the buffer is terminated (in snapshot mode)
                    [frames, dims] = self.camera.hamamatsu.getFrames()

                    # Save frames.
                    for aframe in frames:

                        self.np_data = aframe.getData()
                        self.image = np.reshape(self.np_data, (self.eff_subarrayv, self.eff_subarrayh))
                        if self.settings['save_h5']:
                            self.image_h5[index, :, :] = self.image  # saving to the h5 dataset
                            self.h5file.flush()  # maybe is not necessary

                        if not self.isCameraRun:
                            break
                        index += 1
                        print(index)

                    if not self.isCameraRun:
                        break
                        # index = index + len(frames)
                    # np_data.tofile(bin_fp)
                    self.settings['progress'] = index * 100. / self.camera.hamamatsu.number_image_buffers

            elif self.camera.acquisition_mode.val == "run_till_abort":

                save = True

                while self.isCameraRun:
                    # print('runing loop')
                    [frame, dims] = self.camera.hamamatsu.getLastFrame()
                    self.np_data = frame.getData()
                    self.image = np.reshape(self.np_data, (self.eff_subarrayv, self.eff_subarrayh))

                    if self.settings['record']:
                        self.camera.hamamatsu.stopAcquisition()
                        self.camera.hamamatsu.startRecording()
                        self.camera.hamamatsu.stopRecording()
                        self.cameraInterrupt()

                    if self.settings['save_h5']:

                        if save:
                            self.initH5()
                            save = False  # at next cycle, we don't do initH5 again (we have already created the file)

                        mean_value = np.mean(self.np_data)
                        last_frame_index = self.camera.hamamatsu.buffer_index
                        # print(self.camera.hamamatsu.last_frame_number)
                        if self.debug:
                            print("The mean is: ", mean_value)

                        if mean_value > self.settings['threshold']:

                            print("\n \n ******* \n \n Saving :D !\n \n *******")
                            j = 0
                            # starting_index=last_frame_index
                            stalking_number = 0
                            remaining = False
                            while j < self.camera.number_frames.val:

                                self.get_and_save_Frame(j, last_frame_index)
                                last_frame_index = self.updateIndex(last_frame_index)

                                if self.debug:
                                    print("The last_frame_index is: ", last_frame_index)

                                j += 1

                                if not remaining:
                                    upgraded_last_frame_index = self.camera.hamamatsu.getTransferInfo()[
                                        0]  # upgrades the transfer information
                                    # The stalking_number represents the relative steps the camera has made in acquisition with respect to the saving.
                                    stalking_number = stalking_number + self.camera.hamamatsu.backlog - 1

                                    if self.debug:
                                        print('upgraded_last_frame_index: ', upgraded_last_frame_index)
                                        print('stalking_number: ', stalking_number)
                                        print('The camera is at {} passes from you'.format(
                                            self.camera.hamamatsu.number_image_buffers - stalking_number))

                                    if stalking_number + self.camera.hamamatsu.backlog > self.camera.hamamatsu.number_image_buffers:
                                        self.camera.hamamatsu.stopAcquisitionNotReleasing()  # stop acquisition when we know that at next iteration, some images may be rewritten
                                        remaining = True  # if the buffer reach us, we execute the "while" without the "if not remaining" block.

                            self.cameraInterrupt()
                            self.camera.hamamatsu.stopAcquisition()
                            if self.debug:
                                print("The last_frame_number is: ", self.camera.hamamatsu.last_frame_number)

        finally:
            self.camera.hamamatsu.stopAcquisition()

    def cameraInterrupt(self):
        self.isCameraRun = False

    def cameraStart(self):
        self.isCameraRun = True
