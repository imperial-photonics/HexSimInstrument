import os
import time
from datetime import datetime
from threading import Thread, currentThread, Event
from multiprocessing import Process

import numpy as np
import pyqtgraph as pg
import tifffile as tif
from ScopeFoundry import Measurement
from ScopeFoundry import h5_io
from ScopeFoundry.helper_funcs import sibling_path, load_qt_ui_file
from hexSimProcessor import *

from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtCore import QTimer

class HexSimMeasurement(Measurement):
    name = 'hexsim_measurement'

    def setup(self):

        # load ui file
        self.ui_filename = sibling_path(__file__, "hexsim_measurement.ui")
        self.ui = load_qt_ui_file(self.ui_filename)

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
        self.stage = self.app.hardware['NanoScanHardware']

        self.autoRange = self.settings.autoRange.val
        self.display_update_period = self.settings.refresh_period.val
        self.autoLevels = self.settings.autoLevels.val
        self.level_min = self.settings.level_min.val
        self.level_max = self.settings.level_max.val

        self.isStreamRun = False
        self.isUpdateImageViewer = False
        self.isCameraRun = False
        self.isStandardProcessing = False
        self.isBatchProcessing = False
        self.standardProcessEvent = Event()
        self.standardProcessEvent.clear()
        self.batchProcessEvent = Event()
        self.batchProcessEvent.clear()

    def setup_figure(self):
        # connect ui widgets to measurement/hardware settings or functionss

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

        # Camera
        self.ui.camButton.clicked.connect(self.camButtonPressed)

        # Screen
        self.ui.slmButton.clicked.connect(self.slmButtonPressed)
        self.ui.previousPatternButton.clicked.connect(self.screen.previousPattern)
        self.ui.nextPatternButton.clicked.connect(self.screen.nextPattern)

        # Stage
        self.ui.stagePositionIncrease.clicked.connect(self.stage.moveUpHW)
        self.ui.stagePositionDecrease.clicked.connect(self.stage.moveDownHW)

        # Measure
        self.ui.captureStandardButton.clicked.connect(self.standardAcquisition)

        self.ui.startStreamingButton.clicked.connect(self.streamAcquisitionTimer)
        self.ui.stopStreamingButton.clicked.connect(self.streamAcquisitionTimerStop)

        self.ui.captureBatchButton.clicked.connect(self.batchAcquisition)

        # region Reconstructor settings
        self.ui.debugCheck.stateChanged.connect(self.setReconstructor)
        self.ui.cleanupCheck.stateChanged.connect(self.setReconstructor)
        self.ui.gpuCheck.stateChanged.connect(self.setReconstructor)
        self.ui.axialCheck.stateChanged.connect(self.setReconstructor)
        self.ui.usemodulationCheck.stateChanged.connect(self.setReconstructor)
        self.ui.compactCheck.stateChanged.connect(self.setReconstructor)

        self.ui.magnificationValue.valueChanged.connect(self.setReconstructor)
        self.ui.naValue.valueChanged.connect(self.setReconstructor)
        self.ui.nValue.valueChanged.connect(self.setReconstructor)
        self.ui.wavelengthValue.valueChanged.connect(self.setReconstructor)
        self.ui.pixelsizeValue.valueChanged.connect(self.setReconstructor)

        self.ui.alphaValue.valueChanged.connect(self.setReconstructor)
        self.ui.betaValue.valueChanged.connect(self.setReconstructor)
        self.ui.wValue.valueChanged.connect(self.setReconstructor)
        self.ui.etaValue.valueChanged.connect(self.setReconstructor)
        # endregion

        # Display
        self.ui.rawImageSlider.valueChanged.connect(self.rawImageSliderChanged)
        self.ui.simImageSlider.valueChanged.connect(self.simImageSliderChanged)

        # Test
        self.ui.snapshotButton.clicked.connect(self.snapshotButtonPressed)
        self.ui.snapshotSeqButton.clicked.connect(self.snapshotSeqButtonPressed)

        self.ui.standardSimuButton.clicked.connect(self.standardSimuButtonPressed)
        self.ui.standardSimuUpdate.clicked.connect(self.standardReconstructionUpdate)

        self.ui.streamSimuButton.clicked.connect(self.streamSimuButtonPressed)
        self.ui.streamSimuStop.clicked.connect(self.streamStopPressed)

        self.ui.batchSimuButton.clicked.connect(self.batchSimuButtonPressed)
        self.ui.batchSimuUpdate.clicked.connect(self.batchReconstructionUpdate)

    def update_display(self):
        """
        Displays the numpy array called self.image.
        This function runs repeatedly and automatically during the measurement run,
        its update frequency is defined by self.display_update_period.
        """
        # # update stage position
        # self.ui.stagePositionDisplay.display(self.stage.getPositionABS())

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
        if self.isStreamRun or self.isUpdateImageViewer:
            self.updateImageViewer()
            self.isUpdateImageViewer = False

        # print('Display')

    def run(self):

        self.eff_subarrayh = int(self.camera.subarrayh.val / self.camera.binning.val)
        self.eff_subarrayv = int(self.camera.subarrayv.val / self.camera.binning.val)

        self.image = np.zeros((self.eff_subarrayv, self.eff_subarrayh), dtype=np.uint16)
        self.image[0, 0] = 1  # Otherwise we get the "all zero pixels" error (we should modify pyqtgraph...)

        if not hasattr(self,'h'):
            # create reconstructor
            self.h = hexSimProcessor()
            # Initialize the raw image array
            self.imageRaw = np.zeros((7, self.eff_subarrayv, self.eff_subarrayh), dtype=np.uint16)
            self.setReconstructor()
            self.h.N = self.eff_subarrayh

        if not hasattr(self,'standardProcessThread'):
            self.standardProcessThread = Thread(target=self.standardProcessor)
            self.standardProcessThread.start()

        if not hasattr(self,'batchProcessThread'):
            self.batchProcessThread = Thread(target=self.batchProcessor)
            self.batchProcessThread.start()

        while not self.interrupt_measurement_called:
            if self.isCameraRun:
                self.cameraRunTest()

################   Control  ################
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

    # region Display Functions
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
    # endregion

################    HexSIM  ################
    # Standard reconstruction
    def standardAcquisition(self):
        self.standardMeasureThread = Thread(target=self.standardMeasure)
        self.standardMeasureThread.start()

    def standardMeasure(self):
        # Stop the camera
        self.cameraInterrupt()

        # print(self.camera.hamamatsu.isCapturing())
        self.interrupt()
        # Initialize the raw image array
        self.imageRaw = np.zeros((7, self.eff_subarrayv, self.eff_subarrayh), dtype=np.uint16)

        # Project the patterns and acquire raw images
        for i in range(7):
            self.screen.slm_dev.displayFrameN(i)
            time.sleep(self.getAcquisitionInterval()/1000.0)
            start_time = time.time()
            self.imageRaw[i, :, :] = self.getOneFrame()
            print('Time of acquisition:',time.time()-start_time)
            print('Capture frame:', i)

        self.screen.slm_dev.displayFrameN(0)
        # Recover camera streaming
        self.camera.updateCameraSettings()
        # Restart camera
        self.cameraStart()
        self.start()
        # Reconstruct the SIM image
        # self.standardReconstruction()
        self.standardProcessEvent.set()

    def standardProcessor(self):
        cThread = currentThread()
        while getattr(cThread,"isThreadRun",True):
            self.standardProcessEvent.wait()
            print('Start standard processing...')
            startTime = time.time()
            if self.h.gpuenable:
                self.h.calibrate_fast(self.imageRaw)
                self.imageSIM = self.h.reconstruct_cupy(self.imageRaw).get()

            elif not self.h.gpuenable:
                self.h.calibrate(self.imageRaw)
                self.imageSIM = self.h.reconstruct_rfftw(self.imageRaw)

            self.imageSIM = self.imageSIM[np.newaxis, :, :]
            self.isUpdateImageViewer = True
            # self.isStandardProcessing = False
            print('One SIM image is processed in:', time.time()-startTime,'s')
            self.standardProcessEvent.clear()

    def batchProcessor(self):
        cThread = currentThread()
        while getattr(cThread,"isThreadRun",True):
            self.batchProcessEvent.wait()
            print('Start batch processing...')
            startTime = time.time()
            nStack = len(self.imageRaw)
            # calibrate & reconstruction
            if self.h.gpuenable:
                self.h.calibrate_fast(self.imageRaw[int(nStack // 2):int(nStack // 2 + 7), :, :])
                if self.h.compact:
                    self.imageSIM = self.h.batchreconstructcompact_cupy(self.imageRaw).get()
                elif not self.h.compact:
                    self.imageSIM = self.h.batchreconstruct_cupy(self.imageRaw).get()

            elif not self.h.gpuenable:
                self.h.calibrate(self.imageRaw[int(nStack // 2):int(nStack // 2 + 7), :, :])
                if self.h.compact:
                    self.imageSIM = self.h.batchreconstructcompact(self.imageRaw)
                elif not self.h.compact:
                    self.imageSIM = self.h.batchreconstruct(self.imageRaw)

            ## update display
            self.isUpdateImageViewer = True
            # self.isBatchProcessing = False
            print('Batch reconstruction finished', time.time()-startTime,'s')
            self.batchProcessEvent.clear()

    # region Streaming reconstruction
    def streamAcquisition(self):
        self.streamMeasureThread = Thread(target=self.streamMeasure)
        self.streamMeasureThread.start()

    def streamAcquisitionTimer(self):
        # Initialization
        self.isStreamRun = True
        self.streamIndex = 0
        self.imageRaw = np.zeros((7, self.eff_subarrayv, self.eff_subarrayh), dtype=np.uint16)
        # Calibration should be done by standard measurement first.
        self.cameraInterrupt()
        self.interrupt()
        self.streamTimer = QTimer(self)
        self.streamTimer.timeout.connect(self.streamMeasureTimer)
        self.streamTimer.start(self.getAcquisitionInterval())

    def streamAcquisitionTimerStop(self):
        self.streamTimer.stop()
        self.isStreamRun = True
        # Recover camera streaming
        self.camera.updateCameraSettings()
        self.cameraStart()
        self.start()


    # def streamMeasure(self):
    #     # Initialization
    #     self.isStreamRun = True
    #     self.streamIndex = 0
    #     self.imageRaw = np.zeros((7, self.eff_subarrayv, self.eff_subarrayh), dtype=np.uint16)
    #
    #     # Calibration
    #     self.cameraInterrupt()
    #     self.interrupt()
    #
    #     # Project the patterns and acquire raw images
    #     for i in range(7):
    #         self.screen.slm_dev.displayFrameN(i)
    #         time.sleep(self.getAcquisitionInterval()/1000.0)
    #         self.imageRaw[i, :, :] = self.getOneFrame()
    #         print('Capture frame:', i)
    #         self.streamIndex +=1
    #
    #     # Reconstruct the SIM image
    #     self.standardReconstruction()
    #
    #     while self.isStreamRun:
    #         self.streamIndex +=1
    #         self.screen.slm_dev.displayFrameN(self.streamIndex % 7)
    #         time.sleep(self.getAcquisitionInterval() / 1000.0)
    #         newFrame = self.getOneFrame()
    #         self.streamReconstruction(newFrame[0,:,:], self.streamIndex % 7)
    #         self.updateImageViewer()
    #         print(self.streamIndex)
    #
    #     # Recover camera streaming
    #     self.camera.updateCameraSettings()
    #     self.cameraStart()
    #     self.start()

    def streamMeasureTimer(self):
        print(self.streamIndex)
        self.screen.slm_dev.displayFrameN((self.streamIndex) % 7)
        newFrame = self.getOneFrame()
        self.imageRaw[(self.streamIndex % 7),:,:] = newFrame
        self.streamReconstruction(newFrame[0,:,:], (self.streamIndex % 7))
        self.updateImageViewer()
        self.streamIndex +=1

    # endregion

    # region Batch reconstruction
    def batchAcquisition(self):
        self.batchMeasureThread = Thread(target=self.batchMeasure)
        self.batchMeasureThread.start()
        # self.batchMeasureThread.join()

    def batchMeasure(self):
        # Stop the camera
        self.cameraInterrupt()
        self.interrupt()

        # Initialize the raw image array
        n_stack = self.ui.nStack.value()
        self.imageRaw = np.zeros((n_stack, self.eff_subarrayv, self.eff_subarrayh), dtype=np.uint16)

        # Project the patterns and acquire raw images
        for i in range(n_stack):
            self.screen.slm_dev.displayFrameN(i % 7)
            time.sleep(self.getAcquisitionInterval() / 1000.0)
            self.imageRaw[i, :, :] = self.getOneFrame()
            print('Capture frame:', i)

        # Reconstruct the SIM image
        self.batchProcessEvent.set()
        # self.batchReconstruction()
        # Recover camera streaming
        self.camera.updateCameraSettings()
        # Restart camera
        self.cameraStart()
        self.start()
    # endregion

############## Test #################################
    def standardSimuButtonPressed(self):
        # read data
        # filename = "./data/standardData.tif"
        filename, _ = QFileDialog.getOpenFileName(directory="./data")
        self.imageRaw = np.single(tif.imread(filename))
        self.standardReconstructionUpdate()

    def standardReconstructionUpdate(self):
        self.standardProcessEvent.set()
        # self.isStandardProcessing = True
        # self.standardReconstruction()
        # try:
        #     self.standardSimuThread.join()
        #     self.standardSimuThread = Thread(target=self.standardReconstruction)
        #     self.standardSimuThread.start()
        # except:
        #     self.standardSimuThread = Thread(target=self.standardReconstruction)
        #     self.standardSimuThread.start()

        # self.standardSimuThread.join()

    def standardReconstruction(self):
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
        self.isStreamRun = True
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
        while self.isStreamRun and self.streamIndex<=280:
            self.imageRawFrame = self.imageRawStack[self.streamIndex,:,:]

            self.streamReconstruction(self.imageRawFrame,(self.streamIndex) % 7)
            time.sleep(0.5)
            self.streamIndex +=1
            print(self.streamIndex)

    def streamReconstruction(self,newFrame,index):
        if self.h.gpuenable:
            self.imageSIM = (self.h.reconstructframe_cupy(newFrame, index)).get()
        elif not self.h.gpuenable:
            # print(newFrame.shape)
            self.imageSIM = self.h.reconstructframe_rfftw(newFrame, index)

        self.imageSIM = self.imageSIM[np.newaxis, :, :]

    def streamStopPressed(self):
        self.isStreamRun = False

    def batchSimuButtonPressed(self):
        # filename = "./data/stackData.tif"
        filename, _ = QFileDialog.getOpenFileName(directory="./data")
        self.imageRaw = np.single(tif.imread(filename))
        # Nsize = 256
        # self.imageRaw = np.single(self.imageRaw[:, 256 - Nsize // 2: 256 + Nsize // 2, 256 - Nsize // 2: 256 + Nsize // 2])
        # self.h.N = 256
        # self.h._allocate_arrays()
        self.batchReconstructionUpdate()
        # print('start batch processing')
        # print(self.imageRaw.shape)

    def batchReconstructionUpdate(self):
        self.batchProcessEvent.set()
        # self.isBatchProcessing = True
        # self.batchSimuThread = Thread(target=self.batchReconstruction)
        # self.batchSimuThread.start()
        # self.batchSimuThread.join()
        # self.updateImageViewer()
        # print('One SIM stack is processed.')

    def batchReconstruction(self):
        # create reconstruction object
        self.setReconstructor()
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

    def virtualRecording(self):
        filename, _ = QFileDialog.getOpenFileName(directory="./data")
        self.imageRawStack = np.single(tif.imread(filename))

    def snapshotButtonPressed(self):
        self.cameraInterrupt()
        # time.sleep(0.1)
        self.imageRaw = self.getOneFrame()
        self.camera.updateCameraSettings()
        # self.start()
        self.cameraStart()

        self.imageRaw = self.imageRaw[np.newaxis, :, :]
        self.isUpdateImageViewer = True

    def snapshotSeqButtonPressed(self):
        self.standardSnapshotsThread = Thread(target=self.standardSnapshots)
        self.standardSnapshotsThread.start()

    ############ Methods ############
    ## Acqusition
    def standardSnapshots(self):
        # Stop the camera
        self.cameraInterrupt()
        self.interrupt()

        # initialize the raw image array
        self.imageRaw = np.zeros((7, self.eff_subarrayv, self.eff_subarrayh), dtype=np.uint16)
        # project the patterns and acquire raw images
        for i in range(7):
            self.screen.slm_dev.displayFrameN(i)
            time.sleep(self.getAcquisitionInterval()/1000.0)
            # time.sleep(0.1)
            self.imageRaw[i, :, :] = self.getOneFrame()
            print('Capture frame:', i)

        # Recover camera streaming
        self.camera.updateCameraSettings()
        # Restart camera
        self.cameraStart()
        self.start()
        self.isUpdateImageViewer = True

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

    # <editor-fold desc="Settings">
    def setReconstructor(self):
        self.h.debug = self.ui.debugCheck.isChecked()
        self.h.cleanup = self.ui.cleanupCheck.isChecked()
        self.h.gpuenable = self.ui.gpuCheck.isChecked()
        self.h.axial = self.ui.axialCheck.isChecked()
        self.h.usemodulation = self.ui.usemodulationCheck.isChecked()
        self.h.compact = self.ui.compactCheck.isChecked()

        # self.h.N = len(imgStack[0,:,:])
        self.h.magnification =self.ui.magnificationValue.value()
        self.h.NA = self.ui.naValue.value()
        self.h.n = self.ui.nValue.value()
        self.h.wavelength = self.ui.wavelengthValue.value()
        self.h.pixelsize = self.ui.pixelsizeValue.value()

        self.h.alpha = self.ui.alphaValue.value()
        self.h.beta = self.ui.betaValue.value()
        self.h.w = self.ui.wValue.value()
        self.h.eta = self.ui.etaValue.value()

    def setMagnification(self):
        self.h.magnification =self.ui.magnificationValue.value()
    def setDebug(self):
        self.h.debug = self.ui.debugCheck.isChecked()

    def setCleanup(self):
        self.h.cleanup = self.ui.cleanupCheck.isChecked()

    def setGpuEnable(self):
        self.h.gpuenable = self.ui.gpuCheck.isChecked()

    def setAxial(self):
        self.h.axial = self.ui.axialCheck.isChecked()

    def setUseModulation(self):
        self.h.usemodulation = self.ui.usemodulationCheck.isChecked()

    def setCompact(self):
        self.h.compact = self.ui.compactCheck.isChecked()

    def getAcquisitionInterval(self):
        return float(self.ui.intervalTime.value())

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

    # </editor-fold>


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

                        if not self.isCameraRun and self.interrupt_measurement_called:
                            break
                        index += 1
                        print(index)

                    if not self.isCameraRun and self.interrupt_measurement_called:
                        break
                        # index = index + len(frames)
                    # np_data.tofile(bin_fp)
                    self.settings['progress'] = index * 100. / self.camera.hamamatsu.number_image_buffers

            elif self.camera.acquisition_mode.val == "run_till_abort":

                save = True

                while self.isCameraRun and not self.interrupt_measurement_called:
                    try:
                        [frame, dims] = self.camera.hamamatsu.getLastFrame()
                        self.np_data = frame.getData()

                    except:
                        self.np_data = np.zeros((self.eff_subarrayv, self.eff_subarrayh))
                        print('Camera read data fail.')

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
        # self.camera.hamamatsu.stopAcquisition()
        self.isCameraRun = False
        time.sleep(0.2)

    def cameraStart(self):
        self.isCameraRun = True

