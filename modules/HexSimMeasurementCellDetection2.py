import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pyqtgraph as pg
import tifffile as tif
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QFileDialog
from ScopeFoundry import Measurement
from ScopeFoundry.helper_funcs import load_qt_ui_file
from qtwidgets import Toggle

from HexSimProcessor.SIM_processing.hexSimProcessor import HexSimProcessor
from utils.MessageWindow import CalibrationResults
from utils.StackImageViewer import StackImageViewer
from utils.ImageSegmentation import ImageSegmentation

def add_timer(function):
    """Function decorator to mesaure the execution time of a method.
    To avoid conflicts with QtObjects, it assumes that the method takes no arguments except self
    """
    def inner(cls):
        print(f'\nStarting method "{function.__name__}" ...')
        start_time = time.time()
        result = function(cls)
        end_time = time.time()
        print(f'Execution time for method "{function.__name__}": {end_time - start_time:.6f} s')
        return result

    inner.__name__ = function.__name__
    return inner


def add_update_display(function):
    """Function decorator to to update display at the end of the execution
    To avoid conflicts with QtObjects, it assumes that the method takes no arguments except self
    """
    def inner(cls):
        result = function(cls)
        cls.update_display()
        return result

    inner.__name__ = function.__name__
    return inner

class HexSimMeasurementCellDetection(Measurement):
    name = 'HexSIM_Measurement_Cell_Detection'

    def setup(self):
        # load ui file
        self.ui = load_qt_ui_file(".\\ui\\hexsim_measurement_cell_detection.ui")
        # Connect to hardware components
        self.camera = self.app.hardware['HamamatsuHardware']
        self.screen = self.app.hardware['ScreenHardware']
        self.stage = self.app.hardware['NanoScanHardware']
        self.laser488 = self.app.hardware['Laser488Hardware']
        self.laser561 = self.app.hardware['Laser561Hardware']
        # Measurement component settings
        self.settings.New('refresh_period', dtype=float, unit='s', spinbox_decimals=4, initial=0.02, hardware_set_func=self.setRefresh, vmin=0)
        self.display_update_period = self.settings.refresh_period.val
        # initialize condition labels
        self.isStreamRun = False
        self.isCameraRun = False
        self.isSnapshot = False
        self.isUpdateImageViewer = False
        self.isCalibrated = False
        self.isGpuenable = True  # using GPU for accelerating
        self.isCompact = True  # using compact mode in batch reconstruction to save memory
        self.isFindCarrier = True

        self.action = None

        self.kx_full = np.zeros((3, 1), dtype=np.single) # frequency of full field of view
        self.ky_full = np.zeros((3, 1), dtype=np.single)

        # Image initialization
        self.eff_subarrayh = int(self.camera.subarrayh.val / self.camera.binning.val)
        self.eff_subarrayv = int(self.camera.subarrayv.val / self.camera.binning.val)
        v = self.eff_subarrayv
        h = self.eff_subarrayh
        # Left
        self.imageCAM = np.zeros((v, h), dtype=np.uint16)
        self.imageRaw = np.zeros((7, v, h), dtype=np.uint16)
        self.imageWF = np.zeros((v, h), dtype=np.uint16)
        self.imageWF_ROI = np.zeros((v, h), dtype=np.uint16)
        # Right
        self.imageSIM = np.zeros((2 * v, 2 * h), dtype=np.uint16)
        self.imageSIM_ROI = np.zeros((2 * v, 2 * h), dtype=np.uint16)    # it can be an image or a set of images
        self.wiener_Full = np.zeros((v, h), dtype=np.uint16)
        self.wiener_ROI = np.zeros((v, h), dtype=np.uint16)              # it can be an image or a set of images

        if not hasattr(self, 'h'):
            self.h = HexSimProcessor()  # create reconstruction object
            self.h.opencv =False
            self.imageRaw = np.zeros((7, self.eff_subarrayv, self.eff_subarrayh),dtype=np.uint16)  # Initialize the raw image array
            self.setReconstructor()
            self.h.N = self.eff_subarrayh
            self.h.wienerfilter_store = self.wiener_Full
            self.h.kx_input = np.zeros((3, 1), dtype=np.single)
            self.h.ky_input = np.zeros((3, 1), dtype=np.single)

    def pre_run(self):
        pass

    def setup_figure(self):
        self.ui.imgTab.setCurrentIndex(0)
        # Camera UI
        self.imv = pg.ImageView()
        self.imv.ui.roiBtn.hide()
        self.imv.ui.menuBtn.hide()

        # Image viewers
        self.imvRaw     = StackImageViewer(image_sets=self.imageRaw,set_levels=[1,1])
        self.imvWF      = StackImageViewer(image_sets=self.imageWF,set_levels=[1,1])
        self.imvWF_ROI  = StackImageViewer(image_sets=self.imageWF_ROI,set_levels=[1,1])
        self.imvSIM     = StackImageViewer(image_sets=self.imageSIM, set_levels=[0, 0.8])
        self.imvSIM_ROI = StackImageViewer(image_sets=self.imageSIM_ROI, set_levels=[0, 0.8])

        self.imvCalibration = CalibrationResults(self.h)

        self.imvWiener_ROI = StackImageViewer(image_sets=self.wiener_ROI,set_levels=[1,1])
        self.imvWiener_ROI.imv.ui.histogram.hide()

        # combo lists setting: size of roi
        self.roiRect = []  # list of roi rectangular
        self.roiSizeList = [128,256,512,1024]
        self.ui.roiSizeCombo.addItems(map(str,self.roiSizeList))
        self.ui.roiSizeCombo.setCurrentIndex(1)
        # connect ui widgets to measurement/hardware settings or functions
        self.ui.imgStreamLayout.addWidget(self.imv)
        self.ui.rawImageLayout.addWidget(self.imvRaw)
        self.ui.wfImageLayout.addWidget(self.imvWF)
        self.ui.roiWFLayout.addWidget(self.imvWF_ROI)
        self.ui.simImageLayout.addWidget(self.imvSIM)
        self.ui.roiSIMLayout.addWidget(self.imvSIM_ROI)
        self.ui.calibrationResultLayout.addWidget(self.imvCalibration)
        self.ui.roiWienerfilterLayout.addWidget(self.imvWiener_ROI)
        # Camera
        self.ui.switchCAM = Toggle()
        self.ui.cameraToggleLayout.addWidget(self.ui.switchCAM)
        self.ui.switchCAM.stateChanged.connect(self.controlCAM)
        self.ui.snapshot.clicked.connect(self.snapshotPressed)
        # Screen
        self.ui.slmSlider.valueChanged.connect(self.controlSLM)
        self.ui.previousPatternButton.clicked.connect(self.previousPattern)
        self.ui.nextPatternButton.clicked.connect(self.nextPattern)
        # Stage
        self.ui.stagePositionIncrease.clicked.connect(self.stage.moveUpHW)
        self.ui.stagePositionDecrease.clicked.connect(self.stage.moveDownHW)
        # Reconstructor settings
        self.ui.debugCheck.stateChanged.connect(self.setReconstructor)
        self.ui.cleanupCheck.stateChanged.connect(self.setReconstructor)
        self.ui.gpuCheck.stateChanged.connect(self.setReconstructor)
        self.ui.axialCheck.stateChanged.connect(self.setReconstructor)
        self.ui.usemodulationCheck.stateChanged.connect(self.setReconstructor)
        self.ui.compactCheck.stateChanged.connect(self.setReconstructor)
        self.ui.usePrecalibration.stateChanged.connect(self.setReconstructor)

        self.ui.magnificationValue.valueChanged.connect(self.setReconstructor)
        self.ui.naValue.valueChanged.connect(self.setReconstructor)
        self.ui.nValue.valueChanged.connect(self.setReconstructor)
        self.ui.wavelengthValue.valueChanged.connect(self.setReconstructor)
        self.ui.pixelsizeValue.valueChanged.connect(self.setReconstructor)
        self.ui.alphaValue.valueChanged.connect(self.setReconstructor)
        self.ui.betaValue.valueChanged.connect(self.setReconstructor)
        self.ui.wValue.valueChanged.connect(self.setReconstructor)
        self.ui.etaValue.valueChanged.connect(self.setReconstructor)

        # Measure
        self.ui.captureStandardButton.clicked.connect(self.standardCapturePressed)
        self.ui.captureBatchButton.clicked.connect(self.batchCapturePressed)
        self.ui.startStreamingButton.clicked.connect(self.streamAcquisitionTimer)
        self.ui.stopStreamingButton.clicked.connect(self.streamAcquisitionTimerStop)

        # Operations
        self.ui.calibrationButton.clicked.connect(self.calibrationPressed)
        self.ui.loadCalibrationButton.clicked.connect(self.loadCalibrationResults)
        self.ui.resetButton.clicked.connect(self.resetHexSIM)
        self.ui.findCellButton.clicked.connect(self.findCell)
        self.ui.reconstructionButton.clicked.connect(self.reconstructionPressed)
        self.ui.roiProcessButton.clicked.connect(self.roiprocessPressed)
        self.ui.saveButton.clicked.connect(self.saveMeasurements)

    def update_display(self):
        # update stage position
        try:
            self.ui.stagePositionDisplay.display(self.stage.settings.absolute_position.val)
        except:
            pass
        # update camera viewer
        if self.isStreamRun or self.isCameraRun:
            self.imv.setImage(self.imageCAM.T, autoLevels = True, autoRange = True)
        else:
            pass
        # update hexsim viwer
        if self.isStreamRun or self.isUpdateImageViewer:
            self.updateImageViewer()
            self.isUpdateImageViewer = False
        else:
            pass

        if hasattr(self.screen, 'slm_dev'):
            self.ui.patternNumber.setText(str(self.screen.slm_dev.counter%7))

        if self.isCalibrated:
            self.ui.calibrationProgress.setValue(100)
            self.ui.calibrationProgress.setFormat('Calibrated')
        else:
            self.ui.calibrationProgress.setValue(0)
            self.ui.calibrationProgress.setFormat('Uncalibrated')

    def pre_run(self):
        if hasattr(self,'screen'):
            self.controlSLM()

    def run(self):

        while not self.interrupt_measurement_called:

            if self.isCameraRun:
                self.camera.updateCameraSettings()
                self.eff_subarrayh = int(self.camera.subarrayh.val / self.camera.binning.val)
                self.eff_subarrayv = int(self.camera.subarrayv.val / self.camera.binning.val)
                self.cameraRun()

            if self.action is not None:
                if self.action == 'standard_capture':
                    self.standardCapture()
                    if self.ui.autoCalibration.isChecked():
                        self.calibration()

                elif self.action == 'batch_capture':
                    self.batchCapture()
                    if self.ui.autoCalibration.isChecked():
                        self.calibration()

                elif self.action == 'calibration':
                    self.calibration()
                    self.ui.simTab.setCurrentIndex(2)

                elif self.action == 'standard_process':
                    self.standardReconstruction()
                    self.ui.simTab.setCurrentIndex(0)

                elif self.action == 'standard_process_roi':
                    self.standardReconstructionROI()
                    self.ui.simTab.setCurrentIndex(1)

                elif self.action == 'batch_process':
                    self.batchReconstruction()
                    self.ui.simTab.setCurrentIndex(0)

                elif self.action == 'batch_process_roi':
                    self.batchReconstructionROI()
                    self.ui.simTab.setCurrentIndex(1)

                self.isUpdateImageViewer = True
                self.action = None
                self.controlCAM()
                self.ui.imgTab.setCurrentIndex(2)

            else:
                pass

            time.sleep(0.01)

    def post_run(self):
        if hasattr(self,'camera'):
            self.cameraInterrupt()

################   Control  ################
    def controlCAM(self):
        try:
            if self.ui.switchCAM.isChecked():
                self.isCameraRun = True
            else:
                self.isCameraRun = False
        except Exception as e:
            txtDisplay = f'Camera encountered an error \n{e}'
            self.show_text(txtDisplay)

    def controlSLM(self):
        try:
            if self.ui.slmSlider.value() == 0:
                self.screen.closeSLM()
            elif self.ui.slmSlider.value() == 1:
                self.screen.slm_dev.setPatterns(0.488)
                self.screen.openSLM()
                self.screen.manualDisplay()
                self.ui.wavelengthValue.setValue(0.488)
            elif self.ui.slmSlider.value() == 2:
                self.screen.slm_dev.setPatterns(0.561)
                self.screen.openSLM()
                self.screen.manualDisplay()
                self.ui.wavelengthValue.setValue(0.561)
        except Exception as e:
            txtDisplay = f'SLM encountered an error \n{e}'
            self.show_text(txtDisplay)

    @add_update_display
    def previousPattern(self):
        self.screen.previousPattern()

    @add_update_display
    def nextPattern(self):
        self.screen.nextPattern()

# Functions for HexSIM
    def resetHexSIM(self):
        if hasattr(self, 'h'):
            self.isCalibrated = False
            self.h._allocate_arrays()
            self.imageSIM = np.zeros((self.eff_subarrayv, self.eff_subarrayh), dtype=np.uint16)
            self.updateImageViewer()

    #  Streaming reconstruction
    def streamAcquisitionTimer(self):
        # Calibration should be done by standard measurement firstly.
        if self.isCalibrated:
            # Initialization
            self.isStreamRun = True
            self.streamIndex = 0
            self.imageRaw = np.zeros((7, self.eff_subarrayv, self.eff_subarrayh), dtype=np.uint16)
            self.cameraInterrupt()
            self.interrupt()
            self.streamTimer = QTimer(self)
            self.streamTimer.timeout.connect(self.streamMeasureTimer)
            self.streamTimer.start(self.getAcquisitionInterval())
        elif not self.isCalibrated:
            print('Run calibration first.')

    def streamAcquisitionTimerStop(self):
        self.streamTimer.stop()
        self.isStreamRun = False
        # Recover camera streaming
        self.camera.updateCameraSettings()
        self.cameraStart()
        self.start()

    def streamMeasureTimer(self):
        print(self.streamIndex)
        self.screen.slm_dev.displayFrameN((self.streamIndex) % 7)
        newFrame = self.getOneFrame()
        self.imageRaw[(self.streamIndex % 7), :, :] = newFrame
        self.streamReconstruction(newFrame[0, :, :], (self.streamIndex % 7))
        self.updateImageViewer()
        self.imv.setImage(newFrame.T, autoLevels=True, autoRange=True)
        self.streamIndex += 1

    def streamReconstruction(self, newFrame, index):
        print(index)
        if self.isGpuenable:
            self.imageSIM = self.h.reconstructframe_cupy(newFrame, index)
        elif not self.isGpuenable:
            self.imageSIM = self.h.reconstructframe_rfftw(newFrame, index)

        self.imageSIM = self.imageSIM[np.newaxis, :, :]

    def streamStopPressed(self):
        self.isStreamRun = False

    def setReconstructor(self):
        self.isCompact = self.ui.compactCheck.isChecked()
        self.isGpuenable = self.ui.gpuCheck.isChecked()
        self.isFindCarrier = not self.ui.usePrecalibration.isChecked()

        self.h.debug = self.ui.debugCheck.isChecked()
        self.h.cleanup = self.ui.cleanupCheck.isChecked()
        self.h.axial = self.ui.axialCheck.isChecked()
        self.h.usemodulation = self.ui.usemodulationCheck.isChecked()
        self.h.magnification = self.ui.magnificationValue.value()
        self.h.NA = self.ui.naValue.value()
        self.h.n = self.ui.nValue.value()
        self.h.wavelength = self.ui.wavelengthValue.value()
        self.h.pixelsize = self.ui.pixelsizeValue.value()

        self.h.alpha = self.ui.alphaValue.value()
        self.h.beta = self.ui.betaValue.value()
        self.h.w = self.ui.wValue.value()
        self.h.eta = self.ui.etaValue.value()

        if not self.isFindCarrier:
            self.h.kx = self.kx_input
            self.h.ky = self.ky_input

    # --------------------------New functions--------------------------------------------------------
    def standardCapturePressed(self):
        if not self.screen.slm_dev.isVisible():
            self.show_text('Open SLM!')
        else:
            self.isCameraRun = False
            self.action = 'standard_capture'

    def batchCapturePressed(self):
        if not self.screen.slm_dev.isVisible():
            self.show_text('Open SLM!')
        else:
            self.isCameraRun = False
            self.action = 'batch_capture'

    def calibrationPressed(self):
        self.isCameraRun = False
        self.action = 'calibration'

    def reconstructionPressed(self):
        self.isCameraRun = False
        if len(self.imageRaw)>7:
            self.action = 'batch_process'
        elif len(self.imageRaw)==7:
            self.action = 'standard_process'
        else:
            self.show_text('Raw images are not acquired.')

    def roiprocessPressed(self):
        self.isCameraRun = False
        if len(self.imageRaw_ROI[0])>7:
            self.action = 'batch_process_roi'
        elif len(self.imageRaw_ROI[0])==7:
            self.action = 'standard_process_roi'
        else:
            self.show_text('ROI raw images are not acquired.')

    def standardCapture(self):
        try:
            # Initialize the raw image array
            self.imageRaw = np.zeros((7, self.eff_subarrayv, self.eff_subarrayh), dtype=np.uint16)
            # Project the patterns and acquire 7 raw images
            for i in range(7):
                self.screen.slm_dev.displayFrameN(i)
                time.sleep(self.getAcquisitionInterval() / 1000.0)
                self.imageRaw[i, :, :] = self.getOneFrame()
                txtDisplay = f'Capture frame: {i+1}'
                self.show_text(txtDisplay)

            self.show_text('Standard capture finished.')
            self.imageWF = self.raw2WideFieldImage(self.imageRaw)

        except Exception as e:
            txtDisplay = f'Standard capture encountered an error \n{e}'
            self.show_text(txtDisplay)

    def batchCapture(self):
        try:
            n_stack = 7*self.ui.nStack.value()      # Initialize the raw image array
            step_size = self.stage.settings.stepsize.val
            stage_offset = n_stack*step_size
            pos = 25-stage_offset/2.0
            self.stage.moveAbsolutePositionHW(pos)

            self.imageRaw = np.zeros((n_stack, self.eff_subarrayv, self.eff_subarrayh), dtype=np.uint16)
            # Project the patterns and acquire raw images
            for i in range(n_stack):
                self.screen.slm_dev.displayFrameN(i % 7)
                pos = pos + step_size
                txtDisplay = f'Capture frame: {i+1} of {n_stack}'
                self.show_text(txtDisplay)
                self.imageRaw[i, :, :] = self.getOneFrame()
                self.stage.moveAbsolutePositionHW(pos)

            self.show_text('Batch capture finished.')
            self.stage.moveAbsolutePositionHW(25)    # Move the stage back to the middle position
            self.imageWF = self.raw2WideFieldImage(self.imageRaw)

        except Exception as e:
            txtDisplay = f'Batch capture encountered an error \n{e}'
            self.show_text(txtDisplay)

    @add_timer
    def calibration(self):
        try:
            self.setReconstructor()
            if self.isGpuenable:
                self.h.calibrate_cupy(self.imageRaw, self.isFindCarrier)
            else:
                self.h.calibrate(self.imageRaw, self.isFindCarrier)

            self.isCalibrated = True
            self.show_text('Calibration finished.')
            self.h.wienerfilter_store = self.h.wienerfilter

        except Exception as e:
            txtDisplay = f'Calibration encountered an error \n{e}'
            self.show_text(txtDisplay)

    @add_timer
    def standardReconstruction(self):
        # standard reconstruction
        try:
            if self.isCalibrated:
                if self.isGpuenable:
                    self.imageSIM = self.h.reconstruct_cupy(self.imageRaw)
                elif not self.isGpuenable:
                    self.imageSIM = self.h.reconstruct_rfftw(self.imageRaw)
            else:
                self.calibration()
                if self.isCalibrated:
                    self.standardReconstruction()

            self.show_text('Standard reconstruction finished.')

        except Exception as e:
            txtDisplay = f'Reconstruction encountered an error \n{e}'
            self.show_text(txtDisplay)

    @add_timer
    def standardReconstructionROI(self):
        try:
            if self.isCalibrated:
                self.imageSIM_ROI = []
                self.wiener_ROI = []
                self.kx_roi = []
                self.ky_roi = []
                self.p_roi = []

                for idx in range(self.numSets):
                    image_raw_roi = self.imageRaw_ROI[idx]
                    self.h.kx = self.kx_full
                    self.h.ky = self.ky_full

                    if self.isGpuenable:
                        self.h.calibrate_cupy(image_raw_roi, findCarrier=False)
                        image_sim_roi = self.h.reconstruct_cupy(image_raw_roi)
                    elif not self.isGpuenable:
                        self.h.calibrate(image_raw_roi, findCarrier=False)
                        image_sim_roi = self.h.reconstruct_rfftw(image_raw_roi)

                    self.imageSIM_ROI.append(image_sim_roi[np.newaxis, :, :])
                    self.kx_roi.append(self.h.kx)
                    self.ky_roi.append(self.h.ky)
                    self.p_roi.append(self.h.p)
                    self.wiener_ROI.append(self.h.wienerfilter[np.newaxis, :, :])

                self.show_text('ROI standard reconstruction finished.')
            else:
                self.show_text('Calibration is needed.')
        except Exception as e:
            txtDisplay = f'ROI Reconstruction encountered an error \n{e}'
            self.show_text(txtDisplay)

    @add_timer
    def batchReconstruction(self):
        try:
            if self.isCalibrated:
                # Batch reconstruction
                if self.isGpuenable:
                    if self.isCompact:
                        self.imageSIM = self.h.batchreconstructcompact_cupy(self.imageRaw)
                    elif not self.isCompact:
                        self.imageSIM = self.h.batchreconstruct_cupy(self.imageRaw)
                elif not self.isGpuenable:
                    if self.isCompact:
                        self.imageSIM = self.h.batchreconstructcompact(self.imageRaw)
                    elif not self.isCompact:
                        self.imageSIM = self.h.batchreconstruct(self.imageRaw)
            elif not self.isCalibrated:
                self.calibration()
                if self.isCalibrated:
                    self.batchReconstruction()

            self.show_text('Batch reconstruction finished.')

        except Exception as e:
            txtDisplay = f'Batch reconstruction encountered an error \n{e}'
            self.show_text(txtDisplay)

    @add_timer
    def batchReconstructionROI(self):
        try:
            if self.isCalibrated:
                self.imageSIM_ROI = []
                self.wiener_ROI = []
                self.kx_roi = []
                self.ky_roi = []
                self.p_roi = []
                # Batch reconstruction
                for idx in range(self.numSets):
                    self.h.kx = self.kx_full
                    self.h.ky = self.ky_full
                    image_raw_roi = self.imageRaw_ROI[idx]

                    if self.isGpuenable:
                        self.h.calibrate_cupy(image_raw_roi, findCarrier=False)
                        if self.isCompact:
                            image_sim_roi = self.h.batchreconstructcompact_cupy(image_raw_roi)
                        elif not self.isCompact:
                            image_sim_roi = self.h.batchreconstruct_cupy(image_raw_roi)

                    elif not self.isGpuenable:
                        self.h.calibrate(image_raw_roi, findCarrier=False)
                        if self.isCompact:
                            image_sim_roi = self.h.batchreconstructcompact(image_raw_roi)
                        elif not self.isCompactompact:
                            image_sim_roi = self.h.batchreconstruct(image_raw_roi)

                    self.imageSIM_ROI.append(image_sim_roi)
                    self.kx_roi.append(self.h.kx)
                    self.ky_roi.append(self.h.ky)
                    self.p_roi.append(self.h.p)
                    self.wiener_ROI.append(self.h.wienerfilter[np.newaxis, :, :])

                self.show_text('ROI batch reconstruction finished.')
            else:
                self.show_text('Calibration is needed.')

        except Exception as e:
            txtDisplay = f'Batch reconstruction encountered an error \n{e}'
            self.show_text(txtDisplay)

    def getAcquisitionInterval(self):
        return float(self.ui.intervalTime.value())

# Functions for camera
    def snapshotPressed(self):
        self.isSnapshot = True

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

    def cameraRun(self):

        try:
            self.camera.read_from_hardware()
            self.camera.hamamatsu.startAcquisition()
            index = 0

            if self.camera.acquisition_mode.val == "fixed_length":
                while index < self.camera.hamamatsu.number_image_buffers:
                    # Get frames.
                    # The camera stops acquiring once the buffer is terminated (in snapshot mode)
                    [frames, dims] = self.camera.hamamatsu.getFrames()
                    # Save frames.
                    for aframe in frames:
                        np_data = aframe.getData()
                        self.imageCAM = np.reshape(np_data, (self.eff_subarrayv, self.eff_subarrayh))
                        if not self.isCameraRun:
                            break
                        index += 1
                        print(index)
                    if not self.isCameraRun or self.interrupt_measurement_called:
                        break

            elif self.camera.acquisition_mode.val == "run_till_abort":
                while self.isCameraRun and not self.interrupt_measurement_called:
                    try:
                        [frame, dims] = self.camera.hamamatsu.getLastFrame()
                        np_data = frame.getData()
                    except:
                        np_data = np.zeros((self.eff_subarrayv, self.eff_subarrayh))
                        self.show_text('Camera read data fail.')

                    self.imageCAM = np.reshape(np_data, (self.eff_subarrayv, self.eff_subarrayh))

                    if self.isSnapshot:
                        tif.imwrite('snapshot.tif', np.uint16(self.imageCAM))
                        self.isSnapshot = False
                        self.show_text('Saved one image.')
                    if not self.isCameraRun or self.interrupt_measurement_called:
                        break
        finally:
            self.camera.hamamatsu.stopAcquisition()

    def cameraInterrupt(self):
        self.isCameraRun = False
        self.ui.switchCAM.setChecked(False)

    def cameraStart(self):
        self.isCameraRun = True
        self.ui.switchCAM.setChecked(True)

# Functions for IO
    def saveMeasurements(self):
        t0 = time.time()
        samplename = self.app.settings['sample']
        timestamp = datetime.fromtimestamp(t0)
        timestamp = timestamp.strftime("%Y_%m%d_%H%M")
        if len(samplename):
            samplename = '_' + samplename
        wavelength = '_'+ str(int(self.h.wavelength*1000))+'nm'
        pathname = './measurement/' + timestamp + samplename + wavelength
        Path(pathname).mkdir(parents=True,exist_ok=True)
        rawimagename = pathname + '/' + timestamp + samplename + wavelength + f'_Raw' + '.tif'
        simimagename = pathname + '/' + timestamp + samplename + wavelength + f'_SIM' + '.tif'
        txtname = pathname + '/'+ timestamp + samplename + wavelength + f'_calibration' + '.txt'

        tif.imwrite(rawimagename, np.uint16(self.imageRaw))
        tif.imwrite(simimagename, np.single(self.imageSIM))

        if self.h.wavelength == 0.488:
            laserpower = np.float(self.laser488.power.val)
        elif self.h.wavelength ==0.561:
            laserpower = np.float(self.laser561.power.val)
        else:
            laserpower = 0

        savedictionary = {
            "exposure time (s)":self.camera.exposure_time.val,
            "laser power (mW)": laserpower,
            # System setup:
            "magnification" :   self.h.magnification,
            "NA":               self.h.NA,
            "refractive index": self.h.n,
            "wavelength":       self.h.wavelength,
            "pixelsize":        self.h.pixelsize,
            # Calibration parameters:
            "alpha":            self.h.alpha,
            "beta":             self.h.beta,
            "Wiener filter":    self.h.w,
            "eta":              self.h.eta,
            "cleanup":          self.h.cleanup,
            "axial":            self.h.axial,
            "modulation":       self.h.usemodulation,
            "kx":               self.h.kx,
            "ky":               self.h.ky,
            "phase":            self.h.p,
            "amplitude":        self.h.ampl
            }
        f = open(txtname, 'w+')
        f.write(json.dumps(savedictionary, cls=NumpyEncoder,indent=2))
        self.isCalibrationSaved = True

    def loadCalibrationResults(self):
        try:
            filename, _ = QFileDialog.getOpenFileName(caption="Open file", directory="./measurement", filter="Text files (*.txt)")
            file = open(filename,'r')
            loadResults = json.loads(file.read())
            self.kx_input = np.asarray(loadResults["kx"])
            self.ky_input = np.asarray(loadResults["ky"])
            self.show_text("Calibration results are loaded.")
        except:
            self.show_text("Calibration results are not loaded.")

# Functions for display
    def setRefresh(self, refresh_period):
        self.display_update_period = refresh_period

    def show_text(self, text):
        self.ui.MessageBox.insertPlainText(text+'\n')
        self.ui.MessageBox.ensureCursorVisible()
        print(text)

    def updateImageViewer(self):
        self.imvRaw.setImageSet(self.imageRaw)
        self.imvWF.setImageSet(self.imageWF)
        self.imvWF_ROI.setImageSet(self.imageWF_ROI)
        self.imvSIM.setImageSet(self.imageSIM)
        self.imvSIM_ROI.setImageSet(self.imageSIM_ROI)
        self.imvWiener_ROI.setImageSet(self.wiener_ROI)
        self.imvCalibration.update(self.h)

# Functions for ROI
    def roiSize(self):
        return int(self.ui.roiSizeCombo.currentText())

    def minCellSize(self):
        return int(self.ui.minCellSizeInput.value())

    def findCell(self):
        self.oSegment = ImageSegmentation(self.imageRaw, self.roiSize() // 2, self.minCellSize())
        markpen = pg.mkPen('r', width=1)
        self.removeMarks()
        self.oSegment.min_cell_size = self.minCellSize()
        self.oSegment.roi_half_side = self.roiSize()//2
        self.oSegment.find_cell()
        self.imageRaw_ROI = self.oSegment.roi_creation()
        self.imageWF_ROI = [] # initialize the image sets
        self.numSets = len(self.imageRaw_ROI)
        self.ui.cellNumber.setValue(self.numSets)

        if self.numSets == 0:
            self.imageWF_ROI = np.zeros((self.roiSize(),self.roiSize()), dtype=np.uint16)
        elif self.numSets > 0:
            for idx in range(self.numSets):
                self.imageWF_ROI.append(self.raw2WideFieldImage(self.imageRaw_ROI[idx]))
                # mark the cells with rectangle overlay
                r = pg.ROI(pos = (self.oSegment.cx[idx]-self.oSegment.roi_half_side,
                                  self.oSegment.cy[idx]-self.oSegment.roi_half_side),
                           size=self.roiSize(), pen=markpen, movable=False)
                self.imvWF.imv.getView().addItem(r)
                self.roiRect.append(r)

        self.isUpdateImageViewer = True
        txtDisplay = f'Found cells: {self.numSets}'
        self.show_text(txtDisplay)

    def removeMarks(self):
        if self.roiRect:
            for item in self.roiRect:
                self.imvWF.imv.getView().removeItem(item)
            self.roiRect = []


    def raw2WideFieldImage(self,rawImages):
        wfImages = np.zeros((rawImages.shape[0]//7,rawImages.shape[1],rawImages.shape[2]))
        for idx in range(rawImages.shape[0]//7):
            wfImages[idx,:,:] = np.sum(rawImages[idx*7:(idx+1)*7,:,:],axis=0)/7

        return wfImages

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)