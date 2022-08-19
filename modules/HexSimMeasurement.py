import os, time, h5py
import numpy as np
import pyqtgraph as pg
import tifffile as tif

from pathlib import Path
from PyQt5.QtWidgets import QFileDialog
from ScopeFoundry import Measurement, h5_io
from ScopeFoundry.helper_funcs import load_qt_ui_file
from qtwidgets import Toggle

from HexSimProcessor.SIM_processing.hexSimProcessor import HexSimProcessor
from utils.MessageWindow import CalibrationResults
from utils.StackImageViewer import StackImageViewer, list_equal
from utils.ImageSegmentation import ImageSegmentation

from PyQt5.QtCore import QTimer
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

class HexSimMeasurement(Measurement):
    name = 'HexSIM_Measure'

    def setup(self):
        # load ui file
        self.ui = load_qt_ui_file(".\\ui\\hexsim_measure.ui")
        # Connect to hardware components
        self.camera = self.app.hardware['HamamatsuHardware']
        self.screen = self.app.hardware['ScreenHardware']
        # self.stage = self.app.hardware['NanoDriveHardware']
        self.z_stage = self.app.hardware['MCLNanoDriveHardware']
        self.laser488 = self.app.hardware['Laser488Hardware']
        self.laser561 = self.app.hardware['Laser561Hardware']
        # Measurement component settings
        self.settings.New('refresh_period', dtype=float, unit='s', spinbox_decimals=4, initial=0.02, hardware_set_func=self.setRefresh, vmin=0)
        self.display_update_period = self.settings.refresh_period.val
        self.settings.New('debug', dtype=bool, initial=False,
                          hardware_set_func = self.setReconstructor)
        self.settings.New('cleanup', dtype=bool, initial=False,
                          hardware_set_func = self.setReconstructor)
        self.settings.New('gpu', dtype=bool, initial=True,
                          hardware_set_func = self.setReconstructor)
        self.settings.New('compact', dtype=bool, initial=True,
                          hardware_set_func = self.setReconstructor)
        self.settings.New('axial', dtype=bool, initial=False,
                          hardware_set_func = self.setReconstructor)
        self.settings.New('usemodulation', dtype=bool, initial=True,
                          hardware_set_func = self.setReconstructor)
        self.settings.New('magnification', dtype=int, initial=60,  spinbox_decimals=2,
                          hardware_set_func = self.setReconstructor)
        self.settings.New('NA', dtype=float, initial=1.10,  spinbox_decimals=2,
                          hardware_set_func = self.setReconstructor)
        self.settings.New('n', dtype=float, initial=1.33,  spinbox_decimals=2,
                          hardware_set_func = self.setReconstructor)
        self.settings.New('wavelength', dtype=float, initial=0.523,  spinbox_decimals=3,
                          hardware_set_func = self.setReconstructor)
        self.settings.New('pixelsize', dtype=float, initial=6.50,  spinbox_decimals=3,
                          hardware_set_func = self.setReconstructor)
        self.settings.New('alpha', dtype=float, initial=0.500,  spinbox_decimals=3, description='0th att width',
                          hardware_set_func = self.setReconstructor)
        self.settings.New('beta', dtype=float, initial=0.990,  spinbox_decimals=3,description='0th width',
                          hardware_set_func = self.setReconstructor)
        self.settings.New('w', dtype=float, initial=0.500, spinbox_decimals=2, description='wiener parameter',
                          hardware_set_func = self.setReconstructor)
        self.settings.New('eta', dtype=float, initial=0.70, spinbox_decimals=2,
                          description='must be smaller than the sources radius normalized on the pupil size',
                          hardware_set_func = self.setReconstructor)
        self.settings.New('otf_model',dtype=str, initial = 'none', choices=["none","exp","sph"],
                          hardware_set_func = self.setReconstructor)
        self.settings.New('find_carrier', dtype=bool, initial=True,
                          hardware_set_func=self.setReconstructor)

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

        self.numSets = 0
        self.kx_full = np.zeros((3, 1), dtype=np.single) # frequency of full field of view
        self.ky_full = np.zeros((3, 1), dtype=np.single)

        # image initialization
        self.eff_subarrayh = int(self.camera.subarrayh.val / self.camera.binning.val)
        self.eff_subarrayv = int(self.camera.subarrayv.val / self.camera.binning.val)
        v = self.eff_subarrayv
        h = self.eff_subarrayh
        # left
        self.imageCAM = np.zeros((v, h), dtype=np.uint16)
        self.imageRAW = [np.zeros((7, v, h), dtype=np.uint16), np.zeros((7, v, h), dtype=np.uint16)]
        self.imageRaw_store = [np.zeros((7, v, h), dtype=np.uint16), np.zeros((7, v, h), dtype=np.uint16)]
        self.imageWF = np.zeros((v, h), dtype=np.uint16)
        self.imageWF_ROI = np.zeros((v, h), dtype=np.uint16)
        # right
        self.imageSIM = np.zeros((2 * v, 2 * h), dtype=np.uint16)
        self.imageSIM_ROI = np.zeros((2 * v, 2 * h), dtype=np.uint16)    # it can be an image or a set of images
        self.wiener_Full = np.zeros((v, h), dtype=np.uint16)
        self.wiener_ROI = np.zeros((v, h), dtype=np.uint16)              # it can be an image or a set of images

        if not hasattr(self, 'h'):
            self.h = HexSimProcessor()  # create reconstruction object
            self.h.opencv =False
            self.setReconstructor()
            self.h.N = self.eff_subarrayh
            self.h.N = self.eff_subarrayh
            self.h.wienerfilter_store = self.wiener_Full
            self.h.kx_input = np.zeros((3, 1), dtype=np.single)
            self.h.ky_input = np.zeros((3, 1), dtype=np.single)

    def setup_figure(self):
        self.ui.imgTab.setCurrentIndex(0)
        # camera UI
        self.imv = pg.ImageView()
        self.imv.ui.roiBtn.hide()
        self.imv.ui.menuBtn.hide()

        # image viewers
        self.imvRaw     = StackImageViewer(image_sets=self.imageRAW[0], set_levels=[1, 1])
        self.imvWF      = StackImageViewer(image_sets=self.imageWF,set_levels=[1,1])
        self.imvWF_ROI  = StackImageViewer(image_sets=self.imageWF_ROI,set_levels=[1,1])
        self.imvSIM     = StackImageViewer(image_sets=self.imageSIM, set_levels=[0, 0.8])
        self.imvSIM_ROI = StackImageViewer(image_sets=self.imageSIM_ROI, set_levels=[0, 0.8])

        self.imvCalibration = CalibrationResults(self.h)
        self.imvWiener_ROI = StackImageViewer(image_sets=self.wiener_ROI,set_levels=[1,1])
        self.imvWiener_ROI.imv.ui.histogram.hide()

        # combo lists setting: size of roi
        self.roiRect = []  # list of roi rectangular
        self.roiSizeList = [128,200,256,512,1024]
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
        # camera
        self.ui.switchCAM = Toggle()
        self.ui.cameraToggleLayout.addWidget(self.ui.switchCAM)
        self.ui.switchCAM.stateChanged.connect(self.controlCAM)
        self.ui.snapshot.clicked.connect(self.snapshotPressed)
        # screen
        self.ui.slmSlider.valueChanged.connect(self.controlSLM)
        self.ui.previousPatternButton.clicked.connect(self.previousPattern)
        self.ui.nextPatternButton.clicked.connect(self.nextPattern)
        # stage
        # self.ui.stagePositionIncrease.clicked.connect(self.stage.singleReadZ)
        # self.ui.stagePositionDecrease.clicked.connect(self.stage.moveDownHW)

        # reconstructor settings
        self.settings.debug.connect_to_widget(self.ui.debugCheck)
        self.settings.find_carrier.connect_to_widget(self.ui.usePrecalibration)
        self.settings.cleanup.connect_to_widget(self.ui.cleanupCheck)
        self.settings.axial.connect_to_widget(self.ui.axialCheck)
        self.settings.usemodulation.connect_to_widget(self.ui.usemodulationCheck)
        self.settings.compact.connect_to_widget(self.ui.compactCheck)
        self.settings.gpu.connect_to_widget(self.ui.gpuCheck)

        self.settings.magnification.connect_to_widget(self.ui.magnificationValue)
        self.settings.NA.connect_to_widget(self.ui.naValue)
        self.settings.n.connect_to_widget(self.ui.nValue)
        self.settings.wavelength.connect_to_widget(self.ui.wavelengthValue)
        self.settings.pixelsize.connect_to_widget(self.ui.pixelsizeValue)

        self.settings.alpha.connect_to_widget(self.ui.alphaValue)
        self.settings.beta.connect_to_widget(self.ui.betaValue)
        self.settings.w.connect_to_widget(self.ui.wValue)
        self.settings.eta.connect_to_widget(self.ui.etaValue)
        self.settings.otf_model.connect_to_widget(self.ui.otfModel)
        self.camera.settings.exposure_time.connect_to_widget(self.ui.exposureTime)

        # Measure
        self.ui.captureStandardButton.clicked.connect(self.standardCapturePressed)
        self.ui.captureBatchButton.clicked.connect(self.batchCapturePressed)
        # self.ui.startStreamingButton.clicked.connect(self.streamAcquisitionTimer)
        # self.ui.stopStreamingButton.clicked.connect(self.streamAcquisitionTimerStop)

        # Operations
        self.ui.calibrationButton.clicked.connect(self.calibrationPressed)
        self.ui.loadCalibrationButton.clicked.connect(self.loadCalibrationResults)
        self.ui.resetButton.clicked.connect(self.resetHexSIM)
        self.ui.findCellButton.clicked.connect(self.findCell)
        self.ui.reconstructionButton.clicked.connect(self.reconstructionPressed)
        self.ui.roiProcessButton.clicked.connect(self.roiprocessPressed)
        self.ui.saveButton.clicked.connect(self.saveMeasurements)

        self.imvRaw.ui.cellCombo.currentIndexChanged.connect(self.channelChanged)

    def update_display(self):
        # update stage position
        try:
            self.ui.stagePositionDisplay.display(f'{self.z_stage.settings.absolute_position.val:.2f}')
            self.ui.cellNumber.display(self.numSets)
        except Exception as e:
            print(e)
            pass

        # update camera viewer
        if self.isStreamRun or self.isCameraRun:
            self.imv.setImage(self.imageCAM.T, autoLevels = True, autoRange = True)
        else:
            pass

        # update hexsim viwer
        if self.isUpdateImageViewer:
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
            time.sleep(0.01)
            if self.isCameraRun:
                self.camera.updateCameraSettings()
                self.eff_subarrayh = int(self.camera.subarrayh.val / self.camera.binning.val)
                self.eff_subarrayv = int(self.camera.subarrayv.val / self.camera.binning.val)
                self.cameraRun()

            if self.action is not None:
                if self.action == 'standard_capture':
                    self.standardCapture()
                    self.resetHexSIM()
                    if self.ui.autoCalibration.isChecked():
                        self.calibration()
                    if self.ui.autoSave.isChecked():
                        self.saveMeasurements()

                elif self.action == 'batch_capture':
                    self.batchCapture()
                    self.resetHexSIM()
                    if self.ui.autoCalibration.isChecked():
                        self.calibration()
                    if self.ui.autoSave.isChecked():
                        self.saveMeasurements()

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
                # self.ui.imgTab.setCurrentIndex(2)

    def post_run(self):
        if hasattr(self,'camera'):
            self.cameraInterrupt()

# functions for hardware
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
                self.settings['wavelength'] = 0.523
            elif self.ui.slmSlider.value() == 2:
                self.screen.slm_dev.setPatterns(0.561)
                self.screen.openSLM()
                self.screen.manualDisplay()
                self.settings['wavelength'] = 0.610
        except Exception as e:
            txtDisplay = f'SLM encountered an error \n{e}'
            self.show_text(txtDisplay)

    @add_update_display
    def previousPattern(self):
        self.screen.previousPattern()

    @add_update_display
    def nextPattern(self):
        self.screen.nextPattern()

    def current_channel_caputure(self):
        return self.ui.slmSlider.value() - 1

    def current_channel_display(self):
        return self.imvRaw.ui.cellCombo.currentIndex()
# functions for HexSIM
    def setReconstructor(self, *args):
        self.isFindCarrier = self.settings['find_carrier']
        self.isGpuenable = self.settings['gpu']
        self.isCompact = self.settings['compact']
        self.h.debug = self.settings['debug']
        self.h.cleanup = self.settings['cleanup']
        self.h.axial = self.settings['axial']
        self.h.usemodulation = self.settings['usemodulation']
        self.h.magnification = self.settings['magnification']
        self.h.NA = self.settings['NA']
        self.h.n = self.settings['n']
        self.h.wavelength = self.settings['wavelength']
        self.h.pixelsize = self.settings['pixelsize']
        self.h.alpha = self.settings['alpha']
        self.h.beta = self.settings['beta']
        self.h.w = self.settings['w']
        self.h.eta = self.settings['eta']
        self.h.a_type = self.settings['otf_model']
        if not self.isFindCarrier:
            try:
                self.h.kx = self.kx_input
                self.h.ky = self.ky_input
            except Exception as e:
                self.show_text(f'Load pre-calibration encountered an error \n{e}')

    # @add_update_display
    def resetHexSIM(self):
        if hasattr(self, 'h'):
            self.isCalibrated = False
            self.numSets = 0
            self.h._allocate_arrays()
            self.removeMarks()
            self.imageWF = np.zeros((self.eff_subarrayv, self.eff_subarrayh), dtype=np.uint16)
            self.imageSIM = np.zeros((self.eff_subarrayv, self.eff_subarrayh), dtype=np.uint16)
            self.isUpdateImageViewer = True

    def channelChanged(self):
        self.resetHexSIM()
        if self.current_channel_display() == 0:
            self.settings['wavelength'] = 0.523
            # self.imvRAW.displaySet(0)
            # self.imvRAW.ui.cellCombo.setCurrentIndex(0)
        elif self.current_channel_display() ==1:
            self.settings['wavelength'] = 0.610
            # self.imvRAW.displaySet(1)
            # self.imvRAW.ui.cellCombo.setCurrentIndex(1)

# functions for operation
    def standardCapturePressed(self):
        # if not self.screen.slm_dev.isVisible():
        #     self.show_text('Open SLM!')
        # else:
        #     self.isCameraRun = False
        #     self.channelChanged()
        #     self.action = 'standard_capture'
        self.isCameraRun = False
        self.channelChanged()
        self.action = 'standard_capture'

    def batchCapturePressed(self):
        # if not self.screen.slm_dev.isVisible():
        #     self.show_text('Open SLM!')
        #     pass
        # else:
        #     self.isCameraRun = False
        #     self.channelChanged()
        #     self.action = 'batch_capture'
        self.isCameraRun = False
        self.channelChanged()
        self.action = 'batch_capture'

    def calibrationPressed(self):
        self.isCameraRun = False
        self.action = 'calibration'

    def reconstructionPressed(self):
        self.isCameraRun = False
        if len(self.imageRAW[self.current_channel_display()])>7:
            self.action = 'batch_process'
        elif len(self.imageRAW[self.current_channel_display()])==7:
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

# functions for measurement
    def standardCapture(self):
        try:
            # initialize the raw image array list
            self.imageRAW = [np.zeros((7, self.eff_subarrayv, self.eff_subarrayh), dtype=np.uint16),
                             np.zeros((7, self.eff_subarrayv, self.eff_subarrayh), dtype=np.uint16)]

            if self.ui.dualWavelength.isChecked():
                # switch to 488 nm laser pattern
                # self.screen.slm_dev.setPatterns(0.488)
                # self.screen.openSLM()
                # project the patterns of 488nm pattern and acquire 7 raw images
                for i in range(7):
                    # self.screen.slm_dev.displayFrameN(i)
                    time.sleep(0.05)
                    # time.sleep(self.getAcquisitionInterval() / 1000.0)
                    self.imageRAW[0][i, :, :] = self.getOneFrame()
                    self.show_text(f'[488 nm] Capture frame: {i+1}')
                # switch to 561 nm laser pattern
                # self.screen.slm_dev.setPatterns(0.561)
                self.screen.openSLM()
                # project the patterns of 488nm pattern and acquire 7 raw images
                for i in range(7):
                    # self.screen.slm_dev.displayFrameN(i)
                    time.sleep(0.05)
                    # time.sleep(self.getAcquisitionInterval() / 1000.0)
                    self.imageRAW[1][i, :, :] = self.getOneFrame()
                    self.show_text(f'[561 nm] Capture frame: {i+1}')
                # restore SLM setting
                # self.controlSLM()

            else:
                # project the patterns and acquire 7 raw images
                for i in range(7):
                    # self.screen.slm_dev.displayFrameN(i)
                    time.sleep(0.05)
                    # time.sleep(self.getAcquisitionInterval() / 1000.0)
                    self.imageRAW[self.current_channel_caputure()][i, :, :] = self.getOneFrame()
                    self.show_text(f'Capture frame: {i+1}')

            # self.imageAVG = self.raw2WideFieldImage(self.imageRAW[self.current_channel_display()])
            self.show_text('Standard capture finished.')
            self.isUpdateImageViewer = True

        except Exception as e:
            txtDisplay = f'Standard capture encountered an error \n{e}'
            self.show_text(txtDisplay)

    # def batchCapture(self):
    #     try:
    #         n_stack = 7*self.ui.nStack.value()      # Initialize the raw image array
    #         # initialize the raw image array list
    #         self.imageRAW = [np.zeros((7, self.eff_subarrayv, self.eff_subarrayh), dtype=np.uint16),
    #                          np.zeros((7, self.eff_subarrayv, self.eff_subarrayh), dtype=np.uint16)]
    #
    #         step_size = self.stage.settings.stepsize.val
    #         stage_offset = n_stack*step_size
    #         pos = 25-stage_offset/2.0
    #         self.stage.moveAbsolutePositionHW(pos)
    #
    #         if self.ui.dualWavelength.isChecked():
    #             # extend the raw image storage of stacks
    #             self.imageRAW = [np.zeros((n_stack, self.eff_subarrayv, self.eff_subarrayh), dtype=np.uint16),
    #                              np.zeros((n_stack, self.eff_subarrayv, self.eff_subarrayh), dtype=np.uint16)]
    #             #switch to 488 nm laser pattern and acquire images
    #             self.screen.slm_dev.setPatterns(0.488)
    #             self.screen.openSLM()
    #             pos_tmp = pos
    #             for i in range(n_stack):
    #                 self.screen.slm_dev.displayFrameN(i % 7)
    #                 time.sleep(0.050)
    #                 self.imageRAW[0][i, :, :] = self.getOneFrame()
    #                 # move the stage to position
    #                 pos_tmp = pos_tmp + step_size
    #                 self.stage.moveAbsolutePositionHW(pos_tmp)
    #                 self.show_text(f'[488 nm] Capture frame : {i+1} / {n_stack}')
    #
    #             # switch to 561 nm laser pattern and acquire images
    #             self.screen.slm_dev.setPatterns(0.561)
    #             self.screen.openSLM()
    #             pos_tmp = pos
    #             for i in range(n_stack):
    #                 self.screen.slm_dev.displayFrameN(i % 7)
    #                 time.sleep(0.050)
    #                 self.imageRAW[1][i, :, :] = self.getOneFrame()
    #                 # move the stage to position
    #                 pos_tmp = pos_tmp + step_size
    #                 self.stage.moveAbsolutePositionHW(pos_tmp)
    #                 self.show_text(f'[561 nm] Capture frame: {i+1} / {n_stack}')
    #
    #         else:
    #             ch = self.current_channel_caputure()
    #             self.imageRAW[ch] = np.zeros((n_stack, self.eff_subarrayv, self.eff_subarrayh), dtype=np.uint16)
    #             # print(self.imageRAW[ch].shape)
    #             # project the patterns of CURRENT wavelength pattern and acquire n_stack raw images
    #             for i in range(n_stack):
    #                 self.screen.slm_dev.displayFrameN(i % 7)
    #                 time.sleep(0.050)
    #                 self.imageRAW[ch][i, :, :] = self.getOneFrame()
    #                 # move the stage to position
    #                 pos = pos + step_size
    #                 self.stage.moveAbsolutePositionHW(pos)
    #                 self.show_text(f'Capture frame: {i+1} / {n_stack}')
    #
    #         self.show_text('Batch capture finished.')
    #         self.stage.moveAbsolutePositionHW(25)    # Move the stage back to the middle position
    #         self.isUpdateImageViewer = True
    #
    #     except Exception as e:
    #         self.show_text(f'Batch capture encountered an error \n{e}')

    def batchCapture(self):
        try:
            n_stack = 7 * self.ui.nStack.value()      # Initialize the raw image array
            # initialize the raw image array list
            self.imageRAW = [np.zeros((7, self.eff_subarrayv, self.eff_subarrayh), dtype=np.uint16),
                             np.zeros((7, self.eff_subarrayv, self.eff_subarrayh), dtype=np.uint16)]

            step_size = self.z_stage.stepsize.val
            stage_offset = n_stack*step_size
            pos = self.z_stage.settings.absolute_position.val - stage_offset / 2.0
            self.z_stage.movePositionHW(pos)

            # extend the raw image storage of stacks
            self.imageRAW = [np.zeros((n_stack, self.eff_subarrayv, self.eff_subarrayh), dtype=np.uint16),
                             np.zeros((n_stack, self.eff_subarrayv, self.eff_subarrayh), dtype=np.uint16)]
            pos_tmp = pos
            # self.z_stage.zScanHW(pos_tmp, self.ui.nStack.value())
            # not sure if this is multithreading, maybe we need to excute z-stage manully!!!!!!!!!!!
            # Also how to synchronise the NI_CO signal and the acquisition????
            frames = self.getFrameStack(n_stack)
            print(len(frames))
            # self.camera.hamamatsu.stopAcquisition()
            for i in range(int(n_stack/2)):
                self.imageRAW[0][i, :, :] = frames[2 * i, :, :]
                self.imageRAW[1][i, :, :] = frames[2 * i + 1, :, :]
                # move the stage to position
                # pos_tmp = pos_tmp + step_size
                # self.stage.moveAbsolutePositionHW(pos_tmp)
                # self.show_text(f'Capture frame : {i + 1} / {n_stack}')
            # else:
            #     ch = self.current_channel_caputure()
            #     self.imageRAW[ch] = np.zeros((n_stack, self.eff_subarrayv, self.eff_subarrayh), dtype=np.uint16)
            #     # print(self.imageRAW[ch].shape)
            #     # project the patterns of CURRENT wavelength pattern and acquire n_stack raw images
            #     for i in range(n_stack):
            #         self.screen.slm_dev.displayFrameN(i % 7)
            #         time.sleep(0.050)
            #         self.imageRAW[ch][i, :, :] = self.getOneFrame()
            #         # move the stage to position
            #         pos = pos + step_size
            #         self.z_stage.movePositionHW(pos)
            #         self.show_text(f'Capture frame: {i+1} / {n_stack}')
            #
            # self.show_text('Batch capture finished.')
            # # self.stage.moveAbsolutePositionHW(25)    # Move the stage back to the middle position
            # self.isUpdateImageViewer = True

        except Exception as e:
            self.show_text(f'Batch capture encountered an error: {e}')

# functions for processing
    @add_timer
    def calibration(self):
        try:
            self.setReconstructor()
            image_raw_tmp = self.imageRAW[self.current_channel_display()]
            if self.isGpuenable:
                self.h.calibrate_cupy(image_raw_tmp, self.isFindCarrier)
            else:
                self.h.calibrate(image_raw_tmp, self.isFindCarrier)

            self.isCalibrated = True
            self.show_text('Calibration finished.')
            self.h.wienerfilter_store = self.h.wienerfilter
            self.kx_full = self.h.kx
            self.ky_full = self.h.ky

        except Exception as e:
            self.show_text(f'Calibration encountered an error \n{e}')

    @add_timer
    def standardReconstruction(self):
        # standard reconstruction
        try:
            image_raw_tmp = self.imageRAW[self.current_channel_display()]
            if self.isCalibrated:
                if self.isGpuenable:
                    self.imageSIM = self.h.reconstruct_cupy(image_raw_tmp)
                elif not self.isGpuenable:
                    self.imageSIM = self.h.reconstruct_rfftw(image_raw_tmp)
            else:
                self.calibration()
                if self.isCalibrated:
                    self.standardReconstruction()

            self.show_text('Standard reconstruction finished.')

        except Exception as e:
            self.show_text(f'Reconstruction encountered an error \n{e}')

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
            imageRaw_tmp = self.imageRAW[self.current_channel_display()]
            if self.isCalibrated:
                # Batch reconstruction
                if self.isGpuenable:
                    if self.isCompact:
                        self.imageSIM = self.h.batchreconstructcompact_cupy(imageRaw_tmp)
                    elif not self.isCompact:
                        self.imageSIM = self.h.batchreconstruct_cupy(imageRaw_tmp)
                elif not self.isGpuenable:
                    if self.isCompact:
                        self.imageSIM = self.h.batchreconstructcompact(imageRaw_tmp)
                    elif not self.isCompact:
                        self.imageSIM = self.h.batchreconstruct(imageRaw_tmp)
            elif not self.isCalibrated:
                self.calibration()
                if self.isCalibrated:
                    self.batchReconstruction()

            self.show_text('Batch reconstruction finished.')

        except Exception as e:
            self.show_text(f'Batch reconstruction encountered an error \n{e}')

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

# functions for camera
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

    def getFrameStack(self, n_frames):
        self.camera.hamamatsu.setACQMode("fixed_length", number_frames=n_frames)
        self.camera.hamamatsu.startAcquisition()
        [frames, dims] = self.camera.hamamatsu.getFrames()
        self.camera.hamamatsu.stopAcquisition()
        self.image_stack = np.zeros((n_frames, dims[0], dims[1]))
        if len(frames) == n_frames:
            for i in range(n_frames):
                self.image_stack[i, :, :] = np.reshape(frames[i].getData().astype(np.uint16), dims)
        else:
            self.image_stack = None
            print("Camera buffer empty")

        return self.image_stack

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

# functions for IO
    def saveMeasurements(self):
        if list_equal(self.imageRaw_store, self.imageRAW):
            self.show_text("Raw images are not saved: the same measurement.")
        else:
        #
        # if list_equal(self.imageRaw_store, self.imageRAW):
            timestamp = time.strftime("%y%m%d_%H%M%S", time.localtime())
            sample = self.app.settings['sample']
            if sample == '':
                sample_name = '_'.join([timestamp, self.name])
            else:
                sample_name = '_'.join([timestamp, self.name, sample])
            # create file path for both h5 and other types of files
            pathname = os.path.join(self.app.settings['save_dir'], sample_name)
            Path(pathname).mkdir(parents=True, exist_ok=True)
            self.pathname = pathname
            self.sample_name = sample_name

            # create h5 base file if the h5 file is not exist
            if self.ui.saveH5.isChecked():
                # create h5 file for raw
                fname_raw = os.path.join(pathname, sample_name + '_Raw.h5')
                self.h5file_raw = h5_io.h5_base_file(app=self.app, measurement=self, fname=fname_raw)
                # save measure component settings
                h5_io.h5_create_measurement_group(measurement=self, h5group=self.h5file_raw)
                for ch_idx in range(2):
                    gname = f'data/c{ch_idx}/raw'
                    if np.sum(self.imageRAW[ch_idx]) == 0:  # remove the empty channel
                        self.h5file_raw.create_dataset(gname, data = h5py.Empty("f"))
                        self.show_text("[H5] Raw images are empty.")
                    else:
                        self.h5file_raw.create_dataset(gname, data = self.imageRAW[ch_idx])
                        self.show_text("[H5] Raw images are saved.")

                self.h5file_raw.close()

            if self.ui.saveTif.isChecked():
                for ch_idx in range(2):
                    fname_raw = os.path.join(pathname, sample_name + f'_Raw_Ch{ch_idx}.tif')
                    if np.sum(self.imageRAW[ch_idx]) != 0:  # remove the empty channel
                        tif.imwrite(fname_raw, np.single(self.imageRAW[ch_idx]))
                        self.show_text("[Tif] Raw images are saved.")
                    else:
                        self.show_text("[Tif] Raw images are empty.")

            self.imageRaw_store = self.imageRAW # store the imageRAW for comparision

        if self.isCalibrated:
            if self.ui.saveH5.isChecked():
                fname_pro = os.path.join(self.pathname, self.sample_name + f'_C{self.current_channel_display()}_Processed.h5')
                self.h5file_pro = h5_io.h5_base_file(app=self.app, measurement=self, fname=fname_pro)
                h5_io.h5_create_measurement_group(measurement=self, h5group=self.h5file_pro)

                name = f'data/sim'
                if np.sum(self.imageSIM) == 0:
                    dset = self.h5file_pro.create_dataset(name, data = h5py.Empty("f"))
                    self.show_text("[H5] SIM images are empty.")
                else:
                    dset = self.h5file_pro.create_dataset(name, data=self.imageSIM)
                    self.show_text("[H5] SIM images are saved.")
                dset.attrs['kx'] = self.kx_full
                dset.attrs['ky'] = self.ky_full

                if self.numSets!=0:
                    for idx in range (self.numSets):
                        roi_group_name = f'data/roi/{idx:03}'
                        raw_set = self.h5file_pro.create_dataset(roi_group_name+'/raw', data=self.imageRaw_ROI[idx])
                        raw_set.attrs['cx'] = self.oSegment.selected_cx[idx]
                        raw_set.attrs['cy'] = self.oSegment.selected_cy[idx]
                        sim_set = self.h5file_pro.create_dataset(roi_group_name+'/sim', data=self.imageSIM_ROI[idx])
                        sim_set.attrs['kx'] = self.kx_roi[idx]
                        sim_set.attrs['ky'] = self.ky_roi[idx]
                    self.show_text("[H5] ROI images are saved.")

                self.h5file_pro.close()

            if self.ui.saveTif.isChecked():
                fname_sim = os.path.join(self.pathname,self.sample_name + f'_C{self.current_channel_display()}_SIM' + '.tif')
                fname_ini = os.path.join(self.pathname,self.sample_name + f'_C{self.current_channel_display()}_Settings' + '.ini')
                if np.sum(self.imageSIM) != 0:
                    tif.imwrite(fname_sim, np.single(self.imageSIM))
                    self.app.settings_save_ini(fname_ini, save_ro=False)
                    self.show_text("[Tif] SIM images are saved.")
                else:
                    self.show_text("[Tif] SIM images are empty.")

                if self.numSets != 0:
                    for idx in range(self.numSets):
                        fname_roi = os.path.join(self.pathname, self.sample_name + f'_Roi_C{self.current_channel_display()}_{idx:003}_SIM' + '.tif')
                        tif.imwrite(fname_roi, np.single(self.imageSIM_ROI[idx]))
                    self.show_text("[Tif] ROI images are saved.")

    @add_update_display
    def loadCalibrationResults(self):
        try:
            filename, _ = QFileDialog.getOpenFileName(caption="Open file", directory=self.app.settings['save_dir'],
                                                      filter="H5 files (*Processed.h5)")
            with h5py.File(filename,"r") as f:
                self.h.kx_input = self.kx_input = f["data/sim"].attrs['kx']
                self.h.ky_input = self.ky_input = f["data/sim"].attrs['ky']
            self.setReconstructor()
            self.isUpdateImageViewer = True
            self.show_text("Calibration results are loaded.")
        except:
            self.show_text("Calibration results are not loaded.")

# functions for display
    def setRefresh(self, refresh_period):
        self.display_update_period = refresh_period

    def show_text(self, text):
        self.ui.MessageBox.insertPlainText(text+'\n')
        self.ui.MessageBox.ensureCursorVisible()
        print(text)

    def updateImageViewer(self):
        self.imvRaw.showImageSet(self.imageRAW)
        try:
            self.imageWF = self.raw2WideFieldImage(self.imageRAW[self.current_channel_display()])
            self.imvWF.showImageSet(self.imageWF)
        except:
            pass
        self.imvWF_ROI.showImageSet(self.imageWF_ROI)
        self.imvSIM.showImageSet(self.imageSIM)
        self.imvSIM_ROI.showImageSet(self.imageSIM_ROI)
        self.imvWiener_ROI.showImageSet(self.wiener_ROI)
        self.imvCalibration.update(self.h)

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

# functions for ROI
    def roiSize(self):
        return int(self.ui.roiSizeCombo.currentText())

    def minCellSize(self):
        return int(self.ui.minCellSizeInput.value())

    def findCell(self):
        self.oSegment = ImageSegmentation(self.imageRAW[self.current_channel_display()], self.roiSize() // 2, self.minCellSize())
        markpen = pg.mkPen('r', width=1)
        self.removeMarks()
        self.oSegment.min_cell_size = self.minCellSize()**2
        self.oSegment.roi_half_side = self.roiSize()//2
        self.oSegment.find_cell()
        self.imageRaw_ROI = self.oSegment.roi_creation()
        self.imageWF_ROI = [] # initialize the image sets
        self.numSets = len(self.imageRaw_ROI)
        self.ui.cellNumber.display(self.numSets)

        if self.numSets == 0:
            self.imageWF_ROI = np.zeros((self.roiSize(),self.roiSize()), dtype=np.uint16)
        elif self.numSets > 0:
            for idx in range(self.numSets):
                self.imageWF_ROI.append(self.raw2WideFieldImage(self.imageRaw_ROI[idx]))
                # mark the cells with rectangle overlay
                r = pg.ROI(pos = (self.oSegment.selected_cx[idx]-self.oSegment.roi_half_side,
                                  self.oSegment.selected_cy[idx]-self.oSegment.roi_half_side),
                           size=self.roiSize(), pen=markpen, movable=False)
                self.imvWF.imv.getView().addItem(r)
                self.roiRect.append(r)

        self.isUpdateImageViewer = True
        self.show_text(f'Found cells: {self.numSets}')

# #  Streaming reconstruction
#     def streamAcquisitionTimer(self):
#         # Calibration should be done by standard measurement firstly.
#         if self.isCalibrated:
#             # Initialization
#             self.isStreamRun = True
#             self.streamIndex = 0
#             self.imageRAW = np.zeros((7, self.eff_subarrayv, self.eff_subarrayh), dtype=np.uint16)
#             self.cameraInterrupt()
#             self.interrupt()
#             self.streamTimer = QTimer(self)
#             self.streamTimer.timeout.connect(self.streamMeasureTimer)
#             self.streamTimer.start(self.getAcquisitionInterval())
#         elif not self.isCalibrated:
#             print('Run calibration first.')
#
#     def streamAcquisitionTimerStop(self):
#         self.streamTimer.stop()
#         self.isStreamRun = False
#         # Recover camera streaming
#         self.camera.updateCameraSettings()
#         self.cameraStart()
#         self.start()
#
#     def streamMeasureTimer(self):
#         print(self.streamIndex)
#         self.screen.slm_dev.displayFrameN((self.streamIndex) % 7)
#         newFrame = self.getOneFrame()
#         self.imageRAW[(self.streamIndex % 7), :, :] = newFrame
#         self.streamReconstruction(newFrame[0, :, :], (self.streamIndex % 7))
#         self.updateImageViewer()
#         self.imv.setImage(newFrame.T, autoLevels=True, autoRange=True)
#         self.streamIndex += 1
#
#     def streamReconstruction(self, newFrame, index):
#         print(index)
#         if self.isGpuenable:
#             self.imageSIM = self.h.reconstructframe_cupy(newFrame, index)
#         elif not self.isGpuenable:
#             self.imageSIM = self.h.reconstructframe_rfftw(newFrame, index)
#
#         self.imageSIM = self.imageSIM[np.newaxis, :, :]
#
#     def streamStopPressed(self):
#         self.isStreamRun = False
