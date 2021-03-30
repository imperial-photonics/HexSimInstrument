import json
import os
import time
from datetime import datetime
from pathlib import Path
from threading import Thread, currentThread, Event

import numpy as np
import pyqtgraph as pg
import qimage2ndarray
import tifffile as tif
from PyQt5 import uic
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QWidget, QTableWidgetItem, \
    QHeaderView
from ScopeFoundry import Measurement, h5_io
from ScopeFoundry.helper_funcs import sibling_path, load_qt_ui_file

from QtImageViewer import QtImageViewer
from HexSimProcessor.SIM_processing.hexSimProcessor import HexSimProcessor


class HexSimMeasurement(Measurement):
    name = 'HexSIM_Measurement'

    # def __init__(self):


    def setup(self):
        # Initialize condition labels
        self.isStreamRun = False
        self.isUpdateImageViewer = False
        self.isCameraRun = False
        self.showCalibrationResult = False
        self.isProcessingFinished = False
        self.isCalibrationSaved = False
        self.isSnapshot = False
        # self.isAutoSaveBatchMeasure = True
        # self.isAutoSaveCalibration = True

        # self.isCameraConnect = False
        # self.isLaser488Connect = False
        # self.isLaser561Connect = False
        # self.isScreenConnect = False
        # self.isStageConnect = False
        # Message window
        self.messageWindow = None
        # load ui file
        self.ui_filename = sibling_path(__file__, "hexsim_measurement.ui")
        self.ui = load_qt_ui_file(self.ui_filename)



        # Connect to hardware components
        self.camera = self.app.hardware['HamamatsuHardware']
        self.screen = self.app.hardware['ScreenHardware']
        self.stage = self.app.hardware['NanoScanHardware']
        self.laser488 = self.app.hardware['Laser488Hardware']
        self.laser561 = self.app.hardware['Laser561Hardware']
        # try:
        #     self.camera = self.app.hardware['HamamatsuHardware']
        #     self.isCameraConnect = True
        #     print('Camera is connected.')
        # except:
        #     self.isCameraConnect = False
        #     print('Camera is not connected.')
        #
        # try:
        #     self.screen = self.app.hardware['ScreenHardware']
        #     self.isScreenConnect = True
        #     print('SLM is connected.')
        # except:
        #     self.isScreenConnect = False
        #     print('SLM is not connected.')
        #
        # try:
        #     self.stage = self.app.hardware['NanoScanHardware']
        #     self.isStageConnect = True
        #     print('Stage is connected.')
        # except:
        #     self.isStageConnect = False
        #     print('Stage is not connected.')

        # Measurement component settings
        self.settings.New('record', dtype=bool, initial=False, hardware_set_func=self.setRecord,
                          hardware_read_func=self.getRecord, reread_from_hardware_after_write=True)
        self.settings.New('save_h5', dtype=bool, initial=False, hardware_set_func=self.setSaveH5,
                          hardware_read_func=self.getSaveH5)
        self.settings.New('refresh_period', dtype=float, unit='s', spinbox_decimals=4, initial=0.02,
                          hardware_set_func=self.setRefresh, vmin=0)
        self.settings.New('autoRange', dtype=bool, initial=False, hardware_set_func=self.setautoRange)
        self.settings.New('autoLevels', dtype=bool, initial=True, hardware_set_func=self.setautoLevels)
        self.settings.New('level_min', dtype=int, initial=60, hardware_set_func=self.setminLevel,
                          hardware_read_func=self.getminLevel)
        self.settings.New('level_max', dtype=int, initial=150, hardware_set_func=self.setmaxLevel,
                          hardware_read_func=self.getmaxLevel)
        self.settings.New('threshold', dtype=int, initial=500, hardware_set_func=self.setThreshold)

        self.add_operation('terminate', self.terminate)

        self.autoRange = self.settings.autoRange.val
        self.display_update_period = self.settings.refresh_period.val
        self.autoLevels = self.settings.autoLevels.val
        self.level_min = self.settings.level_min.val
        self.level_max = self.settings.level_max.val



        self.standardMeasureEvent = Event()
        self.standardProcessEvent = Event()
        self.standardProcessFinished = Event()
        self.standardSimulationEvent = Event()

        self.batchMeasureEvent = Event()
        self.batchProcessEvent = Event()
        self.batchProcessFinished = Event()
        self.batchSimulationEvent = Event()

        self.calibrationMeasureEvent = Event()
        self.calibrationProcessEvent = Event()
        self.calibrationFinished = Event()

        self.dlg = QMessageBox()
        self.dlg.setWindowTitle("Message")



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
        self.imageRaw = np.zeros((1, int(self.camera.subarrayv.val), int(self.camera.subarrayh.val)), dtype=np.uint16)
        self.imageSIM = np.zeros((1, 2 * int(self.camera.subarrayv.val), 2 * int(self.camera.subarrayh.val)),
                                 dtype=np.uint16)

        self.imvRaw.setImage((self.imageRaw[0, :, :]).T, autoRange=False, autoLevels=True, autoHistogramRange=True)
        self.imvSIM.setImage((self.imageSIM[0, :, :]).T, autoRange=False, autoLevels=True, autoHistogramRange=True)

        # Camera
        self.ui.camButton.clicked.connect(self.camButtonPressed)
        self.ui.snapshot.clicked.connect(self.snapshotPressed)

        # Screen
        self.ui.slmButton.clicked.connect(self.slmButtonPressed)
        self.ui.previousPatternButton.clicked.connect(self.screen.previousPattern)
        self.ui.nextPatternButton.clicked.connect(self.screen.nextPattern)

        # Stage
        self.ui.stagePositionIncrease.clicked.connect(self.stage.moveUpHW)
        self.ui.stagePositionDecrease.clicked.connect(self.stage.moveDownHW)

        # region Reconstructor settings
        self.ui.debugCheck.stateChanged.connect(self.setReconstructor)
        self.ui.cleanupCheck.stateChanged.connect(self.setReconstructor)
        self.ui.gpuCheck.stateChanged.connect(self.setReconstructor)
        self.ui.axialCheck.stateChanged.connect(self.setReconstructor)
        self.ui.usemodulationCheck.stateChanged.connect(self.setReconstructor)
        self.ui.compactCheck.stateChanged.connect(self.setReconstructor)
        self.ui.useLoadedResultsCheck.stateChanged.connect(self.setReconstructor)

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

        # Measure
        self.ui.captureStandardButton.clicked.connect(self.standardAcquisition)

        self.ui.startStreamingButton.clicked.connect(self.streamAcquisitionTimer)
        self.ui.stopStreamingButton.clicked.connect(self.streamAcquisitionTimerStop)

        self.ui.captureBatchButton.clicked.connect(self.batchAcquisition)

        self.ui.saveButton.clicked.connect(self.saveMeasurements)

        # Test
        self.ui.calibrationButton.clicked.connect(self.calibrationAcquisition)
        self.ui.resetMeasureButton.clicked.connect(self.resetHexSIM)
        self.ui.calibrationResult.clicked.connect(self.showMessageWindow)

        self.ui.calibrationSave.clicked.connect(self.saveMeasurements)
        self.ui.calibrationLoad.clicked.connect(self.loadCalibrationResults)

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
        # update stage position
        self.ui.stagePositionDisplay.display(self.stage.settings.absolute_position.val)

        # update camera viewer
        if self.isStreamRun or self.isCameraRun:
            if not self.autoLevels:
                self.imv.setImage(self.image.T, autoLevels=self.settings.autoLevels.val,
                                  autoRange=self.settings.autoRange.val, levels=(self.level_min, self.level_max))

            else:  # levels should not be sent when autoLevels is True, otherwise the image is displayed with them
                self.imv.setImage(self.image.T, autoLevels=self.settings.autoLevels.val,
                                  autoRange=self.settings.autoRange.val)

                self.settings.level_min.read_from_hardware()
                self.settings.level_max.read_from_hardware()

        # update hexsim viwer
        if self.isStreamRun or self.isUpdateImageViewer:
            self.updateImageViewer()
            self.isUpdateImageViewer = False

        if self.showCalibrationResult:
            self.showMessageWindow()
            self.showCalibrationResult = False
            self.isCalibrationSaved = False

        if self.isCalibrationSaved:
            msg = QMessageBox()
            msg.about(self.ui, "Message", "Results are saved.")
            msg.setIcon(QMessageBox.Information)
            self.isCalibrationSaved = False


    def start_threads(self):

        self.eff_subarrayh = int(self.camera.subarrayh.val / self.camera.binning.val)
        self.eff_subarrayv = int(self.camera.subarrayv.val / self.camera.binning.val)

        self.image = np.zeros((self.eff_subarrayv, self.eff_subarrayh), dtype=np.uint16)
        # self.image[0, 0] = 1  # Otherwise we get the "all zero pixels" error (we should modify pyqtgraph...)

        if not hasattr(self, 'h'):
            self.h = HexSimProcessor()  # create reconstruction object
            self.imageRaw = np.zeros((7, self.eff_subarrayv, self.eff_subarrayh),
                                     dtype=np.uint16)  # Initialize the raw image array
            self.setReconstructor()
            self.h.N = self.eff_subarrayh

        if not hasattr(self, 'calibrationMeasureThread'):
            self.calibrationMeasureThread = Thread(target=self.calibrationMeasure)
            self.calibrationMeasureThread.start()

        if not hasattr(self, 'calibrationProcessThread'):
            self.calibrationProcessThread = Thread(target=self.calibrationProcessor)
            self.calibrationProcessThread.start()

        if not hasattr(self, 'standardMeasureThread'):
            self.standardMeasureThread = Thread(target=self.standardMeasure)
            self.standardMeasureThread.start()

        if not hasattr(self, 'standardProcessThread'):
            self.standardProcessThread = Thread(target=self.standardProcessor)
            self.standardProcessThread.start()

        if not hasattr(self, 'standardSimulationThread'):
            self.standardSimulationThread = Thread(target=self.standardSimulation)
            self.standardSimulationThread.start()

        if not hasattr(self, 'batchMeasureThread'):
            self.batchMeasureThread = Thread(target=self.batchMeasure)
            self.batchMeasureThread.start()

        if not hasattr(self, 'batchProcessThread'):
            self.batchProcessThread = Thread(target=self.batchProcessor)
            self.batchProcessThread.start()

        if not hasattr(self, 'batchSimulationThread'):
            self.batchSimulationThread = Thread(target=self.batchSimulation)
            self.batchSimulationThread.start()

    def run(self):
        self.start_threads()
        # TODO: This while loop will slow the calculation
        while not self.interrupt_measurement_called:
            time.sleep(0.02)
            if self.isCameraRun:
                self.cameraRunTest()

    ################   Control  ################
    def camButtonPressed(self):
        if self.ui.camButton.text() == 'ON':
            try:
                self.cameraStart()
                self.ui.camButton.setText('OFF')
                print('Camera ON')
            except:
                pass

        elif self.ui.camButton.text() == 'OFF':
            try:
                self.cameraInterrupt()
                self.ui.camButton.setText('ON')
                print('Camera OFF')
            except:
                pass

    def slmButtonPressed(self):
        if self.ui.slmButton.text() == 'ON':
            self.screen.openSLM()
            self.screen.manualDisplay()
            self.ui.slmButton.setText('OFF')
            print('Screen OFF')
        elif self.ui.slmButton.text() == 'OFF':
            self.screen.closeSLM()
            self.ui.slmButton.setText('ON')
            print('Screen ON')
        self.ui.wavelengthValue.setValue(self.screen.settings.wavelength.val)

    # region Display Functions
    def rawImageSliderChanged(self):
        self.ui.rawImageSlider.setMinimum(0)
        self.ui.rawImageSlider.setMaximum(self.imageRaw.shape[0] - 1)

        self.imvRaw.setImage((self.imageRaw[int(self.ui.rawImageSlider.value()), :, :]).T, autoRange=False,
                             levels=(self.imageRawMin, self.imageRawMax))

        self.ui.rawImageNth.setText(str(self.ui.rawImageSlider.value() + 1))
        self.ui.rawImageNtotal.setText(str(len(self.imageRaw)))

    def simImageSliderChanged(self):
        self.ui.simImageSlider.setMinimum(0)
        self.ui.simImageSlider.setMaximum(self.imageSIM.shape[0] - 1)
        self.imvSIM.setImage((self.imageSIM[int(self.ui.simImageSlider.value()), :, :]).T,  autoRange=False,
                             levels=(0, 0.7 * self.imageSIMMax))

        self.ui.simImageNth.setText(str(self.ui.simImageSlider.value() + 1))
        self.ui.simImageNtotal.setText(str(len(self.imageSIM)))

    def updateImageViewer(self):
        self.imageRawMax = np.amax(self.imageRaw)
        self.imageRawMin = np.amin(self.imageRaw)
        self.imageSIMMax = np.amax(self.imageSIM)
        self.rawImageSliderChanged()
        self.simImageSliderChanged()

    # endregion

    ################    HexSIM  ################
    def resetHexSIM(self):
        if hasattr(self, 'h'):
            self.h.isCalibrated = False
            self.h._allocate_arrays()
            self.imageSIM = np.zeros((1, self.eff_subarrayv, self.eff_subarrayh), dtype=np.uint16)
            # self.imageRaw = np.zeros((7, self.eff_subarrayv, self.eff_subarrayh), dtype=np.uint16)
            self.updateImageViewer()

    def calibrationAcquisition(self):
        self.calibrationMeasureEvent.set()

    def calibrationMeasure(self):
        cthread = currentThread()
        while getattr(cthread, "isThreadRun", True):
            self.calibrationMeasureEvent.wait()
            # Stop the camera and main run
            self.cameraInterrupt()
            self.interrupt()
            # Initialize the raw image array
            self.imageRaw = np.zeros((7, self.eff_subarrayv, self.eff_subarrayh), dtype=np.uint16)

            # Project the patterns and acquire 7 raw images
            for i in range(7):
                self.screen.slm_dev.displayFrameN(i)
                time.sleep(self.getAcquisitionInterval() / 1000.0)
                self.imageRaw[i, :, :] = self.getOneFrame()
                print('Capture frame:', i)
                # self.ui.processingBar.setValue((i+1)*5)

            # Calibration
            self.calibrationProcessEvent.set()
            self.calibrationMeasureEvent.clear()
            self.calibrationFinished.wait()
            # self.ui.processingBar.setValue(80)
            # Recover camera streaming
            self.camera.updateCameraSettings()
            self.cameraStart()
            self.showCalibrationResult = True
            self.start()
            # self.ui.processingBar.setValue(99.9999)
            self.calibrationFinished.clear()

            self.isProcessingFinished = True

            if self.ui.autoSaveCalibration.isChecked():
                self.saveMeasurements()

    def calibrationProcessor(self):
        cthread = currentThread()
        while getattr(cthread, "isThreadRun", True):
            self.calibrationProcessEvent.wait()
            print('Start calibrating...')
            # self.ui.processingBar.setValue(40)

            startTime = time.time()

            if not self.interrupt_measurement_called:
                self.interrupt()

            if self.h.gpuenable:
                self.h.calibrate_cupy(self.imageRaw)
                # self.ui.processingBar.setValue(75)
                self.imageSIM = self.h.reconstruct_cupy(self.imageRaw)

            elif not self.h.gpuenable:
                self.h.calibrate(self.imageRaw)
                # self.ui.processingBar.setValue(75)
                self.imageSIM = self.h.reconstruct_rfftw(self.imageRaw)

            print('Calibration is processed in:', time.time() - startTime, 's')
            # self.ui.processingBar.setValue(99.99999)

            self.imageSIM = self.imageSIM[np.newaxis, :, :]
            self.isUpdateImageViewer = True

            self.start()

            self.calibrationProcessEvent.clear()
            self.calibrationFinished.set()

    # Standard reconstruction
    def standardAcquisition(self):
        self.standardMeasureEvent.set()

    def standardMeasure(self):
        cthread = currentThread()
        while getattr(cthread, "isThreadRun", True):
            self.standardMeasureEvent.wait()
            # Stop the camera and main run
            self.cameraInterrupt()
            self.interrupt()
            # Initialize the raw image array
            self.imageRaw = np.zeros((7, self.eff_subarrayv, self.eff_subarrayh), dtype=np.uint16)

            # Project the patterns and acquire 7 raw images
            for i in range(7):
                self.screen.slm_dev.displayFrameN(i)
                time.sleep(self.getAcquisitionInterval() / 1000.0)
                self.imageRaw[i, :, :] = self.getOneFrame()
                print('Capture frame:', i)

            # Standard reconstruction
            self.standardProcessEvent.set()
            self.standardMeasureEvent.clear()
            # Recover camera streaming
            self.standardProcessFinished.wait()
            self.camera.updateCameraSettings()
            self.cameraStart()
            self.start()
            self.standardProcessFinished.clear()

    def standardProcessor(self):
        cthread = currentThread()
        while getattr(cthread, "isThreadRun", True):
            self.standardProcessEvent.wait()

            isTemp = self.interrupt_measurement_called

            if not self.interrupt_measurement_called:
                self.interrupt()

            if self.h.isCalibrated:
                print('Start standard processing...')
                startTime = time.time()
                if self.h.gpuenable:
                    self.imageSIM = self.h.reconstruct_cupy(self.imageRaw)

                elif not self.h.gpuenable:
                    self.imageSIM = self.h.reconstruct_rfftw(self.imageRaw)

                print('One SIM image is processed in:', time.time() - startTime, 's')
                self.imageSIM = self.imageSIM[np.newaxis, :, :]

            elif not self.h.isCalibrated:
                self.calibrationProcessEvent.set()

            self.isUpdateImageViewer = True

            if not isTemp:
                self.start()

            self.standardProcessEvent.clear()
            self.standardProcessFinished.set()

    # region Batch reconstruction
    def batchAcquisition(self):
        self.batchMeasureEvent.set()

    def batchMeasure(self):
        cthread = currentThread()
        while getattr(cthread, "isThreadRun", True):
            self.batchMeasureEvent.wait()
            self.cameraInterrupt()
            self.interrupt()

            # Initialize the raw image array
            n_stack = 7*self.ui.nStack.value()
            stage_offset = n_stack*self.stage.settings.stepsize.val
            pos = 25-stage_offset/2.0
            self.stage.moveAbsolutePositionHW(pos)

            self.imageRaw = np.zeros((n_stack, self.eff_subarrayv, self.eff_subarrayh), dtype=np.uint16)

            # Project the patterns and acquire raw images
            for i in range(n_stack):
                self.screen.slm_dev.displayFrameN(i % 7)
                # time.sleep(self.getAcquisitionInterval() / 1000.0)
                self.imageRaw[i, :, :] = self.getOneFrame()
                print('Capture frame:', i)
                pos = pos + 0.05
                self.stage.moveAbsolutePositionHW(pos)

            # self.ui.processingBar.setValue(80)
            self.stage.moveAbsolutePositionHW(25)
            # Reconstruct the SIM image
            self.batchProcessEvent.set()
            self.batchMeasureEvent.clear()
            # Recover camera streaming
            self.batchProcessFinished.wait()
            self.camera.updateCameraSettings()
            self.cameraStart()
            self.start()
            self.batchProcessFinished.clear()
            self.isProcessingFinished = True

            if self.ui.autoSaveBatchCheck.isChecked():
                self.saveMeasurements()

    # endregion

    def batchProcessor(self):
        cthread = currentThread()
        while getattr(cthread, "isThreadRun", True):
            self.batchProcessEvent.wait()
            print('Start batch processing...')
            # self.ui.processingBar.setValue(5)
            isTemp = self.interrupt_measurement_called

            if not self.interrupt_measurement_called:
                self.interrupt()

            if self.h.isCalibrated:
                startTime = time.time()
                # Batch reconstruction
                if self.h.gpuenable:
                    if self.h.compact:
                        self.imageSIM = self.h.batchreconstructcompact_cupy(self.imageRaw)
                    elif not self.h.compact:
                        self.imageSIM = self.h.batchreconstruct_cupy(self.imageRaw)

                elif not self.h.gpuenable:
                    if self.h.compact:
                        self.imageSIM = self.h.batchreconstructcompact(self.imageRaw)
                    elif not self.h.compact:
                        self.imageSIM = self.h.batchreconstruct(self.imageRaw)

            elif not self.h.isCalibrated:
                startTime = time.time()
                nStack = len(self.imageRaw)
                # calibrate & reconstruction
                if self.h.gpuenable:
                    self.h.calibrate_cupy(self.imageRaw[int(nStack // 2):int(nStack // 2 + 7), :, :])
                    # self.ui.processingBar.setValue(25)
                    if self.h.compact:
                        self.imageSIM = self.h.batchreconstructcompact_cupy(self.imageRaw)
                    elif not self.h.compact:
                        self.imageSIM = self.h.batchreconstruct_cupy(self.imageRaw)

                elif not self.h.gpuenable:
                    self.h.calibrate(self.imageRaw[int(nStack // 2):int(nStack // 2 + 7), :, :])
                    # self.ui.processingBar.setValue(25)
                    if self.h.compact:
                        self.imageSIM = self.h.batchreconstructcompact(self.imageRaw)
                    elif not self.h.compact:
                        self.imageSIM = self.h.batchreconstruct(self.imageRaw)

            print('Batch reconstruction finished', time.time() - startTime, 's')
            # self.ui.processingBar.setValue(99.9999)

            self.isUpdateImageViewer = True

            if not isTemp:
                self.start()

            self.batchProcessEvent.clear()
            self.batchProcessFinished.set()

    # region Streaming reconstruction
    def streamAcquisitionTimer(self):
        # Calibration should be done by standard measurement firstly.
        if self.h.isCalibrated:
            # Initialization
            self.isStreamRun = True
            self.streamIndex = 0
            self.imageRaw = np.zeros((7, self.eff_subarrayv, self.eff_subarrayh), dtype=np.uint16)
            self.cameraInterrupt()
            self.interrupt()
            self.streamTimer = QTimer(self)
            self.streamTimer.timeout.connect(self.streamMeasureTimer)
            self.streamTimer.start(self.getAcquisitionInterval())
        elif not self.h.isCalibrated:
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
        # time.sleep(4/60)
        newFrame = self.getOneFrame()
        self.imageRaw[(self.streamIndex % 7), :, :] = newFrame
        self.streamReconstruction(newFrame[0, :, :], (self.streamIndex % 7))
        self.updateImageViewer()
        if not self.autoLevels:
            self.imv.setImage(newFrame.T, autoLevels=self.settings.autoLevels.val,
                              autoRange=self.settings.autoRange.val, levels=(self.level_min, self.level_max))

        else:  # levels should not be sent when autoLevels is True, otherwise the image is displayed with them
            self.imv.setImage(newFrame.T, autoLevels=self.settings.autoLevels.val,
                              autoRange=self.settings.autoRange.val)

            self.settings.level_min.read_from_hardware()
            self.settings.level_max.read_from_hardware()
        self.streamIndex += 1

    def streamReconstruction(self, newFrame, index):
        print(index)
        if self.h.gpuenable:
            self.imageSIM = self.h.reconstructframe_cupy(newFrame, index)
        elif not self.h.gpuenable:
            self.imageSIM = self.h.reconstructframe_rfftw(newFrame, index)

        self.imageSIM = self.imageSIM[np.newaxis, :, :]

    # endregion

    ############## Test #################################
    def standardSimuButtonPressed(self):
        self.runningStateStored = self.interrupt_measurement_called
        # print(self.runningStateStored)

        if not self.interrupt_measurement_called:
            self.interrupt()
        # read data
        filename, _ = QFileDialog.getOpenFileName(directory="./measurement")        # filename = "./data/standardData.tif"

        self.imageRaw = np.single(tif.imread(filename))

        if self.imageRaw.shape[0] == 7:
            self.standardSimulationEvent.set()
        else:
            print('Please input the 7-frame data set.')

    def standardSimulation(self):
        cthread = currentThread()
        while getattr(cthread, "isThreadRun", True):
            self.standardSimulationEvent.wait()
            self.calibrationProcessEvent.set()
            self.standardSimulationEvent.clear()
            self.calibrationFinished.wait()
            self.start()
            self.calibrationFinished.clear()


    def standardReconstructionUpdate(self):
        self.standardSimulationEvent.set()
        # self.calibrationProcessEvent.set()

    def standardReconstruction(self):
        # calibrate & reconstruction
        if self.h.gpuenable:
            self.h.calibrate_cupy(self.imageRaw)
            self.imageSIM = self.h.reconstruct_cupy(self.imageRaw)

        elif not self.h.gpuenable:
            self.h.calibrate(self.imageRaw)
            self.imageSIM = self.h.reconstruct_rfftw(self.imageRaw)

        self.imageSIM = self.imageSIM[np.newaxis, :, :]
        self.isUpdateImageViewer = True
        print('One SIM image is processed.')

    def streamSimuButtonPressed(self):
        self.virtualRecording()
        self.isStreamRun = True
        # create reconstruction object
        [_, width, height] = self.imageRawStack.shape
        self.imageRaw = np.zeros((7, width, height), dtype=np.uint16)
        # h from the first 7 images
        self.streamIndex = 0

        for i in range(7):
            self.imageRaw[i, :, :] = self.imageRawStack[i, :, :]
            self.streamIndex += 1
            time.sleep(0.5)
            print(self.streamIndex)

        self.standardReconstruction()
        self.updateImageViewer()

        self.streamSimuThread = Thread(target=self.streamReconstructionLoop)
        self.streamSimuThread.start()

    def streamReconstructionLoop(self):
        while self.isStreamRun and self.streamIndex <= 280:
            self.imageRawFrame = self.imageRawStack[self.streamIndex, :, :]

            self.streamReconstruction(self.imageRawFrame, (self.streamIndex) % 7)
            time.sleep(0.5)
            self.streamIndex += 1
            print(self.streamIndex)

    def streamStopPressed(self):
        self.isStreamRun = False

    def batchSimuButtonPressed(self):
        self.runningStateStored = self.interrupt_measurement_called
        # print(self.runningStateStored)

        if not self.interrupt_measurement_called:
            self.interrupt()

        filename, _ = QFileDialog.getOpenFileName(directory="./measurement")        # filename = "./data/stackData.tif"
        self.imageRaw = np.single(tif.imread(filename))
        self.batchSimulationEvent.set()

    def batchSimulation(self):
        cthread = currentThread()
        while getattr(cthread, "isThreadRun", True):
            self.batchSimulationEvent.wait()
            self.batchProcessEvent.set()
            self.batchSimulationEvent.clear()
            self.batchProcessFinished.wait()
            self.start()
            self.batchProcessFinished.clear()

    def batchReconstructionUpdate(self):
        self.h.isCalibrated = False
        self.batchProcessEvent.set()
        # self.batchSimulationEvent.set()

    def virtualRecording(self):
        filename, _ = QFileDialog.getOpenFileName(directory="./measurement")
        self.imageRawStack = np.single(tif.imread(filename))

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
        self.h.usePreCalibration = self.ui.useLoadedResultsCheck.isChecked()
        self.h.magnification = self.ui.magnificationValue.value()
        self.h.NA = self.ui.naValue.value()
        self.h.n = self.ui.nValue.value()
        self.h.wavelength = self.ui.wavelengthValue.value()
        self.h.pixelsize = self.ui.pixelsizeValue.value()

        self.h.alpha = self.ui.alphaValue.value()
        self.h.beta = self.ui.betaValue.value()
        self.h.w = self.ui.wValue.value()
        self.h.eta = self.ui.etaValue.value()

    def setMagnification(self):
        self.h.magnification = self.ui.magnificationValue.value()

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

    def snapshotPressed(self):
        self.isSnapshot = True

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

                    if self.isSnapshot:
                        tif.imwrite('live_image.tif', np.uint16(self.image))
                        self.isSnapshot = False
                        print('Saved one image.')

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
            "kx":               self.h.ckx,
            "ky":               self.h.cky,
            "phase":            self.h.p,
            "amplitude":        self.h.ampl
            }
        f = open(txtname, 'w+')
        f.write(json.dumps(savedictionary, cls=NumpyEncoder,indent=2))
        self.isCalibrationSaved = True

    def loadCalibrationResults(self):
        filename, _ = QFileDialog.getOpenFileName(caption="Open file", directory="./measurement", filter="Text files (*.txt)")
        file = open(filename,'r')
        loadResults = json.loads(file.read())
        self.h.ckx_in = np.asarray(loadResults["kx"])
        self.h.cky_in = np.asarray(loadResults["ky"])
        self.h.p_in = np.asarray(loadResults["phase"])
        self.h.ampl_in = np.asarray(loadResults["amplitude"])
        print('Calibration results are loaded.')

    def showMessageWindow(self):
        self.messageWindow = MessageWindow(self.h)
        self.messageWindow.show()


class MessageWindow(QWidget):

    """
    This window display the Winier filter and other debug data
    """

    def __init__(self, h):
        super().__init__()
        self.ui = uic.loadUi('calibration_results.ui',self)
        self.h = h
        self.setWindowTitle('Calibration results')
        self.showCurrentTable()
        # self.showLoadedTable()
        self.showWienerFilter()

        self.ui.wienerfilterLayout.addWidget(self.wienerfilterWidget)

    def showCurrentTable(self):

        self.ui.currentTable.setItem(0, 0, QTableWidgetItem(str(self.h.ckx_in[0]).lstrip('[').rstrip(']')))
        self.ui.currentTable.setItem(0, 1, QTableWidgetItem(str(self.h.ckx_in[1]).lstrip('[').rstrip(']')))
        self.ui.currentTable.setItem(0, 2, QTableWidgetItem(str(self.h.ckx_in[2]).lstrip('[').rstrip(']')))
        #
        self.ui.currentTable.setItem(1, 0, QTableWidgetItem(str(self.h.cky_in[0]).lstrip('[').rstrip(']')))
        self.ui.currentTable.setItem(1, 1, QTableWidgetItem(str(self.h.cky_in[1]).lstrip('[').rstrip(']')))
        self.ui.currentTable.setItem(1, 2, QTableWidgetItem(str(self.h.cky_in[2]).lstrip('[').rstrip(']')))

        # self.ui.currentTable.setItem(0, 0, QTableWidgetItem("k[x]"))
        self.ui.currentTable.setItem(2, 0, QTableWidgetItem(str(self.h.ckx[0]).lstrip('[').rstrip(']')))
        self.ui.currentTable.setItem(2, 1, QTableWidgetItem(str(self.h.ckx[1]).lstrip('[').rstrip(']')))
        self.ui.currentTable.setItem(2, 2, QTableWidgetItem(str(self.h.ckx[2]).lstrip('[').rstrip(']')))
        #
        # # self.ui.currentTable.setItem(1, 0, QTableWidgetItem("k[y]"))
        self.ui.currentTable.setItem(3, 0, QTableWidgetItem(str(self.h.cky[0]).lstrip('[').rstrip(']')))
        self.ui.currentTable.setItem(3, 1, QTableWidgetItem(str(self.h.cky[1]).lstrip('[').rstrip(']')))
        self.ui.currentTable.setItem(3, 2, QTableWidgetItem(str(self.h.cky[2]).lstrip('[').rstrip(']')))
        #
        # # self.ui.currentTable.setItem(2, 0, QTableWidgetItem("Phase"))
        self.ui.currentTable.setItem(4, 0, QTableWidgetItem(str(self.h.p[0]).lstrip('[').rstrip(']')))
        self.ui.currentTable.setItem(4, 1, QTableWidgetItem(str(self.h.p[1]).lstrip('[').rstrip(']')))
        self.ui.currentTable.setItem(4, 2, QTableWidgetItem(str(self.h.p[2]).lstrip('[').rstrip(']')))
        #
        # # self.ui.currentTable.setItem(3, 0, QTableWidgetItem("Amplitude"))
        self.ui.currentTable.setItem(5, 0, QTableWidgetItem(str(self.h.ampl[0]).lstrip('[').rstrip(']')))
        self.ui.currentTable.setItem(5, 1, QTableWidgetItem(str(self.h.ampl[1]).lstrip('[').rstrip(']')))
        self.ui.currentTable.setItem(5, 2, QTableWidgetItem(str(self.h.ampl[2]).lstrip('[').rstrip(']')))

        # Table will fit the screen horizontally
        self.currentTable.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

    # def showLoadedTable(self):
    #     self.ui.loadedTable.setItem(0, 0, QTableWidgetItem(str(self.h.ckx_in[0]).lstrip('[').rstrip(']')))
    #     self.ui.loadedTable.setItem(0, 1, QTableWidgetItem(str(self.h.ckx_in[1]).lstrip('[').rstrip(']')))
    #     self.ui.loadedTable.setItem(0, 2, QTableWidgetItem(str(self.h.ckx_in[2]).lstrip('[').rstrip(']')))
    #     #
    #     self.ui.loadedTable.setItem(1, 0, QTableWidgetItem(str(self.h.cky_in[0]).lstrip('[').rstrip(']')))
    #     self.ui.loadedTable.setItem(1, 1, QTableWidgetItem(str(self.h.cky_in[1]).lstrip('[').rstrip(']')))
    #     self.ui.loadedTable.setItem(1, 2, QTableWidgetItem(str(self.h.cky_in[2]).lstrip('[').rstrip(']')))
    #     #
    #     self.ui.loadedTable.setItem(2, 0, QTableWidgetItem(str(self.h.p_in[0]).lstrip('[').rstrip(']')))
    #     self.ui.loadedTable.setItem(2, 1, QTableWidgetItem(str(self.h.p_in[1]).lstrip('[').rstrip(']')))
    #     self.ui.loadedTable.setItem(2, 2, QTableWidgetItem(str(self.h.p_in[2]).lstrip('[').rstrip(']')))
    #     #
    #     self.ui.loadedTable.setItem(3, 0, QTableWidgetItem(str(self.h.ampl_in[0]).lstrip('[').rstrip(']')))
    #     self.ui.loadedTable.setItem(3, 1, QTableWidgetItem(str(self.h.ampl_in[1]).lstrip('[').rstrip(']')))
    #     self.ui.loadedTable.setItem(3, 2, QTableWidgetItem(str(self.h.ampl_in[2]).lstrip('[').rstrip(']')))
    #
    #     # Table will fit the screen horizontally
    #     self.currentTable.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

    def showWienerFilter(self):
        self.wienerfilterWidget = QtImageViewer()
        self.wienerfilterWidget.aspectRatioMode = Qt.KeepAspectRatio
        im = qimage2ndarray.gray2qimage(self.h.wienerfilter, normalize=True)
        self.wienerfilterWidget.setImage(im)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)