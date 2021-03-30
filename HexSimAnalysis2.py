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
from image_decorr import measure, ImageDecorr

class HexSimAnalysis(Measurement):
    name = 'HexSIM_Analysis'

    def setup(self):

        # load ui file
        self.ui_filename = sibling_path(__file__, "hexsim_analysis.ui")
        self.ui = load_qt_ui_file(self.ui_filename)

        # Message window
        self.messageWindow = None
        self.settings.New('refresh_period', dtype=float, unit='s', spinbox_decimals=4, initial=0.02,
                          hardware_set_func=self.setRefresh, vmin=0)

        # self.add_operation('terminate', self.terminate)
        self.display_update_period = self.settings.refresh_period.val

        # Initialize condition labels
        self.isUpdateImageViewer = False
        self.showCalibrationResult = False
        self.isCalibrationSaved = False
        self.isFileLoad = False
        self.processValue = 0
        self.start_timers()

    def setup_figure(self):
        # connect ui widgets to measurement/hardware settings or functionss
        # Set up pyqtgraph graph_layout in the UI
        self.imvRaw = pg.ImageView()
        self.imvSIM = pg.ImageView()
        self.imvWF = pg.ImageView()

        self.ui.rawImageLayout.addWidget(self.imvRaw)
        self.ui.simImageLayout.addWidget(self.imvSIM)
        self.ui.wfImageLayout.addWidget(self.imvWF)

        # Image initialization
        self.imageRaw = np.zeros((1, 512, 512), dtype=np.uint16)
        self.imageSIM = np.zeros((1, 1024,1024), dtype=np.uint16)
        self.imageWF = np.zeros((1, 512, 512), dtype=np.uint16)

        self.imvRaw.setImage((self.imageRaw[0, :, :]).T, autoRange=False, autoLevels=True, autoHistogramRange=True)
        self.imvWF.setImage((self.imageWF[0, :, :]).T, autoRange=False, autoLevels=True, autoHistogramRange=True)
        self.imvSIM.setImage((self.imageSIM[0, :, :]).T, autoRange=False, autoLevels=True, autoHistogramRange=True)

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
        self.ui.wfImageSlider.valueChanged.connect(self.wfImageSliderChanged)

        # Toolbox
        self.ui.loadFileButton.clicked.connect(self.loadFile)
        self.ui.resetMeasureButton.clicked.connect(self.resetHexSIM)
        self.ui.calibrationResult.clicked.connect(self.showMessageWindow)

        self.ui.calibrationSave.clicked.connect(self.saveMeasurements)
        self.ui.calibrationLoad.clicked.connect(self.loadCalibrationResults)

        self.ui.standardSimuButton.clicked.connect(self.standardSimuButtonPressed)
        self.ui.standardSimuUpdate.clicked.connect(self.standardReconstructionUpdate)

        # self.ui.streamSimuButton.clicked.connect(self.streamSimuButtonPressed)
        # self.ui.streamSimuStop.clicked.connect(self.streamStopPressed)

        self.ui.batchSimuButton.clicked.connect(self.batchSimuButtonPressed)
        self.ui.batchSimuUpdate.clicked.connect(self.batchReconstructionUpdate)

        self.ui.resolutionEstimateButton.clicked.connect(self.resolutionEstimatePressed)

    def update_display(self):
        """
        Displays the numpy array called self.image.
        This function runs repeatedly and automatically during the measurement run,
        its update frequency is defined by self.display_update_period.
        """
        if self.isUpdateImageViewer:
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

        self.ui.processingBar.setValue(self.processValue)


    def start_timers(self):
        if not hasattr(self, 'h'):
            self.h = HexSimProcessor()  # create reconstruction object
            self.setReconstructor()

        if not hasattr(self, 'calibrationProcessTimer'):
            self.calibrationProcessTimer = QTimer(self)
            self.calibrationProcessTimer.setSingleShot(True)
            self.calibrationProcessTimer.setInterval(1)
            self.calibrationProcessTimer.timeout.connect(self.calibrationProcessor)
            # print('Timer here')

        if not hasattr(self, 'standardProcessTimer'):
            self.standardProcessTimer = QTimer(self)
            self.standardProcessTimer.setSingleShot(True)
            self.standardProcessTimer.setInterval(1)
            self.standardProcessTimer.timeout.connect(self.standardProcessor)

        if not hasattr(self, 'batchProcessTimer'):
            self.batchProcessTimer = QTimer(self)
            self.batchProcessTimer.setSingleShot(True)
            self.batchProcessTimer.setInterval(1)
            self.batchProcessTimer.timeout.connect(self.batchProcessor)

        if not hasattr(self, 'resolutionEstimateTimer'):
            self.resolutionEstimateTimer = QTimer(self)
            self.resolutionEstimateTimer.setSingleShot(True)
            self.resolutionEstimateTimer.setInterval(1)
            self.resolutionEstimateTimer.timeout.connect(self.resolutionEstimate)

    def run(self):
        # self.start_threads()

        while not self.interrupt_measurement_called:
            time.sleep(1)

    def loadFile(self):
        filename, _ = QFileDialog.getOpenFileName(directory="./measurement")
        self.imageRaw = np.single(tif.imread(filename))
        self.imageRawShape = np.shape(self.imageRaw)
        self.imageSIM = np.zeros((self.imageRawShape[0]//7,self.imageRawShape[1],self.imageRawShape[2]))
        self.imageWF = np.zeros((self.imageRawShape[0] // 7, self.imageRawShape[1], self.imageRawShape[2]))

        self.raw2WideFieldImage()

        self.filetitle = Path(filename).stem
        self.filepath = os.path.dirname(filename)

        try:
            # get file name of txt file
            for file in os.listdir(self.filepath):
                if file.endswith(".txt"):
                    configFileName = os.path.join(self.filepath, file)

            configFile = open(configFileName, 'r')
            configSet = json.loads(configFile.read())

            self.h.ckx_in = np.asarray(configSet["kx"])
            self.h.cky_in = np.asarray(configSet["ky"])
            self.h.p_in = np.asarray(configSet["phase"])
            self.h.ampl_in = np.asarray(configSet["amplitude"])

            # set value
            self.ui.magnificationValue.setValue(configSet["magnification"])
            self.ui.naValue.setValue(configSet["NA"])
            self.ui.nValue.setValue(configSet["refractive index"])
            self.ui.wavelengthValue.setValue(configSet["wavelength"])
            self.ui.pixelsizeValue.setValue(configSet["pixelsize"])

            try:
                self.exposuretime = configSet["camera exposure time"]
            except:
                self.exposuretime = configSet["exposure time (s)"]

            try:
                self.laserpower = configSet["laser power (mW)"]
            except:
                self.laserpower = 0

            txtDisplay = "File name:\t {}\n" \
                         "Array size:\t {}\n" \
                         "Wavelength:\t {} um\n" \
                         "Exposure time:\t {:.3f} s\n" \
                         "Laser power:\t {} mW".format(self.filetitle, self.imageRawShape, \
                                                                configSet["wavelength"], \
                                                                self.exposuretime, self.laserpower)
            self.ui.fileInfo.setPlainText(txtDisplay)

        except:
            self.ui.fileInfo.setPlainText("No information about this measurement.")

        self.isFileLoad = True
        self.isUpdateImageViewer = True
        self.updateImageViewer()
        self.h.isCalibrated = False
        self.setReconstructor()
        self.h._allocate_arrays()

    def raw2WideFieldImage(self):
        for n_idx in range(self.imageRawShape[0]//7):
            self.imageWF[n_idx,:,:] = np.sum(self.imageRaw[n_idx*7:(n_idx+1)*7,:,:],axis=0)/7

        # print(self.imageWF.shape)

    # region Display Functions
    def rawImageSliderChanged(self):
        self.ui.rawImageSlider.setMinimum(0)
        self.ui.rawImageSlider.setMaximum(self.imageRaw.shape[0] - 1)

        self.imvRaw.setImage((self.imageRaw[int(self.ui.rawImageSlider.value()), :, :]).T, autoRange=False,
                             levels=(self.imageRawMin, self.imageRawMax))

        self.ui.rawImageNth.setText(str(self.ui.rawImageSlider.value() + 1))
        self.ui.rawImageNtotal.setText(str(len(self.imageRaw)))

    def wfImageSliderChanged(self):
        self.ui.wfImageSlider.setMinimum(0)
        self.ui.wfImageSlider.setMaximum(self.imageWF.shape[0] - 1)

        self.imvWF.setImage((self.imageWF[int(self.ui.wfImageSlider.value()), :, :]).T, autoRange=False,
                             levels=(self.imageWFMin, self.imageWFMax))

        self.ui.wfImageNth.setText(str(self.ui.wfImageSlider.value() + 1))
        self.ui.wfImageNtotal.setText(str(len(self.imageWF)))

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
        self.imageWFMax = np.amax(self.imageWF)
        self.imageWFMin = np.amin(self.imageWF)
        self.imageSIMMax = np.amax(self.imageSIM)
        self.rawImageSliderChanged()
        self.wfImageSliderChanged()
        self.simImageSliderChanged()

    # endregion

    ################    HexSIM  ################
    def resetHexSIM(self):
        if hasattr(self, 'h'):
            self.h.isCalibrated = False
            self.h._allocate_arrays()
            self.imageSIM = np.zeros(self.imageRawShape, dtype=np.uint16)
            self.updateImageViewer()

    def calibrationProcessor(self):
        print('Start calibrating...')
        self.processValue = 0

        startTime = time.time()

        if self.h.gpuenable:
            self.h.calibrate_cupy(self.imageRaw)
            self.processValue = 60
            self.imageSIM = self.h.reconstruct_cupy(self.imageRaw)

        elif not self.h.gpuenable:
            self.h.calibrate(self.imageRaw)
            self.processValue = 60
            self.imageSIM = self.h.reconstruct_rfftw(self.imageRaw)

        print('Calibration is processed in:', time.time() - startTime, 's')
        self.processValue = 90

        self.imageSIM = self.imageSIM[np.newaxis, :, :]
        self.isUpdateImageViewer = True

        self.processValue = 100

    def standardProcessor(self):
        self.processValue = 10

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
            self.calibrationProcessTimer.start()

        self.processValue = 90
        self.isUpdateImageViewer = True

        self.processValue = 100

    def batchProcessor(self):
        print('Start batch processing...')
        self.processValue = 10

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
            self.processValue = 80

        elif not self.h.isCalibrated:
            startTime = time.time()
            nStack = len(self.imageRaw)
            # calibrate & reconstruction
            if self.h.gpuenable:
                self.h.calibrate_cupy(self.imageRaw[int(nStack // 2):int(nStack // 2 + 7), :, :])
                self.processValue = 20
                if self.h.compact:
                    self.imageSIM = self.h.batchreconstructcompact_cupy(self.imageRaw)
                elif not self.h.compact:
                    self.imageSIM = self.h.batchreconstruct_cupy(self.imageRaw)
                self.processValue = 80

            elif not self.h.gpuenable:
                self.h.calibrate(self.imageRaw[int(nStack // 2):int(nStack // 2 + 7), :, :])
                self.processValue = 20
                if self.h.compact:
                    self.imageSIM = self.h.batchreconstructcompact(self.imageRaw)
                elif not self.h.compact:
                    self.imageSIM = self.h.batchreconstruct(self.imageRaw)
                self.processValue = 80
        print('Batch reconstruction finished', time.time() - startTime, 's')

        self.isUpdateImageViewer = True
        self.processValue = 100

    def streamReconstruction(self, newFrame, index):
        print(index)
        if self.h.gpuenable:
            self.imageSIM = self.h.reconstructframe_cupy(newFrame, index)
        elif not self.h.gpuenable:
            self.imageSIM = self.h.reconstructframe_rfftw(newFrame, index)

        self.imageSIM = self.imageSIM[np.newaxis, :, :]

    ############## Test #################################
    def standardSimuButtonPressed(self):
        if self.isFileLoad:
            if self.imageRaw.shape[0] == 7:
                print('standard processing')
                self.processValue = 0
                self.standardProcessTimer.start()
            else:
                print('Please input the 7-frame data set.')
        else:
            print('Image is not loaded.')
            self.ui.fileInfo.setPlainText('Image is not loaded.')

    def standardReconstructionUpdate(self):
        self.calibrationProcessTimer.start()

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
        if self.isFileLoad:
            self.batchProcessTimer.start()
            self.processValue = 0
        else:
            print('Image is not loaded.')
            self.ui.fileInfo.setPlainText('Image is not loaded.')

    def batchReconstructionUpdate(self):
        self.h.isCalibrated = False
        self.processValue = 10
        self.batchProcessTimer.start()

    def virtualRecording(self):
        filename, _ = QFileDialog.getOpenFileName(directory="./data")
        self.imageRawStack = np.single(tif.imread(filename))

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

    def getAcquisitionInterval(self):
        return float(self.ui.intervalTime.value())

    def setRefresh(self, refresh_period):
        self.display_update_period = refresh_period

    def saveMeasurements(self):
        t0 = time.time()
        timestamp = datetime.fromtimestamp(t0)
        timestamp = timestamp.strftime("%Y%m%d%H%M")
        pathname = self.filepath + '/reprocess'
        Path(pathname).mkdir(parents=True,exist_ok=True)
        simimagename = pathname + '/' + self.filetitle + timestamp + f'_reprocessed' + '.tif'
        txtname =      pathname + '/' + self.filetitle + timestamp + f'_reprocessed' + '.txt'
        tif.imwrite(simimagename, np.float32(self.imageSIM))
        print(type(self.imageSIM))

        savedictionary = {
            "exposure time (s)":self.exposuretime,
            "laser power (mW)": self.laserpower,
            # "z stepsize (um)":  self.
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

    def resolutionEstimate(self):
        try:
            pixelsizeWF = self.h.pixelsize / self.h.magnification
            ciWF = ImageDecorr(self.imageWF[self.ui.wfImageSlider.value(),:,:], square_crop=True,pixel_size=pixelsizeWF)
            optimWF, resWF = ciWF.compute_resolution()
            ciSIM = ImageDecorr(self.imageSIM[self.ui.simImageSlider.value(),:,:], square_crop=True,pixel_size=pixelsizeWF/2)
            optimSIM, resSIM = ciSIM.compute_resolution()
            txtDisplay = f"Wide field image resolution:\t {ciWF.resolution:.3f} um \
                  \nSIM image resolution:\t {ciSIM.resolution:.3f} um\n"
            self.ui.resolutionEstimation.setPlainText(txtDisplay)
        except:
            pass

    def resolutionEstimatePressed(self):
        self.resolutionEstimateTimer.start()

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

        self.ui.currentTable.setItem(2, 0, QTableWidgetItem(str(self.h.ckx[0]).lstrip('[').rstrip(']')))
        self.ui.currentTable.setItem(2, 1, QTableWidgetItem(str(self.h.ckx[1]).lstrip('[').rstrip(']')))
        self.ui.currentTable.setItem(2, 2, QTableWidgetItem(str(self.h.ckx[2]).lstrip('[').rstrip(']')))
        #
        self.ui.currentTable.setItem(3, 0, QTableWidgetItem(str(self.h.cky[0]).lstrip('[').rstrip(']')))
        self.ui.currentTable.setItem(3, 1, QTableWidgetItem(str(self.h.cky[1]).lstrip('[').rstrip(']')))
        self.ui.currentTable.setItem(3, 2, QTableWidgetItem(str(self.h.cky[2]).lstrip('[').rstrip(']')))
        #
        self.ui.currentTable.setItem(4, 0, QTableWidgetItem(str(self.h.p[0]).lstrip('[').rstrip(']')))
        self.ui.currentTable.setItem(4, 1, QTableWidgetItem(str(self.h.p[1]).lstrip('[').rstrip(']')))
        self.ui.currentTable.setItem(4, 2, QTableWidgetItem(str(self.h.p[2]).lstrip('[').rstrip(']')))
        #
        self.ui.currentTable.setItem(5, 0, QTableWidgetItem(str(self.h.ampl[0]).lstrip('[').rstrip(']')))
        self.ui.currentTable.setItem(5, 1, QTableWidgetItem(str(self.h.ampl[1]).lstrip('[').rstrip(']')))
        self.ui.currentTable.setItem(5, 2, QTableWidgetItem(str(self.h.ampl[2]).lstrip('[').rstrip(']')))

        # Table will fit the screen horizontally
        self.currentTable.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

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