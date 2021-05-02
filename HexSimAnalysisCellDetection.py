import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pyqtgraph as pg
import tifffile as tif
from PyQt5 import uic
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QWidget, QTableWidgetItem, \
    QHeaderView,QLabel
from PyQt5.QtGui import QImage,QPixmap
from ScopeFoundry import Measurement, h5_io
from ScopeFoundry.helper_funcs import sibling_path, load_qt_ui_file

from HexSimProcessor.SIM_processing.hexSimProcessor import HexSimProcessor
from image_decorr import ImageDecorr
from StackImageViewer import StackImageViewer
from ImageSegmentation import ImageSegmentation

class HexSimAnalysisCellDetection(Measurement):
    name = 'HexSIM_Analysis_cell_detection'

    def setup(self):
        # load ui file
        self.ui_filename = sibling_path(__file__, "hexsim_analysis_cell_detection.ui")
        self.ui = load_qt_ui_file(self.ui_filename)

    def setup_figure(self):
        # connect ui widgets to measurement/hardware settings or functionss
        # combo lists setting
        self.roiSizeList = [128,256,512,1024]
        self.ui.roiSizeCombo.addItems(map(str,self.roiSizeList))
        self.ui.roiSizeCombo.setCurrentIndex(1)

        self.ui.imageTabs.setCurrentIndex(1)
        # Set up pyqtgraph graph_layout in the UI
        self.imvRaw = pg.ImageView()
        self.imvRaw.ui.roiBtn.hide()
        self.imvRaw.ui.menuBtn.hide()

        self.imvSIM = pg.ImageView()
        self.imvSIM.ui.roiBtn.hide()
        self.imvSIM.ui.menuBtn.hide()

        self.imvWF = pg.ImageView()
        self.imvWF.ui.roiBtn.hide()
        self.imvWF.ui.menuBtn.hide()
        
        self.imvSeg = QLabel()
        self.imvSeg.setScaledContents(True)

        self.ui.rawImageLayout.addWidget(self.imvRaw)
        self.ui.simImageLayout.addWidget(self.imvSIM)
        self.ui.wfImageLayout.addWidget(self.imvWF)
        self.ui.cellsLayout.addWidget(self.imvSeg)

        # Image initialization
        self.imageRaw = np.zeros((1, 512, 512), dtype=np.uint16)
        self.imageSIM = np.zeros((1, 1024,1024), dtype=np.uint16)
        self.imageWF = np.zeros((1, 512, 512), dtype=np.uint16)
        self.imageSeg = np.zeros((1, 512, 512), dtype=np.uint16)

        self.imvRaw.setImage((self.imageRaw[0, :, :]).T, autoRange=False, autoLevels=True, autoHistogramRange=True)
        self.imvWF.setImage((self.imageWF[0, :, :]).T, autoRange=False, autoLevels=True, autoHistogramRange=True)
        self.imvSIM.setImage((self.imageSIM[0, :, :]).T, autoRange=False, autoLevels=True, autoHistogramRange=True)

        self.imageSetsTemp = [np.zeros((2,256,256)),np.zeros((2,256,256))]

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

        self.ui.wfViewerButton.clicked.connect(self.showCellWFImageSets)
        self.ui.simViewerButton.clicked.connect(self.showCellSIMImageSets)

        # Operation
        self.ui.loadFileButton.clicked.connect(self.loadFile)
        self.ui.calibrationLoad.clicked.connect(self.loadCalibrationResults)
        self.ui.findCellButton.clicked.connect(self.findCell)
        self.ui.calibrationButton.clicked.connect(self.calibrationButtonPressed)
        self.ui.resetMeasureButton.clicked.connect(self.resetHexSIM)
        self.ui.calibrationResult.clicked.connect(self.showMessageWindow)
        self.ui.reconstructionButton.clicked.connect(self.reconstructionButtonPressed)
        self.ui.resolutionEstimateButton.clicked.connect(self.resolutionEstimatePressed)
        self.ui.calibrationSave.clicked.connect(self.saveMeasurements)

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

        if self.isCalibrated:
            self.ui.roiCalibrationProgress.setValue(100)
            self.ui.roiCalibrationProgress.setFormat('Calibrated')
            self.ui.reconstructionButton.setEnabled(True)
            self.ui.calibrationResult.setEnabled(True)
        else:
            self.ui.roiCalibrationProgress.setValue(0)
            self.ui.roiCalibrationProgress.setFormat('Uncalibrated')
            self.ui.reconstructionButton.setEnabled(False)
            self.ui.calibrationResult.setEnabled(False)


    def start_timers(self):
        if not hasattr(self, 'h'):
            self.h = HexSimProcessor()  # create reconstruction object
            self.h.opencv = False
            self.setReconstructor()


        if not hasattr(self, 'calibrationProcessTimerROI'):
            self.calibrationProcessTimerROI = QTimer(self)
            self.calibrationProcessTimerROI.setSingleShot(True)
            self.calibrationProcessTimerROI.setInterval(1)
            self.calibrationProcessTimerROI.timeout.connect(self.calibrationProcessorROI)

        if not hasattr(self, 'standardProcessTimerROI'):
            self.standardProcessTimerROI = QTimer(self)
            self.standardProcessTimerROI.setSingleShot(True)
            self.standardProcessTimerROI.setInterval(1)
            self.standardProcessTimerROI.timeout.connect(self.standardProcessorROI)

        if not hasattr(self, 'batchProcessTimerROI'):
            self.batchProcessTimerROI = QTimer(self)
            self.batchProcessTimerROI.setSingleShot(True)
            self.batchProcessTimerROI.setInterval(1)
            self.batchProcessTimerROI.timeout.connect(self.batchProcessorROI)

        if not hasattr(self, 'resolutionEstimateTimer'):
            self.resolutionEstimateTimer = QTimer(self)
            self.resolutionEstimateTimer.setSingleShot(True)
            self.resolutionEstimateTimer.setInterval(1)
            self.resolutionEstimateTimer.timeout.connect(self.resolutionEstimate)

    def pre_run(self):
        # Message window
        self.messageWindow = None
        self.wfImageViewer = StackImageViewer(image_sets=self.imageSetsTemp,set_levels=[1,0.8],title='Wide field images')
        self.simImageViewer = StackImageViewer(image_sets=self.imageSetsTemp,set_levels=[0,0.8],title='SIM images')

        self.settings.New('refresh_period', dtype=float, unit='s', spinbox_decimals=4, initial=0.02,
                          hardware_set_func=self.setRefresh, vmin=0)

        self.display_update_period = self.settings.refresh_period.val

        # Initialize condition labels
        self.isUpdateImageViewer = False
        self.showCalibrationResult = False
        self.isCalibrated = False
        self.isCalibrationSaved = False
        self.isGpuenable = True     # using GPU for accelerating
        self.isCompact = True       # using compact mode in batch reconstruction to save memory
        self.isFileLoad = False
        self.isFindCarrier = True

        self.kx_input = np.zeros((3, 1), dtype=np.single)
        self.ky_input = np.zeros((3, 1), dtype=np.single)
        self.p_input = np.zeros((3, 1), dtype=np.single)
        self.ampl_input = np.zeros((3, 1), dtype=np.single)

        self.start_timers()

    def run(self):
        while not self.interrupt_measurement_called:
            time.sleep(1)

    def roiSize(self):
        return int(self.ui.roiSizeCombo.currentText())

    def minCellSize(self):
        return int(self.ui.minCellSizeInput.value())

    def image8bit_normalized(self,image):
        level_min = np.amin(image)
        level_max = np.amax(image)
        img_thres = np.clip(image, level_min, level_max)
        return ((img_thres - level_min + 1) / (level_max - level_min + 1) * 255).astype('uint8')

    def loadFile(self):
        try:
            filename, _ = QFileDialog.getOpenFileName(directory="./measurement")
            self.imageRaw = np.single(tif.imread(filename))
            self.imageRawShape = np.shape(self.imageRaw)

            self.imageSIM = np.zeros((self.imageRawShape[0]//7,self.imageRawShape[1],self.imageRawShape[2]))
            self.imageWF = np.zeros((self.imageRawShape[0] // 7, self.imageRawShape[1], self.imageRawShape[2]))

            self.oSegment = ImageSegmentation(self.imageRaw,self.roiSize()//2,self.minCellSize())
            self.imageWF = self.raw2WideFieldImage(self.imageRaw)
            self.filetitle = Path(filename).stem
            self.filepath = os.path.dirname(filename)
            self.isFileLoad = True

            try:
                # get file name of txt file
                for file in os.listdir(self.filepath):
                    if file.endswith(".txt"):
                        configFileName = os.path.join(self.filepath, file)

                configFile = open(configFileName, 'r')
                configSet = json.loads(configFile.read())

                self.kx_input = np.asarray(configSet["kx"])
                self.ky_input = np.asarray(configSet["ky"])
                self.p_input = np.asarray(configSet["phase"])
                self.ampl_input = np.asarray(configSet["amplitude"])

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

        except AssertionError as error:
            print(error)
            self.isFileLoad = False

        if self.isFileLoad:
            self.ui.findCellButton.setEnabled(True)
            self.ui.imageTabs.setCurrentIndex(1)
            self.isUpdateImageViewer = True
            self.isCalibrated = False
            self.updateImageViewer()
            self.setReconstructor()
            self.h._allocate_arrays()
        else:
            print("File is not loaded.")

    def findCell(self):
        self.oSegment.min_cell_size = self.minCellSize()
        self.oSegment.roi_half_side = self.roiSize()//2
        self.oSegment.find_cell()
        self.imageRawSets = self.oSegment.roi_creation()
        self.imageWFSets = []# initialize the image sets
        displayed_image = self.oSegment.draw_contours_on_image(self.oSegment.image8bit)
        self.imageRawROILabled = QImage(displayed_image.data, displayed_image.shape[1], displayed_image.shape[0], QImage.Format_RGB888)
        self.imvSeg.setPixmap(QPixmap.fromImage(self.imageRawROILabled))
        self.numSets = len(self.imageRawSets)
        self.ui.cellNumber.setValue(self.numSets)

        self.ui.cellPickCombo.clear()
        self.ui.cellPickCombo.addItems(map(str,np.arange(self.numSets)))
        self.ui.cellPickCombo.setCurrentIndex(0)
        self.ui.calibrationButton.setEnabled(True)
        self.ui.imageTabs.setCurrentIndex(2)
        for idx in range(self.numSets):
            self.imageWFSets.append(self.raw2WideFieldImage(self.imageRawSets[idx]))

        self.wfImageViewer.image_sets = self.imageWFSets
        self.wfImageViewer.update()

    def raw2WideFieldImage(self,rawImages):
        wfImages = np.zeros((rawImages.shape[0]//7,rawImages.shape[1],rawImages.shape[2]))
        for idx in range(rawImages.shape[0]//7):
            wfImages[idx,:,:] = np.sum(rawImages[idx*7:(idx+1)*7,:,:],axis=0)/7

        return wfImages

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


    ############## Operation #################################
    def calibrationButtonPressed(self):
        if self.isFileLoad:
            self.imageRawROI = self.imageRawSets[self.ui.cellPickCombo.currentIndex()]
            self.calibrationProcessTimerROI.start()
            self.ui.reconstructionButton.setEnabled(True)
        else:
            print('Image is not loaded.')
            self.ui.fileInfo.setPlainText('Image is not loaded.')

    def reconstructionButtonPressed(self):
        if len(self.imageRawROI)>7:
            self.batchProcessTimerROI.start()
        else:
            self.standardProcessTimerROI.start()


    def resetHexSIM(self):
        if hasattr(self, 'h'):
            self.isCalibrated = False
            self.h._allocate_arrays()
            self.updateImageViewer()


    def setReconstructor(self):
        if not hasattr(self, 'h'):
            print('Activate the component.')
        else:
            self.isCompact = self.ui.compactCheck.isChecked()
            self.isGpuenable = self.ui.gpuCheck.isChecked()
            self.isFindCarrier = not self.ui.useLoadedResultsCheck.isChecked()

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

    def getAcquisitionInterval(self):
        return float(self.ui.intervalTime.value())

    def setRefresh(self, refresh_period):
        self.display_update_period = refresh_period

    def saveMeasurements(self):
        t0 = time.time()
        timestamp = datetime.fromtimestamp(t0)
        timestamp = timestamp.strftime("%Y%m%d%H%M")
        pathname = self.filepath + '/segmented_analysis'
        Path(pathname).mkdir(parents=True,exist_ok=True)

        for idx in range(self.numSets):
            suffix = str(idx).zfill(3)
            simimagename = pathname + '/' + self.filetitle + timestamp + f'_segmented_sim' + '_' + suffix + '.tif'
            wfimagename = pathname + '/' + self.filetitle + timestamp + f'_segmented_widefield' + '_' + suffix + '.tif'
            tif.imwrite(simimagename, np.single(self.imageSIMSets[idx]))
            tif.imwrite(wfimagename,np.uint16(self.imageWFSets[idx]))

        txtname =      pathname + '/' + self.filetitle + timestamp + f'_configuration' + '.txt'
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
            "kx":               self.h.kx,
            "ky":               self.h.ky,
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

    def resolutionEstimate(self):
        try:
            pixelsizeWF = self.h.pixelsize / self.h.magnification
            imageWF_temp = self.imageWFSets[self.wfImageViewer.ui.cellCombo.currentIndex()]
            ciWF = ImageDecorr(imageWF_temp[self.wfImageViewer.ui.imgSlider.value(),:,:], square_crop=True,pixel_size=pixelsizeWF)
            optimWF, resWF = ciWF.compute_resolution()
            imageSIM_temp = self.imageSIMSets[self.simImageViewer.ui.cellCombo.currentIndex()]
            ciSIM = ImageDecorr(imageSIM_temp[self.simImageViewer.ui.imgSlider.value(),:,:], square_crop=True,pixel_size=pixelsizeWF/2)
            optimSIM, resSIM = ciSIM.compute_resolution()
            txtDisplay = f"Cell {self.wfImageViewer.ui.cellCombo.currentIndex()}" \
                         f"\nWide field image resolution:\t {ciWF.resolution:.3f} um \
                  \nSIM image resolution:\t {ciSIM.resolution:.3f} um\n"
            self.ui.resolutionEstimation.setPlainText(txtDisplay)
        except:
            pass


    def resolutionEstimatePressed(self):
        self.resolutionEstimateTimer.start()

    def loadCalibrationResults(self):
        try:
            filename, _ = QFileDialog.getOpenFileName(caption="Open file", directory="./measurement", filter="Text files (*.txt)")
            file = open(filename,'r')
            loadResults = json.loads(file.read())
            self.kx_input = np.asarray(loadResults["kx"])
            self.ky_input = np.asarray(loadResults["ky"])
            print("Calibration results are loaded.")
        except:
            print("Calibration results are not loaded.")

    def showMessageWindow(self):
        try:
            self.messageWindow = MessageWindow(self.h, self.kx_input, self.ky_input)
            self.messageWindow.show()
        except AssertionError as error:
            print(error)

    def showCellWFImageSets(self):
        if not self.wfImageViewer.isVisible():
            self.wfImageViewer.show()

    def showCellSIMImageSets(self):
        self.simImageViewer.image_sets = self.imageSIMSets
        self.simImageViewer.update()
        if not self.simImageViewer.isVisible():
            self.simImageViewer.show()

    # Processors

    def calibrationProcessorROI(self):
        print('Start calibrating...')
        startTime = time.time()

        if self.isGpuenable:
            self.h.calibrate_cupy(self.imageRawROI, self.isFindCarrier)

        elif not self.isGpuenable:
            self.h.calibrate(self.imageRawROI, self.isFindCarrier)

        print('Calibration is processed in:', time.time() - startTime, 's')

        self.isUpdateImageViewer = True
        self.isCalibrated = True



    def standardProcessorROI(self):

        if self.isCalibrated:
            print('Start standard processing...')
            startTime = time.time()
            self.imageSIMSets = []

            for idx in range(self.numSets):
                self.imageRawROI = self.imageRawSets[idx]

                if self.isGpuenable:
                    self.imageSIMROI = self.h.reconstruct_cupy(self.imageRawROI)
                elif not self.isGpuenable:
                    self.imageSIMROI = self.h.reconstruct_rfftw(self.imageRawROI)

                self.imageSIMROI = self.imageSIMROI[np.newaxis, :, :]
                self.imageSIMSets.append(self.imageSIMROI)

            print('One SIM image set is processed in:', time.time() - startTime, 's')

        self.isUpdateImageViewer = True
        self.showCellSIMImageSets()


    def batchProcessorROI(self):
        if self.isCalibrated:
            print('Start batch processing...')
            startTime = time.time()
            self.imageSIMSets = []
            # Batch reconstruction
            for idx in range(self.numSets):
                self.imageRawROI = self.imageRawSets[idx]

                if self.isGpuenable:
                    if self.isCompact:
                        self.imageSIMROI = self.h.batchreconstructcompact_cupy(self.imageRawROI)
                    elif not self.isCompact:
                        self.imageSIMROI = self.h.batchreconstruct_cupy(self.imageRawROI)

                elif not self.isGpuenable:
                    if self.isCompact:
                        self.imageSIMROI = self.h.batchreconstructcompact(self.imageRawROI)
                    elif not self.isCompactompact:
                        self.imageSIMROI = self.h.batchreconstruct(self.imageRawROI)

                self.imageSIMSets.append(self.imageSIMROI)


        print('Batch reconstruction is processed in', time.time() - startTime, 's')

        self.isUpdateImageViewer = True
        self.showCellSIMImageSets()

class MessageWindow(QWidget):

    """
    This window display the Winier filter and other debug data
    """

    def __init__(self, h, kx, ky):
        super().__init__()
        self.ui = uic.loadUi('calibration_results.ui',self)
        self.h = h
        self.kx = kx
        self.ky = ky
        self.setWindowTitle('Calibration results')
        self.showCurrentTable()
        self.showWienerFilter()
        self.ui.wienerfilterLayout.addWidget(self.wienerfilterWidget)

    def showCurrentTable(self):

        self.ui.currentTable.setItem(0, 0, QTableWidgetItem(str(self.kx[0]).lstrip('[').rstrip(']')))
        self.ui.currentTable.setItem(0, 1, QTableWidgetItem(str(self.kx[1]).lstrip('[').rstrip(']')))
        self.ui.currentTable.setItem(0, 2, QTableWidgetItem(str(self.kx[2]).lstrip('[').rstrip(']')))
        #
        self.ui.currentTable.setItem(1, 0, QTableWidgetItem(str(self.ky[0]).lstrip('[').rstrip(']')))
        self.ui.currentTable.setItem(1, 1, QTableWidgetItem(str(self.ky[1]).lstrip('[').rstrip(']')))
        self.ui.currentTable.setItem(1, 2, QTableWidgetItem(str(self.ky[2]).lstrip('[').rstrip(']')))

        self.ui.currentTable.setItem(2, 0, QTableWidgetItem(str(self.h.kx[0]).lstrip('[').rstrip(']')))
        self.ui.currentTable.setItem(2, 1, QTableWidgetItem(str(self.h.kx[1]).lstrip('[').rstrip(']')))
        self.ui.currentTable.setItem(2, 2, QTableWidgetItem(str(self.h.kx[2]).lstrip('[').rstrip(']')))
        #
        self.ui.currentTable.setItem(3, 0, QTableWidgetItem(str(self.h.ky[0]).lstrip('[').rstrip(']')))
        self.ui.currentTable.setItem(3, 1, QTableWidgetItem(str(self.h.ky[1]).lstrip('[').rstrip(']')))
        self.ui.currentTable.setItem(3, 2, QTableWidgetItem(str(self.h.ky[2]).lstrip('[').rstrip(']')))
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
        self.wienerfilterWidget = pg.ImageView()
        self.wienerfilterWidget.aspectRatioMode = Qt.KeepAspectRatio
        self.wienerfilterWidget.ui.roiBtn.hide()
        self.wienerfilterWidget.ui.menuBtn.hide()
        self.wienerfilterWidget.ui.histogram.hide()
        self.wienerfilterWidget.setImage(self.h.wienerfilter, autoRange=True, autoLevels=True)
        self.wienerfilterWidget.adjustSize()

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

    # def standardSimuButtonPressed(self):
    #     if self.isFileLoad:
    #         if self.imageRaw.shape[0] == 7:
    #             print('standard processing')
    #             self.standardProcessTimer.start()
    #         else:
    #             print('Please input the 7-frame data set.')
    #     else:
    #         print('Image is not loaded.')
    #         self.ui.fileInfo.setPlainText('Image is not loaded.')

    # def standardReconstructionUpdate(self):
    #     self.calibrationProcessTimer.start()

    # def standardReconstruction(self):
    #     # calibrate & reconstruction
    #     if self.isGpuenable:
    #         self.h.calibrate_cupy(self.imageRaw, self.isFindCarrier)
    #         self.isCalibrated = True
    #         self.imageSIM = self.h.reconstruct_cupy(self.imageRaw)
    #
    #     elif not self.isGpuenable:
    #         self.h.calibrate(self.imageRaw, self.isFindCarrier)
    #         self.isCalibrated = True
    #         self.imageSIM = self.h.reconstruct_rfftw(self.imageRaw)
    #
    #     self.imageSIM = self.imageSIM[np.newaxis, :, :]
    #     self.isUpdateImageViewer = True
    #     print('One SIM image is processed.')
    #
    # def batchSimuButtonPressed(self):
    #     if self.isFileLoad:
    #         self.batchProcessTimer.start()
    #     else:
    #         print('Image is not loaded.')
    #         self.ui.fileInfo.setPlainText('Image is not loaded.')
    #
    # def batchReconstructionUpdate(self):
    #     if self.isFileLoad:
    #         self.isCalibrated = False
    #         self.batchProcessTimer.start()
    #     else:
    #         print('Image is not loaded.')
    #         self.ui.fileInfo.setPlainText('Image is not loaded.')

    # if not hasattr(self, 'calibrationProcessTimer'):
    #     self.calibrationProcessTimer = QTimer(self)
    #     self.calibrationProcessTimer.setSingleShot(True)
    #     self.calibrationProcessTimer.setInterval(1)
    #     self.calibrationProcessTimer.timeout.connect(self.calibrationProcessor)
    #
    # if not hasattr(self, 'standardProcessTimer'):
    #     self.standardProcessTimer = QTimer(self)
    #     self.standardProcessTimer.setSingleShot(True)
    #     self.standardProcessTimer.setInterval(1)
    #     self.standardProcessTimer.timeout.connect(self.standardProcessor)
    #
    # if not hasattr(self, 'batchProcessTimer'):
    #     self.batchProcessTimer = QTimer(self)
    #     self.batchProcessTimer.setSingleShot(True)
    #     self.batchProcessTimer.setInterval(1)
    #     self.batchProcessTimer.timeout.connect(self.batchProcessor)

    # def standardProcessor(self):
    #
    #     if self.isCalibrated:
    #         print('Start standard processing...')
    #         startTime = time.time()
    #
    #         if self.isGpuenable:
    #             self.imageSIM = self.h.reconstruct_cupy(self.imageRaw)
    #
    #         elif not self.isGpuenable:
    #             self.imageSIM = self.h.reconstruct_rfftw(self.imageRaw)
    #
    #         print('One SIM image is processed in:', time.time() - startTime, 's')
    #         self.imageSIM = self.imageSIM[np.newaxis, :, :]
    #
    #     elif not self.isCalibrated:
    #         self.calibrationProcessTimer.start()
    #
    #     self.isUpdateImageViewer = True

    # elif not self.isCalibrated:
    #     startTime = time.time()
    #     nStack = len(self.imageRawROI)
    #     # calibrate & reconstruction
    #     if self.isGpuenable:
    #         self.h.calibrate_cupy(self.imageRaw[int(nStack // 2):int(nStack // 2 + 7), :, :], self.isFindCarrier)
    #         self.isCalibrated = True
    #         if self.isCompact:
    #             self.imageSIM = self.h.batchreconstructcompact_cupy(self.imageRaw)
    #         elif not self.isCompact:
    #             self.imageSIM = self.h.batchreconstruct_cupy(self.imageRaw)
    #
    #     elif not self.isGpuenable:
    #         self.h.calibrate(self.imageRaw[int(nStack // 2):int(nStack // 2 + 7), :, :], self.isFindCarrier)
    #         self.isCalibrated = True
    #         if self.isCompact:
    #             self.imageSIM = self.h.batchreconstructcompact(self.imageRaw)
    #         elif not self.isCompact:
    #             self.imageSIM = self.h.batchreconstruct(self.imageRaw)

    # def batchProcessor(self):
    #     print('Start batch processing...')
    #
    #     if self.isCalibrated:
    #         startTime = time.time()
    #         # Batch reconstruction
    #         if self.isGpuenable:
    #             if self.isCompact:
    #                 self.imageSIM = self.h.batchreconstructcompact_cupy(self.imageRaw)
    #             elif not self.isCompact:
    #                 self.imageSIM = self.h.batchreconstruct_cupy(self.imageRaw)
    #
    #         elif not self.isGpuenable:
    #             if self.isCompact:
    #                 self.imageSIM = self.h.batchreconstructcompact(self.imageRaw)
    #             elif not self.isCompactompact:
    #                 self.imageSIM = self.h.batchreconstruct(self.imageRaw)
    #
    #     elif not self.isCalibrated:
    #         startTime = time.time()
    #         nStack = len(self.imageRaw)
    #         # calibrate & reconstruction
    #         if self.isGpuenable:
    #             self.h.calibrate_cupy(self.imageRaw[int(nStack // 2):int(nStack // 2 + 7), :, :], self.isFindCarrier)
    #             self.isCalibrated = True
    #             if self.isCompact:
    #                 self.imageSIM = self.h.batchreconstructcompact_cupy(self.imageRaw)
    #             elif not self.isCompact:
    #                 self.imageSIM = self.h.batchreconstruct_cupy(self.imageRaw)
    #
    #         elif not self.isGpuenable:
    #             self.h.calibrate(self.imageRaw[int(nStack // 2):int(nStack // 2 + 7), :, :], self.isFindCarrier)
    #             self.isCalibrated = True
    #             if self.isCompact:
    #                 self.imageSIM = self.h.batchreconstructcompact(self.imageRaw)
    #             elif not self.isCompact:
    #                 self.imageSIM = self.h.batchreconstruct(self.imageRaw)
    #
    #     print('Batch reconstruction finished', time.time() - startTime, 's')
    #
    #     self.isUpdateImageViewer = True

    # def calibrationProcessor(self):
    #     print('Start calibrating...')
    #
    #     startTime = time.time()
    #
    #     if self.isGpuenable:
    #         self.h.calibrate_cupy(self.imageRaw, self.isFindCarrier)
    #         self.imageSIM = self.h.reconstruct_cupy(self.imageRaw)
    #
    #     elif not self.isGpuenable:
    #         self.h.calibrate(self.imageRaw, self.isFindCarrier)
    #         self.imageSIM = self.h.reconstruct_rfftw(self.imageRaw)
    #
    #     print('Calibration is processed in:', time.time() - startTime, 's')
    #
    #     self.imageSIM = self.imageSIM[np.newaxis, :, :]
    #     self.isUpdateImageViewer = True
    #     self.isCalibrated = True
