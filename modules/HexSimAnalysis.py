"""
Intro:  GUI for HexSIM data analysis.
        The GUI is based on the framework of ScopeFoundry.
        The data should be acquired by hexSimMeasurement component.
Author: Hai Gong
Email:  h.gong@imperial.ac.uk
Time:   July 2021
Address:Imperial College London
"""

import os,time,h5py
import numpy as np
import cupy as cp
import pyqtgraph as pg
import tifffile as tif

from pathlib import Path
from PyQt5.QtWidgets import QFileDialog
from ScopeFoundry import Measurement, h5_io
from ScopeFoundry.helper_funcs import load_qt_ui_file

from HexSimProcessor.SIM_processing.hexSimProcessor import HexSimProcessor
from utils.MessageWindow import CalibrationResults
from utils.StackImageViewer import StackImageViewer, list_equal
from utils.ImageSegmentation import ImageSegmentation
from utils.image_decorr import ImageDecorr


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
    """Function decorator to update display at the end of the execution
    To avoid conflicts with QtObjects, it assumes that the method takes no arguments except self
    """

    def inner(cls):
        result = function(cls)
        cls.update_display()
        return result

    inner.__name__ = function.__name__
    return inner

def add_run(function):
    """
    Function decorator to run the run QThread at the end of the execution
    """
    def inner(cls):
        result = function(cls)
        # cls.run()
        cls.start()
        return result

    inner.__name__ = function.__name__
    return inner

class HexSimAnalysis(Measurement):
    ''' This HexSim anaysis works on H5 file.'''
    name = 'HexSIM_Analyse'

    def setup(self):
        # load ui file
        self.ui = load_qt_ui_file(".\\ui\\hexsim_analyse.ui")
        self.settings.New('debug', dtype=bool, initial=False,   hardware_set_func = self.setReconstructor)
        self.settings.New('cleanup', dtype=bool, initial=False, hardware_set_func = self.setReconstructor)
        self.settings.New('gpu', dtype=bool, initial=True,      hardware_set_func = self.setReconstructor)
        self.settings.New('compact', dtype=bool, initial=True,  hardware_set_func = self.setReconstructor)
        self.settings.New('axial', dtype=bool, initial=False,   hardware_set_func = self.setReconstructor)
        self.settings.New('usemodulation', dtype=bool, initial=True, hardware_set_func = self.setReconstructor)
        self.settings.New('NA', dtype=float, initial=1.10, spinbox_decimals=2, hardware_set_func = self.setReconstructor)
        self.settings.New('n', dtype=float, initial=1.33, spinbox_decimals=2,  hardware_set_func = self.setReconstructor)
        self.settings.New('magnification', dtype=int, initial=60, spinbox_decimals=2, hardware_set_func = self.setReconstructor)
        self.settings.New('pixelsize', dtype=float, initial=6.50, spinbox_decimals=3, hardware_set_func = self.setReconstructor)
        self.settings.New('wavelength', dtype=float, initial=0.523, spinbox_decimals=3, hardware_set_func = self.setReconstructor)
        self.settings.New('alpha', dtype=float, initial=0.500, spinbox_decimals=3, description='0th att width',
                          hardware_set_func = self.setReconstructor)
        self.settings.New('beta', dtype=float, initial=0.990, spinbox_decimals=3, description='0th width',
                          hardware_set_func = self.setReconstructor)
        self.settings.New('w', dtype=float, initial=0.500, spinbox_decimals=2, description='wiener parameter',
                          hardware_set_func = self.setReconstructor)
        self.settings.New('eta', dtype=float, initial=0.70, spinbox_decimals=2,
                          description='must be smaller than the sources radius normalized on the pupil size',
                          hardware_set_func = self.setReconstructor)
        self.settings.New('find_carrier', dtype=bool, initial=True, hardware_set_func = self.setReconstructor)
        self.settings.New('otf_model',dtype=str, initial = 'none', choices=["none","exp","sph"],
                          hardware_set_func = self.setReconstructor)
        self.settings.New('segment_method',dtype=str, initial = 'watershed', choices=["simple","watershed"])
        self.settings.New('watershed_threshold', dtype=float, initial=0.50, spinbox_decimals=2)

        # initialize condition lables
        self.isUpdateImageViewer = False
        self.isCalibrated = False
        self.isGpuenable = True  # using GPU for accelerating
        self.isCompact = True  # using compact mode in batch reconstruction to save memory
        self.isFindCarrier = True

        self.action = None # the action in run()
        self.selected_dir = None

        self.numSets = 0
        self.kx_full = np.zeros((3, 1), dtype=np.single) # frequency of full field of view
        self.ky_full = np.zeros((3, 1), dtype=np.single)

        # image initialization: it can be an image or a set of images
        v = h = 256
        array_3d = np.zeros((7, v, h), dtype=np.uint16)
        array_2d = np.zeros((v, h), dtype=np.uint16)
        # left
        self.imageRAW     = [array_3d, array_3d]
        self.imageAVG     = array_2d
        self.imageSTD     = array_2d
        self.imageRAW_ROI = array_3d
        self.imageAVG_ROI = array_2d
        self.imageSTD_ROI = array_2d

        # right
        self.imageSIM = array_2d
        self.imageSIM_store = [array_3d, array_3d]
        self.imageSIM_ROI = array_2d
        self.wiener_Full  = array_2d
        self.wiener_ROI   = array_2d

        self.start_sim_processor()

    def setup_figure(self):
        # image viewers
        self.imvRAW     = StackImageViewer(image_sets=self.imageRAW, set_levels=[1, 1])
        self.imvAVG     = StackImageViewer(image_sets=self.imageAVG, set_levels=[1, 1], combo_visbile=False)
        self.imvSTD     = StackImageViewer(image_sets=self.imageSTD, set_levels=[1,1], combo_visbile=False)
        self.imvRAW_ROI = StackImageViewer(image_sets=self.imageRAW_ROI, set_levels=[1, 1])
        self.imvAVG_ROI = StackImageViewer(image_sets=self.imageAVG_ROI, set_levels=[1, 1])
        self.imvSTD_ROI = StackImageViewer(image_sets=self.imageSTD_ROI, set_levels=[1, 1])

        self.imvSIM     = StackImageViewer(image_sets=self.imageSIM, set_levels=[0, 0.8], combo_visbile=False)
        self.imvSIM_ROI = StackImageViewer(image_sets=self.imageSIM_ROI, set_levels=[0, 0.8])
        self.imvCalibration = CalibrationResults(self.h)
        self.imvWiener_ROI = StackImageViewer(image_sets=self.wiener_ROI, set_levels=[1,1])
        self.imvWiener_ROI.imv.ui.histogram.hide()

        # combo lists setting: size of roi
        self.roiRect = [] # list of roi rectangular
        self.roiSizeList = [128,200,256,512,1024]
        self.ui.roiSizeCombo.addItems(map(str,self.roiSizeList))
        self.ui.roiSizeCombo.setCurrentIndex(1)

        # add image viewers to ui
        self.ui.rawImageLayout.addWidget(self.imvRAW)
        self.ui.wfImageLayout.addWidget(self.imvAVG)
        self.ui.stdImageLayout.addWidget(self.imvSTD)
        self.ui.roiWFLayout.addWidget(self.imvAVG_ROI)
        self.ui.roiSTDLayout.addWidget(self.imvSTD_ROI)
        self.ui.roiRAWLayout.addWidget(self.imvRAW_ROI)
        self.ui.simImageLayout.addWidget(self.imvSIM)
        self.ui.roiSIMLayout.addWidget(self.imvSIM_ROI)
        self.ui.calibrationResultLayout.addWidget(self.imvCalibration)
        self.ui.roiWienerfilterLayout.addWidget(self.imvWiener_ROI)

        # connect the measurement widgets to ui widgets
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
        self.settings.segment_method.connect_to_widget(self.ui.segmentMethod)
        self.settings.watershed_threshold.connect_to_widget(self.ui.watershedThreshold)

        # Operation
        self.ui.loadMeasurementButton.clicked.connect(self.loadMeasurement)
        self.ui.calibrationButton.clicked.connect(self.calibrationPressed)
        self.ui.findCellButton.clicked.connect(self.findCell)
        self.ui.reconstructionButton.clicked.connect(self.reconstructionPressed)
        self.ui.roiProcessButton.clicked.connect(self.roiprocessPressed)
        self.ui.loadCalibrationButton.clicked.connect(self.loadCalibrationResults)
        self.ui.resetButton.clicked.connect(self.resetHexSIM)
        self.ui.saveButton.clicked.connect(self.saveMeasurements)
        self.ui.resolutionButton.clicked.connect(self.resolutionEstimatePressed)
        self.ui.browseFolderButton.clicked.connect(self.browseFolder)
        self.ui.batchFileProcessButton.clicked.connect(self.batchFileProcessPressed)
        self.ui.stopFileProcessButton.clicked.connect(self.interrupt)
        self.ui.channelSwitch.valueChanged.connect(self.channelSwitch)
        # self.imvRAW.ui.cellCombo.currentIndexChanged.connect(self.channelChanged)

    def update_display(self):
        if self.isUpdateImageViewer:
            self.updateImageViewer()
            self.isUpdateImageViewer = False

        if self.isCalibrated:
            self.ui.calibrationProgress.setValue(100)
            self.ui.calibrationProgress.setFormat('Calibrated')
        else:
            self.ui.calibrationProgress.setValue(0)
            self.ui.calibrationProgress.setFormat('Uncalibrated')

        self.ui.cellNumber.setValue(self.numSets)

    def run(self):
        if self.action is not None:
            if self.action == 'calibration':
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
                # time.sleep(0.5)
                self.ui.simTab.setCurrentIndex(1)

            elif self.action == 'resolution':
                self.resolutionEstimate()

            elif self.action == 'batch_file':
                self.batchFileProcess()

            self.isUpdateImageViewer = True
            self.action = None
            self.update_display()
        else:
            pass

# functions for HexSIM
    def start_sim_processor(self):
        self.isCalibrated = False
        if not hasattr(self, 'h'):
            self.h = HexSimProcessor()  # create reconstruction object
            self.h.opencv = False
            self.h.kx_input = np.zeros((3, 1), dtype=np.single)
            self.h.ky_input = np.zeros((3, 1), dtype=np.single)
            self.h.wienerfilter_store = self.wiener_Full
            self.setReconstructor()

    def stop_sim_processor(self):
        if hasattr(self, 'h'):
            delattr(self, 'h')

    @add_update_display
    def resetHexSIM(self):
        self.isCalibrated = False
        self.numSets = 0
        self.stop_sim_processor()
        if self.settings['gpu']:
            cp._default_memory_pool.free_all_blocks()
        self.start_sim_processor()
        self.removeMarks()
        self.imageAVG = np.zeros_like(self.imageAVG, dtype=np.uint16)
        self.imageSIM = np.zeros_like(self.imageSIM, dtype=np.uint16)
        self.isUpdateImageViewer = True

    # @add_update_display
    # def channelChanged(self):
    #     if self.current_channel_display() == 0:
    #         self.settings['wavelength'] = 0.523
    #     elif self.current_channel_display() == 1:
    #         self.settings['wavelength'] = 0.610
    #     self.isUpdateImageViewer = True

    def channelSwitch(self):
        if self.current_channel_process() == 0:
            self.imvRAW.displaySet(0)
            self.settings['wavelength'] = 0.523
        elif self.current_channel_process() == 1:
            self.imvRAW.displaySet(1)
            self.settings['wavelength'] = 0.610
        self.resetHexSIM()

    def current_channel_display(self):
        return self.imvRAW.ui.cellCombo.currentIndex()

    def current_channel_process(self):
        return self.ui.channelSwitch.value()

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
                self.show_text(f'[ERROR] Load pre-calibration: \n{e}')

# functions for operation
    @add_run
    def calibrationPressed(self):
        if self.isFileLoad:
            self.action = 'calibration'
        else:
            self.show_text('[WARNING] No loaded file.')

    @add_run
    def reconstructionPressed(self):
        if len(self.imageRAW[self.current_channel_process()]) > 7:
            self.action = 'batch_process'
        elif len(self.imageRAW[self.current_channel_process()]) == 7:
            self.action = 'standard_process'
        else:
            self.show_text('[WARNING] NO RAW images.')

    @add_run
    def roiprocessPressed(self):
        if len(self.imageRAW_ROI[0]) > 7:
            self.action = 'batch_process_roi'
        elif len(self.imageRAW_ROI[0]) == 7:
            self.action = 'standard_process_roi'
        else:
            self.show_text('[WARNING] NO ROI RAW images.')

    @add_run
    def resolutionEstimatePressed(self):
        self.action = 'resolution'

# functions for processing
    @add_timer
    def calibration(self):
        try:
            self.setReconstructor()
            if self.isGpuenable:
                self.h.calibrate_cupy(self.imageRAW[self.current_channel_process()], self.isFindCarrier)
            elif not self.isGpuenable:
                self.h.calibrate(self.imageRAW[self.current_channel_process()], self.isFindCarrier)

            self.isCalibrated = True
            self.h.wienerfilter_store = self.h.wienerfilter
            self.kx_full = self.h.kx
            self.ky_full = self.h.ky
            self.show_text('[DONE] Calibration.')

        except Exception as e:
            self.show_text(f'[ERROR] Calibration: \n{e}')

    @add_timer
    def standardReconstruction(self):
        # standard reconstruction
        try:
            if self.isCalibrated:
                if self.isGpuenable:
                    self.imageSIM = self.h.reconstruct_cupy(self.imageRAW[self.current_channel_process()])
                elif not self.isGpuenable:
                    self.imageSIM = self.h.reconstruct_rfftw(self.imageRAW[self.current_channel_process()])
            else:
                self.calibration()
                if self.isCalibrated:
                    self.standardReconstruction()

            self.show_text('[DONE] Standard reconstruction.')

        except Exception as e:
            self.show_text(f'[ERROR] Reconstruction: \n{e}')

    @add_timer
    def standardReconstructionROI(self):
        try:
            if self.isCalibrated:
                self.imageSIM_ROI = []
                self.wiener_ROI = []
                self.kx_roi = []
                self.ky_roi = []
                self.p_roi = []
                image_sim_roi = None

                for idx in range(self.numSets):
                    image_raw_roi = self.imageRAW_ROI[idx]
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

                self.show_text('[DONE] ROI standard reconstruction.')
            else:
                self.show_text('[WARNING] Uncalibrated.')
        except Exception as e:
            self.show_text(f'[ERROR] ROI Reconstruction: \n{e}')

    @add_timer
    def batchReconstruction(self):
        try:
            if self.isCalibrated:
                # Batch reconstruction
                if self.isGpuenable:
                    if self.isCompact:
                        self.imageSIM = self.h.batchreconstructcompact_cupy(self.imageRAW[self.current_channel_process()])
                    elif not self.isCompact:
                        self.imageSIM = self.h.batchreconstruct_cupy(self.imageRAW[self.current_channel_process()])
                elif not self.isGpuenable:
                    if self.isCompact:
                        self.imageSIM = self.h.batchreconstructcompact(self.imageRAW[self.current_channel_process()])
                    elif not self.isCompact:
                        self.imageSIM = self.h.batchreconstruct(self.imageRAW[self.current_channel_process()])
            elif not self.isCalibrated:
                self.calibration()
                if self.isCalibrated:
                    self.batchReconstruction()

            self.show_text('[DONE] Batch reconstruction.')

        except Exception as e:
            self.show_text(f'[ERROR] Batch reconstruction: \n{e}')

    @add_timer
    def batchReconstructionROI(self):
        try:
            if self.isCalibrated:
                self.imageSIM_ROI = []
                self.wiener_ROI = []
                self.kx_roi = []
                self.ky_roi = []
                self.p_roi = []
                image_sim_roi = None
                # Batch reconstruction
                for idx in range(self.numSets):
                    self.h.kx = self.kx_full
                    self.h.ky = self.ky_full
                    image_raw_roi = self.imageRAW_ROI[idx]

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

                self.show_text('[DONE] ROI batch reconstruction.')
            else:
                self.show_text('[WARNING] Uncalibrated.')

        except Exception as e:
            self.show_text(f'[ERROR] Batch reconstruction: \n{e}')

    @add_timer
    def resolutionEstimate(self):
        try:
            self.show_text('[START] Estimating resolution.')
            pixelsize_avg = self.settings['pixelsize'] / self.settings['magnification']
            image_avg_temp = self.imageAVG[self.imvAVG.ui.imgSlider.value(), :, :]
            ci_avg = ImageDecorr(image_avg_temp, square_crop=True,pixel_size=pixelsize_avg)
            ci_avg.compute_resolution()
            imageSIM_temp = self.imageSIM[self.imvSIM.ui.imgSlider.value(),:,:]
            ci_sim = ImageDecorr(imageSIM_temp, square_crop=True,pixel_size=pixelsize_avg/2)
            ci_sim.compute_resolution()
            txtDisplay = f"\nWide field image resolution:\t {ci_avg.resolution:.3f} um \
                           \nSIM image resolution:\t {ci_sim.resolution:.3f} um\n"
            self.show_text(txtDisplay)
        except Exception as e:
            self.show_text(f'[ERROR] Estimating resolution: \n{e}')

    def browseFolder(self):
        self.selected_dir = QFileDialog.getExistingDirectory(None, 'Select a folder:', self.app.settings['save_dir'],
                                                      QFileDialog.ShowDirsOnly)
        self.ui.selectedFolder.clear()
        self.ui.selectedFolder.insertPlainText(self.selected_dir  + '\n')

    @add_run
    def batchFileProcessPressed(self):
        if self.selected_dir:
            if self.isCalibrated:
                self.action = 'batch_file'
            else:
                self.action = None
                self.show_text('[WARNING] Uncalibrated.')
        else:
            self.action = None
            self.show_text('[WARNING] No selected directory.')

    def batchFileProcess(self):
        self.show_text('\n<START> Batch file processing.')
        file_counter = 0
        cell_counter = 0
        
        self.kx_input = self.kx_full
        self.ky_input = self.ky_full
        self.settings['find_carrier'] = False
        
        for subdir, dirs, files in os.walk(self.selected_dir):
            for filename in files:
                if not self.interrupt_measurement_called:
                    filepath = subdir + os.sep + filename
                    if filepath.endswith("Raw.h5"):
                        # load file
                        self.imageRAW = []
                        with h5py.File(filepath, "r") as f:
                            for ch_idx in range(2):
                                gname = f'data/c{ch_idx}/raw'
                                if f[gname].shape != None:
                                    self.imageRAW.append(np.array(f[gname]))
                                else:
                                    self.imageRAW.append(np.zeros((7, 256, 256), dtype=np.uint16))

                        self.filetitle = Path(filepath).stem[:-4]
                        self.show_text('<FILE> ' + self.filetitle)
                        self.filepath = os.path.dirname(filepath)

                        # find cells
                        self.findCell_no_ui()
                        # full process and/or roi process
                        if len(self.imageRAW[self.current_channel_process()]) > 7:
                            self.imageSIM = np.zeros_like(self.imageSIM)
                            if not self.ui.roiProcessOnlyCheck.isChecked():
                                self.calibration()
                                self.batchReconstruction()
                            self.batchReconstructionROI()

                        elif len(self.imageRAW[self.current_channel_process()]) == 7:
                            self.imageSIM = np.zeros_like(self.imageSIM)
                            if not self.ui.roiProcessOnlyCheck.isChecked():
                                self.calibration()
                                self.standardReconstruction()
                            self.standardReconstructionROI()

                        # save files
                        self.saveMeasurements()
                        file_counter = file_counter + 1
                        cell_counter = cell_counter + self.numSets
                        self.show_text(f'------ Total files: {file_counter:03}')
                        self.show_text(f'------ Total cells: {cell_counter:03} \n')
                else:
                    self.removeMarks()
                    break

        self.show_text('<END> Batch file processing. \n')

# functions for IO
#     @add_update_display
    def loadMeasurement(self):
        try:
            self.filename, _ = QFileDialog.getOpenFileName(caption="Open RAW file", directory=self.app.settings['save_dir'],
                                                           filter="H5 files (*Raw.h5)")
        except Exception as e:
            self.isFileLoad = False
            self.show_text(f'[ERROR] Loading file: \n{e}')

        if self.filename:
            self.isFileLoad = True
            self.ui.imgTab.setCurrentIndex(1)
            self.imageRAW = []

            with h5py.File(self.filename,"r") as f:
                for ch_idx in range(2):
                    group_name = f'data/c{ch_idx}/raw'
                    if f[group_name].shape != None:
                        self.imageRAW.append(np.array(f[group_name]))
                        self.ui.channelSwitch.setValue(ch_idx)  # set to the channel has RAW data
                    else:
                        self.imageRAW.append(np.zeros((7, 256, 256), dtype=np.uint16))

            self.filetitle = Path(self.filename).stem[:-4]
            self.filepath = os.path.dirname(self.filename)
            self.resetHexSIM()
            self.show_text('[LOAD] File name: ' + self.filetitle)

        else:
            if not self.imageRAW:
                self.isFileLoad = False
            self.show_text('[WARNING] File is not loaded.')

    def saveMeasurements(self):
        timestamp = time.strftime("%y%m%d_%H%M%S", time.localtime())
        ch = self.current_channel_process() # selected channel
        if self.isCalibrated:
            if self.ui.saveH5.isChecked():
                if list_equal(self.imageSIM_store, self.imageSIM):
                    self.show_text("[NOT SAVED] SIM images are identical.")
                else:
                    # create file name for the processed file
                    fname_pro = os.path.join(self.filepath, self.filetitle +f'_{timestamp}_C{self.current_channel_process()}_Processed.h5')
                    # create H5 file
                    self.h5file_pro = h5_io.h5_base_file(app=self.app, measurement=self, fname=fname_pro)
                    # create measurement group and save measurement settings
                    h5_io.h5_create_measurement_group(measurement=self, h5group=self.h5file_pro)

                    # name of the sim image group
                    sim_name = f'data/sim'
                    avg_name = f'data/avg'
                    std_name = f'data/std'
                    if np.sum(self.imageSIM) == 0:
                        dset = self.h5file_pro.create_dataset(sim_name, data=h5py.Empty("f"))
                        dset_1 = self.h5file_pro.create_dataset(avg_name, data=h5py.Empty("f"))
                        dset_2 = self.h5file_pro.create_dataset(std_name, data=h5py.Empty("f"))
                        # self.show_text("[UNSAVED] SIM images are empty.")
                    else:
                        dset = self.h5file_pro.create_dataset(sim_name, data=self.imageSIM)
                        dset_1 = self.h5file_pro.create_dataset(avg_name, data=self.imageAVG)
                        dset_2 = self.h5file_pro.create_dataset(std_name, data=self.imageSTD)
                        self.show_text("[SAVED] SIM images to <H5>.")
                    dset.attrs['kx'] = self.kx_full
                    dset.attrs['ky'] = self.ky_full

                    if self.numSets != 0:
                        for idx in range(self.numSets):
                            roi_group_name = f'data/roi/{idx:03}'
                            raw_set = self.h5file_pro.create_dataset(roi_group_name + '/raw', data=self.imageRAW_ROI[idx])
                            raw_set.attrs['cx'] = self.oSegment.selected_cx[idx]
                            raw_set.attrs['cy'] = self.oSegment.selected_cy[idx]
                            sim_set = self.h5file_pro.create_dataset(roi_group_name + '/sim', data=self.imageSIM_ROI[idx])
                            sim_set.attrs['kx'] = self.kx_roi[idx]
                            sim_set.attrs['ky'] = self.ky_roi[idx]
                            avg_set = self.h5file_pro.create_dataset(roi_group_name + '/avg', data=self.imageAVG_ROI[idx])
                            std_set = self.h5file_pro.create_dataset(roi_group_name + '/std', data=self.imageSTD_ROI[idx])
                        self.show_text("[SAVED] ROI images to <H5>.")

                    self.h5file_pro.close()

        if self.ui.saveTif.isChecked():
            fname_sim = os.path.join(self.filepath, self.filetitle
                                     + f'_{timestamp}_C{self.current_channel_process()}_SIM' + '.tif')
            fname_ini = os.path.join(self.filepath, self.filetitle
                                     + f'_{timestamp}_C{self.current_channel_process()}_Settings' + '.ini')
            fname_avg = os.path.join(self.filepath, self.filetitle
                                     + f'_{timestamp}_C{self.current_channel_process()}_AVG' + '.tif')
            fname_std = os.path.join(self.filepath, self.filetitle
                                     + f'_{timestamp}_C{self.current_channel_process()}_STD' + '.tif')

            self.app.settings_save_ini(fname_ini, save_ro=False)
            if np.sum(self.imageSIM) != 0:
                tif.imwrite(fname_sim, np.single(self.imageSIM))
                tif.imwrite(fname_avg, np.single(self.imageAVG))
                tif.imwrite(fname_std, np.single(self.imageSTD))
                self.show_text("[SAVED] SIM images to <TIFF>.")
            # else:
                # self.show_text("[UNSAVED] SIM images are empty.")

            if self.numSets != 0:
                for idx in range(self.numSets):
                    fname_roi_sim = os.path.join(self.filepath, self.filetitle
                                             + f'_{timestamp}_Roi_C{self.current_channel_process()}_{idx:003}_SIM' + '.tif')
                    fname_roi_avg = os.path.join(self.filepath, self.filetitle
                                             + f'_{timestamp}_Roi_C{self.current_channel_process()}_{idx:003}_AVG' + '.tif')
                    fname_roi_std = os.path.join(self.filepath, self.filetitle
                                             + f'_{timestamp}_Roi_C{self.current_channel_process()}_{idx:003}_STD' + '.tif')
                    tif.imwrite(fname_roi_sim, np.single(self.imageSIM_ROI[idx]))
                    tif.imwrite(fname_roi_avg, np.single(self.imageAVG_ROI[idx]))
                    tif.imwrite(fname_roi_std, np.single(self.imageSTD_ROI[idx]))
                self.show_text("[SAVED] ROI images to <TIFF>.")

    @add_update_display
    def loadCalibrationResults(self):
        try:
            filename, _ = QFileDialog.getOpenFileName(caption="Open file", directory=self.app.settings['save_dir'],
                                                      filter="H5 files (*Processed.h5)")
            with h5py.File(filename,"r") as f:
                self.h.kx_input = self.kx_input = f["data/sim"].attrs['kx']
                self.h.ky_input = self.ky_input = f["data/sim"].attrs['ky']
                print(self.ky_input)
            self.setReconstructor()
            self.isUpdateImageViewer = True
            self.show_text("[LOAD] kx, ky are loaded.")
        except:
            self.show_text("[WARNING] kx, ky are NOT loaded.")

# functions for display
    def show_text(self, text):
        self.ui.MessageBox.insertPlainText(text+'\n')
        self.ui.MessageBox.ensureCursorVisible()
        print(text)

    def updateImageViewer(self):
        imageRaw_tmp  = self.imageRAW[self.current_channel_process()]
        self.imageAVG = self.raw2avgImage(imageRaw_tmp)
        self.imageSTD = self.raw2stdImage(imageRaw_tmp)
        self.imvRAW.showImageSet(self.imageRAW)
        self.imvAVG.showImageSet(self.imageAVG)
        self.imvSTD.showImageSet(self.imageSTD)
        self.imvAVG_ROI.showImageSet(self.imageAVG_ROI)
        self.imvSTD_ROI.showImageSet(self.imageSTD_ROI)
        self.imvRAW_ROI.showImageSet(self.imageRAW_ROI)
        self.imvSIM_ROI.showImageSet(self.imageSIM_ROI)
        self.imvSIM.showImageSet(self.imageSIM)
        self.imvWiener_ROI.showImageSet(self.wiener_ROI)
        self.imvCalibration.update(self.h)

    def removeMarks(self):
        if self.roiRect:
            for item in self.roiRect:
                self.imvAVG.imv.getView().removeItem(item)
            self.roiRect = []

    def raw2avgImage(self, raw_images):
        avg_images = np.zeros((raw_images.shape[0] // 7, raw_images.shape[1], raw_images.shape[2]))
        for idx in range(raw_images.shape[0] // 7):
            avg_images[idx,:,:] = np.sum(raw_images[idx * 7:(idx + 1) * 7, :, :], axis=0) / 7
        return avg_images

    def raw2stdImage(self, raw_images):
        std_images = np.zeros((raw_images.shape[0] // 7, raw_images.shape[1], raw_images.shape[2]))
        for idx in range(raw_images.shape[0] // 7):
            std_images[idx,:,:] = np.std(raw_images[idx * 7:(idx + 1) * 7, :, :], axis=0)
        return std_images

# functions for ROI
    def roiSize(self):
        return int(self.ui.roiSizeCombo.currentText())

    def minCellSize(self):
        return int(self.ui.minCellSizeInput.value())

    @add_update_display
    def findCell(self):
        self.oSegment = ImageSegmentation(self.imageRAW[self.current_channel_process()], self.roiSize() // 2,
                                          self.minCellSize(), self.settings['watershed_threshold'])
        markpen = pg.mkPen('r', width=1)
        self.removeMarks()
        self.oSegment.min_cell_size = self.minCellSize()**2
        self.oSegment.roi_half_side = self.roiSize()//2
        self.oSegment.find_cell(method=self.settings['segment_method'])
        self.imageRAW_ROI = self.oSegment.roi_creation()
        self.imageAVG_ROI = []  # initialize the image sets
        self.imageSTD_ROI = []  # initialize the image sets
        self.numSets = len(self.imageRAW_ROI)
        self.ui.cellNumber.setValue(self.numSets)

        if self.numSets == 0:
            self.imageAVG_ROI = np.zeros((self.roiSize(), self.roiSize()), dtype=np.uint16)
            self.imageSTD_ROI = np.zeros((self.roiSize(), self.roiSize()), dtype=np.uint16)
        elif self.numSets > 0:
            for idx in range(self.numSets):
                self.imageAVG_ROI.append(self.raw2avgImage(self.imageRAW_ROI[idx]))
                self.imageSTD_ROI.append(self.raw2stdImage(self.imageRAW_ROI[idx]))
                # mark the cells with rectangle overlay
                r = pg.ROI(pos = (self.oSegment.selected_cx[idx]-self.oSegment.roi_half_side,
                                  self.oSegment.selected_cy[idx]-self.oSegment.roi_half_side),
                           size=self.roiSize(), pen=markpen, movable=False)
                self.imvAVG.imv.getView().addItem(r)
                self.roiRect.append(r)

        self.isUpdateImageViewer = True
        self.ui.imgTab.setCurrentIndex(1)
        self.show_text(f'------ Found cells: {self.numSets:02}')

    def findCell_no_ui(self):
        self.oSegment = ImageSegmentation(self.imageRAW[self.current_channel_process()], self.roiSize() // 2,
                                          self.minCellSize(), self.settings['watershed_threshold'])
        self.oSegment.min_cell_size = self.minCellSize()**2
        self.oSegment.roi_half_side = self.roiSize()//2
        self.oSegment.find_cell(method=self.settings['segment_method'])
        self.imageRAW_ROI = self.oSegment.roi_creation()
        self.imageAVG_ROI = [] # initialize the image sets
        self.imageSTD_ROI = []  # initialize the image sets
        self.numSets = len(self.imageRAW_ROI)

        if self.numSets == 0:
            self.imageAVG_ROI = np.zeros((self.roiSize(), self.roiSize()), dtype=np.uint16)
            self.imageSTD_ROI = np.zeros((self.roiSize(), self.roiSize()), dtype=np.uint16)
        elif self.numSets > 0:
            for idx in range(self.numSets):
                self.imageAVG_ROI.append(self.raw2avgImage(self.imageRAW_ROI[idx]))
                self.imageSTD_ROI.append(self.raw2stdImage(self.imageRAW_ROI[idx]))
        self.show_text(f'------ Found cells: {self.numSets:02}')