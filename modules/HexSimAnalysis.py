import os,time,json,h5py
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
from utils.StackImageViewer import StackImageViewer
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
        cls.run()
        return result

    inner.__name__ = function.__name__
    return inner

class HexSimAnalysis(Measurement):
    name = 'HexSIM_Analyse'

    def setup(self):
        # load ui file
        self.ui = load_qt_ui_file(".\\ui\\hexsim_analyse.ui")
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
        self.settings.New('find_carrier', dtype=bool, initial=True,
                          hardware_set_func = self.setReconstructor)
        self.settings.New('otf_model',dtype=str, initial = 'none', choices=["none","exp","sph"],
                          hardware_set_func = self.setReconstructor)

        # initialize condition lables
        self.isUpdateImageViewer = False
        self.isCalibrated = False

        self.isGpuenable = True  # using GPU for accelerating
        self.isCompact = True  # using compact mode in batch reconstruction to save memory
        self.isFindCarrier = True

        self.action = None # the action in run()

        self.kx_full = np.zeros((3, 1), dtype=np.single) # frequency of full field of view
        self.ky_full = np.zeros((3, 1), dtype=np.single)

        # image initialization
        v = h = 512
        # left
        self.imageRaw = np.zeros((7, v, h), dtype=np.uint16)
        self.imageWF = np.zeros((v, h), dtype=np.uint16)
        self.imageWF_ROI = np.zeros((v, h), dtype=np.uint16)
        self.imageRaw_ROI = np.zeros((7, v, h), dtype=np.uint16)
        # right
        self.imageSIM = np.zeros((2*v,2*h), dtype=np.uint16)
        self.imageSIM_ROI = np.zeros((2 * v, 2 * h), dtype=np.uint16)    # it can be an image or a set of images
        self.wiener_Full = np.zeros((v, h), dtype=np.uint16)
        self.wiener_ROI = np.zeros((v, h), dtype=np.uint16)              # it can be an image or a set of images

        if not hasattr(self, 'h'):
            self.h = HexSimProcessor()  # create reconstruction object
            self.h.opencv = False
            self.imageRaw = np.zeros((7, v, h),dtype=np.uint16)  # Initialize the raw image array
            self.setReconstructor()
            self.h.N = v
            self.h.wienerfilter_store = self.wiener_Full
            self.h.kx_input = np.zeros((3, 1), dtype=np.single)
            self.h.ky_input = np.zeros((3, 1), dtype=np.single)

    def setup_figure(self):
        # image viewers
        self.imvRaw     = StackImageViewer(image_sets=self.imageRaw,set_levels=[1,1])
        self.imvWF      = StackImageViewer(image_sets=self.imageWF,set_levels=[1,1])
        self.imvWF_ROI  = StackImageViewer(image_sets=self.imageWF_ROI,set_levels=[1,1])
        self.imvRaw_ROI = StackImageViewer(image_sets=self.imageRaw_ROI,set_levels=[1,1])
        self.imvSIM     = StackImageViewer(image_sets=self.imageSIM, set_levels=[0, 0.8])
        self.imvSIM_ROI = StackImageViewer(image_sets=self.imageSIM_ROI, set_levels=[0, 0.8])

        self.imvCalibration = CalibrationResults(self.h)
        self.imvWiener_ROI = StackImageViewer(image_sets=self.wiener_ROI,set_levels=[1,1])
        self.imvWiener_ROI.imv.ui.histogram.hide()

        # combo lists setting: size of roi
        self.roiRect = [] # list of roi rectangular
        self.roiSizeList = [128,256,512,1024]
        self.ui.roiSizeCombo.addItems(map(str,self.roiSizeList))
        self.ui.roiSizeCombo.setCurrentIndex(1)

        # add image viewers to ui
        self.ui.rawImageLayout.addWidget(self.imvRaw)
        self.ui.wfImageLayout.addWidget(self.imvWF)
        self.ui.roiWFLayout.addWidget(self.imvWF_ROI)
        self.ui.roiRAWLayout.addWidget(self.imvRaw_ROI)
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

        # Operation
        self.ui.loadMeasurementButton.clicked.connect(self.loadFile)
        self.ui.calibrationButton.clicked.connect(self.calibrationPressed)
        self.ui.findCellButton.clicked.connect(self.findCell)
        self.ui.reconstructionButton.clicked.connect(self.reconstructionPressed)
        self.ui.roiProcessButton.clicked.connect(self.roiprocessPressed) # TODO add function roiprocessPressed

        self.ui.loadCalibrationButton.clicked.connect(self.loadCalibrationResults)
        self.ui.resetButton.clicked.connect(self.resetHexSIM)
        self.ui.saveButton.clicked.connect(self.saveMeasurements)
        self.ui.resolutionButton.clicked.connect(self.resolutionEstimatePressed)

        self.imvRaw.ui.cellCombo.currentIndexChanged.connect(self.channelChanged)

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

    def run(self):
        if self.action is not None:
            if self.action == 'calibration':
                self.calibration()

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

            elif self.action == 'resolution':
                self.resolutionEstimate()

            elif self.action == 'test':
                self.show_text('Test')

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
        # self.isFileLoad = False
        self.isCalibrated = False
        self.stop_sim_processor()
        cp._default_memory_pool.free_all_blocks()
        self.start_sim_processor()
        self.removeMarks()
        self.imageWF = np.zeros(self.imageWFShape, dtype=np.uint16)
        self.imageSIM = np.zeros(self.imageSIMShape, dtype=np.uint16)
        self.isUpdateImageViewer = True

    def channelChanged(self):
        self.resetHexSIM()
        if self.current_channel_display() == 0:
            self.settings['wavelength'] = 0.523
        elif self.current_channel_display() == 1:
            self.settings['wavelength'] = 0.610

    def channelChoose(self):
        if hasattr(self,'imvRaw'):
            if self.current_channel_display() == 0:
                self.settings['wavelength'] = 0.523
            elif self.current_channel_display() == 1:
                self.settings['wavelength'] = 0.610

    def current_channel_display(self):
        return self.imvRaw.ui.cellCombo.currentIndex()

    def setReconstructor(self, *args):
        self.channelChoose()
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
            self.h.kx = self.kx_input
            self.h.ky = self.ky_input

# functions for operation
    @add_run
    def calibrationPressed(self):
        if self.isFileLoad:
            self.action = 'calibration'
        else:
            self.show_text('Measurement is not loaded.')

    @add_run
    def reconstructionPressed(self):
        if len(self.imageRaw) > 7:
            self.action = 'batch_process'
        elif len(self.imageRaw) == 7:
            self.action = 'standard_process'
        else:
            self.show_text('Raw images are not acquired.')

    @add_run
    def roiprocessPressed(self):
        if len(self.imageRaw_ROI[0])>7:
            self.action = 'batch_process_roi'
        elif len(self.imageRaw_ROI[0])==7:
            self.action = 'standard_process_roi'
        else:
            self.show_text('ROI raw images are not acquired.')

    @add_run
    def resolutionEstimatePressed(self):
        self.action = 'resolution'

# functions for processing
    # TODO change the processing functions refer to measurement
    @add_timer
    def calibration(self):
        try:
            self.setReconstructor()
            if self.isGpuenable:
                self.h.calibrate_cupy(self.imageRaw, self.isFindCarrier)
            elif not self.isGpuenable:
                self.h.calibrate(self.imageRaw, self.isFindCarrier)

            self.isCalibrated = True
            self.show_text('Calibration finished.')
            self.h.wienerfilter_store = self.h.wienerfilter
            self.kx_full = self.h.kx
            self.ky_full = self.h.ky

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

    @add_timer
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
            self.show_text(txtDisplay)
        except:
            pass

# functions for IO
    @add_update_display
    def loadFile(self):
            try:
                self.filename, _ = QFileDialog.getOpenFileName(directory=self.app.settings['save_dir'])

                if self.filename.endswith('.tif'):
                    self._fileType = 'tif'

                    self.imageRaw = np.single(tif.imread(self.filename))
                    self.imageWF = self.raw2WideFieldImage(self.imageRaw)

                    self.imageRawShape = np.shape(self.imageRaw)
                    self.imageSIMShape = [self.imageRawShape[0] // 7, self.imageRawShape[1] * 2, self.imageRawShape[2] * 2]
                    self.imageWFShape = [self.imageRawShape[0] // 7, self.imageRawShape[1], self.imageRawShape[2]]

                    self.imageSIM = np.zeros(self.imageSIMShape, dtype=np.uint16)

                elif self.filename.endswith('.h5'):
                    self._fileType = 'h5'
                    pass
                    # TODO transfer h5 data to program

                self.oSegment = ImageSegmentation(self.imageRaw, self.roiSize() // 2, self.minCellSize())

                self.filetitle = Path(self.filename).stem
                self.filepath = os.path.dirname(self.filename)
                # print(self.filename)
                # print(self.filepath)
                # print(self.filetitle)

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
                    self.settings['magnification'] = configSet["magnification"]
                    self.settings['NA'] = configSet["NA"]
                    self.settings['n'] = configSet["refractive index"]
                    self.settings['wavelength'] = configSet["wavelength"]
                    self.settings['pixelsize'] = configSet["pixelsize"]

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
                    self.ui.fileInfo.insertPlainText(txtDisplay)

                except:
                    self.show_text("No information about this measurement.")

            except AssertionError as error:
                print(error)
                self.isFileLoad = False

            if self.isFileLoad:
                self.ui.findCellButton.setEnabled(True)
                self.isUpdateImageViewer = True
                self.isCalibrated = False
                self.setReconstructor()
                self.h._allocate_arrays()
            else:
                self.show_text("File is not loaded.")

    def load_ini_settings(self, fname):
        self.log.info("ini settings loading from " + fname)

        config = configparser.ConfigParser()
        config.optionxform = str
        config.read(fname)

        if 'app' in config.sections():
            for lqname, new_val in config.items('app'):
                lq = self.settings.as_dict().get(lqname)
                if lq:
                    if lq.dtype == bool:
                        new_val = str2bool(new_val)
                    lq.update_value(new_val)


    def saveMeasurements(self):

        timestamp = time.strftime("%Y_%m%d_%H%M%S", time.localtime())
        # sample = self.app.settings['sample']
        # # sample_name = f'{timestamp}_{self.name}_{sample}.h5'
        # if sample == '':
        #     sample_name = '_'.join([timestamp, self.name])
        # else:
        #     sample_name = '_'.join([timestamp, self.name, sample])
        if self._fileType == 'tif':
            fname = os.path.join(self.filepath, self.filetitle + '.h5')
            self.h5file = h5_io.h5_base_file(app=self.app, measurement=self, fname=fname)
            self.h5_group = h5_io.h5_create_measurement_group(measurement=self, h5group=self.h5file)

            name = f't0/image'
            self.image_h5 = self.h5_group.create_dataset(name=name,
                                                         shape=(self.imageWFShape[0], self.imageWFShape[1],
                                                                self.imageWFShape[2]),
                                                         dtype=self.imageWF.dtype,
                                                         chunks=(1, self.imageWFShape[1], self.imageWFShape[2])
                                                         )
            self.h5file.flush()
            self.h5file.close()

        elif self._fileType == 'h5':
            pass
                # TODO: add functions







        # self.initH5()
        # self.image_h5 = self.imageWF

        # self.app.settings_save_h5('test.h5')
        # t0 = time.time()
        # timestamp = datetime.fromtimestamp(t0)
        # timestamp = timestamp.strftime("%Y%m%d%H%M")
        # pathname = self.filepath + '/segmented_analysis'
        # Path(pathname).mkdir(parents=True,exist_ok=True)
        #
        # for idx in range(self.numSets):
        #     suffix = str(idx).zfill(3)
        #     simimagename = pathname + '/' + self.filetitle + timestamp + f'_segmented_sim' + '_' + suffix + '.tif'
        #     wfimagename = pathname + '/' + self.filetitle + timestamp + f'_segmented_widefield' + '_' + suffix + '.tif'
        #     rawimagename =  pathname + '/' + self.filetitle + timestamp + f'_segmented_raw' + '_' + suffix + '.tif'
        #     tif.imwrite(simimagename, np.single(self.imageSIMSets[idx]))
        #     tif.imwrite(wfimagename,np.uint16(self.imageWFSets[idx]))
        #     tif.imwrite(rawimagename, np.uint16(self.imageRawSets[idx]))
        #
        # txtname =      pathname + '/' + self.filetitle + timestamp + f'_configuration' + '.txt'
        # ininame = pathname + '/' + self.filetitle + timestamp + f'_configuration'+'.ini'
        # savedictionary = {
        #     "exposure time (s)":self.exposuretime,
        #     "laser power (mW)": self.laserpower,
        #     # "z stepsize (um)":  self.
        #     # System setup:
        #     "magnification" :   self.h.magnification,
        #     "NA":               self.h.NA,
        #     "refractive index": self.h.n,
        #     "wavelength":       self.h.wavelength,
        #     "pixelsize":        self.h.pixelsize,
        #     # Calibration parameters:
        #     "alpha":            self.h.alpha,
        #     "beta":             self.h.beta,
        #     "Wiener filter":    self.h.w,
        #     "eta":              self.h.eta,
        #     "cleanup":          self.h.cleanup,
        #     "axial":            self.h.axial,
        #     "modulation":       self.h.usemodulation,
        #     "kx":               self.h.kx,
        #     "ky":               self.h.ky,
        #     "phase":            self.h.p,
        #     "amplitude":        self.h.ampl
        #     }
        # f = open(txtname, 'w+')
        # f.write(json.dumps(savedictionary, cls=NumpyEncoder,indent=2))
        # # try:
        # self.app.settings_save_ini(ininame, save_ro=False)
        # # except:
        # #     pass
        # self.isCalibrationSaved = True

    def initH5(self):
        """
        Initialization operations for the h5 file
        """
        self.create_saving_directory()

        # file name creation
        timestamp = time.strftime("%y%m%d_%H%M%S", time.localtime())
        sample = self.app.settings['sample']
        # sample_name = f'{timestamp}_{self.name}_{sample}.h5'
        if sample == '':
            sample_name = '_'.join([timestamp, self.name])
        else:
            sample_name = '_'.join([timestamp, self.name, sample])
        fname = os.path.join(self.filepath, sample_name + '.h5')
        # fname = os.path.join('./measurement/',sample_name+'.h5')
        # fname = self.app.settings['save_dir'] + '/' + sample_name + '.h5'
        print(fname)
        #
        # d1 = np.random.random(size=(1000, 20))
        # d2 = np.random.random(size=(1000, 200))
        #
        # with h5py.File(fname, "w") as f:
        #     dset = f.create_dataset("dataset1", data=d1)
        #     dset = f.create_dataset("dataset2", data=d2)
        # hf = h5py.File(fname,'w')
        # hf.close()
        # file creation
        self.h5file = h5_io.h5_base_file(app=self.app, measurement=self,fname = fname)
        self.h5_group = h5_io.h5_create_measurement_group(measurement=self, h5group=self.h5file)

        name = f't0/image'
        self.image_h5 = self.h5_group.create_dataset(name=name,
                                                     shape=(self.imageWFShape[0], self.imageWFShape[1], self.imageWFShape[2]),
                                                     dtype=self.imageWF.dtype,
                                                     chunks=(1, self.imageWFShape[1],self.imageWFShape[2])
                                                     )
        # img_size = self.im.image[0].shape  # both image[0] and image[1] are valid, since they have the same shape

        # number_of_channels = self.settings.channels_num.val

        # take as third dimension of the file the total number of images collected in the buffer
        # if self.camera.hamamatsu.last_frame_number < self.camera.hamamatsu.number_image_buffers:
        #     length = int((self.camera.hamamatsu.last_frame_number + 1) / number_of_channels)
        # else:
        #     length = self.camera.hamamatsu.number_image_buffers / number_of_channels
        #
        # for ch_index in self.channels:
        #     name = f't0/c{ch_index}/image'
        #
        #     self.image_h5[ch_index] = self.h5_group.create_dataset(name=name,
        #                                                            shape=(length, img_size[0], img_size[1]),
        #                                                            dtype=self.im.image[0].dtype,
        #                                                            chunks=(1, img_size[0], img_size[1])
        #                                                            )
        #
        #     self.image_h5[ch_index].attrs['element_size_um'] = [self.settings['zsampling'], self.settings['ysampling'],
        #                                                         self.settings['xsampling']]
        #     self.image_h5[ch_index].attrs['acq_time'] = timestamp
        #     self.image_h5[ch_index].attrs['flowrate'] = self.settings['flowrate']

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

# functions for display
    def show_text(self, text):
        self.ui.MessageBox.insertPlainText(text+'\n')
        self.ui.MessageBox.ensureCursorVisible()
        print(text)

    def raw2WideFieldImage(self,rawImages):
        wfImages = np.zeros((rawImages.shape[0]//7,rawImages.shape[1],rawImages.shape[2]))
        for idx in range(rawImages.shape[0]//7):
            wfImages[idx,:,:] = np.sum(rawImages[idx*7:(idx+1)*7,:,:],axis=0)/7

        return wfImages

    def updateImageViewer(self):
        self.imvRaw.setImageSet(self.imageRaw)
        self.imvWF.setImageSet(self.imageWF)
        self.imvWF_ROI.setImageSet(self.imageWF_ROI)
        self.imvRaw_ROI.setImageSet(self.imageRaw_ROI)
        self.imvSIM.setImageSet(self.imageSIM)
        self.imvSIM_ROI.setImageSet(self.imageSIM_ROI)
        self.imvWiener_ROI.setImageSet(self.wiener_ROI)
        self.imvCalibration.update(self.h)

    def removeMarks(self):
        if self.roiRect:
            for item in self.roiRect:
                self.imvWF.imv.getView().removeItem(item)
            self.roiRect = []

# functions for ROI
    def roiSize(self):
        return int(self.ui.roiSizeCombo.currentText())

    def minCellSize(self):
        return int(self.ui.minCellSizeInput.value())

    @add_update_display
    def findCell(self):

        markpen = pg.mkPen('r', width=1)
        self.removeMarks()
        self.oSegment.min_cell_size = self.minCellSize()**2
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
                r = pg.ROI(pos = (self.oSegment.selected_cx[idx]-self.oSegment.roi_half_side,
                                  self.oSegment.selected_cy[idx]-self.oSegment.roi_half_side),
                           size=self.roiSize(), pen=markpen, movable=False)
                self.imvWF.imv.getView().addItem(r)
                self.roiRect.append(r)

        self.isUpdateImageViewer = True
        txtDisplay = f'Found cells: {self.numSets}'
        self.show_text(txtDisplay)

