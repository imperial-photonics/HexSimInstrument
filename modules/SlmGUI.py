"""
Intro:  GUI of SLM.
        Model: QXGA SLM with R11 system (Forth Dimension Displays)
Time:   31/08/2022
Modified from:
    Written by Hai Going
    Code for creating the measurement class of ScopeFoundry for the Hamamatsu camera
    10/20
"""
__author__ = "Meizhu Liang @Imperial College London"

import numpy as np
import pyqtgraph as pg
from ScopeFoundry import Measurement
from ScopeFoundry.helper_funcs import load_qt_ui_file
from utils.StackImageViewer import StackImageViewer
from qtwidgets import Toggle

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

class SlmMeasurement(Measurement):
    name = 'HexSIM_Measure'

    def setup(self):
        # load ui file
        self.ui = load_qt_ui_file(".\\ui\\SLM.ui")
        # connect to hardware components
        self.slm = self.app.hardware['SLM_hardware']
        # Measurement component settings
        self.settings.New('refresh_period', dtype=float, unit='s', spinbox_decimals=4, initial=0.02, hardware_set_func=self.setRefresh, vmin=0)
        self.display_update_period = self.settings.refresh_period.val
        self.settings.New('debug', dtype=bool, initial=False,
                          hardware_set_func = self.setReconstructor)
        # initialise condition labels
        self.isSlmRun = False
        self.saveRep = True
        self.sendRep = True
        self.numSets = 0
        self.kx_full = np.zeros((3, 1), dtype=np.single)  # frequency of full field of view
        self.ky_full = np.zeros((3, 1), dtype=np.single)
        # initialise images
        v = 1536
        h = 2048
        self.imageHol = [np.zeros((7, v, h), dtype=np.uint16), np.zeros((7, v, h), dtype=np.uint16)]  # a set of images

    def setup_figure(self):
        # self.ui.imgTab.setCurrentIndex(0)
        # camera UI
        self.imv = pg.ImageView()
        # self.imv.ui.roiBtn.hide()
        # self.imv.ui.menuBtn.hide()

        # image viewers
        self.imvHol= StackImageViewer(image_sets=self.imageHol, set_levels=[1, 1])

        # connect ui widgets to measurement/hardware settings or functions
        self.ui.hLayout.addWidget(self.imv)

        self.ui.switchSLM = Toggle()
        self.ui.slmToggleLayout.addWidget(self.ui.switchSLM)
        self.ui.switchSLM.stateChanged.connect(self.controlSLM)

        # self.ui.slmSlider.valueChanged.connect(self.controlSLM)

        # reconstructor settings



        # Operations
        self.ui.holGenButton.clicked.connect(self.genHolPressed)
        self.ui.selectPushButton.clicked.connect(self.selectPressed)
        self.ui.sendPushButton.clicked.connect(self.sendPressed)

    def update_display(self):
        # update SLM status
        try:
            self.ui.statusDisplay.display(self.settings.activation_state)
            self.ui.repDisplay.display(self.settings.self.settings.rep_name)
            self.ui.acTypeDisplay.display(self.settings.self.settings.activation_type)
            self.ui.roCountDisplay.display(self.settings.self.settings.roIndex)
            self.ui.roNameDisplay.display(self.settings.self.settings.roName)
        except Exception as e:
            txtDisplay = f'update_display error: {e}'
            self.show_text(txtDisplay)

        # update camera viewer
        if self.isSlmRun:
            self.imv.setImage(self.imageHol.T, autoLevels = True, autoRange=True)

    def pre_run(self):
        if hasattr(self, 'slm'):
            self.controlSLM()

    def run(self):
        while not self.interrupt_measurement_called:
            # time.sleep(0.01)
            if self.isSlmRun:
                # self.camera.updateCameraSettings()
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
                    self.batchCapture_test()
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
    def controlSLM(self):
        try:
            if self.ui.switchSLM.isChecked():
                self.isSlmRun = True
            else:
                self.isSlmRun = False
        except Exception as e:
            txtDisplay = f'SLM encountered an error \n{e}'
            self.show_text(txtDisplay)

# functions for display
    def setRefresh(self, refresh_period):
        self.display_update_period = refresh_period

    def show_text(self, text):
        self.ui.messageBox.insertPlainText(text+'\n')
        self.ui.messageBox.ensureCursorVisible()
        print(text)

# functions for operation
    def genHolPressed(self):
        self.isSlmRun = False
        self.action = 'generate_holograms'

    def selectPressed(self):
        self.isSlmRun = False
        self.action = 'select_repertoire'

    def sendPressed(self):
        self.isSlmRun = False
        self.action = 'send_repertoire'

# functions for slm
    def slmRun(self):

        try:
            self.slm.read_from_hardware()
            self.slm.ini
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