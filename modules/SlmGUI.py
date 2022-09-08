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
import tkinter as tk
from tkinter import filedialog

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
    name = 'SLM_Measure'

    def setup(self):
        # load ui file
        self.ui = load_qt_ui_file(".\\ui\\SLM.ui")
        # connect to hardware components
        self.slm = self.app.hardware['SLM_hardware']
        # Measurement component settings
        self.settings.New('refresh_period', dtype=float, unit='s', spinbox_decimals=4, initial=0.02, hardware_set_func=self.setRefresh, vmin=0)
        self.display_update_period = self.settings.refresh_period.val
        # self.settings.New('debug', dtype=bool, initial=False,
        #                   hardware_set_func = self.setReconstructor)
        # initialise condition labels
        self.isSlmRun = False
        self.saveRep = True
        self.sendRep = True
        self.numSets = 0
        self.kx_full = np.zeros((3, 1), dtype=np.single)  # frequency of full field of view
        self.ky_full = np.zeros((3, 1), dtype=np.single)
        # initialise images
        self.v = 1536
        self.h = 2048
        self.imageHol = np.zeros((14, self.v, self.h), dtype=np.uint16)  # a set of images

        self.action = None
        self.isUpdateImageViewer = False

    def setup_figure(self):
        # self.ui.imgTab.setCurrentIndex(0)
        # camera UI
        self.imv = pg.ImageView()
        self.imv.ui.roiBtn.hide()
        self.imv.ui.menuBtn.hide()

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
            self.ui.statusDisplay.insertPlainText(f'{self.slm.settings.activation_state}')
            self.ui.statusDisplay.ensureCursorVisible()
            self.ui.repDisplay.insertPlainText(f'{self.slm.settings.rep_name}')
            self.ui.repDisplay.ensureCursorVisible()
            self.ui.acTypeDisplay.insertPlainText(f'{self.slm.settings.activation_type}')
            self.ui.acTypeDisplay.ensureCursorVisible()
            self.ui.roCountDisplay.display(f'{self.slm.settings.roIndex}')
            self.ui.roNameDisplay.insertPlainText(f'{self.slm.settings.roName}')
            self.ui.roNameDisplay.ensureCursorVisible()
        except Exception as e:
            txtDisplay = f'update_display error: {e}'
            self.show_text(txtDisplay)

        # update camera viewer
        if self.isSlmRun:
            self.imv.setImage(self.imageHol, autoLevels=True, autoRange=True)
        else:
            pass

        # update hexsim viwer
        if self.isUpdateImageViewer:
            self.updateImageViewer()
            self.isUpdateImageViewer = False
        else:
            pass

    def pre_run(self):
        if hasattr(self, 'slm'):
            self.controlSLM()

    def run(self):
        while not self.interrupt_measurement_called:
            # time.sleep(0.01)
            if self.isSlmRun:
            #     # self.camera.updateCameraSettings()
                self.slmRun()

            if self.action is not None:
                if self.action == 'generate_holograms':
                    if self.ui.hex_holRadioButton.isChecked():
                        self.show_text('Start holograms generation')
                        re = self.genHex()
                        if self.ui.repSaveCheckBox.isChecked():
                            self.slm.writeRep('hexagons_'+re[2], len(re[0]), re[0])
                            if self.ui.sendCheckBox.isChecked():
                                self.slm. sendRep('hexagons_'+re[2])
                    elif self.ui.stripe_holRadioButoon.isChecked():
                        re = self.genStripes()
                        if self.ui.repSaveCheckBox.isChecked():
                            self.slm.writeRep('stripes_'+re[1], len(re[0]), re[0])
                            if self.ui.sendCheckBox.isCheked():
                                self.slm. sendRep('stripes_'+re[1])

                elif self.action == 'select_repertoire':
                    selected_rep = self.select()
                    self.slm.sendRep(selected_rep)

                self.isUpdateImageViewer = True
                self.action = None
                self.controlSLM()

    def post_run(self):
        if hasattr(self, 'slm'):
            self.slmInterrupt()

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

    def updateImageViewer(self):
        self.imvHol.showImageSet(self.imageHol)
        # try:
        #     self.imageWF = self.raw2WideFieldImage(self.imageRAW[self.current_channel_display()])
        #     self.imvWF.showImageSet(self.imageWF)
        # except:
        #     pass
        # self.imvWF_ROI.showImageSet(self.imageWF_ROI)
        # self.imvSIM.showImageSet(self.imageSIM)
        # self.imvSIM_ROI.showImageSet(self.imageSIM_ROI)
        # self.imvWiener_ROI.showImageSet(self.wiener_ROI)
        # self.imvCalibration.update(self.h)

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
            self.slm.act()
            while (self.slm.slm.getState() != 0x52) or (self.slm.sle.getState() !=0x53) or (self.slm.sle.getState()!=0x54):
                if not self.isSlmRun or self.interrupt_measurement_called:
                    break
        finally:
            self.slm.deact()

    def slmInterrupt(self):
        self.isSlmRun = False
        self.ui.switchSLM.setChecked(False)

    def genHex(self):
        """generate hexagonal holograms"""
        re1 = self.slm.genHexgans(488)
        re2 = self.slm.genHexgans(561)
        nameList = [None] * 14
        for i in range(7):
            self.imageHol[int(2 * i), :, :] = re1[0][i]
            self.imageHol[int(2 * i + 1), :, :] = re2[0][i]
            nameList[2 * i] = re1[1][i]
            nameList[2 * i + 1] = re2[1][i]
        return nameList, re1[2], re2[2]

    def genStripes(self):
        """generate 7 striped holograms"""
        self.imageHol = np.zeros((7, self.v, self.h), dtype=np.uint16)  # initialise the holograms
        re = self.slm.genStripes()
        self.imageHol = re[0]
        return re[1], re[2]

    def select(self):
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename()
        return file_path
