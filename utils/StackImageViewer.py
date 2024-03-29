# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '.\stack_image_viewer.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets, uic

# this function causes issues in the comparison of images.!!!!!!!
def list_equal(list_a,list_b):
    try:
        eql = all([np.array_equal(list_a[i], list_b[i]) for i in range(len(list_a))])
    except (IndexError, TypeError):
        eql = False
    return eql

class UiViewer(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(680, 640)
        self.verticalLayout = QtWidgets.QVBoxLayout(Form)
        self.verticalLayout.setObjectName("verticalLayout")
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.imageLayout = QtWidgets.QGridLayout()
        self.imageLayout.setObjectName("imageLayout")
        self.verticalLayout.addLayout(self.imageLayout)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.nCurrent = QtWidgets.QLabel(Form)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.nCurrent.sizePolicy().hasHeightForWidth())
        self.nCurrent.setSizePolicy(sizePolicy)
        self.nCurrent.setMinimumSize(QtCore.QSize(25, 0))
        self.nCurrent.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.nCurrent.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignTrailing | QtCore.Qt.AlignVCenter)
        self.nCurrent.setObjectName("nCurrent")
        self.horizontalLayout_2.addWidget(self.nCurrent)
        self.label_18 = QtWidgets.QLabel(Form)
        self.label_18.setObjectName("label_18")
        self.horizontalLayout_2.addWidget(self.label_18)
        self.nTotal = QtWidgets.QLabel(Form)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.nTotal.sizePolicy().hasHeightForWidth())
        self.nTotal.setSizePolicy(sizePolicy)
        self.nTotal.setMinimumSize(QtCore.QSize(25, 0))
        self.nTotal.setObjectName("nTotal")
        self.horizontalLayout_2.addWidget(self.nTotal)
        self.imgSlider = QtWidgets.QSlider(Form)
        self.imgSlider.setMinimumSize(QtCore.QSize(400, 0))
        self.imgSlider.setStyleSheet("QSlider::groove:horizontal {\n"
                                     "border: 1px solid #bbb;\n"
                                     "background: white;\n"
                                     "height: 10px;\n"
                                     "border-radius: 4px;\n"
                                     "}\n"
                                     "\n"
                                     "QSlider::sub-page:horizontal {\n"
                                     "background: qlineargradient(x1: 0, y1: 0,    x2: 0, y2: 1,\n"
                                     "    stop: 0 #66e, stop: 1 #bbf);\n"
                                     "background: qlineargradient(x1: 0, y1: 0.2, x2: 1, y2: 1,\n"
                                     "    stop: 0 #bbf, stop: 1 #55f);\n"
                                     "border: 1px solid #777;\n"
                                     "height: 10px;\n"
                                     "border-radius: 4px;\n"
                                     "}\n"
                                     "\n"
                                     "QSlider::add-page:horizontal {\n"
                                     "background: #fff;\n"
                                     "border: 1px solid #777;\n"
                                     "height: 10px;\n"
                                     "border-radius: 4px;\n"
                                     "}\n"
                                     "\n"
                                     "QSlider::handle:horizontal {\n"
                                     "background: qlineargradient(x1:0, y1:0, x2:1, y2:1,\n"
                                     "    stop:0 #eee, stop:1 #ccc);\n"
                                     "border: 1px solid #777;\n"
                                     "width: 35px;\n"
                                     "margin-top: -2px;\n"
                                     "margin-bottom: -2px;\n"
                                     "border-radius: 4px;\n"
                                     "}\n"
                                     "\n"
                                     "QSlider::handle:horizontal:hover {\n"
                                     "background: qlineargradient(x1:0, y1:0, x2:1, y2:1,\n"
                                     "    stop:0 #fff, stop:1 #ddd);\n"
                                     "border: 1px solid #444;\n"
                                     "border-radius: 4px;\n"
                                     "}\n"
                                     "\n"
                                     "QSlider::sub-page:horizontal:disabled {\n"
                                     "background: #bbb;\n"
                                     "border-color: #999;\n"
                                     "}\n"
                                     "\n"
                                     "QSlider::add-page:horizontal:disabled {\n"
                                     "background: #eee;\n"
                                     "border-color: #999;\n"
                                     "}\n"
                                     "\n"
                                     "QSlider::handle:horizontal:disabled {\n"
                                     "background: #eee;\n"
                                     "border: 1px solid #aaa;\n"
                                     "border-radius: 4px;\n"
                                     "}")
        self.imgSlider.setMaximum(500)
        self.imgSlider.setOrientation(QtCore.Qt.Horizontal)
        self.imgSlider.setObjectName("imgSlider")
        self.horizontalLayout_2.addWidget(self.imgSlider)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.previousButton = QtWidgets.QPushButton(Form)
        self.previousButton.setObjectName("previousButton")
        self.horizontalLayout.addWidget(self.previousButton)
        self.cellCombo = QtWidgets.QComboBox(Form)
        self.cellCombo.setEditable(False)
        self.cellCombo.setCurrentText("")
        self.cellCombo.setMaxVisibleItems(30)
        self.cellCombo.setObjectName("cellCombo")
        self.cellCombo.setEditable(True)
        line_edit = self.cellCombo.lineEdit()
        line_edit.setAlignment(QtCore.Qt.AlignCenter)
        line_edit.setReadOnly(True)
        self.horizontalLayout.addWidget(self.cellCombo)
        self.nextButtonButton = QtWidgets.QPushButton(Form)
        self.nextButtonButton.setObjectName("nextButtonButton")
        self.horizontalLayout.addWidget(self.nextButtonButton)
        self.verticalLayout.addLayout(self.horizontalLayout)

        self.retranslateUi(Form)
        self.cellCombo.setCurrentIndex(-1)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.nCurrent.setText(_translate("Form", "0"))
        self.label_18.setText(_translate("Form", "/"))
        self.nTotal.setText(_translate("Form", "0"))
        self.previousButton.setText(_translate("Form", "<"))
        self.nextButtonButton.setText(_translate("Form", ">"))


class StackImageViewer(QtWidgets.QWidget):

    def __init__(self, image_sets, set_levels = [1, 1], title='ImageViewer', combo_visbile = True):
        super().__init__()
        # set ui
        try:
            self.ui = uic.loadUi(".\\utils\\stack_image_viewer.ui", self)
        except:
            self.ui = UiViewer()
            self.ui.setupUi(self)
        self.setWindowTitle(title)
        # set image viewer
        self.imv = pg.ImageView(view=pg.PlotItem())
        self.imv.ui.roiBtn.hide()
        self.imv.ui.menuBtn.hide()
        self.ui.imageLayout.addWidget(self.imv)
        # set combo box visibility
        if not combo_visbile:
            self.ui.previousButton.setVisible(False)
            self.ui.nextButtonButton.setVisible(False)
            self.ui.cellCombo.setVisible(False)

        else:
            self.ui.previousButton.setVisible(True)
            self.ui.nextButtonButton.setVisible(True)
            self.ui.cellCombo.setVisible(True)

        # operations
        self.ui.imgSlider.valueChanged.connect(self.imageSliderChanged)
        self.ui.previousButton.clicked.connect(self.previousSet)
        self.ui.nextButtonButton.clicked.connect(self.nextSet)
        self.ui.cellCombo.currentIndexChanged.connect(self.displayCurrentSet)

        self.image_sets = []
        self.showImageSet(image_sets, set_levels)

    def showImageSet(self, images, set_levels=None):
        image_sets_tmp = self.converttoList(images)
        eql = list_equal(image_sets_tmp,self.image_sets)
        if not eql:
            self.image_sets = image_sets_tmp
            if set_levels is not None:
                self.set_levels = set_levels  # [0,0.7] for SIM image, [1, 1] for WF image. [0]*min [1]*max
            self.update()

    def update(self):
        self.uiUpdate()
        self.idx_sets = 0
        self.image_tmp = self.image_sets[self.idx_sets]
        num_z, dim_h, dim_w = np.shape(self.image_tmp)
        # set slider
        self.idx_z = int(self.ui.imgSlider.value())  # current image index
        # set z index to maxmium if last z index out of bound
        if self.idx_z >= num_z-1:
            self.idx_z = num_z-1

        self.ui.nTotal.setText(str(num_z))
        self.ui.nCurrent.setText(str(self.idx_z+1))
        self.ui.imgSlider.setMinimum(0)
        self.ui.imgSlider.setMaximum(num_z-1)

        self.level_max = np.amax(self.image_tmp)
        self.level_min = np.amin(self.image_tmp)

        self.imv.setImage((self.image_tmp[self.idx_z, :, :]).T, autoRange=False,
                          levels=(self.set_levels[0] * self.level_min, self.set_levels[1] * self.level_max))

    def uiUpdate(self):
        self.num_sets = len(self.image_sets)  # number of image sets
        # set combo box
        self.ui.cellCombo.clear()
        self.comboList = map(str, np.arange(self.num_sets))
        self.ui.cellCombo.addItems(self.comboList)


    def converttoList(self,input):
        datatype = type(input)
        image_list = []
        if datatype is np.ndarray:
            n_dim = input.ndim
            if  n_dim == 2: # if the data set is a 2d array
                input = input[np.newaxis,:,:]
            elif n_dim == 3:# if the data set is a 3d array
                pass
            image_list.append(input)   #
        elif datatype is list:
            if input:   # if input data is not empty convert each 2d array element to 3d array
                n_dim = input[0].ndim
                if n_dim == 2:
                    n_list = len(input)
                    for idx in range (n_list):
                        input[idx] = input[idx][np.newaxis,:,:]
                elif n_dim == 3:
                    pass
            else:
                pass
            image_list = input

        return image_list

    def imageSliderChanged(self):
        self.idx_z = int(self.ui.imgSlider.value())
        self.imv.setImage((self.image_tmp[self.idx_z, :, :]).T, autoRange=False,autoLevels=False)
        self.ui.nCurrent.setText(str(self.idx_z + 1))

    def previousSet(self):
        self.idx_sets = self.ui.cellCombo.currentIndex()
        self.idx_sets = self.idx_sets - 1
        if self.idx_sets == -1:
            self.idx_sets = self.num_sets - 1
        self.displaySet(self.idx_sets)
        self.ui.cellCombo.setCurrentIndex(self.idx_sets)

    def nextSet(self):
        self.idx_sets = self.ui.cellCombo.currentIndex()
        self.idx_sets = self.idx_sets + 1
        if self.idx_sets == self.num_sets:
            self.idx_sets = 0
        self.displaySet(self.idx_sets)
        self.ui.cellCombo.setCurrentIndex(self.idx_sets)

    def displayCurrentSet(self):
        try:
            self.displaySet(self.ui.cellCombo.currentIndex())
        except:
            pass

    def displaySet(self, idx_set):
        self.image_tmp = self.image_sets[idx_set]
        num_z, dim_h, dim_w = np.shape(self.image_tmp)
        # set slider
        self.idx_z = int(self.ui.imgSlider.value())  # current image index
        if self.idx_z > len(self.image_tmp):
            self.idx_z = len(self.image_tmp)

        self.ui.nTotal.setText(str(num_z))
        self.ui.nCurrent.setText(str(self.idx_z+1))
        self.ui.imgSlider.setMinimum(0)
        self.ui.imgSlider.setMaximum(num_z-1)

        self.level_max = np.amax(self.image_tmp)
        self.level_min = np.amin(self.image_tmp)
        self.imv.setImage((self.image_tmp[self.idx_z, :, :]).T, autoRange=False,
                          levels=(self.set_levels[0] * self.level_min, self.set_levels[1] * self.level_max))

        self.ui.cellCombo.setCurrentIndex(idx_set)

if __name__ == '__main__':
    import sys
    # input data as a list of 3d arrays
    image_sets_1 = [np.random.randint(0, 100, size=(3, 1024, 1024)), np.random.randint(0, 100, size=(3, 1024, 1024)),
                    np.random.randint(0, 100, size=(3, 1024, 1024))]
    # input data as a list of 2d arrays
    image_sets_2 = [np.random.randint(0, 100, size=(30, 30)), np.random.randint(0, 100, size=(30, 30)),
                    np.random.randint(0, 100, size=(30, 30))]
    # input data with a list of only one 3d array
    image_sets_3 = [np.random.randint(0, 100, size=(3, 30, 30))]
    # input data with a list of only one 2d array
    image_sets_4 = [np.random.randint(0, 100, size=(30, 30))]
    # input data with a 3d numpy array
    image_sets_5 = np.random.randint(0, 100, size=(3, 30, 30))
    # input data with a 2d numpy array
    image_sets_6 = np.random.randint(0, 100, size=(30, 30))
    # input contain empty element
    image_sets_7 = [[], np.random.randint(0, 100, size=(3, 30, 30)),
                    np.random.randint(0, 100, size=(3, 30, 30))]
    # input data as a list of 3d arrays with different dimensions
    image_sets_8 = [np.random.randint(0, 100, size=(3, 30, 30)), np.random.randint(0, 100, size=(4, 30, 30)),
                    np.random.randint(0, 100, size=(6, 30, 30))]

    app = QtWidgets.QApplication(sys.argv)

    display_image_widget = StackImageViewer(image_sets=image_sets_1, set_levels=[1, 1])

    display_image_widget.show()
    sys.exit(app.exec_())