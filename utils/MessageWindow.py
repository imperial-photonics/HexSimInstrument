import pyqtgraph as pg
from PyQt5 import uic
from PyQt5.QtWidgets import QWidget, QTableWidgetItem,QHeaderView
from PyQt5.QtCore import Qt

class CalibrationResults(QWidget):

    """
    This window display the Winier filter and other debug data
    """

    def __init__(self, h):
        super().__init__()
        self.ui = uic.loadUi('.\\ui\\calibration_results.ui',self)
        self.h = h
        self.setWindowTitle('Calibration results')
        self.showCurrentTable()
        self.showWienerFilter()
        self.ui.wienerfilterLayout.addWidget(self.wienerfilterWidget)

    def update(self,h):
        # update the widgets with new data
        self.h = h
        self.showCurrentTable()
        self.wienerfilterWidget.setImage(self.h.wienerfilter_store, autoRange=True, autoLevels=True)

    def showCurrentTable(self):
        def table_item(element):
            return QTableWidgetItem(str(element).lstrip('[').rstrip(']'))

        table = self.ui.currentTable

        table.setItem(0, 0, table_item(self.h.kx_input[0]))
        table.setItem(0, 1, table_item(self.h.kx_input[1]))
        table.setItem(0, 2, table_item(self.h.kx_input[2]))

        table.setItem(1, 0, table_item(self.h.ky_input[0]))
        table.setItem(1, 1, table_item(self.h.ky_input[1]))
        table.setItem(1, 2, table_item(self.h.ky_input[2]))

        table.setItem(2, 0, table_item(self.h.kx[0]))
        table.setItem(2, 1, table_item(self.h.kx[1]))
        table.setItem(2, 2, table_item(self.h.kx[2]))

        table.setItem(3, 0, table_item(self.h.ky[0]))
        table.setItem(3, 1, table_item(self.h.ky[1]))
        table.setItem(3, 2, table_item(self.h.ky[2]))

        table.setItem(4, 0, table_item(self.h.p[0]))
        table.setItem(4, 1, table_item(self.h.p[1]))
        table.setItem(4, 2, table_item(self.h.p[2]))

        table.setItem(5, 0, table_item(self.h.ampl[0]))
        table.setItem(5, 1, table_item(self.h.ampl[1]))
        table.setItem(5, 2, table_item(self.h.ampl[2]))

        # Table will fit the screen horizontally
        self.currentTable.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

    def showWienerFilter(self):
        self.wienerfilterWidget = pg.ImageView()
        self.wienerfilterWidget.aspectRatioMode = Qt.KeepAspectRatio
        self.wienerfilterWidget.ui.roiBtn.hide()
        self.wienerfilterWidget.ui.menuBtn.hide()
        self.wienerfilterWidget.ui.histogram.hide()
        self.wienerfilterWidget.setImage(self.h.wienerfilter_store, autoRange=True, autoLevels=True)
        self.wienerfilterWidget.adjustSize()