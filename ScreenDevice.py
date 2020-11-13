import sys
import numpy as np
import time
from numpy import pi,sin,cos,sqrt,floor
from PyQt5 import QtGui
from PyQt5.QtWidgets import QDesktopWidget,QMainWindow,QLabel,QApplication
from PyQt5.QtCore import QTimer,Qt

class ScreenDisplay(QMainWindow):
    def __init__(self,monitor_number = 1, shift_orientation = pi/9, scale = pi/16, update_time=100):
        super().__init__()
        self.writeUpdateTime(update_time)
        self.monitor_number = monitor_number
        self.orientation = shift_orientation # pi/18
        self.scale = scale # 2*pi/32
        self.counter = 0
        self.monitor = QDesktopWidget().screenGeometry(self.monitor_number)
        self.move(self.monitor.left(), self.monitor.top())
        self.img = self.imgGenerate()
        self.screen = QLabel(self)
        self.setCentralWidget(self.screen)
        self.timer = QTimer(self)
        self.timer.stop()

    def writeUpdateTime(self,update_time):
        self._update_time = update_time

    def getScreenWidth(self):
        return self.monitor.width()

    def getScreenHeight(self):
        return self.monitor.height()

    def enableTimer(self):
        self.timer.timeout.connect(self.displayFrame)
        self.timer.start(self._update_time)
        print('Display once')

    def disableTimer(self):
        self.timer.stop()

    def changeTimer(self,update_time):
        self.timer.setInterval(update_time)

    def displayFrameN(self,frame_number):
        'Display a selected pattern'
        img_n = self.img[:,:,frame_number]
        img_buffer = QtGui.QImage(img_n.data.tobytes(), img_n.shape[1], img_n.shape[0], QtGui.QImage.Format_Grayscale8)
        self.screen.setPixmap(QtGui.QPixmap(img_buffer))
        self.showFullScreen()

    def displayFrame(self):
        'Display the patterns in a sequential mode'
        if self.counter >= self.img.shape[2]:
            self.counter = 0
        print(self.counter)
        img_n = self.img[:,:,self.counter]
        img_buffer = QtGui.QImage(img_n.data.tobytes(), img_n.shape[1], img_n.shape[0], QtGui.QImage.Format_Grayscale8)
        self.screen.setPixmap(QtGui.QPixmap(img_buffer))
        self.counter += 1


    def imgGenerate(self):
        'Generate the polka pattern'
        w = self.monitor.width()
        h = self.monitor.height()
        X, Y = np.meshgrid(np.linspace(0, w, w), np.linspace(0, h, h))
        p = 4 * pi / sqrt(3) / self.scale
        r0 = 0.33 * p
        img = np.ones([h,w,7])

        for i in range (7):
            phase = i * 2 * pi / 7
            xr = -X * sin(self.orientation) + Y * cos(self.orientation) - 1.0 * p * phase / (2 * pi)
            yr = -X * cos(self.orientation) - Y * sin(self.orientation) + 2.0 / sqrt(3) * p * phase / (2 * pi)
            yi = floor(yr / (p * sqrt(3) / 2) + 0.5)
            xi = floor((xr / p) - (yi % 2.0) / 2.0 + 0.5) + (yi % 2.0) / 2.0
            y0 = yi * p * sqrt(3) / 2
            x0 = xi * p
            r = sqrt((xr - x0) ** 2 + (yr - y0) ** 2)
            img[:, :, i] = 255*(r < r0)

        return np.require(img, np.uint8, 'C')


    def keyPressEvent(self, input):
        if input.key() ==Qt.Key_Escape:
            self.close()

if __name__=='__main__':
    app = QApplication(sys.argv)
    hex_slm = ScreenDisplay(monitor_number=1, shift_orientation=pi/9, scale=pi/16,update_time=100)

    hex_slm.showFullScreen()

    print('slm started')
    hex_slm.displayFrameN(1)
    print('slm display frame 1')

    time.sleep(5)
    print('sleep over')
    # hex_slm.close()

    hex_slm.enableTimer()
    print('timer enabled')

    # hex_slm.showFullScreen()
    # time.sleep(5)
    # hex_slm.disable_timer()
    # hex_slm.showFullScreen()
    sys.exit(app.exec())