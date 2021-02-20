import sys
import numpy as np
import time
from numpy import pi,sin,cos,sqrt,floor
from PyQt5 import QtGui
from PyQt5.QtWidgets import QDesktopWidget,QMainWindow,QLabel,QApplication,QWidget
from PyQt5.QtCore import QTimer,Qt
from PIL import Image
from skimage.transform import resize
from skimage import img_as_bool

class ScreenDisplay(QMainWindow):
# class ScreenDisplay(QWidget):
    def __init__(self,monitor_number = 2, shift_orientation = pi/18, scale = 1.0, update_time=100,
                 wavelength = 0.561, NA = 0.75, magnification = 40, tune_scale = 40):
        super().__init__()
        self.writeUpdateTime(update_time)
        self.monitor_number = monitor_number
        self.orientation = shift_orientation # pi/18
        self.scale = scale # 2*pi/32
        self.counter = 0
        self.monitor = QDesktopWidget().screenGeometry(self.monitor_number)
        self.move(self.monitor.left(), self.monitor.top())
        self.wavelength = wavelength
        self.NA = NA
        self.M = magnification
        self.factor = tune_scale

        # self.img = self.imgGenerate()
        self.img = np.zeros([self.monitor.height(),self.monitor.width(),7])
        self.img488 = self.imgGenerator(0.488)
        self.img561 = self.imgGenerator(0.561)
        self.setPatterns(wavelength)
        # self.img = self.imgRead()
        self.screen = QLabel(self)
        self.setCentralWidget(self.screen)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.displayFrame)

    def writeUpdateTime(self,update_time):
        self._update_time = update_time

    def getScreenWidth(self):
        return self.monitor.width()

    def getScreenHeight(self):
        return self.monitor.height()

    def enableTimer(self):
        self.counter = 0

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
        self.show()


    def displayFrame(self):
        'Display the patterns in a sequential mode'
        if self.counter >= self.img.shape[2]:
            self.counter = 0
        print(self.counter)
        img_n = self.img[:,:,self.counter]
        img_buffer = QtGui.QImage(img_n.data.tobytes(), img_n.shape[1], img_n.shape[0], QtGui.QImage.Format_Grayscale8)
        self.screen.setPixmap(QtGui.QPixmap(img_buffer))
        self.counter += 1

    # def runningFrame(self,update_time):
    #     'Display the patterns in a sequential mode'
    #     if self.counter >= self.img.shape[2]:
    #         self.counter = 0
    #     print(self.counter)
    #     img_n = self.img[:,:,self.counter]
    #     img_buffer = QtGui.QImage(img_n.data.tobytes(), img_n.shape[1], img_n.shape[0], QtGui.QImage.Format_Grayscale8)
    #     self.screen.setPixmap(QtGui.QPixmap(img_buffer))
    #     self.counter += 1

    def imgGenerate(self):
        'Generate the polka pattern'
        w = self.monitor.width()
        h = self.monitor.height()
        X, Y = np.meshgrid(np.linspace(0, w, w), np.linspace(0, h, h))
        scalefactor = self.factor*self.NA*self.scale/self.wavelength/self.M
        p = 4 * pi / sqrt(3) / scalefactor
        r0 = 0.33 * p
        img = np.ones([h,w,7])
        # 40. * NA * scalefactor / 0.532 / M
        print("Orientation:",self.orientation,"Scalefactor:",scalefactor)
        for i in range (7):
            phase = i * 2 * pi / 7
            xr = -X * sin(-self.orientation) + Y * cos(-self.orientation) - 1.0 * p * phase / (2 * pi)
            yr = -X * cos(-self.orientation) - Y * sin(-self.orientation) + 2.0 / sqrt(3) * p * phase / (2 * pi)
            yi = floor(yr / (p * sqrt(3) / 2) + 0.5)
            xi = floor((xr / p) - (yi % 2.0) / 2.0 + 0.5) + (yi % 2.0) / 2.0
            y0 = yi * p * sqrt(3) / 2
            x0 = xi * p
            r = sqrt((xr - x0) ** 2 + (yr - y0) ** 2)
            img[:, :, i] = 255*(r < r0)

        return np.require(img, np.uint8, 'C')

    def imgGenerator(self, wavelength):
        'Generate the polka pattern'
        w = self.monitor.width()
        h = self.monitor.height()
        X, Y = np.meshgrid(np.linspace(0, w, w), np.linspace(0, h, h))
        scalefactor = self.factor*self.NA*self.scale/wavelength/self.M
        p = 4 * pi / sqrt(3) / scalefactor
        r0 = 0.33 * p
        img = np.ones([h,w,7])
        print("Orientation:",self.orientation,"Scalefactor:",scalefactor)
        for i in range (7):
            phase = i * 2 * pi / 7
            xr = -X * sin(-self.orientation) + Y * cos(-self.orientation) - 1.0 * p * phase / (2 * pi)
            yr = -X * cos(-self.orientation) - Y * sin(-self.orientation) + 2.0 / sqrt(3) * p * phase / (2 * pi)
            yi = floor(yr / (p * sqrt(3) / 2) + 0.5)
            xi = floor((xr / p) - (yi % 2.0) / 2.0 + 0.5) + (yi % 2.0) / 2.0
            y0 = yi * p * sqrt(3) / 2
            x0 = xi * p
            r = sqrt((xr - x0) ** 2 + (yr - y0) ** 2)
            img[:, :, i] = 255*(r < r0)

        return np.require(img, np.uint8, 'C')

    def imgRead(self):
        w = self.monitor.width()
        h = self.monitor.height()

        img = np.zeros((h, w,7))

        for i in range(7):
            name_tmp = str(i).zfill(3)
            file_name = './patterns/'+name_tmp + '.bmp'
            img_tmp = Image.open(file_name)
            img[ :, :,i] = resize(np.sum(np.array(img_tmp), axis=2) / 3,(h,w))

        return np.require(img,np.uint8,'C')

    def setPatterns(self,wavelength):
        self.wavelength = wavelength

        if self.wavelength == 0.488:
            self.img = self.img488

        elif self.wavelength == 0.561:
            self.img = self.img561

    def keyPressEvent(self, input):
        if input.key() ==Qt.Key_Escape:
            self.close()


if __name__=='__main__':
    app = QApplication(sys.argv)
    hex_slm = ScreenDisplay(monitor_number=1, shift_orientation=pi/9, scale=1,update_time=100)

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