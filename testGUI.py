import sys
from PyQt5 import QtWidgets, uic

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self,*args,**kwargs):
        super(MainWindow,self).__init__()
        uic.loadUi("test.ui",self)
        self.camButton.setText('ON')
        self.camButton.clicked.connect(self.camButtonPressed)
        self.textLabel.setText('Not Clicked')
        self.show()

    def camButtonPressed(self):
        if self.camButton.text()=='ON':
            self.camButton.setText('OFF')
            print('OFF')
        elif self.camButton.text()=='OFF':
            self.camButton.setText('ON')
            print('ON')

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)

    window = MainWindow()

    window.show()
    app.exec_()
