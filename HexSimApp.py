""" Code for instrument control of the HexSIM system. The GUI is based on the framework of ScopeFoundry.
Author: Hai Gong
Email: h.gong@imperial.ac.uk
Oct 2020
"""
from ScopeFoundry import BaseMicroscopeApp

class HexSimApp(BaseMicroscopeApp):
    # this is the name of the microscope that ScopeFoundry uses
    # when storing data
    name = 'hex_sim_control'

    def __init__(self, *kwds):
        """
        We need an __init__ since we want to put a new save directory
        """

        super().__init__(*kwds)  # *kwds is needed since in the main we pass as argument sys.argv. Without
        # the *kwds this will give a problem

        # self.settings.save_dir.update_value(QtWidgets.QFileDialog.getExistingDirectory(directory = "D:\\Data\\temp"))
        self.settings['save_dir'] = "D:\\Data\\Test"  # PUT ALWAYS TWO SLASHES!!!!
        self.settings.save_dir.hardware_set_func = self.setDirFunc  # calls set dir func when the save_dir widget is changed

    def setup(self):
        print("Adding Hardware Components")
        from CameraHardware import HamamatsuHardware
        from LaserHardware import Laser488HW,Laser561HW
        from ScreenHardware import ScreenHW

        self.add_hardware(ScreenHW(self))
        self.add_hardware(HamamatsuHardware(self))
        self.add_hardware(Laser488HW(self))
        self.add_hardware(Laser561HW(self))

        print("Adding measurement components")
        from CameraMeasurement import HamamatsuMeasurement

        self.add_measurement(HamamatsuMeasurement(self))

        # show ui
        self.ui.show()
        self.ui.activateWindow()

    def setDirFunc(self, val=None):
        """
        Gets called every time we modify the directory.
        If it does not exist, we create a new one
        """

        if not os.path.isdir(self.settings['save_dir']):
            os.makedirs(self.settings['save_dir'])

    def file_browser(self):

        if self.settings.save_dir.is_dir:
            fname = QtWidgets.QFileDialog.getExistingDirectory(directory=self.settings.save_dir.val)
        else:
            fname, _ = QtWidgets.QFileDialog.getOpenFileName(directory=self.settings.save_dir.val)
        self.settings.save_dir.log.debug(repr(fname))
        if fname:
            self.settings.save_dir.update_value(fname)

    def connect_to_browse_widgets(self, lineEdit, pushButton):
        assert type(lineEdit) == QtWidgets.QLineEdit
        self.settings.save_dir.connect_to_widget(lineEdit)

        assert type(pushButton) == QtWidgets.QPushButton
        pushButton.clicked.connect(self.file_browser)


if __name__ == '__main__':
    import sys

    app = HexSimApp(sys.argv)
    sys.exit(app.exec_())