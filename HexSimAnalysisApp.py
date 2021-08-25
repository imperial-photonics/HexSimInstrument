"""
Intro:  GUI for HexSIM data analysis.
        The GUI is based on the framework of ScopeFoundry.

Author: Hai Gong
Email:  h.gong@imperial.ac.uk
Time:   Aug 2021
Address:Imperial College London
"""
import os
from ScopeFoundry import BaseMicroscopeApp

class HexSimApp(BaseMicroscopeApp):
    name = 'HexSimAnalysisApp'

    def __init__(self, *kwds):

        super().__init__(*kwds)
        self.settings['save_dir'] = "./measurement"
        self.settings.save_dir.hardware_set_func = self.setDirFunc

    def setup(self):
        print("Adding measurement components")
        from modules.HexSimAnalysis import HexSimAnalysis
        self.add_measurement(HexSimAnalysis(self))
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
    from PyQt5.QtGui import QIcon

    app = HexSimApp(sys.argv)
    # collapse the measurement tree
    app.ui.col_splitter.setSizes([0, 0, 100])
    app.ui.splitter.setSizes([0,100])
    # remove logging and console sub window
    app.ui.mdiArea.removeSubWindow(app.console_subwin)
    app.ui.mdiArea.removeSubWindow(app.logging_subwin)

    logo_icon = QIcon('.\\ui\\icon_attribute_pixel_perfect.png')
    app.ui.setWindowIcon(logo_icon)


    sys.exit(app.exec_())
