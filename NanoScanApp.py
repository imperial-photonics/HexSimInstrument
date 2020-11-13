from ScopeFoundry import BaseMicroscopeApp

import logging

class NanoScanApp(BaseMicroscopeApp):
    name = 'NanoScanZ'

    def setup(self):
        from NanoScanHardware import NanoScanHW
        self.add_hardware(NanoScanHW)

        self.ui.show()
        self.ui.activateWindow()

if __name__=='__main__':
    import  sys
    app = NanoScanApp(sys.argv)
    sys.exit(app.exec_())