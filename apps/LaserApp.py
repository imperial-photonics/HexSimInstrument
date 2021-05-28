"""
Intro:  GUI application for laser control with serial commands.
        Laser models: Oxxius LBX Diode laser, LCX DPSS laser
Author: Hai Gong
Email:  h.gong@imperial.ac.uk
Time:   Oct 2020
"""

from ScopeFoundry import BaseMicroscopeApp

class LaserMicroscopeApp(BaseMicroscopeApp):

    name = 'laser_control'
    def setup(self):
        print("Adding Hardware Components")
        from hardware.LaserHardware import Laser488HW,Laser561HW
        self.add_hardware(Laser488HW(self))
        self.add_hardware(Laser561HW(self))
        # show ui
        self.ui.show()
        self.ui.activateWindow()

if __name__ == '__main__':
    import sys
    app = LaserMicroscopeApp(sys.argv)
    sys.exit(app.exec_())