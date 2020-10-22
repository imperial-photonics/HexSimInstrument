from ScopeFoundry import BaseMicroscopeApp

class LaserMicroscopeApp(BaseMicroscopeApp):
    # this is the name of the microscope that ScopeFoundry uses
    # when storing data
    name = 'laser_control'
    def setup(self):
        print("Adding Hardware Components")
        from LaserHardware import Laser488HW,Laser561HW
        self.add_hardware(Laser488HW(self))
        self.add_hardware(Laser561HW(self))
        # show ui
        self.ui.show()
        self.ui.activateWindow()

if __name__ == '__main__':
    import sys

    app = LaserMicroscopeApp(sys.argv)
    sys.exit(app.exec_())