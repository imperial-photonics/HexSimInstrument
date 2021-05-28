from ScopeFoundry import BaseMicroscopeApp


class SLMMicroscopeApp(BaseMicroscopeApp):
    # this is the name of the microscope that ScopeFoundry uses
    # when storing data
    name = 'slm_microscope'

    # You must define a setup function that adds all the
    # capabilities of the microscope and sets default settings
    def setup(self):
        # Add App wide settings

        # Add hardware components
        print("Adding Hardware Components")
        from hardware.ScreenHardware import ScreenHW
        self.add_hardware(ScreenHW(self))

        # show ui
        self.ui.show()
        self.ui.activateWindow()


if __name__ == '__main__':
    import sys

    app = SLMMicroscopeApp(sys.argv)
    sys.exit(app.exec_())
