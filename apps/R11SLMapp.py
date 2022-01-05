from ScopeFoundry import BaseMicroscopeApp

class InSLMApp(BaseMicroscopeApp):
    # this is the name of the microscope that ScopeFoundry uses
    # when storing data
    name = 'Integrated R11slm'

    # You must define a setup function that adds all the
    # capablities of the microscope and sets default settings
    def setup(self):
        # Add hardware components
        print("Adding Hardware Components")
        from hardware.SLM_hardware import SLMHW
        from hardware.camera_hardware import TL_CS2100M_HW
        from hardware.ni_co_hardware import NI_CO_hw
        from hardware.ni_do_hardware import NI_DO_hw

        self.add_hardware(SLMHW(self))
        self.add_hardware(TL_CS2100M_HW(self))
        self.add_hardware(NI_CO_hw(self))
        self.add_hardware(NI_DO_hw(self))

        #Add Measurement components
        from measurements.camera_measurement import ThorlabsMeasurement
        self.add_measurement(ThorlabsMeasurement)

        # show ui
        self.ui.show()
        self.ui.activateWindow()

if __name__ == '__main__':
    import sys

    app = InSLMApp(sys.argv)
    sys.exit(app.exec_())