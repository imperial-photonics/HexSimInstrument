from ScopeFoundry import HardwareComponent
from devices.SLM_device import SLMDev
import numpy as np
from numpy import sin,cos,pi,sqrt,floor
import subprocess, os, tifffile

class SLMHW(HardwareComponent):
    name = 'SLM_hardware'

    def setup(self):
        self.activation_mode = self.settings.New(name='activation_mode', dtype=str,
                                                  choices=['Activate', 'Deactivate'], initial='Deactivate', ro=False)
        self.add_operation(name='Get RO', op_func=self.getRO)
        self.setROindex = self.settings.New(name='Running Order', initial=3, vmax=3, vmin=0, spinbox_step=1,
                                            dtype=int, ro=False)
        # self.getAT = self.settings.New(name='Get Activation Type', dtype=bool,initial=False,ro=False)
        self.add_operation(name='Get Activation Type', op_func=self.getAT)
        self.add_operation(name='Get Activation State', op_func=self.actState)
        self.add_operation(name='Generate rep', op_func=self.repGen)
        self.add_operation(name='Rep to Repz11', op_func=self.repBuild)
        self.add_operation(name='Repsend', op_func=self.sendRep)
        self.sendBitplane = self.settings.New(name='sendBitplane', dtype=str,
                                              choices=[' ', 'Hexagon', 'Stripe'], initial=' ', ro=False)
        self.eraseBitplane = self.settings.New(name='eraseBitplane', dtype=int, ro=False, unit='px')


    def connect(self):
        # create an instance of the Device
        self.slm = SLMDev()
        self.slm.initiate()
        self.slm.open_usb_port()

        # Connect settings to hardware:
        self.activation_mode.hardware_set_func = self.activation
        self.sendBitplane.hardware_set_func = self.sendBitmaps
        self.eraseBitplane.hardware_set_func = self.slm.eraseBitplane
        self.setROindex.hardware_set_func = self.changeRO
        self.read_from_hardware()

    def disconnect(self):
        if hasattr(self, 'slm'):
            self.closeCheck()
            self.slm.close()
            del self.slm


    # define operations
    def activation(self,activation_mode):
        actMode = activation_mode
        if actMode == 'Activate':
            self.slm.activate()
        elif actMode == 'Deactivate':
            self.slm.deactivate()

    def getAT(self):
        self.slm.getActivationType()

    def getRO(self):
        self.slm.getRO()

    def changeRO(self, setROindex):
        self.slm.setRO(setROindex)
        self.settings["activation_mode"] = 'Activate'

    def closeCheck(self):
        self.slm.getState()
        if self.slm.state == 0x56:
            self.slm.deactivate()
            print('Deactivate successfully before close')

    def actState(self):
        self.slm.getState()
        if self.slm.state == 0x52:
            print('Maintenance – Software deactivated')
        elif self.slm.state == 0x53:
            print('Maintenance – Hardware and Software deactivated')
        elif self.slm.state == 0x56:
            print('Active')
        elif self.slm.state == 0x54:
            print('Maintenance – Hardware deactivated')
        else:
            print(self.slm.state)

    def sendBitmaps(self,sendBitplane):
        self.slm.deactivate()
        x = np.zeros(2048)
        y = np.arange(1536)

        nr = np.arange(8)
        for i in range(64):
            x[(i * 32):(i * 32 + 8)] = nr + i * 8
            x[(i * 32 + 8):(i * 32 + 16)] = nr + i * 8 + 512
            x[(i * 32 + 16):(i * 32 + 24)] = nr + i * 8 + 1024
            x[(i * 32 + 24):(i * 32 + 32)] = nr + i * 8 + 1536

        # x is now an array of interleaved x values in the correct places for sending to the SLM

        xv, yv = np.meshgrid(x, y)

        if sendBitplane == 'Stripe':
            p = 20  # pitch of the grating

            k = 2 * np.pi / p

            for i in range(8):
                t = i / 8
                kx = k * np.cos(t * np.pi)
                ky = k * np.sin(t * np.pi)

                grating = np.cos(kx * xv + ky * yv) > 0

                grating_bits = np.packbits(grating, bitorder='little')
                self.slm.sendBitplane(grating_bits, i)
                # use numpy packbits function to turn our array of boolean values into
                # an array of bytes with the bits set according to the boolean values.
                # grating_bits is something that you can now send to the SLM
        elif sendBitplane == 'Hexagon':
            p = 4 * pi / sqrt(3) / 0.2
            r0 = 0.4 * p
            img = np.ones([1536, 2048, 7])
            orientation = pi / 18

            for i in range(7):
                phase = i * 2 * pi / 7
                xr = -xv * sin(orientation) + yv * cos(orientation) - 1.0 * p * phase / (2 * pi)
                yr = -xv * cos(orientation) - yv * sin(orientation) + 2.0 / sqrt(3) * p * phase / (2 * pi)
                yi = floor(yr / (p * sqrt(3) / 2) + 0.5)
                xi = floor((xr / p) - (yi % 2.0) / 2.0 + 0.5) + (yi % 2.0) / 2.0
                y0 = yi * p * sqrt(3) / 2
                x0 = xi * p
                r = sqrt((xr - x0) ** 2 + (yr - y0) ** 2)
                # img[:, :, i] = r < r0
                img[:, :, i] = 255 * (r < r0)
                # img[:, :, i] = 255 * r
                hex_bits = np.packbits(img[:, :, i].astype('int'), bitorder='little')
                self.slm.sendBitplane(hex_bits, i+8)
            tifffile.imsave(f'hexagon.tiff', np.single(img))
        self.slm.repreload()

    def repBuild(self):
        fns = input('rep file name:')
        # output = subprocess.run([RepBuild, self.fns + '.rep', '-c', self.fns + '.repz11'],
        # stdout=subprocess.PIPE).communicate()[0]
        output = subprocess.run(['RepBuild', fns + '.rep', '-c', fns + '.repz11'], shell=True,
                                stdout=subprocess.PIPE)
        if output.returncode == 0:
            print(output.stdout.decode())
            print('repz file created')
        else:
            print(output.stderr)

    def sendRep(self):
        self.slm.close()
        fns = input('repz.11 file name:')

        os.chdir("repertoires")

        # output = subprocess.Popen([filepath, '-z', self.fns, '-d', '0175000881'],
        # stdout=subprocess.PIPE).communicate()[0]
        output = subprocess.run(['RepSender', '-z', fns + '.repz11', '-d', '0175000881'], shell=True,
                                stdout=subprocess.PIPE)
        if output.returncode == 0:
            print(output.stdout.decode())
        else:
            print(output.stderr)

        os.chdir(r"C:" + os.sep + "Users" + os.sep + "ML2618" + os.sep + "PycharmProjects"+ os.sep +
                 "Integrated-R11-SLM")

        self.connect()
        self.settings["activation_mode"] = 'Deactivate'


    def repGen(self):
        fns = input('new rep file name:')
        rep(fns)





