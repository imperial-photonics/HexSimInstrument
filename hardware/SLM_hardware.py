from ScopeFoundry import HardwareComponent
from devices.SLM_device import SLMDev
import numpy as np
from numpy import meshgrid, sin, cos, pi, sqrt, floor
import cv2
import subprocess, os, tifffile

class SLMHW(HardwareComponent):
    name = 'SLM_hardware'

    def setup(self):
        self.activation= self.add_logged_quantity(name='', dtype=str, choices=['Activate', 'Deactivate'],
                                                  initial='Deactivate', ro=False)
        self.settings.activation_state = self.add_logged_quantity(name='State', dtype=str, ro=True)
        self.settings.rep_name = self.add_logged_quantity(name="Repertoire name", dtype=str, ro=True)
        self.settings.activation_type = self.add_logged_quantity(name='Activation type', dtype=str, ro=True)
        self.settings.roIndex = self.add_logged_quantity(name='Running Order index', spinbox_step=1, dtype=int, ro=False)
        self.settings.roName = self.add_logged_quantity(name='Running Order name', dtype=str, ro=True)
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
        self.activation.hardware_set_func = self.setActState
        self.sendBitplane.hardware_set_func = self.sendBitmaps
        self.eraseBitplane.hardware_set_func = self.slm.eraseBitplane
        self.settings.activation_state.connect_to_hardware(read_func=self.getActState)
        self.settings.rep_name.connect_to_hardware(read_func=self.repName)
        self.settings.activation_type.connect_to_hardware(read_func=self.getAT)
        self.settings.roIndex.connect_to_hardware(write_func=self.changeRO, read_func=self.getRoIndex)
        self.settings.roName.connect_to_hardware(read_func=self.getRoName)
        self.read_from_hardware()

    def disconnect(self):
        if hasattr(self, 'slm'):
            self.closeCheck()
            self.slm.close()
            del self.slm


    # define operations
    def setActState(self, setState):
        if hasattr(self, 'slm'):
            if setState == 'Activate':
                self.slm.activate()
            elif setState == 'Deactivate':
                self.slm.deactivate()
            self.updateHardware()

    def repName(self):
        if hasattr(self, 'slm'):
            return self.slm.getRepName()

    def getAT(self):
        if hasattr(self, 'slm'):
            re = self.slm.getActivationType()
            if re == 1:
                return 'Immediate'
            elif re == 2:
                return 'Software'
            elif re == 4:
                return 'Hardware'

    def getRoIndex(self):
        if hasattr(self, 'slm'):
            return self.slm.getRO()[0]

    def getRoName(self):
        if hasattr(self, 'slm'):
            return self.slm.getRO()[1]

    def changeRO(self, setROindex):
        if hasattr(self, 'slm'):
            self.slm.setRO(setROindex)
            self.updateHardware()

    def closeCheck(self):
        if self.slm.getState() == 0x56:
            self.slm.deactivate()
            print('Deactivate successfully before close')

    def getActState(self):
        if hasattr(self, 'slm'):
            state = self.slm.getState()
            if state == 0x50:
                return 'Repertoire loading'
            elif state == 0x51:
                return 'Starting'
            elif state == 0x52:
                return 'Software deactivated'
            elif state == 0x53:
                return 'Hardware and Software deactivated'
            elif state == 0x54:
                return 'Hardware deactivated'
            elif state == 0x55:
                return 'Activating'
            elif state == 0x56:
                return 'Active'
            elif state == 0x57:
                return 'No Repertoire available'
            else:
                raise Exception('Unrecognised activation state')

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
        # rep(fns)

    def genStripes(self):
        """Generate sinusoidal striped holograms"""
        h = 1536
        w = 2048
        p = 10
        x = np.arange(2048)
        y = np.arange(1536)
        nr = np.arange(8)
        xv, yv = np.meshgrid(x, y)
        g = np.zeros((h, w), dtype=np.uint8)
        k = 1 / p
        for i in range(7):
            t = i * pi / 7
            px = k * (i - 3)
            py = 2 * k
            g = g + (cos(px * xv * pi + py * yv * pi) > 0) * (2 ** i)
            # cv2.imwrite('stripes_%d_%.2f.png'%(i,p),(cos(px * xv * np.pi + py * yv * np.pi) > 0)*1, [cv2.IMWRITE_PNG_BILEVEL, 1])
            cv2.imwrite('stripes_%d_%.2f.png' % (i, p), (cos(px * xv * np.pi + py * yv * np.pi) > 0) * 1,
                        [cv2.IMWRITE_PNG_BILEVEL, 1])

    def genHexgans(self):
        """Generate hexagonal holograms hexagons"""
        h = 1536
        w = 2048

        nm = 10 ** -9
        mm = 0.001
        wavelength = 520 * nm
        f = 160 * mm
        p_mask = 2.1 * mm
        # pitch of the pinhole mask
        u_mask = mm * 2.1 * sqrt(3) / 2
        # u_mask: coordinate in x direction in FT plane
        p = 1 / (u_mask / wavelength / f) / (0.0082 * mm)
        # SLM's pixel size: 0.0082mm
        # p: pitch of the hexagonal hologram (distance between two dots)

        r0 = sqrt(sqrt(3) / np.pi) * p / 2
        # The relationship between r0 and p makes sure the black and white part in the hologram has the same area for the
        # maximum diffraction.

        deg_num = 18
        orientation = deg_num * pi / 180

        distort = (1 - (66 / 2 / 132) ** 2) ** 0.5
        # The distorted pattern corresponds to the angle between the laser source and the imaging system
        # (the imaging system was not perpendicular to the SLM)
        x, y = meshgrid(np.arange(w) * distort, np.arange(h))

        for i in range(7):
            phase = i * 2 * pi / 7
            xr = -x * sin(orientation) + y * cos(orientation) - 1.0 * p * phase / (2 * pi)
            yr = -x * cos(orientation) - y * sin(orientation) + 2.0 / sqrt(3) * p * phase / (2 * pi)
            yi = np.floor(yr / (p * sqrt(3) / 2) + 0.5)
            xi = np.floor((xr / p) - (yi % 2.0) / 2.0 + 0.5) + (yi % 2.0) / 2.0
            y0 = yi * p * sqrt(3) / 2
            x0 = xi * p
            r = sqrt((xr - x0) ** 2 + (yr - y0) ** 2)
            print(np.sum(r > r0) / np.sum(r < r0))
            hol = (r < r0) * 1
            # hol: hologram
            mask = np.zeros((h, w))
            for n in range(int(w / 2 - 40), int(w / 2 + 41)):
                mask[:, n] = 1
            for n in range(int(h / 2 - 40), int(h / 2 + 41)):
                mask[n, :] = 1
            hol_mask = hol * mask

            # img[:, :, i] = 255 * r

            cv2.imwrite(f'hex_{i}_p{p_mask}_deg{deg_num}.png', hol_mask,
                        [cv2.IMWRITE_PNG_BILEVEL, 1])
            # print(f'p:{p*8.2/4.8/(sqrt(3)/2)}')


    def updateHardware(self):
        if hasattr(self, 'slm'):
            self.settings.activation_state.read_from_hardware()
            self.settings.rep_name.read_from_hardware()
            self.settings.activation_type.read_from_hardware()
            self.settings.roIndex.read_from_hardware()
            self.settings.roName.read_from_hardware()





