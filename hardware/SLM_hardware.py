from ScopeFoundry import HardwareComponent
from devices.SLM_device import SLMDev
import numpy as np
from numpy import meshgrid, sin, cos, pi, sqrt, floor
import subprocess, os, tifffile, cv2, time
import cupy as cp

use_cupy = True
class SLMHW(HardwareComponent):
    name = 'SLM_hardware'

    def setup(self):
        self.add_operation(name='Activate', op_func=self.act)
        self.add_operation(name='Deactivate', op_func=self.deact)
        self.settings.activation_state = self.add_logged_quantity(name='State', dtype=str, ro=True)
        self.settings.rep_name = self.add_logged_quantity(name="Repertoire name", dtype=str, ro=True)
        self.settings.activation_type = self.add_logged_quantity(name='Activation type', dtype=str, ro=True)
        # self.settings.BPn = self.add_logged_quantity(name='Bitplane count', spinbox_step=1, dtype=int, ro=True)
        self.settings.roIndex = self.add_logged_quantity(name='Running Order index', spinbox_step=1, dtype=int, ro=False)
        self.settings.roName = self.add_logged_quantity(name='Running Order name', dtype=str, ro=True)
        # self.add_operation(name='Rep to Repz11', op_func=self.repBuild)
        # self.add_operation(name='Repsend', op_func=self.sendRep)
        # self.sendBitplane = self.settings.New(name='sendBitplane', dtype=str,
        #                                       choices=[' ', 'Hexagon', 'Stripe'], initial=' ', ro=False)
        # self.eraseBitplane = self.settings.New(name='eraseBitplane', dtype=int, ro=False, unit='px')


    def connect(self):
        # create an instance of the Device
        self.slm = SLMDev()
        self.slm.initiate()
        self.slm.open_usb_port()

        # Connect settings to hardware:
        # self.activation.hardware_set_func = self.setActState
        # self.sendBitplane.hardware_set_func = self.sendBitmaps
        # self.eraseBitplane.hardware_set_func = self.slm.eraseBitplane
        self.settings.activation_state.connect_to_hardware(read_func=self.getActState)
        self.settings.rep_name.connect_to_hardware(read_func=self.repName)
        self.settings.activation_type.connect_to_hardware(read_func=self.getAT)
        self.settings.roIndex.connect_to_hardware(write_func=self.changeRO, read_func=self.getRoIndex)
        self.settings.roName.connect_to_hardware(read_func=self.getRoName)
        self.read_from_hardware()

        # hologram parameters
        self.xpix = self.slm.xpix
        self.ypix = self.slm.ypix

    def disconnect(self):
        if hasattr(self, 'slm'):
            self.closeCheck()
            self.slm.close()
            del self.slm


    # define operations
    # def setActState(self, setState):
    #     if hasattr(self, 'slm'):
    #         if setState == 'Activate':
    #             self.slm.activate()
    #         elif setState == 'Deactivate':
    #             self.slm.deactivate()
    #         self.updateHardware()

    def act(self):
        if hasattr(self, 'slm'):
            self.slm.activate()
            self.updateHardware()

    def deact(self):
        if hasattr(self, 'slm'):
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
        if self.slm.getState() == 0x54 or self.slm.getState() == 0x56:
            self.slm.deactivate()

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

    def repSend(self, fn):
        """Send the repz.11 file to the board. fn: file name."""
        self.slm.close()
        os.chdir('./gen_repertoires')
        output = subprocess.run(['RepSender', '-z', fn, '-d', '0175000881'], shell=True,
                                stdout=subprocess.PIPE)
        if output.returncode == 0:
            print(output.stdout.decode())
        else:
            print(output.stderr)
        os.getcwd()
        os.chdir('..')
        self.connect()

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
        img = []
        imgNameList = []
        for i in range(7):
            t = i * pi / 7
            px = k * (i - 3)
            py = 2 * k
            g = g + (cos(px * xv * pi + py * yv * pi) > 0) * (2 ** i)
            hol = g * 1
            # hol = (cos(px * xv * np.pi + py * yv * np.pi) > 0) * 1
            timestamp = time.strftime("%y%m%d_%H%M%S", time.localtime())
            path = os.path.join('./gen_repertoires')
            imgN = f'stripes_%d_%.2f_{timestamp}.png' % (i, p)
            cv2.imwrite(os.path.join('./gen_repertoires', imgN), hol,
                        [cv2.IMWRITE_PNG_BILEVEL, 1])
            img.append(hol)
            imgNameList.append(imgN)
        return img, imgNameList, timestamp

    def genHexagons(self, lamda, pm, deg_num):
        """Generate hexagonal holograms hexagons"""
        print('genHexgans')
        h = 1536
        w = 2048

        nm = 10 ** -9
        mm = 0.001
        wavelength = lamda * nm
        f = 160 * mm
        p_mask = pm * mm
        # pitch of the pinhole mask
        u_mask = p_mask * sqrt(3) / 2
        # u_mask: coordinate in x direction in FT plane
        p = 1 / (u_mask / wavelength / f) / (0.0082 * mm)
        # SLM's pixel size: 0.0082mm
        # p: pitch of the hexagonal hologram (distance between two dots)

        r0 = sqrt(sqrt(3) / np.pi) * p / 2
        # The relationship between r0 and p makes sure the black and white part in the hologram has the same area for the
        # maximum diffraction.

        orientation = deg_num * pi / 180
        # distort = (1 - (66 / 2 / 132) ** 2) ** 0.5
        distort = 1
        # The distorted pattern corresponds to the angle between the laser source and the imaging system
        # (the imaging system was not perpendicular to the SLM)
        x, y = meshgrid(np.arange(w) * distort, np.arange(h))
        img = []
        imgNameList = []
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
            hol = (r < r0) * 1  # hol: hologram
            # img[:, :, i] = 255 * r
            timestamp = time.strftime("%y%m%d_%H%M%S", time.localtime())
            path = os.path.join('./gen_repertoires')
            imgN = f'hex_{i}_deg{deg_num}_{timestamp}.png'
            cv2.imwrite(os.path.join('./gen_repertoires', imgN), hol,
                        [cv2.IMWRITE_PNG_BILEVEL, 1])
            img.append(hol)
            imgNameList.append(imgN)
        return img, imgNameList, timestamp

    # def genCorrection(self, ast1_f, ast2_f, coma1_f, coma2_f, tref1_f, tref2_f):
    #     os.chdir('./gen_repertoires')
    #     xpix = self.xpix
    #     ypix = self.ypix
    #     beams = 3
    #     N_iterations = 3  # number of iterations
    #     if use_cupy:
    #         x, y = cp.meshgrid(cp.arange(xpix), cp.arange(ypix))
    #         # place the pixels in negative and positive axes
    #         x = (x - xpix / 2) / (xpix)
    #         y = (y - ypix / 2) / (ypix)
    #         Phi = cp.random.random((ypix, xpix)) * 2 * np.pi
    #         Tau = cp.zeros((ypix, xpix, beams), dtype=cp.double)  # phase tilt
    #         Psi = cp.zeros(beams, dtype=cp.double)
    #         F = cp.zeros(beams, dtype=cp.complex_)
    #         G = cp.zeros((ypix, xpix, beams), dtype=cp.complex_)
    #     else:
    #         x, y = np.meshgrid(np.arange(xpix), np.arange(ypix))
    #         # place the pixels in negative and positive axes
    #         x = (x - xpix / 2) / (xpix)
    #         y = (y - ypix / 2) / (ypix)
    #
    #         Phi = np.random.random((ypix, xpix)) * 2 * np.pi
    #         Tau = np.zeros((ypix, xpix, beams), dtype=np.double)  # phase tilt
    #         Psi = np.zeros(beams, dtype=np.double)
    #         F = np.zeros(beams, dtype=np.complex_)
    #         G = np.zeros((ypix, xpix, beams), dtype=np.complex_)
    #
    #     rSqaure = x ** 2 + y ** 2
    #     ast1 = ast1_f * sqrt(6) * (x ** 2 - y ** 2)
    #     ast2 = ast2_f * 2 * sqrt(6) * x * y
    #     coma1 = coma1_f * 2 * sqrt(2) * (3 * rSqaure - 2) * x
    #     coma2 = coma2_f * 2 * sqrt(2) * (3 * rSqaure - 2) * y
    #     tref1 = tref1_f * 2 * sqrt(2) * (4 * x ** 2 - 3 * rSqaure) * x
    #     tref2 = tref2_f * 2 * sqrt(2) * (3 * rSqaure - 4 * y ** 2) * y
    #     abb = ast1 + ast2 + coma1 + coma2 + tref1 + tref2
    #
    #     nm = 10 ** -9
    #     mm = 0.001
    #     wavelength = 520 * nm
    #     f = 250 * mm
    #     p_mask = 2.1 * mm  # pitch of the pinhole mask
    #     u_mask = p_mask * np.sqrt(3) / 2  # coordinate in x direction in FT plane
    #     p = 1 / (u_mask / wavelength / f) / (0.0082 * mm)
    #
    #     # initialise Tau, 6 ramps of increasing angle, with a yramp to shift away from central axis
    #     if use_cupy:
    #         for i in range(0, beams):
    #             xp = xpix / p * 2 * np.pi * cp.cos(2 * i * np.pi / 3 + np.pi / 18)
    #             yp = ypix / p * 2 * np.pi * cp.sin(2 * i * np.pi / 3 + np.pi / 18)
    #             Tau[:, :, i] = x * xp + y * yp + abb
    #         Psi = cp.zeros(3)
    #         for k in range(0, N_iterations):
    #             F = cp.sum(cp.exp(1j * (-Tau + Phi.reshape((xpix, ypix, 1)))),
    #                        (0, 1))  # DFT to find DC term at Fourier plane
    #             # extract just the phase (set amplitude to 1)
    #             Psi[0] = cp.angle(F[0])
    #             Psi[1] = cp.angle(F[0]) + 2 * np.pi / 7
    #             Psi[2] = cp.angle(F[0]) + 6 * np.pi / 7
    #             A = cp.abs(F)  # Amplitude
    #             # G and Phi do the inverse FT
    #             G = cp.exp(1j * (Tau + Psi))  # calculate the terms needed for summation
    #             Phi = np.pi * (cp.real(
    #                 cp.sum(G, axis=2)) < 0)  # sum the terms and take the argument, which is used as next phi
    #             img = Phi.get() * 1
    #     else:
    #         for i in range(0, beams):
    #             xp = xpix / p * 2 * np.pi * np.cos(2 * i * np.pi / 3 + np.pi / 18)
    #             yp = ypix / p * 2 * np.pi * np.sin(2 * i * np.pi / 3 + np.pi / 18)
    #             Tau[:, :, i] = x * xp + y * yp + abb
    #         Psi = np.zeros(3)
    #         for k in range(0, N_iterations):
    #             F = np.sum(np.exp(1j * (-Tau + Phi.reshape((xpix, ypix, 1)))),
    #                        (0, 1))  # DFT to find DC term at Fourier plane
    #             # extract just the phase (set amplitude to 1)
    #             Psi[0] = np.angle(F[0])
    #             Psi[1] = np.angle(F[0]) + 2 * np.pi / 7
    #             Psi[2] = np.angle(F[0]) + 6 * np.pi / 7
    #             A = np.abs(F)  # Amplitude
    #             G = np.exp(1j * (Tau + Psi))  # calculate the terms needed for summation
    #             Phi = np.pi * (np.real(
    #                 np.sum(G, axis=2)) < 0)  # sum the terms and take the argument, which is used as next phi
    #             img = Phi * 1
    #
    #     cv2.imwrite(f'hol.png', img, [cv2.IMWRITE_PNG_BILEVEL, 1])
    #     os.chdir('..')

    def flashCorrection(self, ast1_f, ast2_f, coma1_f, coma2_f, tref1_f, tref2_f):
        t0 = time.time()
        if not hasattr(self, 'xv'):
            self.xv, self.yv = self.slm.interleaving()
        beams = 3
        N_iterations = 20  # number of iterations
        xpix = self.xpix
        ypix = self.ypix
        if use_cupy:
            pass
            # place the pixels in negative and positive axes
            x = (cp.array(self.xv) - xpix / 2) / (xpix)
            y = (cp.array(self.yv) - ypix / 2) / (ypix)
            Phi = cp.random.random((ypix, xpix)) * 2 * np.pi
            Tau = cp.zeros((ypix, xpix, beams), dtype=cp.double)  # phase tilt
            Psi = cp.zeros(beams, dtype=cp.double)
            F = cp.zeros(beams, dtype=cp.complex_)
            G = cp.zeros((ypix, xpix, beams), dtype=cp.complex_)
        else:
            # place the pixels in negative and positive axes
            x = (self.xv - xpix / 2) / (xpix)
            y = (self.yv - ypix / 2) / (ypix)

            Phi = np.random.random((ypix, xpix)) * 2 * np.pi
            Tau = np.zeros((ypix, xpix, beams), dtype=np.double)  # phase tilt
            Psi = np.zeros(beams, dtype=np.double)
            F = np.zeros(beams, dtype=np.complex_)
            G = np.zeros((ypix, xpix, beams), dtype=np.complex_)

        rSqaure = x ** 2 + y ** 2
        ast1 = ast1_f * sqrt(6) * (x ** 2 - y ** 2)
        ast2 = ast2_f * 2 * sqrt(6) * x * y
        coma1 = coma1_f * 2 * sqrt(2) * (3 * rSqaure - 2) * x
        coma2 = coma2_f * 2 * sqrt(2) * (3 * rSqaure - 2) * y
        tref1 = tref1_f * 2 * sqrt(2) * (4 * x ** 2 - 3 * rSqaure) * x
        tref2 = tref2_f * 2 * sqrt(2) * (3 * rSqaure - 4 * y ** 2) * y
        abb = ast1 + ast2 + coma1 + coma2 + tref1 + tref2

        nm = 10 ** -9
        mm = 0.001
        wavelength = 520 * nm
        f = 250 * mm
        p_mask = 2.1 * mm  # pitch of the pinhole mask
        u_mask = p_mask * np.sqrt(3) / 2  # coordinate in x direction in FT plane
        p = 1 / (u_mask / wavelength / f) / (0.0082 * mm)

        # initialise Tau, 6 ramps of increasing angle, with a yramp to shift away from central axis
        if use_cupy:
            for i in range(0, beams):
                xp = xpix / p * 2 * np.pi * cp.cos(2 * i * np.pi / 3 + np.pi / 18)
                yp = ypix / p * 2 * np.pi * cp.sin(2 * i * np.pi / 3 + np.pi / 18)
                Tau[:, :, i] = x * xp + y * yp + abb
            Psi = cp.zeros(3)
            for k in range(0, N_iterations):
                F = cp.sum(cp.exp(1j * (-Tau + Phi.reshape((xpix, ypix, 1)))),
                           (0, 1))  # DFT to find DC term at Fourier plane
                # extract just the phase (set amplitude to 1)
                Psi[0] = cp.angle(F[0])
                Psi[1] = cp.angle(F[0]) + 2 * np.pi / 7
                Psi[2] = cp.angle(F[0]) + 6 * np.pi / 7
                A = cp.abs(F)  # Amplitude
                # G and Phi do the inverse FT
                G = cp.exp(1j * (Tau + Psi))  # calculate the terms needed for summation
                Phi = np.pi * (cp.real(
                    cp.sum(G, axis=2)) < 0)  # sum the terms and take the argument, which is used as next phi
                img = Phi.get() * 1
        else:
            for i in range(0, beams):
                xp = xpix / p * 2 * np.pi * np.cos(2 * i * np.pi / 3 + np.pi / 18)
                yp = ypix / p * 2 * np.pi * np.sin(2 * i * np.pi / 3 + np.pi / 18)
                Tau[:, :, i] = x * xp + y * yp + abb
            Psi = np.zeros(3)
            for k in range(0, N_iterations):
                F = np.sum(np.exp(1j * (-Tau + Phi.reshape((xpix, ypix, 1)))),
                           (0, 1))  # DFT to find DC term at Fourier plane
                # extract just the phase (set amplitude to 1)
                Psi[0] = np.angle(F[0])
                Psi[1] = np.angle(F[0]) + 2 * np.pi / 7
                Psi[2] = np.angle(F[0]) + 6 * np.pi / 7
                A = np.abs(F)  # Amplitude
                G = np.exp(1j * (Tau + Psi))  # calculate the terms needed for summation
                Phi = np.pi * (np.real(
                    np.sum(G, axis=2)) < 0)  # sum the terms and take the argument, which is used as next phi
                img = Phi * 1
        hex_bits = np.packbits(img[:, :].astype('int'), bitorder='little')
        self.slm.sendBitplane(hex_bits, 0)

        self.slm.repReload()


    def writeRep(self, fns, n_frames, imgns):
        """write a text file of the repertoire and save it as .rep format, and then build it to a .repz11 file"""
        os.chdir('./gen_repertoires')
        with open(f'{fns}.txt', 'w') as f:
            data = ("ID\n"
                    '"V1.0 ${date(\\"yyyy-MMM-dd HH:mm:ss\\")}"\n'
                    "ID_END\n\n"
                    "PLATFORM\n"
                    '"R11"\n'
                    "PLATFORM_END\n\n"
                    "DISPLAY\n"
                    '"QXGA"\n'
                    "DISPLAY_END\n\n"
                    "FORMATVERSION\n"
                    '"FV4"\n'
                    "FORMATVERSION_END\n\n"
                    "SEQUENCES\n")
            f.write(data)

            data2 = ('A "48061 2ms 1-bit Balanced.seq11"\n'
                     'SEQUENCES_END\n\n'
                     'IMAGES\n')
            f.write(data2)

            for i in range(n_frames):
                data3 =(f' 1 "{imgns[i]}"\n')
                f.write(data3)

            data4 = ('IMAGES_END\n'
                     'DEFAULT "RO1"\n'
                     '[HWA h\n')
            f.write(data4)

            for i in range(n_frames):
                data5 = ('{f (A,%d)}\n'%i)
                f.write(data5)

            data6 = (']')
            f.write(data6)

        os.rename(f'{fns}.txt', f'{fns}.rep')
        print('New rep file created')

        # build the rep to a repz11 file
        output = subprocess.run(['RepBuild', fns + '.rep', '-c', fns + '.repz11'], shell=True,
                                stdout=subprocess.PIPE)
        if output.returncode == 0:
            print(output.stdout.decode())
            print('repz11 file created')
        else:
            print(output.stderr)
        os.chdir('..')

    def writeCorrRep(self, fns, imgns):
        """write a text file of the SLM correction repertoire and save it as '.rep' format,
        and then build it to a '.repz11' file"""
        os.chdir('./gen_repertoires')
        with open(f'{fns}.txt', 'w') as f:
            data = ("ID\n"
                    '"V1.0 ${date(\\"yyyy-MMM-dd HH:mm:ss\\")}"\n'
                    "ID_END\n\n"
                    "PLATFORM\n"
                    '"R11"\n'
                    "PLATFORM_END\n\n"
                    "DISPLAY\n"
                    '"2Kx2K"\n'
                    "DISPLAY_END\n\n"
                    "FORMATVERSION\n"
                    '"FV4"\n'
                    "FORMATVERSION_END\n\n"
                    "SEQUENCES\n")
            f.write(data)

            data2 = ('A "48160 1ms 1-bit Balanced.seq11"\n'
                     'SEQUENCES_END\n\n'
                     'IMAGES\n')
            f.write(data2)

            data3 =(f' 1 "{imgns}"\n')
            f.write(data3)

            data4 = ('IMAGES_END\n'
                     'DEFAULT "RO1"\n'
                     '[HWA \n')
            f.write(data4)

            data5 = ('{f (A,0)}\n')
            f.write(data5)

            data6 = (']')
            f.write(data6)

        os.rename(f'{fns}.txt', f'{fns}.rep')
        print('New rep file created')

        # build the rep to a repz11 file
        output = subprocess.run(['RepBuild', fns + '.rep', '-c', fns + '.repz11'], shell=True,
                                stdout=subprocess.PIPE)
        if output.returncode == 0:
            print(output.stdout.decode())
            print('repz11 file created')
        else:
            print(output.stderr)
        os.chdir('..')
        return fns

    def updateHardware(self):
        if hasattr(self, 'slm'):
            self.settings.activation_state.read_from_hardware()
            self.settings.rep_name.read_from_hardware()
            self.settings.activation_type.read_from_hardware()
            self.settings.roIndex.read_from_hardware()
            self.settings.roName.read_from_hardware()