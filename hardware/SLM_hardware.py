from ScopeFoundry import HardwareComponent
from devices.SLM_device import SLMDev
import numpy as np
from numpy import meshgrid, sin, cos, pi, sqrt, floor
import subprocess, os, tifffile, cv2, time
import cupy as cp
from random import randrange

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

    def repSendBP(self, fn):
        """Send the repz.11 file to the board. fn: file name."""
        self.slm.close()
        os.chdir('./gen_repertoires')
        output = subprocess.run(['RepSender', '-z', fn, '-d', '0175000881', '-i'], shell=True, stdout=subprocess.PIPE)
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

        def sendBaseRep(self, fns=f'base_{time.strftime("%d%m%y_%H%M%S", time.localtime())}', imgns='hol.png'):
            """write a text file repertoire as a base and save it as '.rep' format,and then build it to a '.repz11' file.
            This repertoire only needs to be sent once."""
            os.chdir('C:/Users/ML2618/Desktop/SLMtests')
            with open(f'{fns}.txt', 'w') as f:
                data = []
                data.append("ID\n"
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
                data.append('A "48163 10ms 1-bit Balanced.seq11"\n'
                            'SEQUENCES_END\n\n'
                            'IMAGES\n')
                for i in range(3):
                    data.append(f' 1 "{imgns}"\n')

                data.append('IMAGES_END\n'
                            f'DEFAULT "RO0"\n'
                            '[HWA h \n')  # hardware trigger
                data.append(f'<t(A,12) (A,13) (A,14)>]\n')
                for i in range(1, 210):
                    # 10/12 * 252 ROs
                    data.append(f'"RO{i}"\n'
                                '[HWA \n')
                    data.append('<t')
                    for k in range(3):
                        data.append(f'(A,{12 + 3 * i + k}) ')
                    data.append('>]\n')

                #                 # no triggering siganl needed
                data.append('"RO210"\n'
                            '[HWA \n'
                            f'< (A,642) (A,643) (A,644)>]\n')
                for i in range(211, 252):
                    # 2/12 * 252 ROs
                    data.append(f'"RO{i}"\n'
                                '[HWA \n')
                    data.append('<')
                    for k in range(3):
                        data.append(f'(A,{12 + 3 * i + k}) ')
                    data.append('>]\n')
                f.write(''.join(data))

            os.rename(f'{fns}.txt', f'{fns}.rep')
            print('New rep file created')

            # build the rep to a repz11 file
            print(os.getcwd())
            output = subprocess.run(['RepBuild', fns + '.rep', '-c', fns + '.repz11'], shell=True,
                                    stdout=subprocess.PIPE)
            if output.returncode == 0:
                print(output.stdout.decode())
                print('repz11 file created')
            else:
                print(f'rep to repz11 failed, error: {output.stderr}')
            self.repSendBP(fns + '.repz11')
            self.slm.reloadSkipImgs()

    def flashCorrection(self, bp_img):
        for k in range(self.n_bp):
            self.slm.sendBitplane(bp_img[k], self.bpIndex + k)

    def updateBp(self, imgs, mode):
        t0 = time.time()
        if mode == 's':
            roIndex = randrange(210)
        else:
            roIndex = randrange(210, 252)
        bp = 12 + roIndex * 3
        self.flashCorrection(imgs, bp)
        self.slm.repReload(bp)
        self.slm.setRO(roIndex)
        print(f'Repertoire updated in {time.time() - t0}s')

    def updateHardware(self):
        if hasattr(self, 'slm'):
            self.settings.activation_state.read_from_hardware()
            self.settings.rep_name.read_from_hardware()
            self.settings.activation_type.read_from_hardware()
            self.settings.roIndex.read_from_hardware()
            self.settings.roName.read_from_hardware()