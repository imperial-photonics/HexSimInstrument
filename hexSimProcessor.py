import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.ndimage import gaussian_filter
from cupy_filters import gaussian_filter as gaussian_filter_cupy
from numpy import exp, pi, cos, sqrt, arccos
from pyczt import pyczt
import multiprocessing
import scipy.io
import time
import math

try:
    import pyfftw
    import pyfftw.interfaces.numpy_fft as fft

    pyfftw.config.NUM_THREADS = multiprocessing.cpu_count()
    pyfftw.interfaces.cache.enable()
    fftw = True
except:
    import numpy.fft as fft

try:
    import cv2

    opencv = True
except:
    opencv = False

try:
    import cupy as cp
    import cupyx
    import cupyx.scipy.ndimage
    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()
    cupy = True
except:
    cupy = False


class hexSimProcessor:
    N = 256  # points to use in fft
    pixelsize = 6.5  # camera pixel size, um
    magnification = 40  # objective magnification
    NA = 0.75  # numerial aperture at sample
    n = 1.33  # refractive index at sample
    wavelength = 0.525  # wavelength, um
    alpha = 0.3  # zero order attenuation width
    beta = 0.999  # zero order attenuation
    w = 0.3  # Winier parameter
    eta = 0.75  # eta is the factor by which the illumination grid frequency
    # exceeds the incoherent cutoff, eta=1 for normal SIM, eta=sqrt(3)/2 to maximise
    # resolution without zeros in TF carrier is 2*kmax*eta
    cleanup = False
    debug = True
    axial = False
    usemodulation = True

    def __init__(self):
        self._lastN = 0

    def _allocate_arrays(self):
        ''' define grids '''
        self._dx = self.pixelsize / self.magnification  # Sampling in image plane
        self._res = self.wavelength / (2 * self.NA)
        self._oversampling = self._res / self._dx
        self._dk = self._oversampling / (self.N / 2)  # Sampling in frequency plane
        # self._kx = np.linspace(-self._dk * self.N / 2, self._dk * self.N / 2 - self._dk, self.N, dtype=np.single)
        self._kx = np.arange(-self._dk * self.N / 2, self._dk * self.N / 2, self._dk, dtype=np.single)
        [self._kx, self._ky] = np.meshgrid(self._kx, self._kx)
        self._dx2 = self._dx / 2

        ''' define matrix '''
        self._reconfactor = np.zeros((7, 2 * self.N, 2 * self.N), dtype=np.single)  # for reconstruction

        self._prefilter = np.zeros((self.N, self.N),
                                   dtype=np.single)  # for prefilter stage, includes otf and zero order supression
        self._postfilter = np.zeros((2 * self.N, 2 * self.N), dtype=np.single)
        self._carray = np.zeros((7, 2 * self.N, 2 * self.N), dtype=np.complex64)
        self._carray1 = np.zeros((7, 2 * self.N, self.N + 1), dtype=np.complex64)

        self._imgbig = np.zeros((7, 2 * self.N, 2 * self.N), dtype=np.single)
        self._imgbig1 = np.zeros((7, 2 * self.N, 2 * self.N), dtype=np.single)
        self._bigimgstore = np.zeros((2 * self.N, 2 * self.N), dtype=np.single)
        if cupy:
            self._prefilter_cp = cp.zeros((self.N, self.N), dtype=np.single)
            self._postfilter_cp = cp.zeros((2 * self.N, 2 * self.N), dtype=np.single)
            self._carray_cp = cp.zeros((7, 2 * self.N, self.N + 1), dtype=np.complex)
            self._reconfactor_cp = cp.zeros((7, 2 * self.N, 2 * self.N), dtype=np.single)
            self._bigimgstore_cp = cp.zeros((2 * self.N, 2 * self.N), dtype=np.single)
        if opencv:
            self._prefilter_ocv = np.zeros((self.N, self.N),
                                           dtype=np.single)  # for prefilter stage, includes otf and zero order supression
            self._postfilter_ocv = np.zeros((2 * self.N, 2 * self.N), dtype=np.single)
            self._carray_ocv = np.zeros((2 * self.N, 2 * self.N), dtype=np.single)
            self._carray_ocvU = cv2.UMat((2 * self.N, 2 * self.N), s=0.0, type=cv2.CV_32FC2)
            self._bigimgstoreU = cv2.UMat(self._bigimgstore)
            self._imgstoreU = [cv2.UMat((2 * self.N, 2 * self.N), s=0.0, type=cv2.CV_32FC2) for i in range(7)]
        self._lastN = self.N

    def calibrate(self, img):
        if self.N != self._lastN:
            self._allocate_arrays()

        kr = np.sqrt(self._kx ** 2 + self._ky ** 2, dtype=np.single)
        # kxbig = np.linspace(-self._dk * self.N, self._dk * self.N - self._dk, 2 * self.N, dtype=np.single)
        kxbig = np.arange(-self._dk * self.N, self._dk * self.N, self._dk, dtype=np.single)
        [kxbig, kybig] = np.meshgrid(kxbig, kxbig)

        '''Separate bands into DC and 3 high frequency bands'''
        M = exp(1j * 2 * pi / 7) ** ((np.arange(0, 4)[:, np.newaxis]) * np.arange(0, 7))

        sum_prepared_comp = np.zeros((4, self.N, self.N), dtype=np.complex)
        wienerfilter = np.zeros((2 * self.N, 2 * self.N), dtype=np.single)

        for k in range(0, 4):
            for l in range(0, 7):
                sum_prepared_comp[k, :, :] = sum_prepared_comp[k, :, :] + img[l, :, :] * M[k, l]

        # minimum search radius in k-space
        mask1 = (kr > 1.8 * self.eta)

        # find parameters
        ckx = np.zeros((3, 1), dtype=np.single)
        cky = np.zeros((3, 1), dtype=np.single)
        p = np.zeros((3, 1), dtype=np.single)
        ampl = np.zeros((3, 1), dtype=np.single)

        for i in range(0, 3):
            ckx[i], cky[i], p[i], ampl[i] = self._findCarrier(sum_prepared_comp[0, :, :],
                                                              sum_prepared_comp[i + 1, :, :], mask1)
        if self.debug:
            print(f'kx = {ckx[0]}, {ckx[1]}, {ckx[2]}')
            print(f'ky = {cky[0]}, {cky[1]}, {cky[2]}')
            print(f'p  = {p[0]}, {p[1]}, {p[2]}')
            print(f'a  = {ampl[0]}, {ampl[1]}, {ampl[2]}')

        ph = np.single(2 * pi * self.NA / self.wavelength)

        xx = np.arange(-self._dx2 * self.N, self._dx2 * self.N, self._dx2, dtype=np.single)
        yy = xx

        if self.axial:
            A = 6
        else:
            A = 12

        for idx_p in range(0, 7):
            pstep = idx_p * 2 * pi / 7
            if self.usemodulation:
                self._reconfactor[idx_p, :, :] = (1 + 4 / ampl[0] * np.outer(exp(1j * ph * cky[0] * yy), exp(
                    1j * (ph * ckx[0] * xx - pstep + p[0]))).real
                                                  + 4 / ampl[1] * np.outer(exp(1j * ph * cky[1] * yy), exp(
                            1j * (ph * ckx[1] * xx - 2 * pstep + p[1]))).real
                                                  + 4 / ampl[2] * np.outer(exp(1j * ph * cky[2] * yy), exp(
                            1j * (ph * ckx[2] * xx - 3 * pstep + p[2]))).real)
            else:
                self._reconfactor[idx_p, :, :] = (1 + A * np.outer(exp(1j * ph * cky[0] * yy),
                                                                   exp(1j * (ph * ckx[0] * xx - pstep + p[0]))).real
                                                  + A * np.outer(exp(1j * ph * cky[1] * yy),
                                                                 exp(1j * (ph * ckx[1] * xx - 2 * pstep + p[1]))).real
                                                  + A * np.outer(exp(1j * ph * cky[2] * yy),
                                                                 exp(1j * (ph * ckx[2] * xx - 3 * pstep + p[2]))).real)
            # self._reconfactor[:,:,idx_p] = (1+12*cos(ph*(ckx[0]*x2+cky[0]*y2)-pstep+p[0])
            #                                 +12*cos(ph*(ckx[1]*x2+cky[1]*y2)-2*pstep+p[1])
            #                                 +12*cos(ph*(ckx[2]*x2+cky[2]*y2)-3*pstep+p[2]))
        # scipy.io.savemat('reconfactor.mat', {'pyr':self._reconfactor})
        # calculate pre-filter factors

        mask2 = (kr < 2)

        self._prefilter = np.single((self._tfm(kr, mask2) * self._attm(kr, mask2)))
        self._prefilter = fft.fftshift(self._prefilter)

        mtot = np.full((2 * self.N, 2 * self.N), False)

        for i in range(0, 3):
            kr = sqrt((kxbig - ckx[i]) ** 2 + (kybig - cky[i]) ** 2)
            mask = (kr < 2)
            mtot = mtot | mask
            wienerfilter = (wienerfilter + mask * ((self._tfm(kr, mask)) ** 2) * self._attm(kr, mask))
            kr = sqrt((kxbig + ckx[i]) ** 2 + (kybig + cky[i]) ** 2)
            mask = (kr < 2)
            mtot = mtot | mask
            wienerfilter = (wienerfilter + mask * ((self._tfm(kr, mask) ** 2) * self._attm(kr, mask)))
        kr = sqrt(kxbig ** 2 + kybig ** 2)
        mask = (kr < 2)
        mtot = mtot | mask
        wienerfilter = (wienerfilter + mask * self._tfm(kr, mask) ** 2 * self._attm(kr, mask))

        if self.debug:
            plt.figure()
            plt.title('WienerFilter')
            plt.imshow(wienerfilter)

        kmax = 1 * (2 + sqrt(ckx[0] ** 2 + cky[0]))
        wienerfilter = mtot * (1 - kr * mtot / kmax) / (wienerfilter * mtot + self.w ** 2)
        self._postfilter = fft.fftshift(wienerfilter)

        if self.cleanup:
            imgo = self.reconstruct_fftw(img)
            kernel = np.ones((5, 5), np.uint8)
            mask_tmp = abs(fft.fftshift(fft.fft2(imgo))) > (10 * gaussian_filter(abs(fft.fftshift(fft.fft2(imgo))), 5))
            mask = scipy.ndimage.morphology.binary_dilation(np.single(mask_tmp), kernel)
            mask[self.N - 12:self.N + 13, self.N - 12:self.N + 13] = np.full((25, 25), False)
            mask_shift = (fft.fftshift(mask))
            self._postfilter[mask_shift.astype(bool)] = 0

        if opencv:
            self._reconfactorU = [cv2.UMat(self._reconfactor[idx_p, :, :]) for idx_p in range(0, 7)]
            self._prefilter_ocv = np.single(cv2.dft(fft.ifft2(self._prefilter).real))
            pf = np.zeros((self.N, self.N, 2), dtype=np.single)
            pf[:, :, 0] = self._prefilter
            pf[:, :, 1] = self._prefilter
            self._prefilter_ocvU = cv2.UMat(np.single(pf))
            self._postfilter_ocv = np.single(cv2.dft(fft.ifft2(self._postfilter).real))
            pf = np.zeros((2 * self.N, 2 * self.N, 2), dtype=np.single)
            pf[:, :, 0] = self._postfilter
            pf[:, :, 1] = self._postfilter
            self._postfilter_ocvU = cv2.UMat(np.single(pf))

        if cupy:
            self._postfilter_cp = cp.asarray(self._postfilter)
 
    def calibrate_fast(self, img):
        if self.N != self._lastN:
            self._allocate_arrays()

        kr = np.sqrt(self._kx ** 2 + self._ky ** 2, dtype=np.single)
        kxbig = np.linspace(-self._dk * self.N, self._dk * self.N - self._dk, 2 * self.N, dtype=np.single)
        [kxbig, kybig] = np.meshgrid(kxbig, kxbig)

        '''Separate bands into DC and 3 high frequency bands'''
        M = exp(1j * 2 * pi / 7) ** ((np.arange(0, 4)[:, np.newaxis]) * np.arange(0, 7))

        sum_prepared_comp = np.zeros((4, self.N, self.N), dtype=np.complex)
        wienerfilter = np.zeros((2 * self.N, 2 * self.N), dtype=np.single)

        for k in range(0, 4):
            for l in range(0, 7):
                sum_prepared_comp[k, :, :] = sum_prepared_comp[k, :, :] + img[l, :, :] * M[k, l]

        # minimum search radius in k-space
        mask1 = (kr > 1.8 * self.eta)

        # find parameters
        ckx = np.zeros((3, 1), dtype=np.single)
        cky = np.zeros((3, 1), dtype=np.single)
        p = np.zeros((3, 1), dtype=np.single)
        ampl = np.zeros((3, 1), dtype=np.single)

        for i in range(0, 3):
            ckx[i], cky[i], p[i], ampl[i] = self._findCarrier_cupy(sum_prepared_comp[0, :, :],
                                                              sum_prepared_comp[i + 1, :, :], mask1)
        #print(ckx,cky,p,ampl)
        ph = np.single(2 * pi * self.NA / self.wavelength)

        xx = np.arange(-self._dx2 * self.N, self._dx2 * self.N, self._dx2, dtype=np.single)
        yy = xx

        if self.axial:
            A = 6
        else:
            A = 12

        for idx_p in range(0, 7):
            pstep = idx_p * 2 * pi / 7
            if self.usemodulation:
                self._reconfactor[idx_p, :, :] = (1 + 4 / ampl[0] * np.outer(exp(1j * ph * cky[0] * yy), exp(
                    1j * (ph * ckx[0] * xx - pstep + p[0]))).real
                                                  + 4 / ampl[1] * np.outer(exp(1j * ph * cky[1] * yy), exp(
                            1j * (ph * ckx[1] * xx - 2 * pstep + p[1]))).real
                                                  + 4 / ampl[2] * np.outer(exp(1j * ph * cky[2] * yy), exp(
                            1j * (ph * ckx[2] * xx - 3 * pstep + p[2]))).real)
            else:
                self._reconfactor[idx_p, :, :] = (1 + A * np.outer(exp(1j * ph * cky[0] * yy),
                                                                   exp(1j * (ph * ckx[0] * xx - pstep + p[0]))).real
                                                  + A * np.outer(exp(1j * ph * cky[1] * yy),
                                                                 exp(1j * (ph * ckx[1] * xx - 2 * pstep + p[1]))).real
                                                  + A * np.outer(exp(1j * ph * cky[2] * yy),
                                                                 exp(1j * (ph * ckx[2] * xx - 3 * pstep + p[2]))).real)
            # self._reconfactor[:,:,idx_p] = (1+12*cos(ph*(ckx[0]*x2+cky[0]*y2)-pstep+p[0])
            #                                 +12*cos(ph*(ckx[1]*x2+cky[1]*y2)-2*pstep+p[1])
            #                                 +12*cos(ph*(ckx[2]*x2+cky[2]*y2)-3*pstep+p[2]))

        # calculate pre-filter factors

        mask2 = (kr < 2)

        self._prefilter = np.single((self._tfm(kr, mask2) * self._attm(kr, mask2)))
        self._prefilter = fft.fftshift(self._prefilter)

        mtot = np.full((2 * self.N, 2 * self.N), False)

        for i in range(0, 3):
            kr = sqrt((kxbig - ckx[i]) ** 2 + (kybig - cky[i]) ** 2)
            mask = (kr < 2)
            mtot = mtot | mask
            wienerfilter = (wienerfilter + mask * ((self._tfm(kr, mask)) ** 2) * self._attm(kr, mask))
            kr = sqrt((kxbig + ckx[i]) ** 2 + (kybig + cky[i]) ** 2)
            mask = (kr < 2)
            mtot = mtot | mask
            wienerfilter = (wienerfilter + mask * ((self._tfm(kr, mask) ** 2) * self._attm(kr, mask)))
        kr = sqrt(kxbig ** 2 + kybig ** 2)
        mask = (kr < 2)
        mtot = mtot | mask
        wienerfilter = (wienerfilter + mask * self._tfm(kr, mask) ** 2 * self._attm(kr, mask))

        if self.debug:
            plt.figure()
            plt.title('WienerFilter')
            plt.imshow(wienerfilter)

        kmax = 1 * (2 + sqrt(ckx[0] ** 2 + cky[0]))
        wienerfilter = mtot * (1 - kr * mtot / kmax) / (wienerfilter * mtot + self.w ** 2)
        self._postfilter = fft.fftshift(wienerfilter)

        if self.cleanup:
            imgo = self.reconstruct_fftw(img)
            kernel = np.ones((5, 5), np.uint8)
            mask_tmp = abs(fft.fftshift(fft.fft2(imgo))) > (10 * gaussian_filter(abs(fft.fftshift(fft.fft2(imgo))), 5))
            mask = scipy.ndimage.morphology.binary_dilation(np.single(mask_tmp), kernel)
            mask[self.N - 12:self.N + 13, self.N - 12:self.N + 13] = np.full((25, 25), False)
            mask_shift = (fft.fftshift(mask))
            self._postfilter[mask_shift.astype(bool)] = 0

        if opencv:
            self._reconfactorU = [cv2.UMat(self._reconfactor[idx_p, :, :]) for idx_p in range(0, 7)]
            self._prefilter_ocv = np.single(cv2.dft(fft.ifft2(self._prefilter).real))
            pf = np.zeros((self.N, self.N, 2), dtype=np.single)
            pf[:, :, 0] = self._prefilter
            pf[:, :, 1] = self._prefilter
            self._prefilter_ocvU = cv2.UMat(np.single(pf))
            self._postfilter_ocv = np.single(cv2.dft(fft.ifft2(self._postfilter).real))
            pf = np.zeros((2 * self.N, 2 * self.N, 2), dtype=np.single)
            pf[:, :, 0] = self._postfilter
            pf[:, :, 1] = self._postfilter
            self._postfilter_ocvU = cv2.UMat(np.single(pf))

        if cupy:
            self._postfilter_cp = cp.asarray(self._postfilter)
            
    def reconstruct_fftw(self, img):
        imf = fft.fft2(img) * self._prefilter
        self._carray[:, 0:self.N // 2, 0:self.N // 2] = imf[:, 0:self.N // 2, 0:self.N // 2]
        self._carray[:, 0:self.N // 2, 3 * self.N // 2:2 * self.N] = imf[:, 0:self.N // 2, self.N // 2:self.N]
        self._carray[:, 3 * self.N // 2:2 * self.N, 0:self.N // 2] = imf[:, self.N // 2:self.N, 0:self.N // 2]
        self._carray[:, 3 * self.N // 2:2 * self.N, 3 * self.N // 2:2 * self.N] = imf[:, self.N // 2:self.N,
                                                                                  self.N // 2:self.N]
        img2 = np.sum(np.real(fft.ifft2(self._carray)).real * self._reconfactor, 0)
        self._imgstore = img
        self._bigimgstore = fft.ifft2(fft.fft2(img2) * self._postfilter).real
        return self._bigimgstore

    def reconstruct_rfftw(self, img):
        imf = fft.rfft2(img) * self._prefilter[:, 0:self.N // 2 + 1]
        self._carray1[:, 0:self.N // 2, 0:self.N // 2 + 1] = imf[:, 0:self.N // 2, 0:self.N // 2 + 1]
        self._carray1[:, 3 * self.N // 2:2 * self.N, 0:self.N // 2 + 1] = imf[:, self.N // 2:self.N, 0:self.N // 2 + 1]
        img2 = np.sum(fft.irfft2(self._carray1) * self._reconfactor, 0)
        return fft.irfft2(fft.rfft2(img2) * self._postfilter[:, 0:self.N + 1])

    def reconstruct_ocv(self, img):
        assert opencv, "No opencv present"
        img2 = np.zeros((2 * self.N, 2 * self.N), dtype=np.single)
        for i in range(0, 7):
            imf = cv2.mulSpectrums(cv2.dft(img[i, :, :]), self._prefilter_ocv, 0)
            # print(imf.shape)
            self._carray_ocv[0:self.N // 2, 0:self.N] = imf[0:self.N // 2, 0:self.N]
            self._carray_ocv[3 * self.N // 2:2 * self.N, 0:self.N] = imf[self.N // 2:self.N, 0:self.N]
            img2 = cv2.add(img2, cv2.multiply(cv2.idft(self._carray_ocv, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT),
                                              self._reconfactor[i, :, :]))
        return cv2.idft(cv2.mulSpectrums(cv2.dft(img2), self._postfilter_ocv, 0),
                        flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)

    def reconstruct_ocvU(self, img):
        assert opencv, "No opencv present"
        img2 = cv2.UMat((2 * self.N, 2 * self.N), s=0.0, type=cv2.CV_32FC1)
        mask = cv2.UMat((self.N // 2, self.N // 2), s=1, type=cv2.CV_8U)
        for i in range(0, 7):
            self._imgstoreU[i] = cv2.UMat(img[i, :, :])
            # print(self._imgstoreU[i].get().shape)
            # print(cv2.dft(self._imgstoreU[i],flags=cv2.DFT_COMPLEX_OUTPUT).get().shape)
            # print(self._prefilter_ocvU.get().shape)
            imf = cv2.multiply(cv2.dft(self._imgstoreU[i], flags=cv2.DFT_COMPLEX_OUTPUT), self._prefilter_ocvU)
            cv2.copyTo(src=cv2.UMat(imf, (0, 0, self.N // 2, self.N // 2)), mask=mask,
                       dst=cv2.UMat(self._carray_ocvU, (0, 0, self.N // 2, self.N // 2)))
            cv2.copyTo(src=cv2.UMat(imf, (0, self.N // 2, self.N // 2, self.N // 2)), mask=mask,
                       dst=cv2.UMat(self._carray_ocvU, (0, 3 * self.N // 2, self.N // 2, self.N // 2)))
            cv2.copyTo(src=cv2.UMat(imf, (self.N // 2, 0, self.N // 2, self.N // 2)), mask=mask,
                       dst=cv2.UMat(self._carray_ocvU, (3 * self.N // 2, 0, self.N // 2, self.N // 2)))
            cv2.copyTo(src=cv2.UMat(imf, (self.N // 2, self.N // 2, self.N // 2, self.N // 2)), mask=mask,
                       dst=cv2.UMat(self._carray_ocvU, (3 * self.N // 2, 3 * self.N // 2, self.N // 2, self.N // 2)))
            img2 = cv2.add(img2, cv2.multiply(cv2.idft(self._carray_ocvU, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT),
                                              self._reconfactorU[i]))
        self._bigimgstoreU = cv2.idft(cv2.multiply(cv2.dft(img2, flags=cv2.DFT_COMPLEX_OUTPUT), self._postfilter_ocvU),
                                      flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)

        return self._bigimgstoreU

    def reconstruct_cupy(self, img):
        assert cupy, "No CuPy present"
        self._imgstore = img
        imf = cp.fft.rfft2(cp.asarray(img)) * cp.asarray(self._prefilter[:, 0:self.N // 2 + 1])
        self._carray_cp[:, 0:self.N // 2, 0:self.N // 2 + 1] = imf[:, 0:self.N // 2, 0:self.N // 2 + 1]
        self._carray_cp[:, 3 * self.N // 2:2 * self.N, 0:self.N // 2 + 1] = imf[:, self.N // 2:self.N,
                                                                            0:self.N // 2 + 1]
        img2 = cp.sum(cp.fft.irfft2(self._carray_cp) * cp.asarray(self._reconfactor), 0)
        return cp.fft.irfft2(cp.fft.rfft2(img2) * cp.asarray(self._postfilter[:, 0:self.N + 1]))

    def reconstructframe_fftw(self, img, i):
        diff = img - self._imgstore[i, :, :]
        imf = fft.fft2(diff) * self._prefilter
        self._carray[0, 0:self.N // 2, 0:self.N // 2] = imf[0:self.N // 2, 0:self.N // 2]
        self._carray[0, 0:self.N // 2, 3 * self.N // 2:2 * self.N] = imf[0:self.N // 2, self.N // 2:self.N]
        self._carray[0, 3 * self.N // 2:2 * self.N, 0:self.N // 2] = imf[self.N // 2:self.N, 0:self.N // 2]
        self._carray[0, 3 * self.N // 2:2 * self.N, 3 * self.N // 2:2 * self.N] = imf[self.N // 2:self.N,
                                                                                  self.N // 2:self.N]
        img2 = fft.ifft2(self._carray[0, :, :]).real * self._reconfactor[i, :, :]
        self._imgstore[i, :, :] = img
        self._bigimgstore = self._bigimgstore + fft.ifft2(fft.fft2(img2) * self._postfilter).real
        return self._bigimgstore

    def reconstructframe_rfftw(self, img, i):
        diff = img - self._imgstore[i, :, :]
        imf = fft.rfft2(diff) * self._prefilter[:, 0:self.N // 2 + 1]
        self._carray1[0, 0:self.N // 2, 0:self.N // 2 + 1] = imf[0:self.N // 2, 0:self.N // 2 + 1]
        self._carray1[0, 3 * self.N // 2:2 * self.N, 0:self.N // 2 + 1] = imf[self.N // 2:self.N, 0:self.N // 2 + 1]
        img2 = fft.irfft2(self._carray1[0, :, :]) * self._reconfactor[i, :, :]
        self._imgstore[i, :, :] = img
        self._bigimgstore = self._bigimgstore + fft.irfft2(fft.rfft2(img2) * self._postfilter[:, 0:self.N + 1])
        return self._bigimgstore

    def reconstructframe_ocv(self, img, i):
        assert opencv, "No opencv present"
        diff = img - self._imgstore[i, :, :]
        imf = cv2.mulSpectrums(cv2.dft(diff), self._prefilter_ocv, 0)
        self._carray_ocv[0:self.N // 2, 0:self.N] = imf[0:self.N // 2, 0:self.N]
        self._carray_ocv[3 * self.N // 2:2 * self.N, 0:self.N] = imf[self.N // 2:self.N, 0:self.N]
        img2 = cv2.multiply(cv2.idft(self._carray_ocv, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT),
                            self._reconfactor[i, :, :])
        self._imgstore[i, :, :] = img
        self._bigimgstore = self._bigimgstore + cv2.idft(cv2.mulSpectrums(cv2.dft(img2), self._postfilter_ocv, 0),
                                                         flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
        return self._bigimgstore

    def reconstructframe_ocvU(self, img, i):
        assert opencv, "No opencv present"
        img2 = cv2.UMat((2 * self.N, 2 * self.N), s=0.0, type=cv2.CV_32FC1)
        mask = cv2.UMat((self.N // 2, self.N // 2), s=1, type=cv2.CV_8U)
        imU = cv2.UMat(img)
        diff = cv2.subtract(imU, self._imgstoreU[i])
        imf = cv2.multiply(cv2.dft(diff, flags=cv2.DFT_COMPLEX_OUTPUT), self._prefilter_ocvU)
        cv2.copyTo(src=cv2.UMat(imf, (0, 0, self.N // 2, self.N // 2)), mask=mask,
                   dst=cv2.UMat(self._carray_ocvU, (0, 0, self.N // 2, self.N // 2)))
        cv2.copyTo(src=cv2.UMat(imf, (0, self.N // 2, self.N // 2, self.N // 2)), mask=mask,
                   dst=cv2.UMat(self._carray_ocvU, (0, 3 * self.N // 2, self.N // 2, self.N // 2)))
        cv2.copyTo(src=cv2.UMat(imf, (self.N // 2, 0, self.N // 2, self.N // 2)), mask=mask,
                   dst=cv2.UMat(self._carray_ocvU, (3 * self.N // 2, 0, self.N // 2, self.N // 2)))
        cv2.copyTo(src=cv2.UMat(imf, (self.N // 2, self.N // 2, self.N // 2, self.N // 2)), mask=mask,
                   dst=cv2.UMat(self._carray_ocvU, (3 * self.N // 2, 3 * self.N // 2, self.N // 2, self.N // 2)))
        img2 = cv2.multiply(cv2.idft(self._carray_ocvU, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT),
                            self._reconfactorU[i])
        self._imgstoreU[i] = imU
        self._bigimgstoreU = cv2.add(self._bigimgstoreU,
                                     cv2.idft(cv2.mulSpectrums(cv2.dft(img2, flags=cv2.DFT_COMPLEX_OUTPUT),
                                                               self._postfilter_ocvU, 0)
                                              , flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT))
        return self._bigimgstoreU

    def reconstructframe_cupy(self, img, i):
        assert cupy, "No CuPy present"
        self._imgstore[i, :, :] = img
        diff = cp.asarray(img) - cp.asarray(self._imgstore[i, :, :])
        imf = cp.fft.rfft2(diff) * cp.asarray(self._prefilter[:, 0:self.N // 2 + 1])
        self._carray_cp[0, 0:self.N // 2, 0:self.N // 2 + 1] = imf[0:self.N // 2, 0:self.N // 2 + 1]
        self._carray_cp[0, 3 * self.N // 2:2 * self.N, 0:self.N // 2 + 1] = imf[self.N // 2:self.N, 0:self.N // 2 + 1]
        img2 = cp.fft.irfft2(self._carray_cp[0, :, :]) * cp.asarray(self._reconfactor[i, :, :])
        self._bigimgstore_cp = self._bigimgstore_cp + cp.fft.irfft2(
            cp.fft.rfft2(img2) * cp.asarray(self._postfilter[:, 0:self.N + 1]))
        return self._bigimgstore_cp

    def batchreconstruct(self, img):
        nim = img.shape[0]
        r = np.mod(nim, 14)
        if r > 0:  # pad with empty frames so total number of frames is divisible by 14
            img = np.concatenate((img, np.zeros((14 - r, self.N, self.N), np.single)))
            nim = nim + 14 - r
        nim7 = nim // 7
        imf = fft.rfft2(img) * self._prefilter[:, 0:self.N // 2 + 1]
        img2 = np.zeros([nim, 2 * self.N, 2 * self.N], dtype=np.single)
        for i in range(0, nim, 7):
            self._carray1[:, 0:self.N // 2, 0:self.N // 2 + 1] = imf[i:i + 7, 0:self.N // 2, 0:self.N // 2 + 1]
            self._carray1[:, 3 * self.N // 2:2 * self.N, 0:self.N // 2 + 1] = imf[i:i + 7, self.N // 2:self.N,
                                                                              0:self.N // 2 + 1]
            img2[i:i + 7, :, :] = fft.irfft2(self._carray1) * self._reconfactor
        img3 = fft.irfft(fft.rfft(img2, nim, 0)[0:nim7 // 2 + 1, :, :], nim7, 0)
        res = fft.irfft2(fft.rfft2(img3) * self._postfilter[:, :self.N + 1])
        return res

    def batchreconstructcompact(self, img):
        nim = img.shape[0]
        r = np.mod(nim, 14)
        if r > 0:  # pad with empty frames so total number of frames is divisible by 14
            img = np.concatenate((img, np.zeros((14 - r, self.N, self.N), np.single)))
            nim = nim + 14 - r
        nim7 = nim // 7
        imf = fft.rfft2(img) * self._prefilter[:, 0:self.N // 2 + 1]
        img2 = np.zeros([nim, 2 * self.N, 2 * self.N], dtype=np.single)
        for i in range(0, nim, 7):
            self._carray1[:, 0:self.N // 2, 0:self.N // 2 + 1] = imf[i:i + 7, 0:self.N // 2, 0:self.N // 2 + 1]
            self._carray1[:, 3 * self.N // 2:2 * self.N, 0:self.N // 2 + 1] = imf[i:i + 7, self.N // 2:self.N,
                                                                              0:self.N // 2 + 1]
            img2[i:i + 7, :, :] = fft.irfft2(self._carray1) * self._reconfactor
        res = np.zeros((nim7, 2 * self.N, 2 * self.N), dtype=np.single)

        imgf = fft.rfft(img2[:, :self.N, :self.N], nim, 0)[:nim7 // 2 + 1, :, :]
        res[:, :self.N, :self.N] = fft.irfft(imgf, nim7, 0)
        imgf = fft.rfft(img2[:, :self.N, self.N:2 * self.N], nim, 0)[:nim7 // 2 + 1, :, :]
        res[:, :self.N, self.N:2 * self.N] = fft.irfft(imgf, nim7, 0)
        imgf = fft.rfft(img2[:, self.N:2 * self.N, :self.N], nim, 0)[:nim7 // 2 + 1, :, :]
        res[:, self.N:2 * self.N, :self.N] = fft.irfft(imgf, nim7, 0)
        imgf = fft.rfft(img2[:, self.N:2 * self.N, self.N:2 * self.N], nim, 0)[:nim7 // 2 + 1, :, :]
        res[:, self.N:2 * self.N, self.N:2 * self.N] = fft.irfft(imgf, nim7, 0)

        res = fft.irfft2(fft.rfft2(res) * self._postfilter[:, :self.N + 1])
        return res

    def batchreconstruct_cupy(self, img):
        assert cupy, "No CuPy present"
        cp._default_memory_pool.free_all_blocks()
        img = cp.asarray(img, dtype=np.float32)
        nim = img.shape[0]
        r = np.mod(nim, 14)
        if r > 0:  # pad with empty frames so total number of frames is divisible by 14
            img = np.concatenate((img, cp.zeros((14 - r, self.N, self.N), np.single)))
            nim = nim + 14 - r
        nim7 = nim // 7
        imf = cp.fft.rfft2(img) * cp.asarray(self._prefilter[:, 0:self.N // 2 + 1])
        img = None
        cp._default_memory_pool.free_all_blocks()
        if self.debug:
            print(mempool.used_bytes())
            print(mempool.total_bytes())

        img2 = cp.zeros((nim, 2 * self.N, 2 * self.N), dtype=np.single)
        bcarray = cp.zeros((7, 2 * self.N, self.N + 1), dtype=np.complex64)
        reconfactor_cp = cp.asarray(self._reconfactor)
        for i in range(0, nim, 7):
            bcarray[:, 0:self.N // 2, 0:self.N // 2 + 1] = imf[i:i + 7, 0:self.N // 2, 0:self.N // 2 + 1]
            bcarray[:, 3 * self.N // 2:2 * self.N, 0:self.N // 2 + 1] = imf[i:i + 7, self.N // 2:self.N,
                                                                        0:self.N // 2 + 1]
            img2[i:i + 7, :, :] = cp.fft.irfft2(bcarray) * reconfactor_cp
        imf = None
        bcarray = None
        cp._default_memory_pool.free_all_blocks()
        if self.debug:
            print(mempool.used_bytes())
            print(mempool.total_bytes())

        img3 = cp.fft.irfft(cp.fft.rfft(img2, nim, 0)[0:nim7 // 2 + 1, :, :], nim7, 0)
        img2 = None
        cp._default_memory_pool.free_all_blocks()
        if self.debug:
            print(mempool.used_bytes())
            print(mempool.total_bytes())
        res = cp.fft.irfft2(cp.fft.rfft2(img3) * cp.asarray(self._postfilter[:, :self.N + 1]))
        return res

    def batchreconstructcompact_cupy(self, img):
        assert cupy, "No CuPy present"
        cp._default_memory_pool.free_all_blocks()
        img = cp.asarray(img, dtype=np.float32)
        nim = img.shape[0]
        r = np.mod(nim, 14)
        if r > 0:  # pad with empty frames so total number of frames is divisible by 14
            img = cp.concatenate((img, cp.zeros((14 - r, self.N, self.N), np.single)))
            nim = nim + 14 - r
        nim7 = nim // 7
        imf = cp.fft.rfft2(img) * cp.asarray(self._prefilter[:, 0:self.N // 2 + 1])
        img = None
        cp._default_memory_pool.free_all_blocks()

        img2 = cp.zeros((nim, 2 * self.N, 2 * self.N), dtype=np.single)
        bcarray = cp.zeros((7, 2 * self.N, self.N + 1), dtype=np.complex64)
        reconfactor_cp = cp.asarray(self._reconfactor)
        for i in range(0, nim, 7):
            bcarray[:, 0:self.N // 2, 0:self.N // 2 + 1] = imf[i:i + 7, 0:self.N // 2, 0:self.N // 2 + 1]
            bcarray[:, 3 * self.N // 2:2 * self.N, 0:self.N // 2 + 1] = imf[i:i + 7, self.N // 2:self.N,
                                                                        0:self.N // 2 + 1]
            img2[i:i + 7, :, :] = cp.fft.irfft2(bcarray) * reconfactor_cp

        # self._carray_cp = None
        cp._default_memory_pool.free_all_blocks()

        imgout = cp.zeros((nim7, 2 * self.N, 2 * self.N), dtype=np.single)
        imf = cp.fft.rfft(img2[:, 0:self.N, 0:self.N], nim, 0)[:nim7 // 2 + 1, :, :]
        imgout[:, 0:self.N, 0:self.N] = cp.fft.irfft(imf, nim7, 0)
        imf = cp.fft.rfft(img2[:, self.N:2 * self.N, 0:self.N], nim, 0)[:nim7 // 2 + 1, :, :]
        imgout[:, self.N:2 * self.N, 0:self.N] = cp.fft.irfft(imf, nim7, 0)
        imf = cp.fft.rfft(img2[:, 0:self.N, self.N:2 * self.N], nim, 0)[:nim7 // 2 + 1, :, :]
        imgout[:, 0:self.N, self.N:2 * self.N] = cp.fft.irfft(imf, nim7, 0)
        imf = cp.fft.rfft(img2[:, self.N:2 * self.N, self.N:2 * self.N], nim, 0)[:nim7 // 2 + 1, :, :]
        imgout[:, self.N:2 * self.N, self.N:2 * self.N] = cp.fft.irfft(imf, nim7, 0)
        imf = None
        cp._default_memory_pool.free_all_blocks()

        res = cp.fft.irfft2(cp.fft.rfft2(imgout) * cp.asarray(self._postfilter[:, :self.N + 1]))
        return res

    def _findCarrier(self, band0, band1, mask):
        band = band0 * band1
        ixf = abs(fft.fftshift(fft.fft2(fft.fftshift(band))))

        if self.debug:
            plt.figure()
            plt.title('Find carrier')
            plt.imshow((ixf - gaussian_filter(ixf, 20)) * mask)

        pyc0, pxc0 = self._findPeak((ixf - gaussian_filter(ixf, 20)) * mask)
        ixfz, Kx, Ky = self._zoomf(band, self.N, self._kx[pyc0, pxc0], self._ky[pyc0, pxc0], 50, self._dk * self.N)
        pyc, pxc = self._findPeak(abs(ixfz))

        if self.debug:
            plt.figure()
            plt.title('Zoon Find carrier')
            plt.imshow(abs(ixfz))
        # phase = np.angle(ixfz[pyc, pxc])
        kx = Kx[pxc]
        ky = Ky[pyc]

        otf_exclude_min_radius = 0.5
        otf_exclude_max_radius = 1.5

        kr = sqrt(self._kx ** 2 + self._ky ** 2)

        m = (kr < 2)
        otf = fft.fftshift(self._tfm(kr, m) + (1 - m))

        otf_mask = (kr > otf_exclude_min_radius) & (kr < otf_exclude_max_radius)
        otf_mask_for_band_common_freq = fft.fftshift(
            otf_mask & scipy.ndimage.shift(otf_mask, (pyc0 - (self.N // 2 + 1), pxc0 - (self.N // 2 + 1)), order=0))
        band0_common = fft.ifft2(fft.fft2(band0) / otf * otf_mask_for_band_common_freq)

        # xx = np.linspace(-self.N / 2 * self._dx, self.N / 2 * self._dx - self._dx, self.N, dtype=np.single)
        xx = np.arange(-self.N / 2 * self._dx, self.N / 2 * self._dx, self._dx, dtype=np.single)
        phase_shift_to_xpeak = exp(-1j * kx * xx * 2 * pi * self.NA / self.wavelength)
        phase_shift_to_ypeak = exp(-1j * ky * xx * 2 * pi * self.NA / self.wavelength)

        band1_common = fft.ifft2(fft.fft2(band1) / otf * otf_mask_for_band_common_freq) * np.outer(
            phase_shift_to_ypeak, phase_shift_to_xpeak)

        scaling = 1 / np.sum(band0_common * np.conjugate(band0_common))

        cross_corr_result = np.sum(band0_common * band1_common) * scaling

        ampl = np.abs(cross_corr_result) * 2
        phase = np.angle(cross_corr_result)
        return kx, ky, phase, ampl

    def _findCarrier_cupy(self, band0, band1, mask):
        band0 = cp.asarray(band0)
        band1 = cp.asarray(band1)
        mask = cp.asarray(mask)
        
        band = band0 * band1
        ixf = abs(cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(band))))
        pyc0, pxc0 = self._findPeak((ixf - gaussian_filter_cupy(ixf, 20)) * mask)
        # print(pyc0,pxc0)
        # print(cp.asarray(self._kx)[pyc0, pxc0])
        ixfz, Kx, Ky = self._zoomf_cupy(band, self.N, cp.asarray(self._kx)[pyc0, pxc0], cp.asarray(self._ky)[pyc0, pxc0], 100, self._dk * self.N)
        pyc, pxc = self._findPeak(abs(ixfz))
        # phase = np.angle(ixfz[pyc, pxc])
        kx = Kx[pxc]
        ky = Ky[pyc]

        otf_exclude_min_radius = 0.5
        otf_exclude_max_radius = 1.5

        kr = cp.sqrt(cp.asarray(self._kx) ** 2 + cp.asarray(self._ky) ** 2)

        m = (kr < 2)
        otf = cp.fft.fftshift(self._tfm(kr, m) + (1 - m))

        otf_mask = (kr > otf_exclude_min_radius) & (kr < otf_exclude_max_radius)
        otf_mask_for_band_common_freq = cp.fft.fftshift(
            otf_mask & cupyx.scipy.ndimage.shift(otf_mask, (pyc0 - (self.N // 2 + 1), pxc0 - (self.N // 2 + 1)), order=0))
        band0_common = cp.fft.ifft2(cp.fft.fft2(band0) / otf * otf_mask_for_band_common_freq)

        xx = cp.linspace(-self.N / 2 * self._dx, self.N / 2 * self._dx - self._dx, self.N, dtype=np.single)
        phase_shift_to_xpeak = cp.exp(-1j * kx * xx * 2 * pi * self.NA / self.wavelength)
        phase_shift_to_ypeak = cp.exp(-1j * ky * xx * 2 * pi * self.NA / self.wavelength)

        band1_common = cp.fft.ifft2(cp.fft.fft2(band1) / otf * otf_mask_for_band_common_freq) * cp.outer(
            phase_shift_to_ypeak, phase_shift_to_xpeak)

        scaling = 1 / cp.sum(band0_common * cp.conj(band0_common))

        cross_corr_result = cp.sum(band0_common * band1_common) * scaling

        ampl = cp.abs(cross_corr_result) * 2
        phase = cp.angle(cross_corr_result)
        return kx.get(), ky.get(), phase.get(), ampl.get()
    
    def _findPeak(self, in_array):
        xp = cp.get_array_module(in_array)
        return xp.unravel_index(xp.argmax(in_array, axis=None), in_array.shape)

    def _zoomf(self, in_arr, M, kx, ky, mag, kmax):
        resy = pyczt(in_arr, M, exp(-1j * 2 * pi / (mag * M)), exp(-1j * pi * (1 / mag - 2 * ky / kmax)))
        res = pyczt(resy.T, M, exp(-1j * 2 * pi / (mag * M)), exp(-1j * pi * (1 / mag - 2 * kx / kmax))).T
        kyarr = -kmax * (1 / mag - 2 * ky / kmax) / 2 + (kmax / (mag * (M))) * np.arange(0, M)
        kxarr = -kmax * (1 / mag - 2 * kx / kmax) / 2 + (kmax / (mag * (M))) * np.arange(0, M)
        dim = np.shape(in_arr)
        # remove phase tilt from (0,0) offset in spatial domain
        res = res * (exp(1j * (kyarr) * dim[0] * pi / kmax)[:, np.newaxis])
        res = res * (exp(1j * (kxarr) * dim[0] * pi / kmax)[np.newaxis, :])
        return res, kxarr, kyarr
    
    def _zoomf_cupy(self, in_arr, M, kx, ky, mag, kmax):
        resy = self._pyczt_cupy(in_arr, M, cp.exp(-1j * 2 * pi / (mag * M)), cp.exp(-1j * pi * (1 / mag - 2 * ky / kmax)))
        res = self._pyczt_cupy(resy.T, M, cp.exp(-1j * 2 * pi / (mag * M)), cp.exp(-1j * pi * (1 / mag - 2 * kx / kmax))).T
        kyarr = -kmax * (1 / mag - 2 * ky / kmax) / 2 + (kmax / (mag * (M))) * cp.arange(0, M)
        kxarr = -kmax * (1 / mag - 2 * kx / kmax) / 2 + (kmax / (mag * (M))) * cp.arange(0, M)
        dim = cupyx.scipy.sparse.csr_matrix.get_shape(in_arr)
        # remove phase tilt from (0,0) offset in spatial domain
        res = res * (cp.exp(1j * (kyarr) * dim[0] * pi / kmax)[:, cp.newaxis])
        res = res * (cp.exp(1j * (kxarr) * dim[0] * pi / kmax)[cp.newaxis, :])
        return res, kxarr, kyarr
    
    def _att(self, kr):
        atf = (1 - self.beta * exp(-kr ** 2 / (2 * self.alpha ** 2)))
        return atf

    def _attm(self, kr, mask):
        atf = np.zeros_like(kr).flatten()
        mf = mask.flatten()
        atff = atf.flatten()
        krf = kr.flatten()[mf]
        atff[mf] = self._att(krf)
        return atff.reshape(kr.shape)

    def _tf(self, kr):
        xp = cp.get_array_module(kr)
        otf = (1 / pi * (xp.arccos(kr / 2) - kr / 2 * xp.sqrt(1 - kr ** 2 / 4)))
        return otf

    def _tfm(self, kr, mask):
        xp = cp.get_array_module(kr)
        otf = xp.zeros_like(kr).flatten()
        mf = mask.flatten()
        otff = otf.flatten()
        krf = kr.flatten()[mf]
        otff[mf] = self._tf(krf)
        return otff.reshape(kr.shape)
    
    def _pyczt_cupy(self, x, k=None, w=None, a=None):
        olddim = x.ndim
    
        if olddim == 1:
            x = x[:, cp.newaxis]
    
        (m, n) = x.shape
        oldm = m
    
        if m == 1:
            x = x.transpose()
            (m, n) = x.shape
    
        if k is None:
            k = len(x)
        if w is None:
            w = cp.exp(-1j * 2 * pi / k)
        if a is None:
            a = 1.
    
        # %------- Length for power-of-two fft.
    
        nfft = int(2**cp.ceil(cp.log2(abs(m+k-1))))
    
        # %------- Premultiply data.
    
        kk = cp.arange(-m+1, max(k, m))[:, cp.newaxis]
        kk2 = (kk ** 2) / 2
        ww = w ** kk2   # <----- Chirp filter is 1./ww
        nn = cp.arange(0, m)[:, cp.newaxis]
        aa = a ** (-nn)
        aa = aa * ww[m+nn-1, 0]
        y = (x * aa).astype(np.complex64)
    
        # %------- Fast convolution via FFT.
    
        fy = cp.fft.fft(y, nfft, axis=0)
        fv = cp.fft.fft(1 / ww[0: k-1+m], nfft, axis=0)   # <----- Chirp filter.
        fy = fy * fv
        g = cp.fft.ifft(fy, axis=0)
    
        # %------- Final multiply.
    
        g = g[m-1:m+k-1, :] * ww[m-1:m+k-1]
    
        if oldm == 1:
            g = g.transpose()
    
        if olddim == 1:
            g = g.squeeze()
    
        return g
