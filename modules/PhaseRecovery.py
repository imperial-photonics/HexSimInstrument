"""A class to correct the aberration mainly introcuded by the R11 SLM. This work is based on Mark Neil's Matlab code.
    A Thorlabs scientific camera(CS2100M-USB) is used to obtain PSF images."""

import numpy as np
import cupy as cp
import opt_einsum as oe
from scipy import interpolate
import time

__author__ = "Meizhu Liang"

class Phase_correction:
    xp = cp
    xpix = 2048
    ypix = 2048

    # Scaling calculation all distances in µm
    d_c = 5.04  # camera pixel size
    N = 256  # camera roi size
    d_s = 8.2  # slm pixel size
    n_s = 2048  # slm roi size
    fl = 250e3  # fourier focal length
    l = 0.488  # wavelength

    # Sampling in pupil plane
    s_p = fl * l / (N * d_c)

    # pupil plane diameter in pixels
    d_p = n_s * d_s / s_p

    # Scaling from pupil plane to SLM plane
    s_fac = n_s / d_p

    # Set up arrays and Chebyshev polynomials
    xx = xp.linspace(-N / d_p, N / d_p, N)
    m = np.abs(xx) < 1
    ne = (N - sum(m)) // 2  # 1/2 number of elements along side of pupil
    xx *= m

    # weight function along 1-d
    wtx = (1 / xp.sqrt(1 - (xx ** 2))) * m
    # find correctiom to end element that makes intergral(Ch[0]*Ch[2]*wtx) = 0
    s = xp.sum(wtx * (2 * xx ** 2 - 1))
    df = s / (2 * xx[ne] ** 2 - 1) / 2
    wtx[ne] -= df
    wtx[-ne - 1] = wtx[ne]

    # Now calculate 2-d weights
    x, y = xp.meshgrid(xx, xx)
    circ = xp.outer(m, m)
    wt = xp.outer(wtx, wtx)
    r = xp.sqrt(x ** 2 + y ** 2)

    # weight the target PSF towards the centre (or not if all ones)
    G = xp.ones((int(N), int(N)))

    n_c = 6
    n_p = 4

    hex_bits0 = [None] * 3
    def set_C(self, xc, yc, n_c, Nc, circ0):
        ch = np.zeros((n_c, n_c, Nc, Nc))
        if self.xp == cp:
            xc = xc.get()
            yc = yc.get()
            for n in range(n_c):
                for k in range(n_c):
                    c = np.zeros((n + 1, k + 1))
                    c[n, k] = 1
                    ch[n, k, :, :] = np.polynomial.chebyshev.chebval2d(xc, yc, c) * circ0.get()
            ch = cp.array(ch)
        else:
            for n in range(n_c):
                for k in range(n_c):
                    c = np.zeros((n + 1, k + 1))
                    c[n, k] = 1
                    ch[n, k, :, :] = np.polynomial.chebyshev.chebval2d(xc, yc, c) * circ0
        return ch

    def set_Cs(self, xc, yc, n_s):
        if self.xp == cp:
            xc = xc.get()
            yc = yc.get()
        ch = np.zeros((n_s, n_s, n_s + 1, n_s + 1))
        for n in range(n_s):
            for k in range(n_s):
                c = np.zeros((n + 1, k + 1))
                c[n, k] = 1
                ch[n, k, :, :] = np.polynomial.chebyshev.chebval2d(xc, yc, c)
        if self.xp == cp:
            ch = cp.array(ch)
        return ch

    def interleaving(self, x_dis):
        x0 = np.zeros(self.xpix)
        y0 = np.arange(self.ypix)

        nr = np.arange(8)
        for i in range(64):
            x0[(i * 32):(i * 32 + 8)] = nr + i * 8
            x0[(i * 32 + 8):(i * 32 + 16)] = nr + i * 8 + 512
            x0[(i * 32 + 16):(i * 32 + 24)] = nr + i * 8 + 1024
            x0[(i * 32 + 24):(i * 32 + 32)] = nr + i * 8 + 1536
        # x0 is now an array of interleaved x values in the correct places for sending to the SLM

        x, y = np.meshgrid(x0 * x_dis, y0)
        return x, y
    def __init__(self):
        chs = self.set_C(self.x, self.y, self.n_c, self.N, self.circ)

        c_a_p = self.xp.reshape(chs, (self.n_c ** 2, self.N, self.N))  # Chebyshev aberration for phase
        n_c_i = 4
        c_a_i = self.xp.reshape(chs[:n_c_i, :n_c_i, :, :], (n_c_i ** 2, self.N, self.N))  # Chebyshev aberration for intensity
        self.normval_p = 1 / oe.contract('ijk, ijk, jk -> i', c_a_p, c_a_p, self.wt)
        self.normval_i = 1 / oe.contract('ijk, ijk, jk -> i', c_a_i, c_a_i, self.wt)

        # Orthononality matrix diagonal gives normalisation
        # plt.matshow(np.abs(oe.contract('ijk, mjk, jk -> im', c_a_p, c_a_p, self.wt).get()) ** 0.5)

        # Chebyshev expansion by orthogonality at nodal points
        p_s = np.polynomial.chebyshev.chebpts1(self.n_c + 1)
        if self.xp == cp:
            ch_xs, ch_ys = self.xp.meshgrid(cp.array(p_s), cp.array(p_s))
        else:
            ch_xs, ch_ys = self.xp.meshgrid(p_s, p_s)

        chs_s = self.set_Cs(ch_xs, ch_ys, self.n_c)

        c_a_ps = self.xp.reshape(chs_s, (self.n_c ** 2, self.n_c + 1, self.n_c + 1))  # Chebyshev aberration for phase
        c_a_is = self.xp.reshape(chs_s[:self.n_p, :self.n_p, :, :],
                            (self.n_p ** 2, self.n_c + 1, self.n_c + 1))  # Chebyshev aberration for intensity
        normval_ps = 1 / oe.contract('ijk, ijk -> i', c_a_ps, c_a_ps)
        normval_is = 1 / oe.contract('ijk, ijk -> i', c_a_is, c_a_is)
        self.nm = int(self.xp.sum(self.m).item())

        # Orthononality matrix diagonal gives normalisation
        # plt.matshow((self.xp.sqrt(self.xp.abs(oe.contract('ijk, mjk, i -> im', c_a_ps, c_a_ps, normval_ps)))).get())
        # with np.printoptions(precision=3, suppress=True):
        #     print(oe.contract('ijk, mjk, i -> im', c_a_ps, c_a_ps, normval_ps))

        self.xi = self.xp.zeros((self.n_c + 1, self.n_c + 1, 2))
        self.xi[:, :, 0] = ch_ys
        self.xi[:, :, 1] = ch_xs
        c_interp = np.zeros_like(c_a_ps.get())
        if self.xp == cp:
            for i in range(self.n_c * self.n_c):
                c_interp[i, :, :] = interpolate.interpn((self.xx[self.m].get(), self.xx[self.m].get()),
                                                        c_a_p.get()[i, self.circ.get()].reshape((self.nm, self.nm)),
                                                        self.xi.get(), method='splinef2d')
            c_interp = cp.array(c_interp)
        else:
            for i in range(self.n_c * self.n_c):
                c_interp[i, :, :] = interpolate.interpn((self.xx[self.m], self.xx[self.m]),
                                                        c_a_p[i, self.circ].reshape((self.nm, self.nm)),
                                                        self.xi, method='splinef2d')

        # Initial input
        distortion = np.sqrt(1 - (1.5 / 7) ** 2)  # distortion in x
        xv, yv = self.interleaving(distortion)
        if self.xp == cp:
            # place the pixels in negative and positive axes
            self.xSLM = (self.xp.array(xv) - self.xpix / 2) / (self.xpix)
            self.ySLM = (self.xp.array(yv) - self.ypix / 2) / (self.ypix)
        else:
            # place the pixels in negative and positive axes
            self.xSLM = (xv - self.xpix / 2) / (self.xpix)
            self.ySLM = (yv - self.ypix / 2) / (self.ypix)

        # Calculte Chebyshev polynomials for 2048 * 2048 pixels
        ND = 2048
        xd = self.xp.linspace(-1, 1, ND)
        xD = xd[self.xp.array(yv, dtype=int)]
        yD = -xd[self.xp.array(xv, dtype=int)]
        mD = np.abs(xd) < 1
        self.circD = self.xp.outer(mD, mD)
        chsD = self.set_C(xD, yD, self.n_c, ND, self.circD)
        self.c_a_pD = self.xp.reshape(chsD, (self.n_c ** 2, ND, ND))  # Chebyshev aberration for phase

        self.bias = [4 * (chs[0, 2] + chs[2, 0]), 8 * chs[2, 2], -4 * (chs[0, 2] + chs[2, 0])]
        self.biasD = self.xp.zeros((3, self.ypix, self.xpix), dtype=self.xp.double)
        self.biasD[0] = 4 * (chsD[0, 2] + chsD[2, 0])
        self.biasD[1] = 8 * chsD[2, 2]
        self.biasD[2] = -4 * (chsD[0, 2] + chsD[2, 0])


        self.ri2 = (-18 * chs[1, 1] + 5 * chs[2, 2]
               - 7 * chs[3, 1] + 11 * chs[3, 3]) * self.circ / 4  # starting random input phase
        self.chs = chs
        self.c_a_p = c_a_p
        self.c_a_i = c_a_i
        self.c_a_ps = c_a_ps
        self.normval_ps = normval_ps
if __name__ == '__main__':
    t = time.time()
    phc = Phase_correction()
    t1 = time.time() - t
    print(t1)
    t2 = time.time()
    print(phc.Tau)
    print(f'{(time.time() - t2)}s')





