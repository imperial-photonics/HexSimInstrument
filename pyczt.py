import numpy as np
import math

try :
    import pyfftw
    import pyfftw.interfaces.numpy_fft as fft
    pyfftw.config.NUM_THREADS = multiprocessing.cpu_count()
    pyfftw.interfaces.cache.enable()
except:
    import numpy.fft as fft


def pyczt(x, k=None, w=None, a=None):
    # Chirp z-transform ported from Matlab implementation (see comment below)
    # By Mark Neil Apr 2020
    # %CZT  Chirp z-transform.
    # %   G = CZT(X, M, W, A) is the M-element z-transform of sequence X,
    # %   where M, W and A are scalars which specify the contour in the z-plane
    # %   on which the z-transform is computed.  M is the length of the transform,
    # %   W is the complex ratio between points on the contour, and A is the
    # %   complex starting point.  More explicitly, the contour in the z-plane
    # %   (a spiral or "chirp" contour) is described by
    # %       z = A * W.^(-(0:M-1))
    # %
    # %   The parameters M, W, and A are optional; their default values are
    # %   M = length(X), W = exp(-j*2*pi/M), and A = 1.  These defaults
    # %   cause CZT to return the z-transform of X at equally spaced points
    # %   around the unit circle, equivalent to FFT(X).
    # %
    # %   If X is a matrix, the chirp z-transform operation is applied to each
    # %   column.
    # %
    # %   See also FFT, FREQZ.
    #
    # %   Author(s): C. Denham, 1990.
    # %   	   J. McClellan, 7-25-90, revised
    # %   	   C. Denham, 8-15-90, revised
    # %   	   T. Krauss, 2-16-93, updated help
    # %   Copyright 1988-2002 The MathWorks, Inc.
    # %       $Revision: 1.7.4.1 $  $Date: 2007/12/14 15:04:15 $
    #
    # %   References:
    # %     [1] Oppenheim, A.V. & R.W. Schafer, Discrete-Time Signal
    # %         Processing,  Prentice-Hall, pp. 623-628, 1989.
    # %     [2] Rabiner, L.R. and B. Gold, Theory and Application of
    # %         Digital Signal Processing, Prentice-Hall, Englewood
    # %         Cliffs, New Jersey, pp. 393-399, 1975.

    olddim = x.ndim

    if olddim == 1:
        x = x[:, np.newaxis]

    (m, n) = x.shape
    oldm = m

    if m == 1:
        x = x.transpose()
        (m, n) = x.shape

    if k is None:
        k = len(x)
    if w is None:
        w = np.exp(-1j * 2 * math.pi / k)
    if a is None:
        a = 1.

    # %------- Length for power-of-two fft.

    nfft = int(2**np.ceil(math.log2(abs(m+k-1))))

    # %------- Premultiply data.

    kk = np.arange(-m+1, max(k, m))[:, np.newaxis]
    kk2 = (kk ** 2) / 2
    ww = w ** kk2   # <----- Chirp filter is 1./ww
    nn = np.arange(0, m)[:, np.newaxis]
    aa = a ** (-nn)
    aa = aa * ww[m+nn-1, 0]
    # y = (x * aa)
    y = (x * aa).astype(np.complex64)
    # print(y.dtype)
    # %------- Fast convolution via FFT.

    fy = fft.fft(y, nfft, axis=0)
    fv = fft.fft(1 / ww[0: k-1+m], nfft, axis=0)   # <----- Chirp filter.
    fy = fy * fv
    g = fft.ifft(fy, axis=0)

    # %------- Final multiply.

    g = g[m-1:m+k-1, :] * ww[m-1:m+k-1]

    if oldm == 1:
        g = g.transpose()

    if olddim == 1:
        g = g.squeeze()

    return g
