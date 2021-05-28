"""
Python implementation of the Image Resolution Estimation algorithm by A. Descloux et al.

Descloux, A., K. S. Grußmayer, et A. Radenovic. _Parameter-Free Image
Resolution Estimation Based on Decorrelation Analysis_. Nature Methods
16, nᵒ 9 (septembre 2019):
918‑24. https://doi.org/10.1038/s41592-019-0515-7.

Original source code in matlab™ and ImageJ plugin are available at https://github.com/Ades91/ImDecorr

"""
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize_scalar
from scipy.fft import fftn, fftshift, ifftn, ifftshift
from scipy.signal import general_gaussian

def _fft(image):
    """shifted fft 2D
    """
    return fftshift(fftn(fftshift(image)))


def _ifft(im_fft):
    """shifted ifft 2D
    """
    return ifftshift(ifftn(ifftshift(im_fft)))


# apodImRect.m
def apodise(image, border, order=8):
    """
    Parameters
    ----------

    image: np.ndarray
    border: int, the size of the boreder in pixels

    Note
    ----
    The image is assumed to be of float datatype, no datatype management
    is performed.

    This is different from the original apodistation method,
    which multiplied the image borders by a quater of a sine.
    """
    # stackoverflow.com/questions/46211487/apodization-mask-for-fast-fourier-transforms-in-python
    nx, ny = image.shape
    # Define a general Gaussian in 2D as outer product of the function with itself
    window = np.outer(
        general_gaussian(nx, order, nx // 2 - border),
        general_gaussian(ny, order, ny // 2 - border),
    )
    ap_image = window * image

    return ap_image


def fft_dist(nx, ny):

    uu2, vv2 = np.meshgrid(np.fft.fftfreq(ny) ** 2, np.fft.fftfreq(nx) ** 2)
    dist = (uu2 + vv2) ** 0.5
    return dist  # / dist.sum()


def measure(image, metadata):
    """Estimates SNR and resolution of an image based on the Image Resolution Estimation
    algorithm by A. Descloux et al.


    Descloux, A., K. S. Grußmayer, et A. Radenovic. _Parameter-Free Image
    Resolution Estimation Based on Decorrelation Analysis_. Nature Methods
    16, nᵒ 9 (septembre 2019) 918‑24. https://doi.org/10.1038/s41592-019-0515-7.

    Parameters
    ----------
    image : the 2D image to be evaluated
    metadata : image metadata (the key physicalSizeX will be use as pixel size)

    Returns
    -------
    measured_data : dict
        the evaluated SNR and resolution

    """
    pixel_size = metadata.get("physicalSizeX", 1.0)
    imdecor = ImageDecorr(image, pixel_size)
    imdecor.compute_resolution()
    return {"SNR": imdecor.snr0, "resolution": imdecor.resolution}


class ImageDecorr:
    pod_size = 100
    pod_order = 10

    def __init__(self, image, pixel_size=1.0, square_crop=True):
        """ Creates an ImageDecorr contrainer class

        Parameters
        ----------
        image: 2D np.ndarray
        """

        self.image = apodise(image, self.pod_size, self.pod_order)
        self.pixel_size = pixel_size
        nx, ny = self.image.shape

        if square_crop:
            # odd number of pixels, square image
            n = min(nx, ny)
            n = n - (1 - n % 2)
            self.image = self.image[:n, :n]
            self.size = n ** 2
            xx, yy = np.meshgrid(np.linspace(-1, 1, n), np.linspace(-1, 1, n))
        else:

            nx = nx - (1 - nx % 2)
            ny = ny - (1 - nx % 2)
            self.image = self.image[:nx, :ny]
            self.size = nx * ny
            xx, yy = np.meshgrid(np.linspace(-1, 1, ny), np.linspace(-1, 1, nx))

        self.disk = xx ** 2 + yy ** 2
        self.mask0 = self.disk < 1.0

        im_fft0 = _fft(self.image)
        im_fft0 /= np.abs(im_fft0)
        im_fft0[~np.isfinite(im_fft0)] = 0

        self.im_fft0 = im_fft0 * self.mask0  # I in original code
        image_bar = (self.image - self.image.mean()) / self.image.std()
        im_fftk = _fft(image_bar) * self.mask0  # Ik
        self.im_invk = _ifft(im_fftk).real  # imr

        self.im_fftr = _masked_fft(self.im_invk, self.mask0, self.size)  # Ir

        self.snr0, self.kc0 = self.maximize_corcoef(self.im_fftr).values()  # A0, res0
        self.max_width = 2 / self.kc0
        self.kc = None
        self.resolution = None

    def corcoef(self, radius, im_fftr, c1=None):
        """Computes the normed correlation coefficient between
        the two FFTS of eq. 1 in Descloux et al.
        """
        mask = self.disk < radius ** 2
        f_im_fft = (mask * self.im_fft0).ravel()[: self.size // 2]
        if c1 is None:
            c1 = np.linalg.norm(im_fftr)
        c2 = np.linalg.norm(f_im_fft)

        return (im_fftr * f_im_fft.conjugate()).real.sum() / (c1 * c2)

    def maximize_corcoef(self, im_fftr, r_min=0, r_max=1):
        """Finds the cutoff radius corresponding to the maximum of the correlation coefficient for
        image fft im_fftr (noted r_i in the article)

        Returns
        -------
        result : dict
            the key 'snr' is the value of self.corcoef at the maximum
            the key 'kc' corresponds to the argmax of self.corcoef
        """
        # cost function
        def anti_cor(radius):
            c1 = np.linalg.norm(im_fftr)
            cor = self.corcoef(radius, im_fftr, c1=c1)
            return 1 - cor

        res = minimize_scalar(
            anti_cor, bounds=(r_min, r_max), method="bounded", options={"xatol": 1e-4}
        )

        if not res.success:
            return {"snr": 0.0, "kc": 1.0}

        if (r_max - res.x) / r_max < 1e-3:
            return {"snr": 0.0, "kc": r_max}

        return {"snr": 1 - res.fun, "kc": res.x}

    def all_corcoefs(self, num_rs, r_min=0, r_max=1, num_ws=0):
        """Computes decorrelation data for num_rs radius and num_ws filter widths

        This allows to produce plots similar to those of the imagej plugin
        or e.g. fig 1b

        Parameters
        ----------
        num_rs : int
            the number of mask radius
        r_min, r_max : floats
            min and max of the mask radii
        num_ws : float
            number of Gaussian blur filters

        Returns
        -------
        data : dict of ndarrays


        """

        radii = np.linspace(r_min, r_max, num_rs)
        c1 = np.linalg.norm(self.im_fftr)
        d0 = np.array([self.corcoef(radius, self.im_fftr, c1=c1) for radius in radii])
        if not num_ws:
            return {"radii": radii, "ds": d0}

        ds = [d0]
        snr, kc = self.maximize_corcoef(self.im_fftr, r_min, r_max).values()
        snrs = [snr]
        kcs = [kc]

        widths = np.concatenate(
            [[0,], np.logspace(-1, np.log10(self.max_width), num_ws)]
        )
        for width in widths[1:]:
            f_im = self.im_invk - gaussian_filter(self.im_invk, width)
            f_im_fft = _masked_fft(f_im, self.mask0, self.size)
            c1 = np.linalg.norm(f_im_fft)
            d = np.array([self.corcoef(radius, f_im_fft, c1=c1) for radius in radii])
            ds.append(d)
            snr, kc = self.maximize_corcoef(f_im_fft, r_min, r_max).values()
            snrs.append(snr)
            kcs.append(kc)

        data = {
            "radius": np.array(radii),
            "d": np.array(ds),
            "snr": np.array(snrs),
            "kc": np.array(kcs),
            "widths": widths,
        }
        return data

    def filtered_decorr(self, width, returm_gm=True):
        """Computes the decorrelation cutoff for a given
        filter widh

        If return_gm is True, returns 1 minus the geometric means,
        to be used as a cost function, else, returns the snr
        and the cutoff.
        """
        f_im = self.im_invk - gaussian_filter(self.im_invk, width)
        f_im_fft = _masked_fft(f_im, self.mask0, self.size)
        res = self.maximize_corcoef(f_im_fft)

        if returm_gm:
            if (1 - res["kc"]) < 1e-1:
                return 1 + width
            return 1 - (res["kc"] * res["snr"]) ** 0.5
        return res

    def compute_resolution(self):
        """Finds the filter width giving the maximum of the geometric
        mean (kc * snr)**0.5 (eq. 2)


        """

        res = minimize_scalar(
            self.filtered_decorr,
            method="bounded",
            bounds=(0.15, self.max_width),
            options={"xatol": 1e-3},
        )
        width = res.x
        max_cor = self.filtered_decorr(width, returm_gm=False)

        self.kc = max_cor["kc"]
        if self.kc:
            self.resolution = 2 * self.pixel_size / self.kc
        else:
            self.resolution = np.inf
        return res, max_cor


def _masked_fft(im, mask, size):
    return (mask * _fft(im)).ravel()[: size // 2]
