# pylint: disable=redefined-builtin
import cupy


# ######## Convolutions and Correlations ##########

def correlate(input, weights, output=None, mode='reflect', cval=0.0, origin=0):
    """Multi-dimensional correlate.
    The array is correlated with the given kernel.
    Args:
        input (cupy.ndarray): The input array.
        weights (cupy.ndarray): Array of weights, same number of dimensions as
            input.
        output (cupy.ndarray, dtype or None): The array in which to place the
            output. Default is is same dtype as the input.
        mode (str): The array borders are handled according to the given mode
            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,
            ``'wrap'``). Default is ``'reflect'``.
        cval (scalar): Value to fill past edges of input if mode is
            ``'constant'``. Default is ``0.0``.
        origin (int or sequence of int): The origin parameter controls the
            placement of the filter, relative to the center of the current
            element of the input. Default of 0 is equivalent to
            ``(0,)*input.ndim``.
    Returns:
        cupy.ndarray: The result of the correlating.
    .. seealso:: :func:`scipy.ndimage.correlate`
    """
    return _correlate_or_convolve(input, weights, output, mode, cval, origin,
                                  False)


def convolve(input, weights, output=None, mode='reflect', cval=0.0, origin=0):
    """Multi-dimensional convolution.
    The array is convolved with the given kernel.
    Args:
        input (cupy.ndarray): The input array.
        weights (cupy.ndarray): Array of weights, same number of dimensions as
            input.
        output (cupy.ndarray, dtype or None): The array in which to place the
            output. Default is is same dtype as the input.
        mode (str): The array borders are handled according to the given mode
            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,
            ``'wrap'``). Default is ``'reflect'``.
        cval (scalar): Value to fill past edges of input if mode is
            ``'constant'``. Default is ``0.0``.
        origin (int or sequence of int): The origin parameter controls the
            placement of the filter, relative to the center of the current
            element of the input. Default of 0 is equivalent to
            ``(0,)*input.ndim``.
    Returns:
        cupy.ndarray: The result of the convolution.
    .. seealso:: :func:`scipy.ndimage.convolve`
    """
    return _correlate_or_convolve(input, weights, output, mode, cval, origin,
                                  True)


def _correlate_or_convolve(input, weights, output, mode, cval, origin,
                           convolution):
    origins, int_type = _check_nd_args(input, weights, mode, origin)
    if weights.size == 0:
        return cupy.zeros_like(input)
    if convolution:
        weights = weights[tuple([slice(None, None, -1)] * weights.ndim)]
        origins = list(origins)
        for i, wsize in enumerate(weights.shape):
            origins[i] = -origins[i]
            if wsize % 2 == 0:
                origins[i] -= 1
        origins = tuple(origins)
    kernel = _get_correlate_kernel(mode, weights.shape, int_type,
                                   origins, cval)
    return _call_kernel(kernel, input, weights, output)


@cupy.memoize()
def _get_correlate_kernel(mode, wshape, int_type, origins, cval):
    return _get_nd_kernel('correlate',
                          'W sum = (W)0;',
                          'sum += (W){value} * wval;',
                          'y = (Y)sum;',
                          mode, wshape, int_type, origins, cval)


def correlate1d(input, weights, axis=-1, output=None, mode="reflect", cval=0.0,
                origin=0):
    """One-dimensional correlate.
    The array is correlated with the given kernel.
    Args:
        input (cupy.ndarray): The input array.
        weights (cupy.ndarray): One-dimensional array of weights
        axis (int): The axis of input along which to calculate. Default is -1.
        output (cupy.ndarray, dtype or None): The array in which to place the
            output. Default is is same dtype as the input.
        mode (str): The array borders are handled according to the given mode
            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,
            ``'wrap'``). Default is ``'reflect'``.
        cval (scalar): Value to fill past edges of input if mode is
            ``'constant'``. Default is ``0.0``.
        origin (int): The origin parameter controls the placement of the
            filter, relative to the center of the current element of the
            input. Default is ``0``.
    Returns:
        cupy.ndarray: The result of the 1D correlation.
    .. seealso:: :func:`scipy.ndimage.correlate1d`
    """
    weights, origins = _convert_1d_args(input.ndim, weights, origin, axis)
    return correlate(input, weights, output, mode, cval, origins)


def convolve1d(input, weights, axis=-1, output=None, mode="reflect", cval=0.0,
               origin=0):
    """One-dimensional convolution.
    The array is convolved with the given kernel.
    Args:
        input (cupy.ndarray): The input array.
        weights (cupy.ndarray): One-dimensional array of weights
        axis (int): The axis of input along which to calculate. Default is -1.
        output (cupy.ndarray, dtype or None): The array in which to place the
            output. Default is is same dtype as the input.
        mode (str): The array borders are handled according to the given mode
            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,
            ``'wrap'``). Default is ``'reflect'``.
        cval (scalar): Value to fill past edges of input if mode is
            ``'constant'``. Default is ``0.0``.
        origin (int): The origin parameter controls the placement of the
            filter, relative to the center of the current element of the
            input. Default is ``0``.
    Returns:
        cupy.ndarray: The result of the 1D convolution.
    .. seealso:: :func:`scipy.ndimage.convolve1d`
    """
    weights = weights[::-1]
    origin = -origin
    if not len(weights) & 1:
        origin -= 1
    return correlate1d(input, weights, axis, output, mode, cval, origin)


def uniform_filter1d(input, size, axis=-1, output=None, mode="reflect",
                     cval=0.0, origin=0):
    """One-dimensional uniform filter along the given axis.

    The lines of the array along the given axis are filtered with a uniform
    filter of the given size.

    Args:
        input (cupy.ndarray): The input array.
        size (int): Length of the uniform filter.
        axis (int): The axis of input along which to calculate. Default is -1.
        output (cupy.ndarray, dtype or None): The array in which to place the
            output. Default is is same dtype as the input.
        mode (str): The array borders are handled according to the given mode
            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,
            ``'wrap'``). Default is ``'reflect'``.
        cval (scalar): Value to fill past edges of input if mode is
            ``'constant'``. Default is ``0.0``.
        origin (int): The origin parameter controls the placement of the
            filter, relative to the center of the current element of the
            input. Default is ``0``.
    Returns:
        cupy.ndarray: The result of the filtering.
    .. seealso:: :func:`scipy.ndimage.uniform_filter1d`
    """
    return correlate1d(input, cupy.ones(size) / size, axis, output, mode, cval,
                       origin)


def uniform_filter(input, size=3, output=None, mode="reflect", cval=0.0,
                   origin=0):
    """Multi-dimensional uniform filter.

    Args:
        input (cupy.ndarray): The input array.
        size (int or sequence of int): Lengths of the uniform filter for each
            dimension. A single value applies to all axes.
        output (cupy.ndarray, dtype or None): The array in which to place the
            output. Default is is same dtype as the input.
        mode (str): The array borders are handled according to the given mode
            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,
            ``'wrap'``). Default is ``'reflect'``.
        cval (scalar): Value to fill past edges of input if mode is
            ``'constant'``. Default is ``0.0``.
        origin (int or sequence of int): The origin parameter controls the
            placement of the filter, relative to the center of the current
            element of the input. Default of ``0`` is equivalent to
            ``(0,)*input.ndim``.
    Returns:
        cupy.ndarray: The result of the filtering.
    .. seealso:: :func:`scipy.ndimage.uniform_filter`
    """
    sizes = _fix_sequence_arg(size, input.ndim, 'size', int)

    def get(size):
        return None if size <= 1 else cupy.ones(size) / size

    return _nd_correlate(input, sizes, get, output, mode, cval, origin)


def gaussian_filter1d(input, sigma, axis=-1, order=0, output=None,
                      mode="reflect", cval=0.0, truncate=4.0):
    """One-dimensional Gaussian filter along the given axis.

    The lines of the array along the given axis are filtered with a Gaussian
    filter of the given standard deviation.

    Args:
        input (cupy.ndarray): The input array.
        sigma (scalar): Standard deviation for Gaussian kernel.
        axis (int): The axis of input along which to calculate. Default is -1.
        order (int): An order of ``0``, the default, corresponds to convolution
            with a Gaussian kernel. A positive order corresponds to convolution
            with that derivative of a Gaussian.
        output (cupy.ndarray, dtype or None): The array in which to place the
            output. Default is is same dtype as the input.
        mode (str): The array borders are handled according to the given mode
            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,
            ``'wrap'``). Default is ``'reflect'``.
        cval (scalar): Value to fill past edges of input if mode is
            ``'constant'``. Default is ``0.0``.
        truncate (float): Truncate the filter at this many standard deviations.
            Default is ``4.0``.
    Returns:
        cupy.ndarray: The result of the filtering.
    .. seealso:: :func:`scipy.ndimage.gaussian_filter1d`
    """
    radius = int(float(truncate) * float(sigma) + 0.5)
    weights = _gaussian_kernel1d(sigma, int(order), radius)
    return correlate1d(input, weights, axis, output, mode, cval)


def gaussian_filter(input, sigma, order=0, output=None, mode="reflect",
                    cval=0.0, truncate=4.0):
    """Multi-dimensional Gaussian filter.

    Args:
        input (cupy.ndarray): The input array.
        sigma (scalar or sequence of scalar): Standard deviations for each axis
            of Gaussian kernel. A single value applies to all axes.
        order (int or sequence of scalar): An order of ``0``, the default,
            corresponds to convolution with a Gaussian kernel. A positive order
            corresponds to convolution with that derivative of a Gaussian. A
            single value applies to all axes.
        output (cupy.ndarray, dtype or None): The array in which to place the
            output. Default is is same dtype as the input.
        mode (str): The array borders are handled according to the given mode
            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,
            ``'wrap'``). Default is ``'reflect'``.
        cval (scalar): Value to fill past edges of input if mode is
            ``'constant'``. Default is ``0.0``.
        truncate (float): Truncate the filter at this many standard deviations.
            Default is ``4.0``.
    Returns:
        cupy.ndarray: The result of the filtering.
    .. seealso:: :func:`scipy.ndimage.gaussian_filter`
    """
    sigmas = _fix_sequence_arg(sigma, input.ndim, 'sigma', float)
    orders = _fix_sequence_arg(order, input.ndim, 'order', int)
    truncate = float(truncate)

    def get(param):
        sigma, order = param
        if sigma <= 1e-15:
            return None
        radius = int(truncate * float(sigma) + 0.5)
        return _gaussian_kernel1d(sigma, order, radius)

    return _nd_correlate(input, list(zip(sigmas, orders)), get, output, mode,
                         cval)


def _gaussian_kernel1d(sigma, order, radius):
    """
    Computes a 1-D Gaussian correlation kernel.
    """
    import numpy
    if order < 0:
        raise ValueError('order must be non-negative')
    sigma2i = -1 / (sigma * sigma)
    x = numpy.arange(-radius, radius+1)
    phi_x = numpy.exp(0.5 * sigma2i * x ** 2)
    phi_x *= 1 / phi_x.sum()

    if order == 0:
        return cupy.asarray(phi_x)
    
    # f(x) = q(x) * phi(x) = q(x) * exp(p(x))
    # f'(x) = (q'(x) + q(x) * p'(x)) * phi(x)
    # p'(x) = -1 / sigma ** 2
    # Implement q'(x) + q(x) * p'(x) as a matrix operator and apply to the
    # coefficients of q(x)
    exponent_range = numpy.arange(order + 1)
    q = numpy.zeros(order + 1)
    q[0] = 1
    D = numpy.diag(exponent_range[1:], 1)  # D @ q(x) = q'(x)
    P = numpy.diag(numpy.ones(order) * sigma2i, -1)  # P @ q(x) = q(x) * p'(x)
    Q_deriv = D + P
    for _ in range(order):
        q = Q_deriv.dot(q)
    q = (x[:, None] ** exponent_range).dot(q)
    return cupy.asarray((q * phi_x)[::-1])


def prewitt(input, axis=-1, output=None, mode="reflect", cval=0.0):
    """Compute a Prewitt filter along the given axis.

    Args:
        input (cupy.ndarray): The input array.
        axis (int): The axis of input along which to calculate. Default is -1.
        output (cupy.ndarray, dtype or None): The array in which to place the
            output. Default is is same dtype as the input.
        mode (str): The array borders are handled according to the given mode
            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,
            ``'wrap'``). Default is ``'reflect'``.
        cval (scalar): Value to fill past edges of input if mode is
            ``'constant'``. Default is ``0.0``.
    Returns:
        cupy.ndarray: The result of the filtering.
    .. seealso:: :func:`scipy.ndimage.prewitt`
    """
    return _prewitt_or_sobel(input, axis, output, mode, cval, cupy.ones(3))


def sobel(input, axis=-1, output=None, mode="reflect", cval=0.0):
    """Compute a Sobel filter along the given axis.

    Args:
        input (cupy.ndarray): The input array.
        axis (int): The axis of input along which to calculate. Default is -1.
        output (cupy.ndarray, dtype or None): The array in which to place the
            output. Default is is same dtype as the input.
        mode (str): The array borders are handled according to the given mode
            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,
            ``'wrap'``). Default is ``'reflect'``.
        cval (scalar): Value to fill past edges of input if mode is
            ``'constant'``. Default is ``0.0``.
    Returns:
        cupy.ndarray: The result of the filtering.
    .. seealso:: :func:`scipy.ndimage.sobel`
    """
    return _prewitt_or_sobel(input, axis, output, mode, cval,
                             cupy.array([1, 2, 1]))


def _prewitt_or_sobel(input, axis, output, mode, cval, weights):
    axis = _check_axis(axis, input.ndim)
    output = _get_output(output, input)
    modes = _fix_sequence_arg(mode, input.ndim, 'mode', _check_mode)
    correlate1d(input, cupy.array([-1, 0, 1]), axis, output, modes[axis], cval)
    if input.ndim == 1:
        return output
    input, output = output, _get_output(output.dtype, input)
    for _axis in range(input.ndim):
        if _axis == axis:
            continue
        correlate1d(output, weights, _axis, output, modes[_axis], cval)
        input, output = output, input
    return input


# ######## Derivatives of Convolution Filters ##########

def generic_laplace(input, derivative2, output=None, mode="reflect",
                    cval=0.0, extra_arguments=(), extra_keywords=None):
    """Multi-dimensional Laplace filter using a provided second derivative
    function.

    Args:
        input (cupy.ndarray): The input array.
        derivative2 (callable): Function or other callable with the following
            signature that is called once per axis::

                derivative2(input, axis, output, mode, cval,
                            *extra_arguments, **extra_keywords)

            where ``input`` and ``output`` are ``cupy.ndarray``, ``axis`` is an
            ``int`` from ``0`` to the number of dimensions, and ``mode``,
            ``cval``, ``extra_arguments``, ``extra_keywords`` are the values
            given to this function.
        output (cupy.ndarray, dtype or None): The array in which to place the
            output. Default is is same dtype as the input.
        mode (str): The array borders are handled according to the given mode
            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,
            ``'wrap'``). Default is ``'reflect'``.
        cval (scalar): Value to fill past edges of input if mode is
            ``'constant'``. Default is ``0.0``.
        extra_arguments (sequence, optional):
            Sequence of extra positional arguments to pass to ``derivative2``.
        extra_keywords (dict, optional):
            dict of extra keyword arguments to pass ``derivative2``.
    Returns:
        cupy.ndarray: The result of the filtering.
    .. seealso:: :func:`scipy.ndimage.generic_laplace`
    """
    if extra_keywords is None:
        extra_keywords = {}
    modes = _fix_sequence_arg(mode, input.ndim, 'mode', _check_mode)
    output = _get_output(output, input)
    if input.ndim == 0:
        output[...] = input[...]
        return output
    derivative2(input, 0, output, modes[0], cval,
                *extra_arguments, **extra_keywords)
    if input.ndim > 1:
        tmp = _get_output(output.dtype, input)
        for i in range(1, input.ndim):
            derivative2(input, i, tmp, modes[i], cval,
                        *extra_arguments, **extra_keywords)
            output += tmp
    return output


def laplace(input, output=None, mode="reflect", cval=0.0):
    """Multi-dimensional Laplace filter based on approximate second
    derivatives.

    Args:
        input (cupy.ndarray): The input array.
        output (cupy.ndarray, dtype or None): The array in which to place the
            output. Default is is same dtype as the input.
        mode (str): The array borders are handled according to the given mode
            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,
            ``'wrap'``). Default is ``'reflect'``.
        cval (scalar): Value to fill past edges of input if mode is
            ``'constant'``. Default is ``0.0``.
    Returns:
        cupy.ndarray: The result of the filtering.
    .. seealso:: :func:`scipy.ndimage.laplace`
    """
    weights = cupy.array([1, -2, 1], dtype=cupy.float64)

    def derivative2(input, axis, output, mode, cval):
        return correlate1d(input, weights, axis, output, mode, cval)

    return generic_laplace(input, derivative2, output, mode, cval)


def gaussian_laplace(input, sigma, output=None, mode="reflect",
                     cval=0.0, **kwargs):
    """Multi-dimensional Laplace filter using Gaussian second derivatives.

    Args:
        input (cupy.ndarray): The input array.
        sigma (scalar or sequence of scalar): Standard deviations for each axis
            of Gaussian kernel. A single value applies to all axes.
        output (cupy.ndarray, dtype or None): The array in which to place the
            output. Default is is same dtype as the input.
        mode (str): The array borders are handled according to the given mode
            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,
            ``'wrap'``). Default is ``'reflect'``.
        cval (scalar): Value to fill past edges of input if mode is
            ``'constant'``. Default is ``0.0``.
        kwargs (dict, optional):
            dict of extra keyword arguments to pass ``gaussian_filter()``.
    Returns:
        cupy.ndarray: The result of the filtering.
    .. seealso:: :func:`scipy.ndimage.gaussian_laplace`
    """
    def derivative2(input, axis, output, mode, cval):
        order = [0] * input.ndim
        order[axis] = 2
        return gaussian_filter(input, sigma, order, output, mode, cval,
                               **kwargs)
    return generic_laplace(input, derivative2, output, mode, cval)


def generic_gradient_magnitude(input, derivative, output=None,
                               mode="reflect", cval=0.0,
                               extra_arguments=(), extra_keywords=None):
    """Multi-dimensional gradient magnitude filter using a provided derivative
    function.

    Args:
        input (cupy.ndarray): The input array.
        derivative (callable): Function or other callable with the following
            signature that is called once per axis::

                derivative(input, axis, output, mode, cval,
                           *extra_arguments, **extra_keywords)

            where ``input`` and ``output`` are ``cupy.ndarray``, ``axis`` is an
            ``int`` from ``0`` to the number of dimensions, and ``mode``,
            ``cval``, ``extra_arguments``, ``extra_keywords`` are the values
            given to this function.
        output (cupy.ndarray, dtype or None): The array in which to place the
            output. Default is is same dtype as the input.
        mode (str): The array borders are handled according to the given mode
            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,
            ``'wrap'``). Default is ``'reflect'``.
        cval (scalar): Value to fill past edges of input if mode is
            ``'constant'``. Default is ``0.0``.
        extra_arguments (sequence, optional):
            Sequence of extra positional arguments to pass to ``derivative2``.
        extra_keywords (dict, optional):
            dict of extra keyword arguments to pass ``derivative2``.
    Returns:
        cupy.ndarray: The result of the filtering.
    .. seealso:: :func:`scipy.ndimage.generic_gradient_magnitude`
    """
    if extra_keywords is None:
        extra_keywords = {}
    modes = _fix_sequence_arg(mode, input.ndim, 'mode', _check_mode)
    output = _get_output(output, input)
    if input.ndim == 0:
        output[...] = input[...]
        return output
    derivative(input, 0, output, modes[0], cval,
               *extra_arguments, **extra_keywords)
    output *= output
    if input.ndim > 1:
        tmp = _get_output(output.dtype, input)
        for i in range(1, input.ndim):
            derivative(input, i, tmp, modes[i], cval,
                       *extra_arguments, **extra_keywords)
            tmp *= tmp
            output += tmp
    return cupy.sqrt(output, output, casting='unsafe')


def gaussian_gradient_magnitude(input, sigma, output=None, mode="reflect",
                                cval=0.0, **kwargs):
    """Multi-dimensional gradient magnitude using Gaussian derivatives.

    Args:
        input (cupy.ndarray): The input array.
        sigma (scalar or sequence of scalar): Standard deviations for each axis
            of Gaussian kernel. A single value applies to all axes.
        output (cupy.ndarray, dtype or None): The array in which to place the
            output. Default is is same dtype as the input.
        mode (str): The array borders are handled according to the given mode
            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,
            ``'wrap'``). Default is ``'reflect'``.
        cval (scalar): Value to fill past edges of input if mode is
            ``'constant'``. Default is ``0.0``.
        kwargs (dict, optional):
            dict of extra keyword arguments to pass ``gaussian_filter()``.
    Returns:
        cupy.ndarray: The result of the filtering.
    .. seealso:: :func:`scipy.ndimage.gaussian_gradient_magnitude`
    """
    def derivative(input, axis, output, mode, cval):
        order = [0] * input.ndim
        order[axis] = 1
        return gaussian_filter(input, sigma, order, output, mode, cval,
                               **kwargs)
    return generic_gradient_magnitude(input, derivative, output, mode, cval)


# ######## Rank-Base Filters ##########

def minimum_filter(input, size=None, footprint=None, output=None,
                   mode="reflect", cval=0.0, origin=0):
    """Multi-dimensional minimum filter.

    Args:
        input (cupy.ndarray): The input array.
        size (int or sequence of int): One of ``size`` or ``footprint`` must be
            provided. If ``footprint`` is given, ``size`` is ignored. Otherwise
            ``footprint = cupy.ones(size)`` with ``size`` automatically made to
            match the number of dimensions in ``input``.
        footprint (cupy.ndarray): a boolean array which specifies which of the
            elements within this shape will get passed to the filter function.
        output (cupy.ndarray, dtype or None): The array in which to place the
            output. Default is is same dtype as the input.
        mode (str): The array borders are handled according to the given mode
            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,
            ``'wrap'``). Default is ``'reflect'``.
        cval (scalar): Value to fill past edges of input if mode is
            ``'constant'``. Default is ``0.0``.
        origin (int or sequence of int): The origin parameter controls the
            placement of the filter, relative to the center of the current
            element of the input. Default of 0 is equivalent to
            ``(0,)*input.ndim``.
    Returns:
        cupy.ndarray: The result of the filtering.
    .. seealso:: :func:`scipy.ndimage.minimum_filter`
    """
    return _min_or_max_filter(input, size, footprint, output, mode, cval,
                              origin, 'min')


def maximum_filter(input, size=None, footprint=None, output=None,
                   mode="reflect", cval=0.0, origin=0):
    """Multi-dimensional maximum filter.

    Args:
        input (cupy.ndarray): The input array.
        size (int or sequence of int): One of ``size`` or ``footprint`` must be
            provided. If ``footprint`` is given, ``size`` is ignored. Otherwise
            ``footprint = cupy.ones(size)`` with ``size`` automatically made to
            match the number of dimensions in ``input``.
        footprint (cupy.ndarray): a boolean array which specifies which of the
            elements within this shape will get passed to the filter function.
        output (cupy.ndarray, dtype or None): The array in which to place the
            output. Default is is same dtype as the input.
        mode (str): The array borders are handled according to the given mode
            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,
            ``'wrap'``). Default is ``'reflect'``.
        cval (scalar): Value to fill past edges of input if mode is
            ``'constant'``. Default is ``0.0``.
        origin (int or sequence of int): The origin parameter controls the
            placement of the filter, relative to the center of the current
            element of the input. Default of 0 is equivalent to
            ``(0,)*input.ndim``.
    Returns:
        cupy.ndarray: The result of the filtering.
    .. seealso:: :func:`scipy.ndimage.maximum_filter`
    """
    return _min_or_max_filter(input, size, footprint, output, mode, cval,
                              origin, 'max')


def _min_or_max_filter(input, size, ftprnt, output, mode, cval, origin, func):
    sizes, ftprnt, sep = \
        _check_size_or_ftprnt(input.ndim, size, ftprnt, 3, True)
    
    if sep:
        fltr = minimum_filter1d if func == 'min' else maximum_filter1d
        return _nd_filter([fltr if size > 1 else None for size in sizes],
                          input, sizes, output, mode, cval, origin)
    
    origins, int_type = _check_nd_args(input, ftprnt, mode, origin, 'footprint')
    if ftprnt.size == 0:
        return cupy.zeros_like(input)
    kernel = _get_min_or_max_kernel(mode, ftprnt.shape, func,
                                    origins, float(cval), int_type)
    return _call_kernel(kernel, input, ftprnt, output, bool)


def minimum_filter1d(input, size, axis=-1, output=None, mode="reflect",
                     cval=0.0, origin=0):
    """Compute the minimum filter along a single axis.

    Args:
        input (cupy.ndarray): The input array.
        size (int): Length of the minimum filter.
        axis (int): The axis of input along which to calculate. Default is -1.
        output (cupy.ndarray, dtype or None): The array in which to place the
            output. Default is is same dtype as the input.
        mode (str): The array borders are handled according to the given mode
            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,
            ``'wrap'``). Default is ``'reflect'``.
        cval (scalar): Value to fill past edges of input if mode is
            ``'constant'``. Default is ``0.0``.
        origin (int): The origin parameter controls the placement of the
            filter, relative to the center of the current element of the
            input. Default is ``0``.
    Returns:
        cupy.ndarray: The result of the filtering.
    .. seealso:: :func:`scipy.ndimage.minimum_filter1d`
    """
    return _max_or_min_1d(input, size, axis, output, mode, cval, origin, 'min')


def maximum_filter1d(input, size, axis=-1, output=None, mode="reflect",
                     cval=0.0, origin=0):
    """Compute the maximum filter along a single axis.

    Args:
        input (cupy.ndarray): The input array.
        size (int): Length of the maximum filter.
        axis (int): The axis of input along which to calculate. Default is -1.
        output (cupy.ndarray, dtype or None): The array in which to place the
            output. Default is is same dtype as the input.
        mode (str): The array borders are handled according to the given mode
            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,
            ``'wrap'``). Default is ``'reflect'``.
        cval (scalar): Value to fill past edges of input if mode is
            ``'constant'``. Default is ``0.0``.
        origin (int): The origin parameter controls the placement of the
            filter, relative to the center of the current element of the
            input. Default is ``0``.
    Returns:
        cupy.ndarray: The result of the filtering.
    .. seealso:: :func:`scipy.ndimage.maximum_filter1d`
    """
    return _max_or_min_1d(input, size, axis, output, mode, cval, origin, 'max')


def _max_or_min_1d(input, size, axis=-1, output=None, mode="reflect", cval=0.0,
                   origin=0, func='min'):
    ftprnt = cupy.ones(size, dtype=bool)
    ftprnt, origins = _convert_1d_args(input.ndim, ftprnt, origin, axis)
    origins, int_type = _check_nd_args(input, ftprnt, mode, origins, 'footprint')
    kernel = _get_min_or_max_kernel(mode, ftprnt.shape, func, origins,
                                    float(cval), int_type, False)
    return _call_kernel(kernel, input, None, output, bool)


@cupy.memoize()
def _get_min_or_max_kernel(mode, wshape, func, origins, cval, int_type, has_weights=True):
    return _get_nd_kernel(
        func, 'X value = x[i];',
        'value = {func}((X){{value}}, value);'.format(func=func),
        'y = (Y)value;', mode, wshape, int_type, origins, cval,
        has_weights=has_weights)


def rank_filter(input, rank, size=None, footprint=None, output=None,
                mode="reflect", cval=0.0, origin=0):
    """Multi-dimensional rank filter.

    Args:
        input (cupy.ndarray): The input array.
        rank (int): The rank of the element to get. Can be negative to count
            from the largest value, e.g. ``-1`` indicates the largest value.
        size (int or sequence of int): One of ``size`` or ``footprint`` must be
            provided. If ``footprint`` is given, ``size`` is ignored. Otherwise
            ``footprint = cupy.ones(size)`` with ``size`` automatically made to
            match the number of dimensions in ``input``.
        footprint (cupy.ndarray): a boolean array which specifies which of the
            elements within this shape will get passed to the filter function.
        output (cupy.ndarray, dtype or None): The array in which to place the
            output. Default is is same dtype as the input.
        mode (str): The array borders are handled according to the given mode
            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,
            ``'wrap'``). Default is ``'reflect'``.
        cval (scalar): Value to fill past edges of input if mode is
            ``'constant'``. Default is ``0.0``.
        origin (int or sequence of int): The origin parameter controls the
            placement of the filter, relative to the center of the current
            element of the input. Default of 0 is equivalent to
            ``(0,)*input.ndim``.
    Returns:
        cupy.ndarray: The result of the filtering.
    .. seealso:: :func:`scipy.ndimage.rank_filter`
    """
    rank = int(rank)
    return _rank_filter(input, lambda fs: rank+fs if rank < 0 else rank,
                        size, footprint, output, mode, cval, origin)


def median_filter(input, size=None, footprint=None, output=None,
                  mode="reflect", cval=0.0, origin=0):
    """Multi-dimensional median filter.

    Args:
        input (cupy.ndarray): The input array.
        size (int or sequence of int): One of ``size`` or ``footprint`` must be
            provided. If ``footprint`` is given, ``size`` is ignored. Otherwise
            ``footprint = cupy.ones(size)`` with ``size`` automatically made to
            match the number of dimensions in ``input``.
        footprint (cupy.ndarray): a boolean array which specifies which of the
            elements within this shape will get passed to the filter function.
        output (cupy.ndarray, dtype or None): The array in which to place the
            output. Default is is same dtype as the input.
        mode (str): The array borders are handled according to the given mode
            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,
            ``'wrap'``). Default is ``'reflect'``.
        cval (scalar): Value to fill past edges of input if mode is
            ``'constant'``. Default is ``0.0``.
        origin (int or sequence of int): The origin parameter controls the
            placement of the filter, relative to the center of the current
            element of the input. Default of 0 is equivalent to
            ``(0,)*input.ndim``.
    Returns:
        cupy.ndarray: The result of the filtering.
    .. seealso:: :func:`scipy.ndimage.median_filter`
    """
    return _rank_filter(input, lambda fs: fs//2,
                        size, footprint, output, mode, cval, origin)


def percentile_filter(input, percentile, size=None, footprint=None,
                      output=None, mode="reflect", cval=0.0, origin=0):
    """Multi-dimensional percentile filter.

    Args:
        input (cupy.ndarray): The input array.
        percentile (scalar): The percentile of the element to get (from ``0``
            to ``100``). Can be negative, thus ``-20`` equals ``80``.
        size (int or sequence of int): One of ``size`` or ``footprint`` must be
            provided. If ``footprint`` is given, ``size`` is ignored. Otherwise
            ``footprint = cupy.ones(size)`` with ``size`` automatically made to
            match the number of dimensions in ``input``.
        footprint (cupy.ndarray): a boolean array which specifies which of the
            elements within this shape will get passed to the filter function.
        output (cupy.ndarray, dtype or None): The array in which to place the
            output. Default is is same dtype as the input.
        mode (str): The array borders are handled according to the given mode
            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,
            ``'wrap'``). Default is ``'reflect'``.
        cval (scalar): Value to fill past edges of input if mode is
            ``'constant'``. Default is ``0.0``.
        origin (int or sequence of int): The origin parameter controls the
            placement of the filter, relative to the center of the current
            element of the input. Default of 0 is equivalent to
            ``(0,)*input.ndim``.
    Returns:
        cupy.ndarray: The result of the filtering.
    .. seealso:: :func:`scipy.ndimage.percentile_filter`
    """
    percentile = float(percentile)
    if percentile < 0.0:
        percentile += 100.0
    if percentile < 0.0 or percentile > 100.0:
        raise RuntimeError('invalid percentile')
    get_rank = lambda fs: int(float(fs) * percentile / 100.0)
    if percentile == 100.0:
        get_rank = lambda fs: fs - 1
    return _rank_filter(input, get_rank,
                        size, footprint, output, mode, cval, origin)


def _rank_filter(input, get_rank, size=None, ftprnt=None, output=None,
                 mode="reflect", cval=0.0, origin=0):
    ftprnt = _check_size_or_ftprnt(input.ndim, size, ftprnt, 3)
    origins, int_type = _check_nd_args(input, ftprnt, mode, origin,
                                       'footprint')
    if ftprnt.size == 0:
        return cupy.zeros_like(input)
    filter_size = int(ftprnt.sum())
    rank = get_rank(filter_size)
    if rank < 0 or rank >= filter_size:
        raise RuntimeError('rank not within filter footprint size')
    if rank == 0:
        return _min_or_max_filter(input, None, ftprnt, output, mode, cval,
                                  origins, True)
    if rank == filter_size - 1:
        return _min_or_max_filter(input, None, ftprnt, output, mode, cval,
                                  origins, False)
    kernel = _get_rank_kernel(filter_size, rank, mode, ftprnt.shape,
                              origins, float(cval), int_type)
    return _call_kernel(kernel, input, ftprnt, output, bool)


__SHELL_SORT = '''
__device__ void sort(X *array, int size) {{
    int gap = {gap};
    while (gap > 1) {{
        gap /= 3;
        for (int i = gap; i < size; ++i) {{
            X value = array[i];
            int j = i - gap;
            while (j >= 0 && value < array[j]) {{
                array[j + gap] = array[j];
                j -= gap;
            }}
            array[j + gap] = value;
        }}
    }}
}}'''


__SELECTION_SORT = '''
__device__ void sort(X *array, int size) {
    for (int i = 0; i < size; ++i) {
        int min_val = array[i];
        int min_idx = i;
        for (int j = i+1; j < size; ++j) {
            int val_j = array[j];
            if (val_j < min_val) {
                min_idx = j;
                min_val = val_j;
            }
        }
        if (i != min_idx) {
            array[min_idx] = array[i];
            array[i] = min_val;
        }
    }
}'''


@cupy.memoize()
def _get_shell_gap(filter_size):
    gap = 1
    while gap < filter_size:
        gap = 3*gap+1
    return gap


@cupy.memoize()
def _get_rank_kernel(filter_size, rank, mode, wshape, origins, cval, int_type):
    # Below 225 (15x15 median filter) selection sort is 1.5-2.5x faster
    # Above, shell sort does progressively better (by 3025 (55x55) it is 9x)
    # Also tried insertion sort, which is always slower than either one
    sorter = __SELECTION_SORT if filter_size <= 225 else \
        __SHELL_SORT.format(gap=_get_shell_gap(filter_size))
    return _get_nd_kernel(
        'rank_{}_{}'.format(filter_size, rank),
        'int iv = 0;\nX values[{}];'.format(filter_size),
        'values[iv++] = {value};',
        'sort(values, {});\ny = (Y)values[{}];'.format(filter_size, rank),
        mode, wshape, int_type, origins, cval, preamble=sorter)


# ######## Derivatives of Convolution Filters ##########

def generic_filter(input, function, size=None, footprint=None,
                   output=None, mode="reflect", cval=0.0, origin=0):
    """Compute a multi-dimensional filter using the provided reduction kernel,
    fused function that performs a reduction (or a function that can be fused),
    or one of the built-in cupy reduction functions.

    Unlike the scipy.ndimage function, this does not support the
    ``extra_arguments`` or ``extra_keywordsdict`` arguments and has significant
    restrictions on the ``function`` provided.

    Args:
        input (cupy.ndarray): The input array.
        function (cupy.ReductionKernel, function, or cupy.core.fusion.Fusion):
            The reduction kernel or function to apply to each region.
        size (int or sequence of int): One of ``size`` or ``footprint`` must be
            provided. If ``footprint`` is given, ``size`` is ignored. Otherwise
            ``footprint = cupy.ones(size)`` with ``size`` automatically made to
            match the number of dimensions in ``input``.
        footprint (cupy.ndarray): a boolean array which specifies which of the
            elements within this shape will get passed to the filter function.
        output (cupy.ndarray, dtype or None): The array in which to place the
            output. Default is is same dtype as the input.
        mode (str): The array borders are handled according to the given mode
            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,
            ``'wrap'``). Default is ``'reflect'``.
        cval (scalar): Value to fill past edges of input if mode is
            ``'constant'``. Default is ``0.0``.
        origin (int): The origin parameter controls the placement of the
            filter, relative to the center of the current element of the
            input. Default is ``0``.
    Returns:
        cupy.ndarray: The result of the filtering.
    .. seealso:: :func:`scipy.ndimage.generic_filter`
    """
    function = _get_reduction_kernel(function, input.dtype)
    footprint = _check_size_or_ftprnt(input.ndim, size, footprint, 2)
    origins, int_type = \
        _check_nd_args(input, footprint, mode, origin, 'footprint')
    if footprint.size == 0:
        return cupy.zeros_like(input)
    output = _get_output(output, input)
    kernel = _get_generic_filter(function, int(footprint.sum()), mode,
                                 footprint.shape, origins, float(cval), int_type)
    return _call_kernel(kernel, input, footprint, output)


@cupy.memoize()
def _get_generic_filter(kernel, filter_size, mode, wshape, origins, cval,
                        int_type):
    return _get_nd_kernel(
        'generic_{}_{}'.format(filter_size, kernel.name),
        'int iv = 0;\nX values[{}];'.format(filter_size),
        'values[iv++] = {value};',
        '{}(values,{},y);'.format(kernel.name, filter_size),
        mode, wshape, int_type, origins, cval,
        preamble=_reduction_kernel_code(kernel),
        options=getattr(kernel, 'options', ()))


def _get_reduction_kernel(func, dtype):
    """
    Takes the "function" given given to generic_filter or generic_filter1d and
    returns a ReductionKernel. This supports:
     * cupy.ReductionKernel or cupy.core._kernel.simple_reduction_function
       checks that there is a single input and output
     * cupy.core.fusion.Fusion
       computes the underlying kernel and sends it back through this function
     * any callable
       attempts to fuse it then and sends it back through this function
    The dtype argument is the data type used for function fusions.
    """
    # pylint: disable=protected-access
    if isinstance(func, (cupy.ReductionKernel,
                         cupy.core._kernel.simple_reduction_function)):  # TODO: update this for v8.0 compatibility: cupy.core._kernel._SimpleReductionKernel or maybe _AbstractReductionKernel
        if func.nin != 1 or func.nout != 1:
            raise TypeError('cupyx.scipy.ndimage.generic_filter only accepts '
                            'ReductionKernels with a single input and output')
        return func
    elif isinstance(func, cupy.ElementwiseKernel):
        # special error message for ElementwiseKernels
        raise TypeError('cupyx.scipy.ndimage.generic_filter only accepts '
                        'ReductionKernels and not ElementwiseKernels')
    elif isinstance(func, cupy.core.fusion.Fusion):
        key = (dtype.char, 1)
        if key not in func._memo:
            arg = [cupy.zeros(1, dtype=dtype)]
            try:
                history = cupy.core.fusion._FusionHistory()
                cupy.core._kernel._thread_local.history = history
                func._memo[key] = history.get_fusion(func.func, arg, func.name)
            finally:
                cupy.core._kernel._thread_local.history = None
        kernel, kwargs = func._memo[key]
        if kwargs:
            # The only two kwargs don't make sense: axis and out
            raise TypeError('cupyx.scipy.ndimage.generic_filter only accepts '
                            'fused reduction functions without axis and out')
        return _get_reduction_kernel(kernel, dtype)
    elif callable(func):
        return _get_reduction_kernel(cupy.fuse(func), dtype)
    raise TypeError('function')


def _reduction_kernel_code(kernel):
    # pylint: disable=protected-access
    srk = isinstance(kernel, cupy.core._kernel._SimpleReductionKernel)
    in_param = kernel._in_params[0] if srk else kernel.in_params[0]
    out_param = kernel._out_params[0] if srk else kernel.out_params[0]
    if srk:
        # TODO: intelligently look up? at this point we don't know the dtype...
        pre_map_expr, reduce_expr, post_map_expr, reduce_type = kernel._ops.ops[0]
    else:
        pre_map_expr, reduce_expr, post_map_expr, reduce_type = (
            kernel.map_expr, kernel.reduce_expr, kernel.post_map_expr,
            kernel.reduce_type)
    if reduce_type is None or reduce_type == out_param.ctype:
        reduce_type = 'double'  # for scipy compatibility, otherwise 'Y'
    type_preamble = 'typedef X {};'.format(in_param.ctype)
    if in_param.ctype != out_param.ctype:
        type_preamble += '\ntypedef Y {};'.format(out_param.ctype)
    return '''{type_preamble}
{preamble}
__device__ void {name}({const} X *_array, const int _size, Y &{out_name}) {{
  #define PRE_MAP({in_name}) ({pre_map_expr})
  #define REDUCE(a, b) ({reduce_expr})
  #define POST_MAP(a) ({post_map_expr})
  typedef {reduce_type} _type_reduce;
  _type_reduce _s = _type_reduce({identity});
  for (int _j = 0; _j < _size; _j += 1) {{
    _type_reduce _a = static_cast<_type_reduce>PRE_MAP(_array[_j]);
    _s = REDUCE(_s, _a);
  }}
  POST_MAP(_s);
}}'''.format(
        name=kernel.name,
        type_preamble=type_preamble,
        preamble=kernel._preamble if srk else kernel.preamble,
        identity='' if kernel.identity is None else kernel.identity,
        const='const' if in_param.is_const else '',
        in_name=in_param.name,
        out_name=out_param.name,
        pre_map_expr=pre_map_expr,
        reduce_expr=reduce_expr,
        post_map_expr=post_map_expr,
        reduce_type=reduce_type,
    )


LT_RK = cupy.ReductionKernel('T x', 'int32 y',
                             '127 < x', 'a + b', 'y = a', '0',
                             'lt_rk', reduce_type='int')
LT_RK = cupy.ReductionKernel('raw T x', 'int32 y',
                             'x[x.size()/2] < x[i]', 'a + b', 'y = a', '0',
                             'lt_rk', reduce_type='int')


# ######## Utility Functions ##########

def _get_output(output, input, shape=None):
    if shape is None:
        shape = input.shape
    if isinstance(output, cupy.ndarray):
        if output.shape != tuple(shape):
            raise ValueError('output shape is not correct')
    else:
        dtype = input.dtype if output is None else output
        output = cupy.zeros(shape, dtype)
    return output


def _fix_sequence_arg(arg, ndim, name, conv=lambda x: x):
    if hasattr(arg, '__iter__') and not isinstance(arg, str):
        lst = [conv(x) for x in arg]
        if len(lst) != ndim:
            msg = "{} must have length equal to input rank".format(name)
            raise RuntimeError(msg)
    else:
        lst = [conv(arg)] * ndim
    return lst


def _check_origin(origin, width):
    origin = int(origin)
    if (width // 2 + origin < 0) or (width // 2 + origin >= width):
        raise ValueError('invalid origin')
    return origin


def _check_mode(mode):
    if mode not in ('reflect', 'constant', 'nearest', 'mirror', 'wrap'):
        msg = 'boundary mode not supported (actual: {}).'.format(mode)
        raise RuntimeError(msg)
    return mode


def _check_axis(axis, ndim):
    axis = int(axis)
    if axis < 0:
        axis += ndim
    if axis < 0 or axis >= ndim:
        raise ValueError('invalid axis')
    return axis


def _check_size_or_ftprnt(ndim, size, ftprnt, stacklevel, check_sep=False):
    import warnings
    if (size is not None) and (ftprnt is not None):
        warnings.warn("ignoring size because footprint is set",
                      UserWarning, stacklevel=stacklevel+1)
    if ftprnt is None:
        if size is None:
            raise RuntimeError("no footprint or filter size provided")
        sizes = _fix_sequence_arg(size, ndim, 'size', int)
        if check_sep:
            return sizes, None, True
        ftprnt = cupy.ones(sizes, dtype=bool)
    else:
        ftprnt = cupy.ascontiguousarray(ftprnt, dtype=bool)
        if not ftprnt.any():
            raise ValueError("All-zero footprint is not supported.")
        if check_sep:
            if ftprnt.all():
                return ftprnt.shape, None, True
            return None, ftprnt, False
    return ftprnt


def _convert_1d_args(ndim, weights, origin, axis):
    if weights.ndim != 1 or weights.size < 1:
        raise RuntimeError('incorrect filter size')
    axis = _check_axis(axis, ndim)
    wshape = [1]*ndim
    wshape[axis] = weights.size
    weights = weights.reshape(wshape)
    origins = [0]*ndim
    origins[axis] = _check_origin(origin, weights.size)
    return weights, tuple(origins)


def _check_nd_args(input, weights, mode, origins, wghts_name='filter weights'):
    if input.dtype.kind == 'c':
        raise TypeError('Complex type not supported.')
    _check_mode(mode)
    # The integer type to use for positions in input
    # We will always assume that wsize is int32 however
    int_type = 'size_t' if input.size > 1 << 31 else 'int'
    weight_dims = [x for x in weights.shape if x != 0]
    if len(weight_dims) != input.ndim:
        raise RuntimeError('{} array has incorrect shape'.format(wghts_name))
    origins = _fix_sequence_arg(origins, len(weight_dims), 'origin', int)
    for origin, width in zip(origins, weight_dims):
        _check_origin(origin, width)
    return tuple(origins), int_type


def _call_kernel(kernel, input, weights, output,
                 weight_dtype=cupy.float64):
    """
    Calls a constructed ElementwiseKernel. The kernel must take an input image,
    an array of weights, and an output array.

    The weights are the only optional part and can be passed as None and then
    one less argument is passed to the kernel. If the output is given as None
    then it will be allocated in this function.

    This function deals with making sure that the weights are contiguous and
    float64 or bool*, that the output is allocated and appriopate shaped. This
    also deals with the situation that the input and output arrays overlap in
    memory.

    * weights is always casted to float64 or bool in order to get an output
    compatible with SciPy, though float32 might be sufficient when input dtype
    is low precision.
    """
    if weights is not None:
        weights = cupy.ascontiguousarray(weights, weight_dtype)
    output = _get_output(output, input)
    needs_temp = cupy.shares_memory(output, input, 'MAY_SHARE_BOUNDS')
    if needs_temp:
        output, temp = _get_output(output.dtype, input), output
    if weights is None:
        kernel(input, output)
    else:
        kernel(input, weights, output)
    if needs_temp:
        temp[...] = output[...]
        output = temp
    return output


def _nd_filter(filters, input, args, output, mode, cval, origin=0):
    """
    Runs a series of 1D filters forming an nd filter. The filters must be a
    list of callables that take input, arg, axis, output, mode, cval, origin.
    The args is a list of values that get past for the arg value to the filter.
    Individual filters can be None causing that axis to be skipped.
    """
    output_orig = output
    output = _get_output(output, input)
    modes = _fix_sequence_arg(mode, input.ndim, 'mode', _check_mode)
    origins = _fix_sequence_arg(origin, input.ndim, 'origin', int)
    n_filters = sum(filter is not None for filter in filters)
    if n_filters == 0:
        output[...] = input[...]
        return output
    # We can't operate in-place efficiently, so use a 2-buffer system
    temp = _get_output(output.dtype, input) if n_filters > 1 else None
    first = True
    iterator = zip(filters, args, modes, origins)
    for axis, (fltr, arg, mode, origin) in enumerate(iterator):
        if fltr is None:
            continue
        fltr(input, arg, axis, output, mode, cval, origin)
        input, output = output, temp if first else input
    if output_orig is not None and input is not output_orig:
        output_orig[...] = input
        input = output_orig
    return input


def _nd_correlate(input, params, get_weights, output, mode, cval, origin=0):
    """
    Enhanced version of _nd_filter that uses correlate1d as the filter
    function. The params are a list of values to pass to the get_weights
    callable given. If duplicate param values are found, the weights are
    reused from the first invocation of get_weights. The get_weights callable
    must return a 1D array of weights to give to correlate1d.
    """
    weights = {}
    for param in params:
        if param not in weights:
            weights[param] = get_weights(param)
    weights = [weights[param] for param in params]
    return _nd_filter([None if w is None else correlate1d for w in weights],
                      input, weights, output, mode, cval, origin)



# ######## Generating Elementwise Kernels ##########

def _generate_boundary_condition_ops(mode, ix, xsize):
    if mode == 'reflect':
        ops = '''
        if ({ix} < 0) {{
            {ix} = -1 - {ix};
        }}
        {ix} %= {xsize} * 2;
        {ix} = min({ix}, 2 * {xsize} - 1 - {ix});'''.format(ix=ix, xsize=xsize)
    elif mode == 'mirror':
        ops = '''
        if ({ix} < 0) {{
            {ix} = -{ix};
        }}
        if ({xsize} == 1) {{
            {ix} = 0;
        }} else {{
            {ix} = 1 + ({ix} - 1) % (({xsize} - 1) * 2);
            {ix} = min({ix}, 2 * {xsize} - 2 - {ix});
        }}'''.format(ix=ix, xsize=xsize)
    elif mode == 'nearest':
        ops = '''
        {ix} = min(max({ix}, 0), {xsize} - 1);'''.format(ix=ix, xsize=xsize)
    elif mode == 'wrap':
        ops = '''
        if ({ix} < 0) {{
            {ix} += (1 - ({ix} / {xsize})) * {xsize};
        }}
        {ix} %= {xsize};'''.format(ix=ix, xsize=xsize)
    elif mode == 'constant':
        ops = '''
        if ({ix} >= {xsize}) {{
            {ix} = -1;
        }}'''.format(ix=ix, xsize=xsize)
    return ops


def _get_nd_kernel(name, pre, found, post, mode, wshape, int_type,
                   origins, cval, preamble='', options=(), has_weights=True):
    ndim = len(wshape)
    in_params = 'raw X x, raw W w'
    out_params = 'Y y'

    inds = _generate_indices_ops(
        ndim, int_type, 'xsize_{j}',
        [' - {}'.format(wshape[j]//2 + origins[j]) for j in range(ndim)])
    sizes = ['{type} xsize_{j}=x.shape()[{j}], xstride_{j}=x.strides()[{j}];'.
             format(j=j, type=int_type) for j in range(ndim)]
    cond = ' || '.join(['(ix_{0} < 0)'.format(j) for j in range(ndim)])
    expr = ' + '.join(['ix_{0}'.format(j) for j in range(ndim)])

    if has_weights:
        weights_init = 'const W* weights = (const W*)&w[0];\nint iw = 0;'
        weights_check = 'W wval = weights[iw++];\nif (wval)'
    else:
        in_params = 'raw X x'
        weights_init = weights_check = ''

    loops = []
    for j in range(ndim):
        if wshape[j] == 1:
            loops.append('{{ {type} ix_{j} = ind_{j} * xstride_{j};'.format(j=j, type=int_type))
        else:
            boundary = _generate_boundary_condition_ops(mode, 'ix_{}'.format(j),
                                                        'xsize_{}'.format(j))
            loops.append('''
        for (int iw_{j} = 0; iw_{j} < {wsize}; iw_{j}++)
        {{
            {type} ix_{j} = ind_{j} + iw_{j};
            {boundary}
            ix_{j} *= xstride_{j};
            '''.format(j=j, wsize=wshape[j], boundary=boundary, type=int_type))

    value = '(*(X*)&data[{expr}])'.format(expr=expr)
    if mode == 'constant':
        value = '(({cond}) ? (X){cval} : {value})'.format(
            cond=cond, cval=cval, value=value)
    found = found.format(value=value)

    operation = '''
    {sizes}
    {inds}
    // don't use a CArray for indexing (faster to deal with indexing ourselves)
    const unsigned char* data = (const unsigned char*)&x[0];
    {weights_init}
    {pre}
    {loops}
        // inner-most loop
        {weights_check} {{
            {found}
        }}
    {end_loops}
    {post}
    '''.format(sizes='\n'.join(sizes), inds=inds, pre=pre, post=post,
               weights_init=weights_init, weights_check=weights_check,
               loops='\n'.join(loops), found=found, end_loops='}'*ndim)

    name = 'cupy_ndimage_{}_{}d_{}_w{}'.format(
        name, ndim, mode, '_'.join(['{}'.format(j) for j in wshape]))
    if int_type == 'size_t':
        name += '_i64'
    return cupy.ElementwiseKernel(in_params, out_params, operation, name,
                                  reduce_dims=False, preamble=preamble,
                                  options=options)


def _generate_indices_ops(ndim, int_type, xsize='x.shape()[{j}]', extras=None):
    if extras is None:
        extras = ('',)*ndim
    code = '{type} ind_{j} = _i % ' + xsize + '{extra}; _i /= ' + xsize + ';'
    body = [code.format(type=int_type, j=j, extra=extras[j])
            for j in range(ndim-1, 0, -1)]
    return '{type} _i = i;\n{body}\n{type} ind_0 = _i{extra};'.format(
        type=int_type, body='\n'.join(body), extra=extras[0])