# cython: boundscheck=False, wraparound=False, cdivision=True

"""Interpolation methods."""

from cython.view cimport contiguous
from libc.math cimport M_PI, sin, floor, ceil

import numpy as np


cdef inline double _sinc(double x) noexcept nogil:
    """Normalized sinc function."""

    if x != 0.0:
        return sin(x*M_PI) / (x*M_PI)

    return 1.0


def sinc_interpolate(
    double[::contiguous] y_sampled, double[::contiguous] x_interp,
    double[::contiguous] y_interp = None, int window=100
):
    """Perform sinc interpolation.

    As a consequence of the Nyquist theorem, sinc interpolation allows
    to reconstruct a continuous-time function $x(t)$ up to a certain
    bandwidth $1/T$ from a sequence of real numbers $x[n]$:

    $$
    x(t) = \sum \limits_n x[n] ~ {\\rm sinc} \\frac {t - n T}{T}
    $$

    This can be used to find the optimal edge position between samples
    with the fast timing discriminators in this module. In particular
    for fast slopes consisting of only a few sample points, this will
    generally yield significantly better results than linear
    interpolation, which tends to shift points towards the middle. For
    the constant fraction discriminator specifically, this can also
    be used to enable the use of real delay values.

    Args:
        y_sampled (ArrayLike): Sampled input data.
        x_interp (ArrayLike): Real index positions to interpolate at.
        y_interp (ArrayLike, optional): Output array for interpolated
            result, a new one is allocated.
        window (int, optional): Number of samples before and after the
            interpolation points actually used to evaluate $x(t)$, i.e.
            the finite boundaries to approximate the infinite sum.
            By default, up to 100 samples in each direction are used.

    Returns:
        y_interp (np.ndarray): Interpolated output data.
    """

    if y_interp is None:
        y_interp = np.zeros(len(x_interp), dtype=np.float64)

    cdef int i, k, \
        interp_len = min(x_interp.shape[0], y_interp.shape[0]), \
        sampling_start = max(<int>floor(x_interp[0]) - window, 0), \
        sampling_end = min(<int>ceil(x_interp[0]) + window, y_sampled.shape[0])

    for i in range(interp_len):
        for k in range(sampling_start, sampling_end):
            y_interp[i] += y_sampled[k] * _sinc(k - x_interp[i])

    return np.asarray(y_interp)
