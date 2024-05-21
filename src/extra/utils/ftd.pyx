# cython: boundscheck=False, wraparound=False, cdivision=True

"""Fast timing disciminators.

This module contains native software implementations for fast timing
discrimination that mark the arrival time of pulses on the digitized
recording of a continguous analog signal.
"""


from cython cimport floating
from cython.view cimport contiguous
from libc.math cimport M_PI, sin, fabs, floor, ceil, round
from libc.stdlib cimport malloc, free

import numpy as np


ctypedef floating data_t
ctypedef floating[::contiguous] data_array_t

cpdef enum EdgeInterpolation:
    # Interpolation method to find edge position.

    NEAREST = 0
    LINEAR = 1
    SPLINE = 2
    SINC = 3

"""Sinc interpolation parameters."""
cdef int sinc_window = 200
cdef int sinc_search_iterations = 10


def config_sinc_interpolation(window=None, search_iterations=None):
    """Configure sinc interpolation.

    As a consequence of the Nyquist theorem, sinc interpolation allows
    to reconstruct a continuous-time function $x(t)$ up to a certain
    bandwidth $1/T$ from a sequence of real numbers $x[n]$:

    $$
    x(t) = \sum_{n=-\infty}^{\infty} x[n] \, {\rm sinc}\left(\frac{t - nT}{T}\right)
    $$

    This can be used to find the optimal edge position between samples
    with the fast timing discriminators in this module. In particular
    for fast slopes consisting of only a few sample points, this will
    generally yield significantly better results than linear
    interpolation, which tends to shift points towards the middle. For
    the constant fraction discriminator specifically, this can also
    be used to enable the use of real delay values.

    This method has generally a large performance impact, but can be
    somewhat tuned by parameters:

    * `window` specifies the number of samples before and after the
        interpolation points actually used to evaluate $x(t), i.e. the
        finite boundaries to approximate the infinite sum above. By
        default, up to 200 samples in each direction are used.

    * `search_iterations` specifies the number of binary search steps
        taken to find the optimal edge position from interpolated values
        in between two samples. The maximal resolution in samples the
        interpolation can therefore achieve is $2^-{\rm search_iterations}$.

    When set, these parameters apply to all discriminator
    implementations and all their use of sinc interpolation.

    Args:
        window (int, optional): Sample window used around the
            interpolated point, unchanged if omitted.
        search_iterations (int, optional): Number of iterations used in
            binary search for closest function argument, unchanged if
            omitted.

    Returns
        (window, search_iterations) Tuple of current values.
    """

    global sinc_window, sinc_search_iterations

    if window is not None:
        sinc_window = <int>window

    if search_iterations is not None:
        sinc_search_iterations = <int>search_iterations

    return dict(window=sinc_window, search_iterations=sinc_search_iterations)


cdef inline double _sinc(double x) nogil:
    """Normalized sinc function."""

    if x != 0.0:
        return sin(x*M_PI) / (x*M_PI)
    else:
        return 1.0


cdef double _sinc_interp(data_t[::contiguous] y_sampled, double x_interp) nogil:
    cdef int k, \
        sampling_start = max(<int>floor(x_interp) - sinc_window, 0), \
        sampling_end = min(<int>ceil(x_interp) + sinc_window, y_sampled.shape[0])

    cdef double y_interp = 0.0

    for k in range(sampling_start, sampling_end):
        y_interp += <double>y_sampled[k] * _sinc(k - x_interp)

    return y_interp


cdef data_t _cfd_sinc_interp(
    data_array_t signal, int i, int j,
    data_t delay, data_t fraction, data_t walk
) nogil:
    cdef int int_delay = <int>ceil(delay)
    cdef bint is_integer_delay = int_delay == delay

    cdef int k, \
        sampling_start = max(i - sinc_window, 0), \
        sampling_end = min(j + sinc_window, signal.shape[0])

    # Interpolation is always done on double precision, as single
    # precision is almost certain to cause rounding errors in the sum.
    cdef double left_pos = <double>i, right_pos = <double>j, \
        middle_pos = 0.0, middle_value

    cdef data_t *interp_buf = NULL

    if not is_integer_delay:
        # For non-integer delays, the sinc interpolation includes
        # the same interpolated values in its own summation for every
        # iteration. Computing these values only once and keep them in
        # a small buffer significantly increases performance.
        interp_buf = <data_t*>malloc(sizeof(delay) * (2 * sinc_window + 1))

        for k in range(sampling_start, sampling_end):
            interp_buf[k - sampling_start] = _sinc_interp(signal, k - delay)

    for _ in range(1, sinc_search_iterations+1):
        middle_pos = (left_pos + right_pos) / 2

        middle_value = 0.0

        if is_integer_delay:
            for k in range(sampling_start, sampling_end):
                middle_value += (
                    (signal[k - int_delay] - fraction * signal[k])
                    * _sinc(k - middle_pos)
                )
        else:
            for k in range(sampling_start, sampling_end):
                middle_value += (
                    (interp_buf[k - sampling_start] - fraction * signal[k])
                    * _sinc(k - middle_pos)
                )

        if middle_value > walk:
            left_pos = middle_pos
        else:
            right_pos = middle_pos

    if not is_integer_delay:
        free(interp_buf)

    return <data_t>middle_pos


cdef data_t _dled_sinc_interp(
    data_array_t signal, bint negative, int ratio_idx, data_t ratio_value,
) nogil:
    cdef int k, \
        sampling_start = max(ratio_idx - sinc_window, 0), \
        sampling_end = min(ratio_idx + 1 + sinc_window, signal.shape[0])

    if negative:
        ratio_value = -ratio_value  # Invert for negative traces.

    # Interpolation is always done on double precision, see above in the
    # implementation for CFD.
    cdef double cmp_value = <double>ratio_value, \
        middle_value, middle_pos = 0.0, \
        left_pos = <double>ratio_idx, right_pos = <double>(ratio_idx + 1)

    for _ in range(1, sinc_search_iterations+1):
        middle_pos = (left_pos + right_pos) / 2

        middle_value = 0.0
        for k in range(sampling_start, sampling_end):
            middle_value += <double>signal[k] * _sinc(k - middle_pos)

        if (
            (not negative and middle_value < cmp_value) or
            (negative and middle_value > cmp_value)
        ):
            left_pos = middle_pos
        else:
            right_pos = middle_pos

    return <data_t>middle_pos


def cfd(
    data_array_t signal, data_t threshold, data_t delay,
    int width=0, data_t fraction=1.0, data_t walk=0.0,
    int interp=EdgeInterpolation.LINEAR,
    data_array_t edges = None,
    data_array_t amplitudes = None
):
    """Constant fraction discriminator.

    The pulse shape is assumed to always grow away from zero, i.e. the
    rising slope of positive pulses is positive and that of negative
    pulses is negative. Correspondigly, a positive threshold implies the
    pulse to peak at its largest value while a negative threshold
    implies the pulse to peak at its smallest value.

    Args:
        signal (array_like): 1D input array with analog signal.
        threshold (data-type): Trigger threshold, positive values imply
            a positive pulse slope while negative values correspondingly
            imply a negative pulse slope.
        delay (int): Delay between the raw and inverted signal.
        width (int, optional): Minimal distance between found edges,
            none by default.
        fraction (data-type, optional): Fraction of the inverted signal,
            1.0 by default.
        walk (data-type, optional): Point of intersection in the
            inverted signal, 0.0 by default.
        interp (EdgeInterpolation, optional): Interpolation mode to
            locate the edge position, linear by default.
        edges (ArrayLike, optional): 1D output array to hold the
            positions of found edges, a new one is allocated if None is
            passed.
        amplitudes (ArrayLike, optional): 1D output array to hold the
            pulse amplitudes corresponding to found edges, a new one is
            allocated if None is passed.

    Returns:
        (ArrayLike, ArrayLike, int) 1D arrays containing the edge
            positions and amplitudes, number of found edges.
    """

    if edges is None:
        edges = np.zeros(
            len(amplitudes) if amplitudes is not None else len(signal) // 100,
            dtype=np.asarray(signal).dtype)

    if amplitudes is None:
        amplitudes = np.zeros_like(edges, dtype=np.asarray(signal).dtype)

    cdef int i, j, edge_idx = 0, next_edge = -1, \
        max_edge = min(edges.shape[0], amplitudes.shape[0])
    cdef data_t cfd_i, cfd_j, edge_pos

    # Negative thresholds require inverted inequality relations when
    # comparing those to signals. For optimal performance without
    # duplicating most of the function body, only the first operations
    # per iteration are handled separately, while they shape their
    # output in such a way to assume the signal is always positive.
    cdef bint negative = threshold < 0

    # For "positive" comparisons to work with a negative-turned-positive
    # signal, the walk must be inverted.
    cdef data_t orig_walk = walk
    if negative:
        walk = -walk

    # A rounded-up delay is required to control various loops, and also
    # used instead of the floating delay if its value is actually an
    # integer. Omitting sinc interpolation in these cases is a huge
    # performance increase.
    cdef int int_delay = <int>ceil(delay)
    cdef bint is_integer_delay = int_delay == delay

    with nogil:
        for i in range(int_delay, signal.shape[0] - 1):
            if negative:
                if signal[i] >= threshold:
                    continue
            elif signal[i] <= threshold:
                continue

            j = i + 1

            if not is_integer_delay:
                if negative:
                    cfd_i = fraction * signal[i] - _sinc_interp(
                        signal, <data_t>i - delay)
                    cfd_j = fraction * signal[j] - _sinc_interp(
                        signal, <data_t>j - delay)
                else:
                    cfd_i = _sinc_interp(
                        signal, <data_t>i - delay) - fraction * signal[i]
                    cfd_j = _sinc_interp(
                        signal, <data_t>j - delay) - fraction * signal[j]
            elif negative:
                cfd_i = fraction * signal[i] - signal[i - int_delay]
                cfd_j = fraction * signal[j] - signal[j - int_delay]
            else:
                cfd_i = signal[i - int_delay] - fraction * signal[i]
                cfd_j = signal[j - int_delay] - fraction * signal[j]

            # From this point on, the CFD values are computed in such a
            # way as if the signal and threshold would be positive.

            if cfd_i < walk and cfd_j > walk and i > next_edge:
                if interp == EdgeInterpolation.NEAREST:
                    edge_pos = i if fabs(cfd_j - walk) > fabs(cfd_i - walk) \
                        else j
                elif interp == EdgeInterpolation.LINEAR:
                    edge_pos = i + (cfd_i - walk) / (cfd_i - cfd_j)
                elif interp == EdgeInterpolation.SPLINE:
                    raise NotImplementedError('spline interpolation')
                elif interp == EdgeInterpolation.SINC:
                    # As sinc interpolation still makes use of the
                    # original (possibly negative!) signal, it also
                    # requires the original walk value for comparions.
                    edge_pos = _cfd_sinc_interp(signal, i, j, delay, fraction,
                                                orig_walk)
                else:
                    raise ValueError('invalid interpolation mode')

                edges[edge_idx] = edge_pos
                next_edge = i + width
                edge_idx += 1

                if edge_idx == max_edge:
                    break

    return np.asarray(edges)[:edge_idx], np.asarray(amplitudes)[:edge_idx], \
        edge_idx


def dled(
    data_array_t signal, data_t threshold, data_t ratio_max=0.6, data_t width=0,
    int interp = EdgeInterpolation.LINEAR,
    data_array_t edges = None, data_array_t amplitudes = None
):
    """Dynamic leading edge discriminator.

    This algorithm finds the leading edge of pulses in a 1D signal
    located at a certain ratio to the pulse's peak value.

    As with the constant fraction discriminator, it is assumed the
    rising pulse slope points away from zero.

    Args:
        signal (ArrayLike): 1D input array with analog signal.
        threshold (data-type): Trigger threshold, positive values imply
            a positive pulse slope while negative values correspondingly
            imply a negative pulse slope.
        ratio_max (data-type, optional): Ratio of leading edge to peak
            value, 0.6 by default.
        width (data-type, optional): Minimal distance between found
            edges, none by default.
        interp (int, optional): Interpolation mode to locate the edge
            position, linear by default.
        edges (ArrayLike, optional): 1D output array to hold the
            positions of found edges, a new one is allocated if None is
            passed.
        amplitudes (ArrayLike, optional): 1D output array to hold the
            pulse amplitudes corresponding to found edges, a new one is
            allocated if None is passed.
    Returns:
        (ArrayLike, ArrayLike, int) 1D arrays containing the edge
            positions and amplitudes, number of found edges.
    """

    if edges is None:
        edges = np.zeros(
            len(amplitudes) if amplitudes is not None else len(signal) // 100,
            dtype=np.asarray(signal).dtype)

    if amplitudes is None:
        amplitudes = np.zeros_like(edges, dtype=np.asarray(signal).dtype)

    # Negative thresholds require inverted inequality relations when
    # comparing those to signals. For optimal performance without
    # duplicating most of the function body, the threshold and signal
    # are inverted instead (essentially making the signal always
    # positive) or handled specifically in other cases.
    cdef bint negative = threshold < 0
    cdef data_t s = 1

    if negative:
        threshold = -threshold
        s = -1

    cdef int signal_idx, signal_len = signal.shape[0], \
        edge_idx = 0, max_edge = min(edges.shape[0], edges.shape[0]), \
        ratio_idx, peak_idx, last_peak_idx = 0

    cdef data_t cur_value, ratio_pos, ratio_value, peak_value = 0.0, \
        last_ratio_pos = -width
    cdef bint beyond_threshold = False

    with nogil:
        for signal_idx in range(signal_len):
            cur_value = signal[signal_idx]

            if negative:
                cur_value = -cur_value

            if not beyond_threshold:
                # Looking for a region beyond threshold right now.

                if cur_value > threshold:
                    # Crossed the threshold, switch mode.
                    beyond_threshold = True

                    # Start with current value as peak.
                    peak_idx = signal_idx
                    peak_value = cur_value
                else:
                    # Still below threshold, ignore
                    pass

            else:
                # Currently in a region beyond threshold.

                if cur_value > threshold:
                    # Still beyond threshold, check if the peak value
                    # increases and continue

                    if cur_value > peak_value:
                        # New peak value.
                        peak_idx = signal_idx
                        peak_value = cur_value
                else:
                    # Dropped below threshold, find the edge.
                    beyond_threshold = False
                    ratio_value = peak_value * ratio_max

                    ratio_idx = peak_idx

                    # Profiling indicates that entirely separate loops based
                    # on sign improve total runtime by more than 10%, as
                    # this loop is likely too hot for an additional branch
                    # in every ieration.
                    if negative:
                        while ratio_idx >= last_peak_idx:
                            if -signal[ratio_idx] < ratio_value:
                                break
                            ratio_idx -= 1
                        else:
                            continue
                    else:
                        while ratio_idx >= last_peak_idx:
                            if signal[ratio_idx] < ratio_value:
                                break
                            ratio_idx -= 1
                        else:
                            continue

                    if interp == EdgeInterpolation.NEAREST:
                        if fabs(s * signal[ratio_idx] - ratio_value) > \
                                fabs(s * signal[ratio_idx+1] - ratio_value):
                            ratio_pos = <data_t>(ratio_idx+1)
                        else:
                            ratio_pos = <data_t>ratio_idx

                    elif interp == EdgeInterpolation.LINEAR:
                        ratio_pos = <data_t>ratio_idx \
                            + (s * ratio_value - signal[ratio_idx]) \
                            / (signal[ratio_idx+1] - signal[ratio_idx])

                    elif interp == EdgeInterpolation.SPLINE:
                        raise NotImplementedError('spline interpolation')

                    elif interp == EdgeInterpolation.SINC:
                        ratio_pos = <data_t>_dled_sinc_interp(
                            signal, negative, ratio_idx, ratio_value)

                    else:
                        raise ValueError('invalid interpolation mode')

                    if (ratio_pos - width) < last_ratio_pos:
                        # Reject this edge within the dead time.
                        continue

                    edges[edge_idx] = ratio_pos
                    amplitudes[edge_idx] = s * peak_value
                    last_peak_idx = peak_idx
                    last_ratio_pos = ratio_pos
                    edge_idx += 1

                    if edge_idx == max_edge:
                        # Abort condition if the buffer is full.
                        break

    return np.asarray(edges)[:edge_idx], np.asarray(amplitudes)[:edge_idx], \
        edge_idx


def sinc_interpolate(
    data_t[::contiguous] y_sampled, data_t[::contiguous] x_interp,
    double[::contiguous] y_interp = None, int window=100
):
    # TODO: Useful function that should go somewhere, but doesn't
    # actually belong to this module.

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
