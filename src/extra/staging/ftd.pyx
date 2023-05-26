# cython: boundscheck=False, wraparound=False, cdivision=True

from cython cimport floating
from cython.view cimport contiguous
from libc.math cimport M_PI, sin, fabs, floor, ceil

import numpy as np


cpdef enum EdgeInterpolation:
    NEAREST = 0
    LINEAR = 1
    SPLINE = 2
    SINC = 3


cdef inline double sinc(double x) nogil:
    """Normalized sinc function."""

    if x != 0.0:
        return sin(x*M_PI) / (x*M_PI)
    else:
        return 1.0


def sinc_interpolate(
    double[::contiguous] y_sampled, double[::contiguous] x_interp,
    double[::contiguous] y_interp = None, int window=100
):
    if y_interp is None:
        y_interp = np.zeros(len(x_interp), dtype=np.float64)

    cdef int i, k, \
        sampling_start = max(<int>floor(x_interp[0]) - window, 0), \
        sampling_end = min(<int>ceil(x_interp[0]) + window, y_sampled.shape[0])

    for i in range(y_interp.shape[0]):
        for k in range(sampling_start, sampling_end):
            y_interp[i] += y_sampled[k] * sinc(k - x_interp[i])

    return y_interp


def cfd(
    double[::contiguous] signal,
    double[::contiguous] edges = None,  # TODO: Move back
    double threshold = 1, int delay = 1,  # TODO: Remove defaults
    int width=0, double fraction=1.0, double walk=0.0,
    int interp = EdgeInterpolation.LINEAR,
    # edges should be here.
    double[::contiguous] amplitudes = None
):
    """Constant fraction discriminator.

    This implementation is slightly faster than the Python-compatible
    implementation in cfd_full, but does not generate the analog monitor
    signal. Depending on the sign of threshold, it calls either
    cfd_fast_pos or cfd_fast_neg.

    Args:
        signal (array_like): 1D input array with analog signal.
        threshold (data-type): Trigger threshold.
        delay (int): Delay between the raw and inverted signal.
        width (int): Minimal distance between found edges.
        fraction (double): Fraction of the inverted signal.
        walk (double): Point of intersection in the inverted signal.
        interp (EdgeInterpolation): Interpolation mode to locate the
            edge position, 0 for nearest, 1 for linear, 2 for spline,
            3 for sinc.
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
            dtype=np.float64)

    if amplitudes is None:
        amplitudes = np.zeros_like(edges, dtype=np.asarray(signal).dtype)

    with nogil:
        if threshold > 0:
            num_edges = cfd_fast_pos(signal, edges, amplitudes, threshold, delay,
                                    width, fraction, walk, interp)
        elif threshold < 0:
            num_edges = cfd_fast_neg(signal, edges, amplitudes, threshold, delay,
                                    width, fraction, walk, interp)
        else:
            raise ValueError('threshold must be non-zero')

    return np.asarray(edges)[:num_edges], np.asarray(amplitudes)[:num_edges], \
        num_edges


cdef int cfd_fast_pos(double[::contiguous] signal,
                      double[::contiguous] edges,
                      double[::contiguous] amplitudes,
                      double threshold, int delay,
                      int width=0, double fraction=1.0, double walk=0.0,
                      int interp = EdgeInterpolation.LINEAR) nogil:
    """Fast constant fraction discriminator for positive signals.

    Args:
        signal (array_like): 1D input array with analog signal.
        edges (ArrayLike): 1D output array to hold the positions of
            found edges.
        amplitudes (ArrayLike): 1D output array to hold the pulse
            amplitudes correspondig to found edges.
        threshold (data-type): Trigger threshold, must be positive.
        delay (int): Delay between the raw and inverted signal.
        width (int): Minimal distance between found edges.
        fraction (double): Fraction of the inverted signal.
        walk (double): Point of intersection in the inverted signal.
        interp (int): Interpolation mode to locate the edge position,
            0 for nearest, 1 for linear, 2 for spline, 3 for sinc.

    Returns:
        (int) Number of found edges.
    """

    cdef int i, j, edge_idx = 0, next_edge = -1, \
        max_edge = min(edges.shape[0], amplitudes.shape[0])
    cdef double cfd_i, cfd_j

    for i in range(delay, signal.shape[0] - 1):
        if signal[i] <= threshold:
            continue

        j = i + 1

        cfd_i = signal[i] - fraction * signal[i - delay]
        cfd_j = signal[j] - fraction * signal[j - delay]

        if cfd_i > walk and cfd_j < walk and i > next_edge:
            edges[edge_idx] = i + (cfd_i - walk) / (cfd_i - cfd_j)

            next_edge = i + width
            edge_idx += 1

            if edge_idx == max_edge:
                break

    return edge_idx


cdef int cfd_fast_neg(double[::contiguous] signal,
                      double[::contiguous] edges,
                      double[::contiguous] amplitudes,
                      double threshold, int delay,
                      int width=0, double fraction=1.0, double walk=0.0,
                      int interp = EdgeInterpolation.LINEAR) nogil:
    """Fast constant fraction discriminator for negative signals.

    Args:
        signal (array_like): 1D input array with analog signal.
        edges (ArrayLike): 1D output array to hold the positions of
            found edges.
        amplitudes (ArrayLike): 1D output array to hold the pulse
            amplitudes correspondig to found edges.
        threshold (data-type): Trigger threshold, must be negative.
        delay (int): Delay between the raw and inverted signal.
        width (int): Minimal distance between found edges.
        fraction (double): Fraction of the inverted signal.
        walk (double): Point of intersection in the inverted signal.
        interp (int): Interpolation mode to locate the edge position,
            0 for nearest, 1 for linear, 2 for spline, 3 for sinc.

    Returns:
        (int) Number of found edges.
    """

    cdef int i, j, edge_idx = 0, next_edge = -1, \
        max_edge = min(edges.shape[0], amplitudes.shape[0])
    cdef double cfd_i, cfd_j

    for i in range(delay, signal.shape[0] - 1):
        if signal[i] >= threshold:
            continue

        j = i + 1

        cfd_i = signal[i] - fraction * signal[i - delay]
        cfd_j = signal[j] - fraction * signal[j - delay]

        if cfd_i < walk and cfd_j > walk and i > next_edge:
            edges[edge_idx] = i + (walk - cfd_i) / (cfd_j - cfd_i)

            next_edge = i + width
            edge_idx += 1

            if edge_idx == max_edge:
                break

    return edge_idx


def dled(
    floating[::contiguous] signal,
    floating[::contiguous] edges = None,  # TODO: Move back
    floating threshold = 100,  # TODO: Remove default
    floating ratio_max = 0.6, int interp = EdgeInterpolation.LINEAR,
    # edges should be here
    floating[::contiguous] amplitudes = None,
    int sinc_window=200, int sinc_iterations=15
):
    """Dynamic leading edge discriminator.

    This algorithm finds the leading edge of pulses in a 1D signal
    located at a certain ratio to the pulse's peak value.

    Args:
        signal (ArrayLike): 1D input array with analog signal.
        threshold (data-type): Trigger threshold.
        ratio_max (double, optional): Ratio of leading edge to peak
            value, 0.6 by default.
        interp (int, optional): Interpolation mode to locate the edge
            position, 0 for nearest (default), 1 for linear, 2 for
            spline, 3 for sinc.
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
    cdef floating s = 1

    if negative:
        threshold = -threshold
        s = -1

    cdef int signal_idx, signal_len = signal.shape[0], \
        edge_idx = 0, max_edge = min(edges.shape[0], edges.shape[0]), \
        ratio_idx, peak_idx, last_peak_idx = 0

    cdef floating cur_value, ratio_pos, ratio_value, peak_value = 0.0
    cdef bint beyond_threshold = False

    #with nogil:
    if True:
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
                            ratio_pos = <floating>(ratio_idx+1)
                        else:
                            ratio_pos = <floating>ratio_idx

                    elif interp == EdgeInterpolation.LINEAR:
                        ratio_pos = <floating>ratio_idx \
                            + (s * ratio_value - signal[ratio_idx]) \
                            / (signal[ratio_idx+1] - signal[ratio_idx])

                    elif interp == EdgeInterpolation.SPLINE:
                        ratio_pos = <floating>ratio_idx

                    elif interp == EdgeInterpolation.SINC:
                        ratio_pos = <floating>_dle_sinc_interp(
                            signal, negative, ratio_idx, ratio_value,
                            sinc_window, sinc_iterations)

                    else:
                        raise ValueError('invalid interpolation mode')

                    edges[edge_idx] = ratio_pos
                    amplitudes[edge_idx] = s * peak_value
                    last_peak_idx = peak_idx
                    edge_idx += 1

                    if edge_idx == max_edge:
                        # Abort condition if the buffer is full.
                        break

    return np.asarray(edges)[:edge_idx], np.asarray(amplitudes)[:edge_idx], \
        edge_idx


cdef double _dle_sinc_interp(floating[::contiguous] signal, bint negative,
                             int ratio_idx, floating ratio_value,
                             int sinc_window, int sinc_iterations):
    cdef int k, \
        sampling_start = max(ratio_idx - sinc_window, 0), \
        sampling_end = min(ratio_idx + 1 + sinc_window, signal.shape[0])

    if negative:
        ratio_value = -ratio_value  # Invert for negative traces.

    cdef double cmp_value = <double>ratio_value, \
        middle_value, middle_pos = 0.0, \
        left_pos = <double>ratio_idx, right_pos = <double>(ratio_idx + 1)

    for _ in range(1, sinc_iterations+1):
        middle_pos = (left_pos + right_pos) / 2

        middle_value = 0.0
        for k in range(sampling_start, sampling_end):
            middle_value += <double>signal[k] * sinc(k - middle_pos)

        if (
            (not negative and middle_value < cmp_value) or
            (negative and middle_value > cmp_value)
        ):
            left_pos = middle_pos
        else:
            right_pos = middle_pos

    return middle_pos
