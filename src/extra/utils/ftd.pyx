# cython: boundscheck=False, wraparound=False, cdivision=True

"""Fast timing disciminators.

This module contains native software implementations for fast timing
discrimination that mark the arrival time of pulses on the digitized
recording of a continguous analog signal.
"""


from cython cimport floating
from cython.view cimport contiguous
from libc.math cimport M_PI, sin, fabs, floor, ceil

import numpy as np


"""Edge interpolation method."""
cpdef enum EdgeInterpolation:
    NEAREST = 0
    LINEAR = 1
    SPLINE = 2
    SINC = 3

ctypedef floating data_t
ctypedef floating[::contiguous] data_array_t


cdef inline double _sinc(double x) nogil:
    """Normalized sinc function."""

    if x != 0.0:
        return sin(x*M_PI) / (x*M_PI)
    else:
        return 1.0


cdef int _cfd_fast_pos(
    data_array_t signal, data_array_t edges, data_array_t amplitudes,
    data_t threshold, int delay, int width, data_t fraction, data_t walk,
    int interp
) nogil:
    """Fast constant fraction discriminator for positive signals."""

    cdef int i, j, edge_idx = 0, next_edge = -1, \
        max_edge = min(edges.shape[0], amplitudes.shape[0])
    cdef data_t cfd_i, cfd_j, edge_pos

    for i in range(delay, signal.shape[0] - 1):
        if signal[i] <= threshold:
            continue

        j = i + 1

        cfd_i = signal[i] - fraction * signal[i - delay]
        cfd_j = signal[j] - fraction * signal[j - delay]

        if cfd_i > walk and cfd_j < walk and i > next_edge:
            if interp == EdgeInterpolation.NEAREST:
                edge_pos = i if fabs(cfd_j - walk) > fabs(cfd_i - walk) else j
            elif interp == EdgeInterpolation.LINEAR:
                edge_pos = i + (cfd_i - walk) / (cfd_i - cfd_j)
            elif interp == EdgeInterpolation.SPLINE:
                raise NotImplementedError('spline interpolation')
            elif interp == EdgeInterpolation.SINC:
                raise NotImplementedError('sinc interpolation')
            else:
                raise ValueError('invalid interpolation mode')

            edges[edge_idx] = edge_pos
            next_edge = i + width
            edge_idx += 1

            if edge_idx == max_edge:
                break

    return edge_idx


cdef int _cfd_fast_neg(
    data_array_t signal, data_array_t edges, data_array_t amplitudes,
    data_t threshold, int delay, int width, data_t fraction, data_t walk,
    int interp
) nogil:
    """Fast constant fraction discriminator for negative signals."""

    cdef int i, j, edge_idx = 0, next_edge = -1, \
        max_edge = min(edges.shape[0], amplitudes.shape[0])
    cdef data_t cfd_i, cfd_j, edge_pos

    for i in range(delay, signal.shape[0] - 1):
        if signal[i] >= threshold:
            continue

        j = i + 1

        cfd_i = signal[i] - fraction * signal[i - delay]
        cfd_j = signal[j] - fraction * signal[j - delay]

        if cfd_i < walk and cfd_j > walk and i > next_edge:
            if interp == EdgeInterpolation.NEAREST:
                edge_pos = i if fabs(cfd_j - walk) > fabs(cfd_i - walk) else j
            elif interp == EdgeInterpolation.LINEAR:
                edge_pos = i + (walk - cfd_i) / (cfd_j - cfd_i)
            elif interp == EdgeInterpolation.SPLINE:
                raise NotImplementedError('spline interpolation')
            elif interp == EdgeInterpolation.SINC:
                raise NotImplementedError('sinc interpolation')
            else:
                raise ValueError('invalid interpolation mode')

            edges[edge_idx] = edge_pos
            next_edge = i + width
            edge_idx += 1

            if edge_idx == max_edge:
                break

    return edge_idx


cdef inline data_t _cfd_interp_edge(
    data_array_t signal, int i, int j,
    data_t cfd_i, data_t cfd_j, data_t walk, int interp
) nogil:
    if interp == EdgeInterpolation.NEAREST:
        return i if fabs(cfd_j - walk) > fabs(cfd_i - walk) else j
    elif interp == EdgeInterpolation.LINEAR:
        return i + (cfd_i - walk) / (cfd_i - cfd_j)
    elif interp == EdgeInterpolation.SPLINE:
        raise NotImplementedError('spline interpolation')
    elif interp == EdgeInterpolation.SINC:
        raise NotImplementedError('sinc interpolation')
    else:
        raise ValueError('invalid interpolation mode')


cdef double _dled_sinc_interp(
    data_array_t signal, bint negative, int ratio_idx, data_t ratio_value,
    int sinc_window, int sinc_iterations
) nogil:
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
            middle_value += <double>signal[k] * _sinc(k - middle_pos)

        if (
            (not negative and middle_value < cmp_value) or
            (negative and middle_value > cmp_value)
        ):
            left_pos = middle_pos
        else:
            right_pos = middle_pos

    return middle_pos


def cfd_unified_loop(
    data_array_t signal, data_t threshold, int delay,
    int width=0, data_t fraction=1.0, data_t walk=0.0,
    int interp = EdgeInterpolation.LINEAR,
    data_array_t edges = None,
    data_array_t amplitudes = None
):
    # CFD implementation with a single loop for both signs, branching
    # as necessary.

    if edges is None:
        edges = np.zeros(
            len(amplitudes) if amplitudes is not None else len(signal) // 100,
            dtype=np.asarray(signal).dtype)

    if amplitudes is None:
        amplitudes = np.zeros_like(edges, dtype=np.asarray(signal).dtype)

    cdef int i, j, edge_idx = 0, next_edge = -1, \
        max_edge = min(edges.shape[0], amplitudes.shape[0])
    cdef data_t cfd_i, cfd_j, edge_pos

    cdef bint negative = threshold < 0

    if negative:
        walk = (<data_t>-1) * walk

    with nogil:
        for i in range(delay, signal.shape[0] - 1):
            if negative:
                if signal[i] >= threshold:
                    continue
            elif signal[i] <= threshold:
                continue

            j = i + 1

            if negative:
                cfd_i = fraction * signal[i - delay] - signal[i]
                cfd_j = fraction * signal[j - delay] - signal[j]
            else:
                cfd_i = signal[i] - fraction * signal[i - delay]
                cfd_j = signal[j] - fraction * signal[j - delay]

            if cfd_i > walk and cfd_j < walk and i > next_edge:
                if interp == EdgeInterpolation.NEAREST:
                    edge_pos = i if fabs(cfd_j - walk) > fabs(cfd_i - walk) \
                        else j
                elif interp == EdgeInterpolation.LINEAR:
                    edge_pos = i + (cfd_i - walk) / (cfd_i - cfd_j)
                elif interp == EdgeInterpolation.SPLINE:
                    raise NotImplementedError('spline interpolation')
                elif interp == EdgeInterpolation.SINC:
                    raise NotImplementedError('sinc interpolation')
                else:
                    raise ValueError('invalid interpolation mode')

                edges[edge_idx] = edge_pos
                next_edge = i + width
                edge_idx += 1

                if edge_idx == max_edge:
                    break

    return np.asarray(edges)[:edge_idx], np.asarray(amplitudes)[:edge_idx], \
        edge_idx


def cfd_dual_loop_call(
    data_array_t signal, data_t threshold, int delay,
    int width=0, data_t fraction=1.0, data_t walk=0.0,
    int interp = EdgeInterpolation.LINEAR,
    data_array_t edges = None,
    data_array_t amplitudes = None
):
    """Constant fraction discriminator.

    Args:
        signal (array_like): 1D input array with analog signal.
        threshold (data-type): Trigger threshold.
        delay (int): Delay between the raw and inverted signal.
        width (int, optional): Minimal distance between found edges,
            none by default.
        fraction (data-type, optional): Fraction of the inverted signal,
            1.0 by default.
        walk (data-type, optional): Point of intersection in the
            inverted signal, 0.0 by default.
        interp (EdgeInterpolation, optional): Interpolation mode to
            locate the edge position, linear by default.
            TODO: Always linear at the moment.
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

    cdef bint negative = threshold < 0

    with nogil:
        if threshold > 0:
            for i in range(delay, signal.shape[0] - 1):
                if signal[i] <= threshold:
                    continue

                j = i + 1

                cfd_i = signal[i] - fraction * signal[i - delay]
                cfd_j = signal[j] - fraction * signal[j - delay]

                if cfd_i > walk and cfd_j < walk and i > next_edge:
                    edges[edge_idx] = _cfd_interp_edge(
                        signal, i, j, cfd_i, cfd_j, walk, interp)
                    next_edge = i + width
                    edge_idx += 1

                    if edge_idx == max_edge:
                        break

        else:
            for i in range(delay, signal.shape[0] - 1):
                if signal[i] >= threshold:
                    continue

                j = i + 1

                cfd_i = signal[i] - fraction * signal[i - delay]
                cfd_j = signal[j] - fraction * signal[j - delay]

                if cfd_i < walk and cfd_j > walk and i > next_edge:
                    edges[edge_idx] = _cfd_interp_edge(
                        signal, i, j, -cfd_i, -cfd_j, -walk, interp)
                    next_edge = i + width
                    edge_idx += 1

                    if edge_idx == max_edge:
                        break

    return np.asarray(edges)[:edge_idx], np.asarray(amplitudes)[:edge_idx], \
        edge_idx


def cfd_dual_loop_inline(
    data_array_t signal, data_t threshold, int delay,
    int width=0, data_t fraction=1.0, data_t walk=0.0,
    int interp = EdgeInterpolation.LINEAR,
    data_array_t edges = None,
    data_array_t amplitudes = None
):
    # CFD implementation with separate loops per sign and inlined
    # interpolation for position.

    if edges is None:
        edges = np.zeros(
            len(amplitudes) if amplitudes is not None else len(signal) // 100,
            dtype=np.asarray(signal).dtype)

    if amplitudes is None:
        amplitudes = np.zeros_like(edges, dtype=np.asarray(signal).dtype)

    cdef int i, j, edge_idx = 0, next_edge = -1, \
        max_edge = min(edges.shape[0], amplitudes.shape[0])
    cdef data_t cfd_i, cfd_j

    with nogil:
        if threshold > 0:
            for i in range(delay, signal.shape[0] - 1):
                if signal[i] <= threshold:
                    continue

                j = i + 1

                cfd_i = signal[i] - fraction * signal[i - delay]
                cfd_j = signal[j] - fraction * signal[j - delay]

                if cfd_i > walk and cfd_j < walk and i > next_edge:
                    if interp == EdgeInterpolation.NEAREST:
                        edges[edge_idx] = i \
                            if fabs(cfd_j - walk) > fabs(cfd_i - walk) else j
                    elif interp == EdgeInterpolation.LINEAR:
                        edges[edge_idx] = i + (cfd_i - walk) / (cfd_i - cfd_j)
                    elif interp == EdgeInterpolation.SPLINE:
                        raise NotImplementedError('spline interpolation')
                    elif interp == EdgeInterpolation.SINC:
                        raise NotImplementedError('sinc interpolation')
                    else:
                        raise ValueError('invalid interpolation mode')

                    next_edge = i + width
                    edge_idx += 1

                    if edge_idx == max_edge:
                        break

        else:
            for i in range(delay, signal.shape[0] - 1):
                if signal[i] >= threshold:
                    continue

                j = i + 1

                cfd_i = signal[i] - fraction * signal[i - delay]
                cfd_j = signal[j] - fraction * signal[j - delay]

                if cfd_i < walk and cfd_j > walk and i > next_edge:
                    if interp == EdgeInterpolation.NEAREST:
                        edges[edge_idx] = i \
                            if fabs(cfd_j - walk) > fabs(cfd_i - walk) else j
                    elif interp == EdgeInterpolation.LINEAR:
                        edges[edge_idx] = i + (walk - cfd_i) / (cfd_j - cfd_i)
                    elif interp == EdgeInterpolation.SPLINE:
                        raise NotImplementedError('spline interpolation')
                    elif interp == EdgeInterpolation.SINC:
                        raise NotImplementedError('sinc interpolation')
                    else:
                        raise ValueError('invalid interpolation mode')

                    next_edge = i + width
                    edge_idx += 1

                    if edge_idx == max_edge:
                        break

    return np.asarray(edges)[:edge_idx], np.asarray(amplitudes)[:edge_idx], \
        edge_idx


def cfd_redirect_call(
    data_array_t signal, data_t threshold, int delay,
    int width=0, data_t fraction=1.0, data_t walk=0.0,
    int interp = EdgeInterpolation.LINEAR,
    data_array_t edges = None,
    data_array_t amplitudes = None
):
    # Original CFD implementation calling into two separate functions
    # per sign.

    if edges is None:
        edges = np.zeros(
            len(amplitudes) if amplitudes is not None else len(signal) // 100,
            dtype=np.asarray(signal).dtype)

    if amplitudes is None:
        amplitudes = np.zeros_like(edges, dtype=np.asarray(signal).dtype)

    with nogil:
        if threshold > 0:
            num_edges = _cfd_fast_pos(signal, edges, amplitudes,
                                      threshold, delay, width, fraction, walk,
                                      interp)
        elif threshold < 0:
            num_edges = _cfd_fast_neg(signal, edges, amplitudes,
                                      threshold, delay, width, fraction, walk,
                                      interp)
        else:
            raise ValueError('threshold must be non-zero')

    return np.asarray(edges)[:num_edges], np.asarray(amplitudes)[:num_edges], \
        num_edges


def dled(
    data_array_t signal, data_t threshold, data_t ratio_max, data_t width,
    int interp = EdgeInterpolation.LINEAR,
    data_array_t edges = None, data_array_t amplitudes = None,
    int sinc_window=200, int sinc_iterations=15
):
    """Dynamic leading edge discriminator.

    This algorithm finds the leading edge of pulses in a 1D signal
    located at a certain ratio to the pulse's peak value.

    Args:
        signal (ArrayLike): 1D input array with analog signal.
        threshold (data-type): Trigger threshold.
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
                            signal, negative, ratio_idx, ratio_value,
                            sinc_window, sinc_iterations)

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
    double[::contiguous] y_sampled, double[::contiguous] x_interp,
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
