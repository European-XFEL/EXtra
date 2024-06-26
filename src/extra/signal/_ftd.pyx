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
    """Edge interpolation methods.

    The fast timing discriminators in this module can use several
    different interpolation methods to estimate the real edge position
    in between the actual samples. These differ in terms of their
    interpolation quality and performance requirements.

    The default setting is to apply `EdgeInterpolation.LINEAR`, which
    offers good results for not-too-fast rise times at close to no
    cost in performance.

    For particularly fast signals however, i.e. with large curvature,
    linear interpolation tends to bias edge positions towards the sample
    positions at either end. This can be improved with
    `EdgeInterpolation.SINC`, but at a significant performance hit.

    If the discriminated edge positions should always fall on the most
    actual sample positions, `EdgeInterpolation.NEAREST` chooses the
    closest such position. Note that this offers no relevant performance
    advantage over linear interpolation.

    Some of the interpolation methods can be configured via
    [config_ftd_interpolation][extra.signal.config_ftd_interpolation].

    Attributes:
        NEAREST: Chose the sample position whose value is closest to the
            value at the true edge position, no cost in performance and
            will always return an integer result.
        LINEAR: Perform a linear interpolation between the two bordering
            samples to find the position.
        SINC: Perform sinc interpolation on a region of the signal
            around the edge position to find the position, please see
            [sinc_interpolate][extra.signal.sinc_interpolate] for more
            details.
    """

    NEAREST = 0
    LINEAR = 1
    SPLINE = 2
    SINC = 3


"""Sinc interpolation parameters."""
cdef int _sinc_window = 200
cdef int _sinc_search_iterations = 10


def config_ftd_interpolation(sinc_window=None, sinc_search_iterations=None):
    """Configure fast timing discriminator interpolation.

    Some of the interpolation methods used as part of fast timing
    discrimination in this module may be configured in terms of
    performance or precision:

    * `sinc_window` specifies the number of samples before and after the
        interpolation points actually used to evaluate $x(t)$, i.e. the
        finite boundaries to approximate the infinite sum. By default,
        up to 200 samples in each direction are used.

    * `sinc_search_iterations` specifies the number of binary search
        steps taken to find the optimal edge position from interpolated
        values in between two samples. The maximal resolution in samples
        the interpolation can therefore achieve with $N$ iterations is
        $2^{-N}$.

    When set, these parameters apply to all discriminator
    implementations and all their use of interpolation.

    For more details on interpolation, please refer to
    [EdgeInterpolation][extra.signal.EdgeInterpolation].

    Args:
        sinc_window (int, optional): Sample window used around the
            interpolated point, unchanged if omitted.
        sinc_search_iterations (int, optional): Number of iterations
            used in binary search for closest function argument,
            unchanged if omitted.

    Returns:
        config (dict): Mapping of current values with keys
            `sinc_window`, `sinc_search_iterations`.
    """

    global _sinc_window, _sinc_search_iterations

    if sinc_window is not None:
        _sinc_window = <int>sinc_window

    if sinc_search_iterations is not None:
        _sinc_search_iterations = <int>sinc_search_iterations

    return dict(sinc_window=_sinc_window,
                sinc_search_iterations=_sinc_search_iterations)


cdef inline double _sinc(double x) noexcept nogil:
    """Normalized sinc function."""

    if x != 0.0:
        return sin(x*M_PI) / (x*M_PI)

    return 1.0


cdef double _sinc_interp(
    data_t[::contiguous] y_sampled, double x_interp
) noexcept nogil:
    cdef int k, \
        sampling_start = max(<int>floor(x_interp) - _sinc_window, 0), \
        sampling_end = min(<int>ceil(x_interp) + _sinc_window,
                           y_sampled.shape[0])

    cdef double y_interp = 0.0

    for k in range(sampling_start, sampling_end):
        y_interp += <double>y_sampled[k] * _sinc(k - x_interp)

    return y_interp


cdef int _cfd_sinc_interp(
    data_array_t signal, int i, int j,
    data_t delay, data_t fraction, data_t walk, data_t* result
) except -1 nogil:
    # Note that the return value of this function is solely used for
    # exception propagation, the result is returned by pointer.

    cdef int int_delay = <int>ceil(delay)
    cdef bint is_integer_delay = int_delay == delay

    cdef int k, \
        sampling_start = max(i - _sinc_window, int_delay), \
        sampling_end = min(j + _sinc_window, signal.shape[0])

    # Interpolation is always done on double precision, as single
    # precision is almost certain to cause rounding errors in the sum.
    cdef double left_pos = <double>i, right_pos = <double>j, \
        middle_pos = 0.0, middle_value

    cdef data_t *interp_buf = NULL

    assert i >= int_delay or j >= int_delay, 'indices smaller than delay'

    if not is_integer_delay:
        # For non-integer delays, the sinc interpolation includes
        # the same interpolated values in its own summation for every
        # iteration. Computing these values only once and keep them in
        # a small buffer significantly increases performance.
        interp_buf = <data_t*>malloc(sizeof(delay) * (2 * _sinc_window + 1))

        assert interp_buf != NULL, 'unable to allocate interpolation buffer'

        for k in range(sampling_start, sampling_end):
            interp_buf[k - sampling_start] = _sinc_interp(signal, k - delay)

    for _ in range(_sinc_search_iterations):
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

    result[0] = <data_t>middle_pos


cdef data_t _dled_sinc_interp(
    data_array_t signal, bint negative, int ratio_idx, data_t ratio_value,
) noexcept nogil:
    cdef int k, \
        sampling_start = max(ratio_idx - _sinc_window, 0), \
        sampling_end = min(ratio_idx + 1 + _sinc_window, signal.shape[0])

    if negative:
        ratio_value = -ratio_value  # Invert for negative traces.

    # Interpolation is always done on double precision, see above in the
    # implementation for CFD.
    cdef double cmp_value = <double>ratio_value, \
        middle_value, middle_pos = 0.0, \
        left_pos = <double>ratio_idx, right_pos = <double>(ratio_idx + 1)

    for _ in range(_sinc_search_iterations):
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
    data_t width=0.0, data_t fraction=1.0, data_t walk=0.0,
    int interp=EdgeInterpolation.LINEAR,
    data_array_t edges = None, data_array_t amplitudes = None
):
    """Constant fraction discriminator.

    The pulse shape is assumed to always grow away from zero, i.e. the
    rising slope of positive pulses is positive and that of negative
    pulses is negative. Correspondigly, a positive threshold implies the
    pulse to peak at its largest value while a negative threshold
    implies the pulse to peak at its smallest value.

    This discriminator can use sinc interpolation both to find the
    optimal walk crossing as well as to enable real delay values. Please
    see [config_ftd_interpolation][extra.signal.config_ftd_interpolation]
    for more details.

    Note that enabling both these features at the same time makes the
    discrimination much more expensive, as repeated nested
    interpolations are necessary.

    Args:
        signal (ArrayLike): 1D input array with analog signal.
        threshold (np.float32 or np.float64): Trigger threshold,
            positive values imply a positive pulse slope while negative
            values correspondingly imply a negative pulse slope.
        delay (np.float32 or np.float64): Delay between the raw and
            inverted signal.
        width (int, optional): Minimal distance between found edges,
            none by default.
        fraction (np.float32 or np.float64, optional): Fraction of the
            inverted signal, 1.0 by default.
        walk (np.float32 or np.float64, optional): Point of intersection
            in the inverted signal, 0.0 by default.
        interp (EdgeInterpolation, optional): Interpolation mode to
            locate the edge position, linear by default.
        edges (ArrayLike, optional): 1D output array to hold the
            positions of found edges, a new one is allocated if None is
            passed.
        amplitudes (ArrayLike, optional): 1D output array to hold the
            pulse amplitudes corresponding to found edges, a new one is
            allocated if None is passed.

    Returns:
        (ArrayLike, ArrayLike, int): 1D arrays containing the edge
            positions and amplitudes, number of found edges.
    """

    if delay <= 0:
        raise ValueError('delay must be positive')

    if width < 0:
        raise ValueError('width must be non-negative')

    if fraction <= 0:
        raise ValueError('fraction must be positive')

    if edges is None:
        edges = np.zeros(
            len(amplitudes) if amplitudes is not None else len(signal) // 100,
            dtype=np.asarray(signal).dtype)

    if amplitudes is None:
        amplitudes = np.zeros_like(edges, dtype=np.asarray(signal).dtype)

    cdef int i, j, k, edge_idx = 0, \
        max_edge = min(edges.shape[0], amplitudes.shape[0])
    cdef data_t s, cfd_i, cfd_j, edge_pos = -1, next_edge_pos = -1, \
        amplitude

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

            if cfd_i < walk and cfd_j > walk:
                amplitude = signal[i - int_delay]

                if negative:
                    for k in range(j - int_delay, i + int_delay):
                        if signal[k] < amplitude:
                            amplitude = signal[k]
                else:
                    for k in range(j - int_delay, i + int_delay):
                        if signal[k] > amplitude:
                            amplitude = signal[k]

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
                    _cfd_sinc_interp(signal, i, j, delay, fraction,
                                     orig_walk, &edge_pos)
                else:
                    raise ValueError('invalid interpolation mode')

                if edge_pos < next_edge_pos:
                    # Reject this edge within the dead time.
                    continue

                edges[edge_idx] = edge_pos
                amplitudes[edge_idx] = amplitude
                next_edge_pos = edge_pos + width
                edge_idx += 1

                if edge_idx == max_edge:
                    break

    return np.asarray(edges)[:edge_idx], np.asarray(amplitudes)[:edge_idx], \
        edge_idx


def dled(
    data_array_t signal, data_t threshold,
    data_t ratio_max=0.6, data_t width=0.0,
    int interp=EdgeInterpolation.LINEAR,
    data_array_t edges = None, data_array_t amplitudes = None
):
    """Dynamic leading edge discriminator.

    This algorithm finds the leading edge of pulses in a 1D signal
    located at a certain ratio to the pulse's peak value.

    As with the constant fraction discriminator, it is assumed the
    rising pulse slope points away from zero.

    This discriminator can use sinc interpolation to find the optimal
    edge position, please see
    [config_ftd_interpolation][extra.signal.config_ftd_interpolation]
    for more details.

    Args:
        signal (ArrayLike): 1D input array with analog signal.
        threshold (np.float32 or np.float64): Trigger threshold,
            positive values imply a positive pulse slope while negative
            values correspondingly imply a negative pulse slope.
        ratio_max (np.float32 or np.float64, optional): Ratio of leading
            edge to peak value, must be in (0.0, 1.0] and is 0.6 by
            default.
        width (np.float32 or np.float64, optional): Minimal distance
            between found edges, none by default.
        interp (int, optional): Interpolation mode to locate the edge
            position, linear by default.
        edges (ArrayLike, optional): 1D output array to hold the
            positions of found edges, a new one is allocated if None is
            passed.
        amplitudes (ArrayLike, optional): 1D output array to hold the
            pulse amplitudes corresponding to found edges, a new one is
            allocated if None is passed.

    Returns:
        (ArrayLike, ArrayLike, int): 1D arrays containing the edge
            positions and amplitudes, number of found edges.
    """

    if not (0.0 < ratio_max <= 1.0):
        raise ValueError('ratio_max must be in (0.0, 1.0]')

    if width < 0:
        raise ValueError('width must be non-negative')

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
        edge_idx = 0, max_edge = min(edges.shape[0], amplitudes.shape[0]), \
        ratio_idx, peak_idx, last_peak_idx = 0

    cdef data_t cur_value, ratio_pos, ratio_value, peak_value = 0, \
        next_ratio_pos = 0
    cdef bint beyond_threshold = False

    with nogil:
        for signal_idx in range(signal_len):
            cur_value = s * signal[signal_idx]

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

                    if ratio_pos < next_ratio_pos:
                        # Reject this edge within the dead time.
                        continue

                    edges[edge_idx] = ratio_pos
                    amplitudes[edge_idx] = s * peak_value
                    next_ratio_pos = ratio_pos + width
                    last_peak_idx = peak_idx
                    edge_idx += 1

                    if edge_idx == max_edge:
                        # Abort condition if the buffer is full.
                        break

    return np.asarray(edges)[:edge_idx], np.asarray(amplitudes)[:edge_idx], \
        edge_idx
