
# cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False

from cython cimport numeric
from libc.limits cimport INT_MAX
from libc.string cimport memcpy
from numpy.math cimport NAN


def _reshape_flat_pulses(
    numeric[:, :] traces, numeric[:, :] out, int[:] pulse_ids,
    int samples_per_pulse, int first_pulse_offset, int clock_ratio
):
    """Separate trace by pulse into 1D array."""

    if pulse_ids.shape[0] > out.shape[0]:
        raise ValueError('pulse output buffer smaller than pulse count')

    cdef int start, end, i, j = -1, k, \
        cur_pid, prev_pid = INT_MAX, pid_offset = 0, \
        num_trains = traces.shape[0], trace_len = traces.shape[1]

    cdef size_t bytes_per_pulse = samples_per_pulse * sizeof(traces[0, 0])

    # Prepare the placeholder value for float and int types, using the
    # fact that NAN != NAN for the true float value while becoming equal
    # after integer conversion.
    cdef numeric placeholder_val = -1 if <numeric>NAN == <numeric>NAN \
        else <numeric>NAN

    for i in range(pulse_ids.shape[0]):
        cur_pid = pulse_ids[i]

        if cur_pid < prev_pid:
            # Pulse ID decreasing means a new train started.
            pid_offset = cur_pid
            j += 1

            if j >= num_trains:
                raise ValueError('train input buffer smaller than train count')

        prev_pid = cur_pid
        start = first_pulse_offset + (cur_pid - pid_offset) * clock_ratio
        end = start + samples_per_pulse

        if end > trace_len:
            # The end of this pulse is beyond the trace length, try to
            # copy what remains and fill with placeholder as needed.

            if start < trace_len:
                # There is still some data to copy into this pulse.
                memcpy(&out[i, 0], &traces[j, start],
                       (trace_len - start) * sizeof(traces[0, 0]))

            # Fill in the rest of this pulse with the placeholder value.
            for k in range(max(trace_len - start, 0), samples_per_pulse):
                out[i, k] = placeholder_val
        else:
            memcpy(&out[i, 0], &traces[j, start], bytes_per_pulse)
