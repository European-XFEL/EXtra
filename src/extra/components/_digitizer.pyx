
# cython: boundscheck=False, wraparound=False, cdivision=True

from cython cimport numeric
from cython.view cimport contiguous
from libc.limits cimport LONG_MAX
from libc.string cimport memcpy
from libc.stdio cimport printf

def _slice_to_pulses(
    numeric[:, :] data_by_train, numeric[:, :] data_by_pulse,
    unsigned int[:] pulse_ids, int first_pulse_offset, int clock_ratio
):
    assert data_by_pulse.shape[0] == pulse_ids.shape[0], 'differing pulse data arrays'
    
    cdef int pulse_idx, train_idx = -1, pulse_offset = 0, pulse_start, pulse_end
    cdef long cur_pulse_id, prev_pulse_id = LONG_MAX
    cdef int num_pulses = data_by_pulse.shape[0], \
        samples_per_pulse = data_by_pulse.shape[1], \
        num_trains = data_by_train.shape[0], \
        trace_len = data_by_train.shape[1]
    
    cdef size_t pulse_nbytes = samples_per_pulse * sizeof(data_by_train[0, 0])
        
    for pulse_idx in range(num_pulses):
        cur_pulse_id = pulse_ids[pulse_idx]
        
        if cur_pulse_id < prev_pulse_id:
            # New train.
            train_idx += 1
            assert train_idx < num_trains, 'not enough train data'
            
            pulse_offset = cur_pulse_id
                
        prev_pulse_id = cur_pulse_id
        pulse_start = first_pulse_offset + (cur_pulse_id - pulse_offset) * clock_ratio
        pulse_end = pulse_start + samples_per_pulse
        
        if pulse_end > trace_len:
            raise ValueError(f'trace axis too short for pulses located at {pulse_start}:{pulse_end}')
            
        memcpy(&data_by_pulse[pulse_idx, 0],
               &data_by_train[train_idx, pulse_start],
               pulse_nbytes)
