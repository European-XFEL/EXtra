
from copy import copy
import re

import numpy as np

from extra.staging.timing import PulseTiming


class AdqChannel:
    _adq_pipeline_re = re.compile(r'(\w+)\/ADC\/(\d):network')
    _adq_channel_re = re.compile(r'digitizers.channel_(\d)_([A-Z]).raw.samples')
    
    # Set of digitizers using 3G boards with a 392 clock ratio
    _3g_digitizer = {'SQS_DIGITIZER_UTC2/ADC/1:network'}
    
    def __init__(self, data, channel, digitizer=None, interleaved=None,
                 name=None, clock_ratio=None, timeserver=None):
        if digitizer is None or digitizer not in data.instrument_sources:
            digitizer = self._find_adq_pipeline(data, digitizer or '')
        
        self.instrument_src = data[digitizer]
            
        if channel not in self.instrument_src:
            channel = self._find_adq_channel(self.instrument_src, channel)
            
        self.raw_key = self.instrument_src[channel]
        
        # Try to find control source.
        device_id = digitizer[:digitizer.index(':')]

        if device_id in data.control_sources:
            self.control_src = data[device_id]
        else:
            self.control_src = None
        
        if interleaved is None:
            if self.control_src is None:
                # Cannot be inferred without control source in data.
                raise ValueError(f'data is missing control source '
                                 f'{device_id}, please pass explicit '
                                 f'interleaved flag')

            interleaved = self._is_interleaved(self.control_src, channel)
            
        self.interleaved = interleaved
        
        if clock_ratio is None:
            # FIX: Assumes ADQ412 are used with control source not
            # telling whether it's a -3G or -4G board.
            clock_ratio = self._get_clock_ratio(self.instrument_src)
        
        self.clock_ratio = clock_ratio * (2 if interleaved else 1)
        
        self.timing = PulseTiming(data, timeserver=timeserver, bam=False)
        
        if name is None:
            name = f'{self.raw_key.source}:{self.raw_key.key}'
            
        self.name = name
        
    @classmethod
    def _find_adq_pipeline(cls, data, prefix=''):
        digitizers = set()
        
        for source in data.instrument_sources:
            m = cls._adq_pipeline_re.match(source)
            if m is not None and source.startswith(prefix):
                digitizers.add(m[0])
                
        if not digitizers:
            raise ValueError(f'no digitizer source found for prefix={prefix} '
                             f'and pattern={cls._adq_pipeline_re.pattern}, '
                             f'please pass explicit instrument source')
        elif len(digitizers) > 1:
            raise ValueError('multiple digitzer sources found for '
                             'prefix={} and pattern={}:\n{}'.format(
                                prefix, cls._adq_pipeline_re.pattern,
                                ', '.join(sorted(digitizers))))
            
        return digitizers.pop()
    
    @classmethod
    def _find_adq_channel(cls, source, channel):
        if len(channel) == 2:
            key = f'digitizers.channel_{channel[0]}_{channel[1]}.raw.samples'
        elif len(channel) == 3:
            key = f'digitizers.channel_{channel}.raw.samples'
        else:
            key = channel
            
        if key not in source:
            if key != channel:
                raise ValueError(f'key {key} for channel {channel} not found '
                                 f'in digitizer source {source.source}')
            else:
                raise ValueError(f'channel key {key} not found in digitizer '
                                 f'source {source.source}')    
                
        return key
    
    @classmethod
    def _get_clock_ratio(cls, instrument_src):
        """Determine clock ratio of sampling (without interleaving!)."""
        if instrument_src in cls._3g_digitizer:
            return 392
        else:
            return 440
        
    @classmethod
    def _is_interleaved(cls, control_src, channel_key):
        m = cls._adq_channel_re.match(channel_key)
        
        if not m:
            raise ValueError(f'channel key {channel_key} does not match '
                             f'pattern {cls._adq_channel_re.pattern}, please '
                             f'pass explicit interleaved flag')
            
        board = m[1]
        
        return bool(
            control_src[f'board{board}.interleavedMode'].as_single_value())
    
    @property
    def sampling_rate(self):
        """Sampling rate in Hz."""
        return PulseTiming._ppt_clock * self.clock_ratio
    
    @property
    def sampling_period(self):
        """Period between samples in seconds."""
        return 1 / self.sampling_rate
    
    @property
    def trace_shape(self):
        """Shape of a single trace."""
        return self.raw_key.entry_shape[0]
    
    @property
    def trace_duration(self):
        """Duration of a single trace in seconds."""
        return self.trace_shape * self.sampling_period
    
    def select_trains(self, trains):
        res = copy(self)
        
        if res.control_src is not None:
            res.control_src = self.control_src.select_trains(trains)
        
        res.instrument_src = self.instrument_src.select_trains(trains)
        res.raw_key = self.raw_key.select_trains(trains)
        res.timing = self.timing.select_trains(trains)
        
        return res
    
    def get_samples_per_pulse(self, repetition_rate=None, pulse_period=None,
                              pulse_ids=None, fractional=False):
        """Get number of samples per pulse.
        
        Computes the number of samples corresponding to a time period
        
        TODO: Define better order of interpreting overlapping arguments
        
        Args:
            repetition_rate (float): Pulse repetition rate in Hz.
            pulse_period (int or float): Pulse period in PPT units (int)
                or seconds (float).
            fractional (bool, optional): Whether to round to possible
                EuXFEL repetition rates (default) or return the full
                fractional value.
            
        Returns:
            (int or float) Number of samples per pulse, float if
                fractional=True.
        """
        
        if pulse_ids is not None:
            pulse_period = set(pulse_ids[1:] - pulse_ids[:-1])
            
            if len(pulse_period) > 1:
                raise ValueError('more than one period between pulse IDs')
                
            pulse_period = int(pulse_period.pop())
        
        if repetition_rate is not None:
            pulse_period = AdqChannel._master_clock / repetition_rate
        
        elif isinstance(pulse_period, float):
            # Float are interpreted as pulse period in seconds, convert
            # to units of 4.5 MHz.
            pulse_period = AdqChannel._master_clock * pulse_period
            
        elif not isinstance(pulse_period, int):
            raise ValueError('must pass either pulse_period, repetition_rate '
                             'or pulse_ids')
            
        if not fractional:
            pulse_period = int(round(pulse_period, 1))
            
        return self.clock_ratio * pulse_period
    
    def reshape_to_pulses(self, data, first_pulse_offset=1000,
                          pulse_ids=None, samples_per_pulse=None,
                          num_pulses=None, **samples_per_pulse_kwargs):
        """Reshape train data to pulse data.
        
        TODO: Define better order of interpreting overlapping arguments
        
        Returns:
            data (ArrayLike): Digitizer trace(s) for one or more trains,
                last axis is assumed to be sample axis and on a train
                boundary.
            first_pulse_offset (int):
            samples_per_pulse (int): 
            num_pulses (int, optional):
        """
            
        trace_len = data.shape[-1]
        pulses_start = first_pulse_offset
        
        if (
            pulse_ids is None and 
            (samples_per_pulse is None or num_pulses is None)
        ):
            # If nothing was passed, get pulse IDs from timing component.
            pulse_ids = self.timing.get_pulse_ids()
        
        if pulse_ids is not None:
            if samples_per_pulse is None:
                samples_per_pulse = self.get_samples_per_pulse(
                    pulse_ids=pulse_ids, **samples_per_pulse_kwargs)
                
            if num_pulses is None:
                num_pulses = len(pulse_ids)
        
        if samples_per_pulse is None:
            samples_per_pulse = self.get_samples_per_pulse(
                **samples_per_pulse_kwargs)
            
        if not isinstance(samples_per_pulse, int):
            # Non-integer pulse spacings are possible in principle, but
            # not common at EuXFEL given the clock synchronization of
            # FEL and digitizers.
            # Something to be done for later, also requires Cython code
            # for acceptable performance.
            raise ValueError('only integer samples_per_pulse supported '
                             'currently, please contact da-support@xfel.eu if '
                             'you require fractional values')
            
        if num_pulses is None:
            num_pulses = (trace_len - first_pulse_offset) // samples_per_pulse
        
        pulses_end = first_pulse_offset + num_pulses * samples_per_pulse
        
        if pulses_end > trace_len:
            raise ValueError(f'trace axis too short for {num_pulses} pulses '
                             f'[trace_len={trace_len}, '
                             f'pulses={pulses_start}:{pulses_end}]')
 
        return data[..., pulses_start:pulses_end].reshape(
            *data.shape[:-1], num_pulses, samples_per_pulse)

    @staticmethod
    def _correct_cm_by_train(signal, period, baseline=np.s_[:1000],
                             baselevel=None, out=None):
        """Correct common mode in signal by each train trace."""

        if isinstance(baseline, slice):
            baseline = signal[..., baseline]
            
        if baselevel is not None:
            np.add(baseline, baselevel, out=baseline)

        if out is None:
            out = np.empty_like(signal)

        for offset in range(period):
            sel = np.s_[offset::period]
            np.subtract(
                signal[..., sel],
                baseline[..., sel].mean(axis=signal.ndim - 1)[..., None],
                out=out[..., sel], casting='safe')

        return out

    @staticmethod
    def _correct_cm_by_mean(signal, period, baseline=np.s_[:1000],
                            baselevel=None, out=None):
        """Correct common mode in signal by the mean trace."""
        
        if isinstance(baseline, slice):
            baseline = signal[..., baseline].mean(axis=0)
            
        if baselevel is not None:
            np.add(baseline, baselevel, out=baseline)

        if out is None:
            out = np.empty_like(signal)

        for offset in range(period):
            sel = np.s_[offset::period]
            np.subtract(signal[..., sel], baseline[sel].mean(),
                        out=out[..., sel], casting='safe')

        return out
    
    def ndarray(
        self, by_pulse=True, train_roi=(), out=None, dtype=np.float32,
        corrected=True, cm_period=None, baseline=np.s_[:1000], baselevel=0,
        **reshape_pulses_kwargs
    ):
        # TODO: casting can be made more efficient by first allocating a proper
        # array with f32, use its first half to load in data as u16 and then cast
        # it in-place to f32.
        
        # TODO: Implement parallelized version with pasha.
        
        data_by_train = self.raw_key.ndarray(
            roi=train_roi, out=out).astype(dtype)
        
        if corrected:  
            if cm_period is None:
                cm_period = 16 if self.interleaved else 8
                
            self._correct_cm_by_train(data_by_train, cm_period, baseline,
                                      baselevel, out=data_by_train)
        
        if by_pulse:
            data_by_pulse = self.reshape_to_pulses(
                data=data_by_train, **reshape_pulses_kwargs)
            
            return data_by_pulse
        else:
            return data_by_train
    
    def xarray(
        self, by_pulse=True, train_roi=(), out=None, name=None,
        in_pulse_ids=False, in_physical_time=False,
        corrected=True, cm_period=None, baseline=np.s_[:1000], baselevel=0,
        **reshape_pulses_kwargs
    ):
        import xarray
        
        if (
            by_pulse and
            in_pulse_ids and
            ('pulse_ids' not in reshape_pulses_kwargs)
        ):
            # If data is to be pulse separated with proper indices and
            # no explicit indices were passed, load them already now and
            # add them to the kwargs for reshape_pulses. This way
            # another read can be avoided.
            reshape_pulses_kwargs['pulse_ids'] = \
                self.timing.get_pulse_ids()
        
        data = self.ndarray(
            by_pulse=by_pulse, train_roi=train_roi, out=out,
            corrected=corrected, cm_period=cm_period,
            baseline=baseline, baselevel=baselevel,
            **reshape_pulses_kwargs)
        
        dims = ['trainId']
        coords = {'trainId': self.raw_key.train_id_coordinates()}
        
        if by_pulse:
            if in_pulse_ids:
                dims.append('pulseId')
                coords['pulseId'] = reshape_pulses_kwargs['pulse_ids']
            else:
                dims.append('pulseIndex')
                coords['pulseIndex'] = np.arange(data.shape[1], dtype=np.int32)
            
        samples = np.arange(data.shape[-1], dtype=np.int32)
            
        if in_physical_time:
            dims.append('time')
            coords['time'] = samples * self.sampling_period
        else:
            dims.append('sample')
            coords['sample'] = samples
            
        if name is None:
            name = self.name
            
        return xarray.DataArray(data, dims=dims, coords=coords, name=name)
        
    def trains(self, by_pulse=True, corrected=True):
        raise NotImplementedError('trains')
    
    @staticmethod
    def _shape_digital_result(array, outer_shape, truncated, flattened):
        if flattened:
            array = array.flatten()
            array = np.ascontiguousarray(array[np.isfinite(array)])
        else:
            array = array.reshape(*outer_shape, array.shape[-1])
            
            if truncated:
                max_hits = np.isfinite(array).sum(axis=-1).max()
                array = np.ascontiguousarray(array[..., :max_hits])
            
        return array
     
    def digitize(self, data, method='dled', max_hits=None,
                 truncated=True, flattened=False, with_amplitudes=False,
                 num_workers=None, in_physical_time=False,
                 **digitize_kwargs):
        try:
            from extra.staging import ftd
            ftd_func = getattr(ftd, method)
        except (ImportError, AttributeError):
            raise RuntimeError(f'ftd package not available or `{method}` not '
                               f'implemented')
            
        outer_shape = data.shape[:-1] if data.ndim > 1 else tuple()
        data = data.reshape(-1, data.shape[-1])  # Remove all outer axes
        
        if max_hits is None:
            max_hits = data.shape[1] // 100
            
        import pasha
        
        if num_workers is None:
            from os import cpu_count
            num_workers = min(cpu_count() // 4, 10)

        psh_cls = pasha.ProcessContext \
            if num_workers > 1 else pasha.SerialContext
        psh = psh_cls(num_workers=num_workers)
        edges = psh.alloc(shape=(data.shape[0], max_hits), dtype=data.dtype,
                          fill=np.nan)
        
        if with_amplitudes:
            amplitudes = psh.alloc(shape=(data.shape[0], max_hits),
                                   dtype=data.dtype, fill=np.nan)
        else:
            amplitudes = np.zeros(max_hits, dtype=data.dtype)
        
        def digitize_hits(wid, index, trace):
            _, _, num_edges = ftd_func(
                trace, edges=edges[index],
                amplitudes=amplitudes[index] if with_amplitudes else amplitudes,
                **digitize_kwargs)

        # Allow conversion of edges to time.
        # When given or returning xarray, include digitizer parameters
        # as attributes
            
        psh.map(digitize_hits, data)

        if in_physical_time:
            edges *= self.sampling_period
        
        shaping_args = (outer_shape, truncated, flattened)
        
        if with_amplitudes:
            return self._shape_digital_result(edges, *shaping_args), \
                self._shape_digital_result(amplitudes, *shaping_args)
        else:
            return self._shape_digital_result(edges, *shaping_args)
