
from copy import copy
import re

import numpy as np

from ._digitizer import _slice_to_pulses
from .pulses import PulsePattern, XrayPulses


class AdqChannel:
    _adq_pipeline_re = re.compile(r'(\w+)\/ADC\/(\d):network')
    _adq_channel_re = re.compile(r'digitizers.channel_(\d)_([A-Z]).raw.samples')

    # Set of digitizers using 3G boards with a 392 clock ratio
    _3g_digitizer = {'SQS_DIGITIZER_UTC2/ADC/1:network'}

    def __init__(self, data, channel, digitizer=None, interleaved=None,
                 name=None, clock_ratio=None, pulses=None):
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

        if name is None:
            name = f'{self.raw_key.source}:{self.raw_key.key}'

        self.name = name

        if clock_ratio is None:
            # FIX: Assumes ADQ412 are used with control source not
            # telling whether it's a -3G or -4G board.
            clock_ratio = self._get_clock_ratio(self.instrument_src)

        self.clock_ratio = clock_ratio * (2 if interleaved else 1)

        self.pulses = pulses

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

    def _reshaped_pulses_shape(self, pulse_ids):
        num_pulses = pulse_ids.groupby(level=0).count().max()

        pulse_delta = np.diff(np.asarray(pulse_ids))
        samples_per_pulse = self.get_samples_per_pulse(
            pulse_period=np.unique(pulse_delta[pulse_delta > 0]).min())

        return num_pulses, samples_per_pulse

    def _sliced_pulses_shape(self, pulse_ids):
        # Collect all unique pulse distance values.
        pulse_delta = np.diff(np.asarray(pulse_ids))
        samples_per_pulse = self.get_samples_per_pulse(
            pulse_period=np.unique(pulse_delta[pulse_delta > 0]).min())

        return len(pulse_ids), samples_per_pulse

    def _reshape_to_pulses(self, data, num_pulses, samples_per_pulse, first_pulse_offset):
        pulses_start = first_pulse_offset
        pulses_end = first_pulse_offset + num_pulses * samples_per_pulse

        if pulses_end > data.shape[-1]:
            raise ValueError(f'trace axis too short for {num_pulses} pulses '
                             f'located at {pulses_start}:{pulses_end}]')

        return data[..., pulses_start:pulses_end].reshape(
            *data.shape[:-1], num_pulses, samples_per_pulse)

    @property
    def sampling_rate(self):
        """Sampling rate in Hz."""
        return self.pulses.bunch_repetition_rate * self.clock_ratio

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

        if self.pulses is not None:
            res.pulses = self.pulses.select_trains(trains)

        return res

    def get_samples_per_pulse(self, pulse_period=None, repetition_rate=None,
                              pulse_ids=None, fractional=False):
        """Get number of samples per pulse.

        Computes the number of samples corresponding to a time period.

        Calling with no arguments tries to use pulse information initialized with.

        Args:
            repetition_rate (float): Pulse repetition rate in Hz.
            pulse_period (int or float): Pulse period in PPT units (int)
                or seconds (float).
            pulse_ids (array_like, optional): Pulse IDs for a single train.
            fractional (bool, optional): Whether to round to possible
                EuXFEL repetition rates (default) or return the full
                fractional value.

        Returns:
            (int or float) Number of samples per pulse, float if
                fractional=True.
        """

        if isinstance(pulse_period, float):
            # Float are interpreted as pulse period in seconds, convert
            # to units of 4.5 MHz.
            pulse_period = AdqChannel._master_clock * pulse_period

        elif not isinstance(pulse_period, int):
            # Any non-int value assumes the pulse period has to be
            # determined from other values.

            if repetition_rate is not None:
                pulse_period = AdqChannel._master_clock / repetition_rate
            elif pulse_ids is None and self.pulses is not None:
                pulse_ids = self.pulses.peek_pulse_ids()

            if pulse_ids is not None:
                # May either be passed directly or come from pulses
                # component.
                try:
                    pulse_period = set(pulse_ids[1:] - pulse_ids[:-1])
                except IndexError:
                    raise ValueError('two or more pulses requires to infer '
                                     'pulse period') from None

                if len(pulse_period) > 1:
                    raise ValueError('more than one period between pulse IDs')

                pulse_period = int(pulse_period.pop())

        if pulse_period is None:
            raise ValueError('must pass either pulse_period, repetition_rate, '
                             'pulse_ids or initialize component with pulse '
                             'information')

        if not fractional:
            pulse_period = int(round(pulse_period, 1))

        return self.clock_ratio * pulse_period

    def reshape_to_pulses(self, data, first_pulse_offset=1000):
        """Reshape train data to pulse data.

        Assumes constant pulse length
        Assumes constant number of pulses per train
        Uses first train

        Returns:
            data (ArrayLike): Digitizer trace(s) for one or more trains,
                last axis is assumed to be sample axis and on a train
                boundary.
            first_pulse_offset (int):
            samples_per_pulse (int):
            num_pulses (int, optional):
        """

        if self.pulses is None:
            raise RuntimeError('component must be initialized with pulse '
                               'information')

        if not isinstance(data, np.ndarray):
            data = np.asarray(data)

        pulse_ids = self.pulses.peek_pulse_ids()

        num_pulses = len(pulse_ids)
        samples_per_pulse = self.get_samples_per_pulse(pulse_ids=pulse_ids)

        return self._reshape_to_pulses(data, num_pulses, samples_per_pulse, first_pulse_offset)

    def slice_to_pulses(self, data, first_pulse_offset=1000, out=None):
        """Reshape train data to pulse data.

        Assumes digitizer triggers with first pulse
        Assumes constant pulse length

        Returns:
            data (ArrayLike): Digitizer trace(s) for one or more trains,
                last axis is assumed to be sample axis and on a train
                boundary.
            first_pulse_offset (int, optional):
            inplace (bool, optional): Whether the memory for data may be
                re-used or not, may lead to changes of input data!
        """

        if self.pulses is None:
            raise RuntimeError('component must be initialized with pulse '
                               'information')

        trace_len = data.shape[-1]

        # By train for non-constant pattern.
        all_pids = self.pulses.get_pulse_ids(labelled=True)

        shape = self._sliced_pulses_shape(all_pids)

        if out is None:
            out = np.zeros(shape, dtype=data.dtype)
        elif out is data:
            out = data.reshape(-1)[:np.prod(shape)].reshape(shape)
        elif any([a < b for a, b in zip(out.shape, shape)]):
            raise ValueError(f'out must at least be of shape {shape}')

        _slice_to_pulses(data, out, all_pids.to_numpy(),
                        first_pulse_offset, self.clock_ratio)

        return out

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
        self, train_roi=(), out=None,
        by_pulse='slice', first_pulse_offset=10000,
        corrected=True, cm_period=None, baseline=np.s_[:1000], baselevel=0
    ):
        if by_pulse:
            pulse_ids = self.pulses.get_pulse_ids(labelled=True)
            pulse_ids_numpy = pulse_ids.to_numpy()

            if by_pulse == 'slice':
                req_shape = self._sliced_pulses_shape(pulse_ids)
            elif by_pulse == 'reshape':
                req_shape = self.raw_key.shape[:1] \
                    + self._reshaped_pulses_shape(pulse_ids)
            else:
                raise ValueError('by_pulse may be `slice` or `reshape`')

        elif not by_pulse:
            from extra_data.read_machinery import roi_shape
            req_shape = self.raw_key.shape[:1] + roi_shape(
                self.raw_key.entry_shape, train_roi)

        dtype = np.float32 if corrected else self.raw_key.dtype

        if out is None:
            out = np.empty(req_shape, dtype=dtype)
        elif out is not None and out.shape != req_shape:
            raise ValueError(f'requires output array of shape {req_shape}')

        if corrected and cm_period is None:
            cm_period = 16 if self.interleaved else 8

        tmp = None
        dest_cursor = 0

        # Looping over contiguous chunks of instrument data.
        for cg_chunk in self.raw_key._data_chunks_nonempty:
            dset = cg_chunk.dataset
            hdf_chunk = dset.chunks

            # Looping over HDF chunks it is composed of.
            for start in range(0, cg_chunk.total_count, hdf_chunk[0]):
                end = min(start + hdf_chunk[0], cg_chunk.total_count)
                train_ids = cg_chunk.train_ids[start:end]

                dest_chunk_end = dest_cursor + (end - start)
                slices = (slice(cg_chunk.first + np.uint64(start),
                                cg_chunk.first + np.uint64(end)),) + train_roi

                if by_pulse:
                    pulse_slice = pulse_ids.index.slice_indexer(
                        (train_ids[0], 0), (train_ids[-1], 79))

                    if tmp is None:
                        # Read into a temporary buffer for slicing.
                        tmp = np.zeros(hdf_chunk, dtype=dtype)

                    chunk_out = tmp[:(end-start)]

                    if by_pulse == 'slice':
                        data_out = out[pulse_slice]
                    elif by_pulse == 'reshape':
                        data_out = out[dest_cursor:dest_chunk_end, :]
                else:
                    # Read directly into the output buffer.
                    chunk_out = out[dest_cursor:dest_chunk_end]

                dset.read_direct(chunk_out, source_sel=slices)
                dest_cursor = dest_chunk_end

                if corrected:
                    self._correct_cm_by_train(chunk_out, cm_period, baseline,
                                              baselevel, out=chunk_out)

                if by_pulse == 'slice':
                    _slice_to_pulses(
                        chunk_out, data_out, pulse_ids_numpy[pulse_slice],
                        first_pulse_offset, self.clock_ratio
                    )

                elif by_pulse == 'reshape':
                    data_out[:] = self._reshape_to_pulses(
                        chunk_out, *req_shape[1:], first_pulse_offset)

        return out

    def xarray(
        self, train_roi=(), out=None, name=None,
        by_pulse='slice', first_pulse_offset=10000,
        pulse_dim='pulseId', sample_dim='sample',
        corrected=True, cm_period=None, baseline=np.s_[:1000], baselevel=0
    ):
        """

        Args:
            pulse_dim (str, optional): May be pulseId, pulseNumber
            sample_dim (str, optional): May be sample, time
        """

        import xarray

        data = self.ndarray(
            train_roi=train_roi, out=out,
            by_pulse=by_pulse, first_pulse_offset=first_pulse_offset,
            corrected=corrected, cm_period=cm_period,
            baseline=baseline, baselevel=baselevel)

        dims = []
        coords = {}

        if by_pulse == 'slice':
            dims.append('pulse')
            coords['pulse'] = self.pulses.get_pulse_index()

        elif by_pulse == 'reshape':
            dims.append('trainId')
            coords['trainId'] = self.raw_key.train_id_coordinates()

            if pulse_dim == 'pulseId':
                dims.append('pulseId')
                coords['pulseId'] = self.pulses.peek_pulse_ids()
            elif pulse_dim == 'pulseNumber':
                dims.append('pulseNumber')
                coords['pulseNumber'] = np.arange(data.shape[1],
                                                  dtype=np.int32)

        else:
            dims.append('trainId')
            coords['trainId'] = self.raw_key.train_id_coordinates()

        samples = np.arange(data.shape[-1], dtype=np.int32)

        if sample_dim == 'time':
            dims.append('time')
            coords['time'] = samples * self.sampling_period
        elif sample_dim == 'sample':
            dims.append('sample')
            coords['sample'] = samples
        else:
            raise ValueError('sample_dim must be `time` or `sample`')

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
            from extra.utils import ftd
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