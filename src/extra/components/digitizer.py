
from copy import copy
import re

import numpy as np

from ._digitizer import _slice_to_pulses

"""
# OPEN
- Should .ndarray()/.xarray() contain reference to raw?
- How would a zero/APD/peak method be called?

# TODO
- Support for zero-suppressed data
    - Reconstructing traces in memory
    - Load into sparse arrays?
    - Immediate digitization
"""


class AdqChannel:
    """A high-level interface to ADQ digitizer channels.

    Args:
        data (extra.data.DataCollection): Data to access digitizer from.
        channel (str): Channel name with or without underscore, e.g.
            1_A or 3C.
        digitizer (str, optional): Source name of an ADQ digitizer, only
            needed if the data includes more than one such device or
            none could be detected automatically.
        interleaved (bool, optional): Whether this digitizer channel was
            interleaving samples or not, only needed if it could not be
            detected automatically.
        name (str, optional): Name for this component, by default
            consists of the digitizer source and channel key.
        clock_ratio (int, optional): Digitizer sampling clock as
            multiple of the bunch repetition rate (4.5 MHz).
        pulses (extra.components.pulses.PulsePattern): Pulse component
            to pull pulse information, none by default. Some pulse
            separation operations may require this to be present.
    """

    # 4.5 Mhz, see extra.components.pulses.
    _bunch_repetition_rate = 1.3e9 / 288

    _adq_pipeline_re = re.compile(r'(\w+)\/ADC\/(\d):network')
    _adq_channel_re = re.compile(
        r'digitizers.channel_(\d)_([A-Z]).raw.samples')

    # Set of digitizers using 3G boards with a 392 clock ratio that do
    # not advertise as such.
    _3g_digitizer = {'SQS_DIGITIZER_UTC2/ADC/1:network'}

    def __init__(self, data, channel, digitizer=None, interleaved=None,
                 name=None, clock_ratio=None, pulses=None):
        if digitizer is None or digitizer not in data.instrument_sources:
            digitizer = self._find_adq_pipeline(data, digitizer or '')

        self._instrument_src = data[digitizer]

        if channel not in self._instrument_src:
            channel = self._find_adq_channel(self._instrument_src, channel)

        self._raw_key = self._instrument_src[channel]

        # Try to find control source.
        device_id = digitizer[:digitizer.index(':')]

        if device_id in data.control_sources:
            self._control_src = data[device_id]
        else:
            self._control_src = None

        if interleaved is None:
            if self._control_src is None:
                # Cannot be inferred without control source in data.
                raise ValueError(f'data is missing control source '
                                 f'{device_id}, please pass explicit '
                                 f'interleaved flag')

            interleaved = self._is_interleaved(self._control_src, channel)

        self._interleaved = interleaved

        if name is None:
            name = f'{self._raw_key.source}:{self._raw_key.key}'

        self._name = name

        if clock_ratio is None:
            # FIX: Assumes ADQ412 are used with control source not
            # telling whether it's a -3G or -4G board.
            clock_ratio = self._get_clock_ratio(self._instrument_src)

        self._clock_ratio = clock_ratio * (2 if interleaved else 1)

        self._pulses = pulses

    @classmethod
    def _find_adq_pipeline(cls, data, prefix=''):
        """Try to find ADQ digitizer instrument source name."""

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
        """Try to find raw sample key for digitizer channel."""

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
        """Determine if ADQ channel was interleaving samples."""

        m = cls._adq_channel_re.match(channel_key)

        if not m:
            raise ValueError(f'channel key {channel_key} does not match '
                             f'pattern {cls._adq_channel_re.pattern}, please '
                             f'pass explicit interleaved flag')

        board = m[1]

        return bool(
            control_src[f'board{board}.interleavedMode'].as_single_value())

    @staticmethod
    def _correct_cm_by_train(signal, out, period, baseline, baselevel=None):
        """Correct common mode in signal by each train trace."""

        if isinstance(baseline, slice):
            baseline = signal[..., baseline]

        if baselevel is not None:
            np.add(baseline, baselevel, out=baseline)

        for offset in range(period):
            sel = np.s_[offset::period]
            np.subtract(
                signal[..., sel],
                baseline[..., sel].mean(axis=signal.ndim - 1)[..., None],
                out=out[..., sel], casting='safe')

    @staticmethod
    def _correct_cm_by_mean(signal, out, period, baseline, baselevel=None):
        """Correct common mode in signal by the mean trace."""

        if isinstance(baseline, slice):
            baseline = signal[..., baseline].mean(axis=0)

        if baselevel is not None:
            np.add(baseline, baselevel, out=baseline)

        for offset in range(period):
            sel = np.s_[offset::period]
            np.subtract(signal[..., sel], baseline[sel].mean(),
                        out=out[..., sel], casting='safe')

    def _expect_pulses(self):
        """Checks if internal pulse information is available."""

        if self._pulses is None:
            raise RuntimeError('component must be initialized with pulse '
                               'information')

    def _reshape_to_pulses(self, data, num_pulses, samples_per_pulse,
                           first_pulse_offset):
        pulses_start = first_pulse_offset
        pulses_end = pulses_start + num_pulses * samples_per_pulse

        if pulses_end > data.shape[-1]:
            raise ValueError(f'trace axis too short for {num_pulses} pulses '
                             f'located at {pulses_start}:{pulses_end}]')

        return data[..., pulses_start:pulses_end].reshape(
            *data.shape[:-1], num_pulses, samples_per_pulse)

    def _slice_to_pulses(self, data, out, pulse_ids, samples_per_pulse,
                         first_pulse_offset):
        return _slice_to_pulses(data, out, pulse_ids, samples_per_pulse,
                                first_pulse_offset, self.clock_ratio)

    def _iter_h5_chunks(self):
        """Iterate over contiguous regions up to a full HDF chunk."""

        out_start = 0

        # Calling private EXtra-data APIs, use with caution.
        for chunk in self._raw_key._data_chunks_nonempty:
            h5_chunk_len = chunk.dataset.chunks[0]

            for local_start in range(0, chunk.total_count, h5_chunk_len):
                local_end = min(local_start + h5_chunk_len, chunk.total_count)
                out_end = out_start + (local_end - local_start)
                first = int(chunk.first)

                yield chunk.dataset, \
                    np.s_[first+local_start:first+local_end], \
                    np.s_[out_start:out_end], \
                    chunk.train_ids[local_start:local_end]

                out_start = out_end

    @property
    def control_source(self):
        """Control source of this digitizer, if found in data."""
        return self._control_src

    @property
    def instrument_source(self):
        """Instrument source of this digitizer."""
        return self._instrument_src

    def channel_key(self, suffix):
        # TODO
        return self._instrument_src['digitizers.channel_{}_{suffix}']

    @property
    def raw_samples_key(self):
        """Raw samples key."""
        return self._raw_key

    @property
    def interleaved(self):
        """Whether this channel is interleaved."""
        return self._interleaved

    @property
    def clock_ratio(self):
        """Ratio between bunch repetition clock (4.5 MHz) and sample clock."""
        return self._clock_ratio

    @property
    def sampling_rate(self):
        """Sampling rate in Hz."""
        return AdqChannel._bunch_repetition_rate * self._clock_ratio

    @property
    def sampling_period(self):
        """Period between samples in seconds."""
        return 1 / self.sampling_rate

    @property
    def trace_shape(self):
        """Shape of a single trace."""
        return self._raw_key.entry_shape[0]

    @property
    def trace_duration(self):
        """Duration of a single trace in seconds."""
        return self.trace_shape * self.sampling_period

    def select_trains(self, trains):
        """Select a subset of trains in this data.

        This method accepts the same type of arguments as
        [DataCollection.select_trains][extra_data.DataCollection.select_trains].
        """

        res = copy(self)

        if self._control_src is not None:
            res._control_src = self._control_src.select_trains(trains)

        res._instrument_src = self._instrument_src.select_trains(trains)
        res._raw_key = self._raw_key.select_trains(trains)

        if self._pulses is not None:
            res._pulses = self._pulses.select_trains(trains)

        return res

    def get_samples_per_pulse(self, pulse_period=None, repetition_rate=None,
                              pulse_ids=None, fractional=False):
        """Get number of samples per pulse.

        TODO
        Computes the number of samples corresponding to a time period.

        Calling with no arguments tries to use pulse information
        initialized with.

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
            pulse_period = AdqChannel._bunch_repetition_rate * pulse_period

        elif not isinstance(pulse_period, int):
            # Any non-int value assumes the pulse period has to be
            # determined from other values.

            if repetition_rate is not None:
                pulse_period = AdqChannel._bunch_repetition_rate \
                    / repetition_rate
            elif pulse_ids is None and self._pulses is not None:
                pulse_ids = self._pulses.peek_pulse_ids()

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

        return self._clock_ratio * pulse_period

    def reshape_to_pulses(self, data, first_pulse_offset=1000):
        """Reshape train data to pulse data.

        TODO
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

        if not isinstance(data, np.ndarray):
            data = np.asarray(data)

        self._expect_pulses()
        pulse_ids = self._pulses.peek_pulse_ids()

        num_pulses = len(pulse_ids)
        samples_per_pulse = self.get_samples_per_pulse(pulse_ids=pulse_ids)

        return self._reshape_to_pulses(data, num_pulses, samples_per_pulse,
                                       first_pulse_offset)

    def slice_to_pulses(self, data, first_pulse_offset=1000, out=None):
        """Reshape train data to pulse data.

        TODO
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

        self._expect_pulses()
        pulse_ids = self._pulses.get_pulse_ids(labelled=True)

        try:
            pulse_period = int(pulse_ids.groupby(level=0).diff().min())
        except ValueError:
            raise ValueError('every train contains only a single pulse')

        shape = (len(pulse_ids), self.get_samples_per_pulse(
            pulse_period=pulse_period))

        if out is None:
            out = np.zeros(shape, dtype=data.dtype)
        elif out is data:
            # Special mode of performing slicing in-place. This is
            # guaranteed to always work given the assumption of a
            # constant first pulse offset.
            out = data.reshape(-1)[:np.prod(shape)].reshape(shape)
        elif any([a < b for a, b in zip(out.shape, shape)]):
            raise ValueError(f'out must at least be of shape {shape}')

        self._slice_to_pulses(data, out, pulse_ids.to_numpy(),
                              shape[1], first_pulse_offset)

        return out

    def correct_common_mode(self, data, period=None, baseline=np.s_[:1000],
                            baselevel=None, out=None):
        """Apply common mode correction.
        """

        if out is None:
            out = np.empty_like(data)

        # TODO

    def ndarray(
        self, train_roi=(), out=None, dtype=None,
        pulse_layout=None, first_pulse_offset=10000, single_pulse_length=None,
        corrected=True, cm_period=None, baseline=np.s_[:1000], baselevel=0
    ):
        """Load this channel's data as a numpy array.

        Extends the regular [KeyData.ndarray][extra_data.KeyData.ndarray]
        method with the option to efficiently apply both common-mode
        corrections and pulse separation at the same time as reading the
        data from disk.

        Pulse separation is always done using the pulse information the
        component was initialized with and can be done by slicing pulses
        into a linear pulse dimension or reshaping trains into
        two-dimensional trains and pulses. Please see
        [AdqChannel.reshape_to_pulses] and [AdqChannel.slice_to_pulses]
        for further details. Note that this method will also incur a
        copy when reshaping to pulses for memory efficiency.

        Common mode correction will compute a baseline for every nth
        sample up to the configured period within the baseline region
        and substract it from every nth pixel across the entire trace.
        This will change the native data type to float32 if none is
        specified explicitly or an output array passed. Note that this
        type of common mode correction will often not yield useful
        results for integer types.

        Args:
            train_roi (slice, optional): Part of the trace of each train
                to read, applied before common mode correction if
                enabled. The entire train trace is read if omitted.
            out (array_like, optional): Array to read into, a new is
                allocated if omitted.
            dtype (dtype, optional): Data type for new array if
                allocated, by default float32 with common mode
                corrections or same type as saved data.
            pulse_layout ({'none', 'slice', 'reshape'}, optional): How
                data is seperated into pulses during readout, by default
                pulses are sliced if this component is initialized with
                pulse information and kept as trains if not.
            first_pulse_offset (int, optional): Sample where the first
                pulse begins, 10000 by default.
            single_pulse_length (int, optional): Samples per pulse for
                the case all trains only contain a single pulse.
            corrected (bool, optional): Whether data should be common
                mode corrected during readout, enabled by default.
            cm_period (int, optional): Sample periodicity for common
                mode correction, by default 8 or 16 when interleaved.
            baseline (slice, optional): Part of the trace of each train
                to determine common mode, :1000 by default.
            baselevel (number, optional): ADU value to pull the baseline
                to after common mode correction, 0 by default.

        Returns:
            (numpy.ndarray) Digitizer traces.
        """

        num_trains = self._raw_key.shape[0]

        if pulse_layout is None:
            pulse_layout = 'slice' if self._pulses is not None else 'none'

        if pulse_layout in {'slice', 'reshape'}:
            by_pulse = True

            # Obtain labelled series of pulse IDs and group by train.
            self._expect_pulses()
            pulse_ids = self._pulses.get_pulse_ids(labelled=True)
            pids_by_train = pulse_ids.groupby(level=0)

            # Number of pulses per train.
            num_pulses = pids_by_train.count()

            # Samples per pulse based on the shortest difference between
            # pulses if available. All code below using this value is
            # protected against out-of-bounds access.
            try:
                pulse_period = int(pids_by_train.diff().min())
            except ValueError:
                # Not a number means all trains contain a single pulse.#
                if single_pulse_length is None:
                    raise ValueError('only single pulse trains, please pass '
                                     'explicit single_pulse_length') from None

                samples_per_pulse = single_pulse_length
            else:
                samples_per_pulse = self.get_samples_per_pulse(
                    pulse_period=pulse_period)

            if pulse_layout == 'slice':
                shape = (len(pulse_ids), samples_per_pulse)
            elif pulse_layout == 'reshape':
                shape = (num_trains, num_pulses.max(), samples_per_pulse)

            # Generate offsets of first pulse and last pulse of each
            # train relative to all pulses.
            pulse_first = num_pulses.cumsum() - num_pulses.iloc[0]
            pulse_last = pulse_first + num_pulses

            # Convert to unlabelled array for use in native code later.
            pulse_ids = pulse_ids.to_numpy()

        elif pulse_layout == 'none':
            by_pulse = False

            # Identical to KeyData.ndarray().
            from extra_data.read_machinery import roi_shape
            shape = (num_trains,) + roi_shape(self._raw_key.entry_shape,
                                              train_roi)

        else:
            raise ValueError('pulse_layout must be `none`, `slice`, reshape`')

        if dtype is None:
            dtype = np.float32 if corrected else self._raw_key.dtype
        elif corrected and not np.issubdtype(dtype, np.floating):
            from warnings import warn
            warn('Common mode correction may yield incorrect results with '
                 'non-floating data types')

        if out is None:
            out = np.empty(shape, dtype=dtype)
        elif any([a < b for a, b in zip(out.shape, shape)]):
            raise ValueError(f'requires at least output array shape {shape}')

        if corrected and cm_period is None:
            cm_period = 16 if self._interleaved else 8

        tmp = None  # Temporary buffer for a single HDF chunk.

        # Loop over contiguous chunks of raw instrument data.
        for dset, row_sel, out_sel, train_ids in self._iter_h5_chunks():
            if by_pulse:
                pulse_sel = np.s_[pulse_first.loc[train_ids[0]]:
                                  pulse_last.loc[train_ids[-1]]]

                if tmp is None or tmp.shape != dset.shape:
                    # Allocate a new buffer is none yet exists or the
                    # chunk size has changed, which *should* not happen.
                    tmp = np.zeros(dset.chunks, dtype=dtype)

                # Read into the temporary buffer.
                chunk_out = tmp[:(row_sel.stop - row_sel.start)]
            else:
                # Read directly into the output buffer.
                chunk_out = out[out_sel]

            dset.read_direct(chunk_out, source_sel=(row_sel, *train_roi))

            if corrected:
                self._correct_cm_by_train(chunk_out, chunk_out, cm_period,
                                          baseline, baselevel)

            if pulse_layout == 'slice':
                self._slice_to_pulses(
                    chunk_out, out[pulse_sel], pulse_ids[pulse_sel],
                    samples_per_pulse, first_pulse_offset)

            elif pulse_layout == 'reshape':
                out[out_sel, :] = self._reshape_to_pulses(
                    chunk_out, *shape[1:], first_pulse_offset)

        return out

    def xarray(
        self, train_roi=(), out=None, dtype=None,
        pulse_layout=None, first_pulse_offset=10000, single_pulse_length=None,
        corrected=True, cm_period=None, baseline=np.s_[:1000], baselevel=0,
        name=None, pulse_dim='pulseId', sample_dim='sample'
    ):
        """Load this channel's data as a labelled array.

        Extends the regular [KeyData.xarray][extra_data.KeyData.xarray]
        method with the option to efficiently apply both common-mode
        corrections and pulse separation at the same time as reading the
        data from disk. Please see [AdqChannel.ndarray] for more details
        on this functionality and its related arguments.

        Without pulse separation, the returned array is labeled by
        trains and samples. Otherwise depending on the pulse layout, the
        non-sample dimensions are either using a multi index or separate
        dimensions for trains and pulses. Train are always labeled by
        their train ID.

        Args:
            train_roi (slice, optional): Part of the trace of each train
                to read, applied before common mode correction if
                enabled. The entire train trace is read if omitted.
            out (array_like, optional): Array to read into, a new is
                allocated if omitted.
            dtype (dtype, optional): Data type for new array if
                allocated, by default float32 with common mode
                corrections or same type as saved data.
            pulse_layout ({'none', 'slice', 'reshape'}, optional): How
                data is seperated into pulses during readout, by default
                pulses are sliced if this component is initialized with
                pulse information and kept as trains if not.
            first_pulse_offset (int, optional): Sample where the first
                pulse begins, 10000 by default.
            single_pulse_length (int, optional): Samples per pulse for
                the case all trains only contain a single pulse.
            corrected (bool, optional): Whether data should be common
                mode corrected during readout, enabled by default.
            cm_period (int, optional): Sample periodicity for common
                mode correction, by default 8 or 16 when interleaved.
            baseline (slice, optional): Part of the trace of each train
                to determine common mode, :1000 by default.
            baselevel (number, optional): ADU value to pull the baseline
                to after common mode correction, 0 by default.
            name (str, optional): Name of the resulting array, by
                default the name of the component.
            pulse_dim ({'pulseId', 'pulseNumber'}, optional): Label
                for pulse dimension, pulse ID by default.
            sample_dim ({'sample', 'time'}, optional): Label for
                sample dimension, sample number by default.

        Returns:
            (xarray.DataArray) Digitizer traces.
        """

        import xarray

        data = self.ndarray(
            train_roi, out, dtype,
            pulse_layout, first_pulse_offset, single_pulse_length,
            corrected, cm_period, baseline, baselevel)

        dims = []
        coords = {}

        if pulse_layout == 'slice':
            dims.append('pulse')
            coords['pulse'] = self._pulses.get_pulse_index()

        elif pulse_layout == 'reshape':
            dims.append('trainId')
            coords['trainId'] = self._raw_key.train_id_coordinates()

            if pulse_dim == 'pulseId':
                dims.append('pulseId')
                coords['pulseId'] = self._pulses.peek_pulse_ids()
            elif pulse_dim == 'pulseNumber':
                dims.append('pulseNumber')
                coords['pulseNumber'] = np.arange(data.shape[1],
                                                  dtype=np.int32)

        else:
            dims.append('trainId')
            coords['trainId'] = self._raw_key.train_id_coordinates()

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
            name = self._name

        return xarray.DataArray(data, dims=dims, coords=coords, name=name)

    def trains(self, pulse_layout=None, corrected=True):
        # TODO
        raise NotImplementedError('trains')
