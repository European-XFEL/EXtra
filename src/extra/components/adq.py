
from copy import copy
from inspect import signature, Parameter
import re

import numpy as np
import pandas as pd

from extra_data import by_id
from extra_data.read_machinery import roi_shape
from .pulses import XrayPulses
from .utils import _isinstance_no_import
from ._adq import _reshape_flat_pulses


class AdqRawChannel:
    """A high-level interface to raw output of ADQ digitizer channels.

    Fast ADQ digitizers are used to acquire analog signals at GHz sample
    rates. Their onboard FPGA are able to generate different kinds of
    output from the signal they acquire, like raw data, zero suppressed
    data or peak integration.

    This component is able to access the raw data saved with these
    devices in various representations Tand data formats.

    All retrieval methods can perform implicit corrections and other
    steps like pulse separation, which can be controlled by parameters
    when initializing this component.

    By default, common mode corrections are applied to compensate for
    non-uniform baselines as a result of the ADQ's parallel readout
    architecture. The used sample periodicity is 8 samples, or 16 samples
    if interleaved. To disable, the component can be initialized with
    `cm_period` being 0 or less. These corrections also pull the baseline
    to zero unless specified otherwise with `baselevel`. If common mode
    corrections are disabled, the signal is still pulled to a `baselevel`
    if specified. If no baselevel is specified and common mode
    corrections are disabled, the data remains unchanged.

    Typical configurations for these corrections are therefore:

      * `cm_period > 0 and baselevel is not None`: Common mode
        correction is performed with baseline pulled to `baselevel`.

      * `cm_period > 0 and baselevel is None` (default): Common mode
        correction is performed with baseline at `0`.

      * `cm_period == 0 and baselevel is not None`: Baseline is pulled
        to `baselevel`.

      * `cm_period == 0 and baselevel is None`: No modification is done.

    The baseline can be any contiguous slice of the raw data, by default
    the first 1000 samples are used.

    All processing performed implicitly by the retrieval methods is also
    available to run explicitly on a set of data.

    Args:
        data (extra_data.DataCollection): Data to access digitizer from.
        channel (str): Channel name with or without underscore, e.g.
            1_A or 3C.
        digitizer (str, optional): Source name of an ADQ digitizer, only
            needed if the data includes more than one such device or
            none could be detected automatically.
        pulses (extra.components.pulses.PulsePattern): Pulse component
            to pull pulse information, by default a
            [XrayPulses][extra.components.XrayPulses] object is
            constructed unless pulse information are explicitly disabled
            by passing `False`. Most pulse separation operations require
            this to be present.
        interleaved (bool, optional): Whether this digitizer channel was
            interleaving samples or not, only needed if it could not be
            detected automatically.
        clock_ratio (int, optional): Digitizer sampling clock as
            multiple of the bunch repetition rate (4.5 MHz).
        sample_dim ('sample' or 'time', optional): Coordinates for sample
            dimension if a labelled result is returned, sample by default.
        first_pulse_offset (int, optional): Sample where the first
            pulse begins, 10000 by default. This is used to locate the
            beginning of the first pulse when pulse separation is used.
        single_pulse_length (int, optional): Samples per pulse for
            the case all trains only contain a single pulse, when it
            cannot be inferred from pulse repetition rate.
        cm_period (int, optional): Apply common mode correction with
            specified sample periodicity, disabled for non-positive
            values and by default 8 or 16 when interleaved.
        baseline (slice or numpy.typing.ArrayLike, optional): Contiguous
            1D slice of the trace of each train to determine common mode
            or baseline or direct baseline data to use, :1000 by default.
        baselevel (float, optional): ADU value to pull the baseline
            to, None by default. Note that common mode corrections
            if enabled always pull the baselevel to zero unless
            specified otherwise here.
        extra_cm_period (list, optional): Apply the common mode correction
            sequentially with the settings in the list.
    """

    # 4.5 Mhz, see extra.components.pulses.
    _bunch_repetition_rate = 1.3e9 / 288

    _adq_pipeline_re = re.compile(r'(\w+)\/ADC\/(\d):network')
    _adq_channel_re = re.compile(
        r'digitizers.channel_(\d)_([A-Z]).raw.samples')

    # Set of digitizers using 3G boards with a 392 clock ratio that do
    # not advertise as such.
    _3g_digitizer = {'SQS_DIGITIZER_UTC2/ADC/1'}

    def __init__(self, data, channel, digitizer=None, pulses=None,
                 interleaved=None, clock_ratio=None, sample_dim='sample',
                 first_pulse_offset=10000, single_pulse_length=25000,
                 cm_period=None, baselevel=None, baseline=np.s_[:1000],
                 extra_cm_period = list()):
        if digitizer is None or digitizer not in data.instrument_sources:
            digitizer = self._find_adq_pipeline(data, digitizer or '')

        self._instrument_src = data[digitizer]

        if len(channel) == 2:
            self._channel_board = int(channel[0])
            self._channel_letter = channel[1].upper()
        elif len(channel) == 3:
            self._channel_board = int(channel[0])
            self._channel_letter = channel[2].upper()
        else:
            raise ValueError('channel expected to be 2 or 3 characters, '
                             'e.g. 1A or 1_A')

        self._channel_number = ord(self._channel_letter) - ord('A') + 1
        self._channel_name = f'{self._channel_board}_{self._channel_letter}'

        key = f'digitizers.channel_{self._channel_name}.raw.samples'

        if key not in self._instrument_src:
            raise ValueError(f'key {key} for channel {channel} not found '
                             f'in digitizer instrument source {digitizer}')

        self._raw_key = self._instrument_src[key]

        # Try to find control source.
        device_id = digitizer[:digitizer.index(':')]

        if device_id in data.control_sources:
            self._control_src = data[device_id]
        else:
            self._control_src = None

        if pulses is False:
            pulses = None  # Explicitly disabled.
        elif pulses is None:
            # If none, try to auto-detect XrayPulses for this data.
            try:
                pulses = XrayPulses(data)
            except ValueError:
                raise ValueError('could not auto-detect pulse information, '
                                 'please pass explicit pulses object or '
                                 'disable explicitly with pulses=False')

        self._pulses = pulses

        if interleaved is None:
            if self._control_src is None:
                # Cannot be inferred without control source in data.
                raise ValueError(f'data is missing control source '
                                 f'{device_id}, please pass explicit '
                                 f'interleaved flag')

            interleaved = bool(self._control_src.run_value(
                f'board{self.board}.interleavedMode'))

        self._interleaved = interleaved

        if clock_ratio is None:
            if device_id in self._3g_digitizer:
                clock_ratio = 392
            else:
                clock_ratio = 440

        self._clock_ratio = clock_ratio * (2 if interleaved else 1)

        if sample_dim not in {'sample', 'time'}:
            raise ValueError('sample_dim must be one of `samples`, `time`')

        self._sample_dim = sample_dim

        self._first_pulse_offset = first_pulse_offset
        self._single_pulse_length = single_pulse_length

        if cm_period is None:
            cm_period = 16 if interleaved else 8
        else:
            cm_period = int(cm_period)

        self._cm_period = cm_period
        self._extra_cm_period = list(extra_cm_period)
        self._baselevel = baselevel
        self._baseline = baseline

    def __repr__(self):
        source = self._instrument_src.source
        return "<{} {} {}>".format(
            type(self).__name__, source[:source.find(':')],
            self._channel_name)

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

    @staticmethod
    def _correct_cm_by_train(signal, out, period, baseline, baselevel=None):
        """Correct common mode in signal by each train trace."""

        if isinstance(baseline, slice):
            baseline = signal[..., baseline]

        if baselevel is not None:
            baseline = baseline - baselevel

        # Make sure the dtypes match, otherwise baseline is likely
        # going to be float64 and not castable via `safe`.
        baseline = baseline.astype(out.dtype, copy=False)

        if isinstance(period, int):
            period = [period]
        for p in period:
            for offset in range(p):
                sel = np.s_[offset::p]
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
            baseline = baseline - baselevel

        for offset in range(period):
            sel = np.s_[offset::period]
            np.subtract(signal[..., sel], baseline[sel].mean(),
                        out=out[..., sel], casting='safe')

    @staticmethod
    def _pull_to_baselevel(signal, out, baseline, baselevel=None):
        """Pull baseline to a certain level."""

        if isinstance(baseline, slice):
            baseline = signal[..., baseline]

        correction = baseline.mean(axis=signal.ndim - 1)
        if baselevel is not None:
            correction -= baselevel

        np.subtract(signal, correction[..., None], out=out, casting='unsafe')

    @staticmethod
    def _minimize_ragged_array(array):
        """Minimize ragged axis in an array."""

        max_entries = np.isfinite(array).sum(axis=-1).max()
        return np.ascontiguousarray(array[..., :max_entries])

    @staticmethod
    def _prepare_pasha(parallel):
        """Prepare pasha context."""

        import pasha

        if parallel is True:
            # Catch True explicitly to not cause an int cast to 1 and
            # disabling below.
            parallel = None

        if parallel is not None and int(parallel) < 2:
            return pasha.SerialContext()
        else:
            return pasha.ProcessContext(parallel)

    @staticmethod
    def _validate_edge_method(edge_func, edge_kw):
        """Validate edge finding method arguments."""

        if edge_func is None:
            from extra.signal import dled as edge_func

        # Track which keys are actually used
        used_keys = set()

        # Skip first parameter for input signal.
        next(parameters_it := iter(signature(edge_func).parameters.items()))

        for name, param in parameters_it:
            if name in edge_kw:
                used_keys.add(name)
            elif param.default is Parameter.empty:
                raise ValueError(f'missing required parameter `{name}` '
                                 f'for edge finding method')

        if (unknown_keys := (edge_kw.keys() - used_keys)):
            raise ValueError('unknown parameters for edge finding method '
                             + ', '.join(unknown_keys))

        return edge_func

    def _validate_out(self, out, shape):
        """Validate output arguments."""

        is_corrected = self._cm_period > 0 or self._baselevel is not None

        if out is None:
            out = np.empty(shape, dtype=np.float32)
        elif any([a < b for a, b in zip(out.shape, shape)]):
            raise ValueError(f'requires at least output array shape {shape}')
        elif is_corrected and not np.issubdtype(out.dtype, np.floating):
            from warnings import warn
            warn('Common mode correction or baselevel pull may yield '
                 'incorrect results with non-floating data types',
                 stacklevel=2)

        return out

    def _preprocess(self, data, out):
        """Apply default preprocessing of this component."""

        if data.dtype != out.dtype:
            # If dtypes do not match, use output buffer as temporary
            # storage for cast result.
            out[:] = data
            data = out

        if self._cm_period > 0:
            # Apply common mode corrections (includes baselevel).
            self._correct_cm_by_train(data, out, [self._cm_period] + list(self._extra_cm_period),
                                      self._baseline, self._baselevel)

        elif self._baselevel is not None:
            self._pull_to_baselevel(data, out, self._baseline, self._baselevel)

        return out

    def _prepare_pulses(self, train_ids):
        """Prepare pulse information."""

        if self._pulses is None:
            raise RuntimeError('component must be initialized with pulse '
                               'information for this operation')

        aligned_pulses = self._pulses.select_trains(by_id[train_ids])

        pulse_ids = aligned_pulses.pulse_ids(labelled=True)
        num_pulses = aligned_pulses.pulse_counts()

        # Ensure pulse data is available for all trains.
        try:
            num_pulses.loc[train_ids]
        except KeyError:
            raise ValueError('missing pulse information for one or more '
                             'trains') from None

        # Samples per pulse based on the shortest difference between
        # pulses if available. All code below using this value is
        # protected against out-of-bounds access.
        try:
            # Beware, pulse_period is not aligned to train_ids here!
            pulse_period = int(pulse_ids.groupby(level=0).diff().min())
        except ValueError:
            samples_per_pulse = self._single_pulse_length
        else:
            samples_per_pulse = self.samples_per_pulse(
                pulse_period=pulse_period)

        # Generate offsets of first pulse and last pulse of each
        # train relative to all pulses.
        pulse_last = num_pulses.cumsum()
        pulse_first = pulse_last - num_pulses

        # Combine pulse layout into a single dataframe.
        # TODO: samples_per_pulses is currently assumed to be constant
        pulse_layout = pd.DataFrame({'count': num_pulses,
                                     'first': pulse_first,
                                     'last': pulse_last,
                                     'length': samples_per_pulse})

        return aligned_pulses, pulse_layout

    def _reshape_pulses_by_train(self, data, num_pulses, samples_per_pulse,
                                 first_pulse_offset=None):
        """Reshape traces by train into traces by train and pulses."""

        pulses_start = first_pulse_offset or self._first_pulse_offset
        pulses_end = pulses_start + num_pulses * samples_per_pulse

        if pulses_end > data.shape[-1]:
            raise ValueError(f'trace axis too short for {num_pulses} pulses '
                             f'located at {pulses_start}:{pulses_end}')

        return data[..., pulses_start:pulses_end].reshape(
            *data.shape[:-1], num_pulses, samples_per_pulse)

    def _reshape_flat_pulses(self, data, out, pulse_ids, samples_per_pulse,
                             first_pulse_offset=None):
        """Reshape traces by train into traces by pulses."""

        return _reshape_flat_pulses(
            data, out, pulse_ids, samples_per_pulse,
            first_pulse_offset or self._first_pulse_offset, self._clock_ratio)

    def _build_sample_coords(self, data):
        """Build xarray coordinates for samples."""

        samples = np.arange(data.shape[-1], dtype=np.int32)

        if self._sample_dim == 'time':
            return {'time': samples * self.sampling_period}
        elif self._sample_dim == 'sample':
            return {'sample': samples}

    def _shape_edges(self, edges, amplitudes, index):
        """Build pandas object from edge/amplitude array."""

        from . import DelayLineDetector

        edges = DelayLineDetector._build_reduced_pd(
            None, edges, index, entry_level='edgeIndex')
        amplitudes = DelayLineDetector._build_reduced_pd(
            None, amplitudes, index, entry_level='edgeIndex')

        return pd.DataFrame({'edge': edges, 'amplitude': amplitudes})

    def _shape_edge_array(self, edges, amplitudes, coords,
                          labelled, squeeze_edges):
        """Shape output data of edge array finding methods."""

        if squeeze_edges:
            edges = self._minimize_ragged_array(edges)
            amplitudes = self._minimize_ragged_array(amplitudes)

        if not labelled:
            return edges, amplitudes

        if coords is None:
            # Generate generic coordinates.
            coords = {f'dim_{i}': np.arange(edges.shape[i])
                      for i in range(edges.ndim - 1)}

        coords['edge'] = np.arange(edges.shape[-1])

        import xarray as xr
        return xr.Dataset(dict(
            edges=xr.DataArray(edges, coords=coords),
            amplitudes=xr.DataArray(amplitudes, coords=coords)))

    @property
    def control_source(self):
        """Control source of this digitizer, if found in data."""
        return self._control_src

    @property
    def instrument_source(self):
        """Instrument source of this digitizer."""
        return self._instrument_src

    @property
    def board(self):
        """Board number."""
        return self._channel_board

    @property
    def letter(self):
        """Channel letter."""
        return self._channel_letter

    @property
    def number(self):
        """Channel number."""
        return self._channel_number

    @property
    def name(self):
        """Full channel name, e.g. 4_C"""
        return self._channel_name

    def channel_key(self, suffix):
        """Instrument KeyData object of this channel."""
        return self._instrument_src[f'digitizers.channel_{self._channel_name}'
                                    f'.{suffix}']

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
    def first_pulse_offset(self):
        """Sample position of first pulse on digitizer trace."""
        return self._first_pulse_offset

    @property
    def single_pulse_length(self):
        """Length in samples in case of a single pulse."""
        return self._single_pulse_length

    @property
    def sampling_rate(self):
        """Sampling rate in Hz."""
        return AdqRawChannel._bunch_repetition_rate * self._clock_ratio

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

    @property
    def board_parameters(self):
        """All board-related parameters."""

        if self._control_src is None:
            raise ValueError('component must be initialized with control '
                             'source data')

        board_node = f'board{self.board}'
        board_vals = {}

        for k, v in self._control_src.run_values().items():
            if k.endswith('.timestamp') or not k.startswith(board_node):
                # Skip timestamps and non-board nodes.
                continue

            if '.diag.' in k or '.temperature.' in k or 'channel_' in k:
                # Skip diagnostics, temperatures or channel-specifics.
                continue

            # Strip board prefix and .value suffix.
            board_vals[k[7:-6]] = v

        return board_vals

    @property
    def channel_parameters(self):
        """All channel-related parameters."""

        if self._control_src is None:
            raise ValueError('component must be initialized with control '
                             'source data')

        board_node = f'board{self.board}'
        channel_node = f'channel_{self.number}'
        channel_vals = {}

        for k, v in self._control_src.run_values().items():
            if k.endswith('.timestamp') or not k.startswith(board_node):
                # Skip timestamps and non-board nodes.
                continue

            if '.diag.' in k or '.temperature.' in k:
                # Skip diagnostics and temperatures.
                continue

            if channel_node in k:
                # Strip channel node if present
                k = k.replace(f'.{channel_node}', '')
            elif 'channel_' in k:
                # Skip other channel's settings.
                continue

            # Strip board prefix and .value suffix.
            channel_vals[k[7:-6]] = v

        return channel_vals

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

    def samples_per_pulse(self, pulse_period=None, pulse_duration=None,
                          repetition_rate=None, pulse_ids=None,
                          fractional=False):
        """Get number of samples per pulse.

        Digitizer traces are acquired by train, by may contain the data
        of several pulses, with each pulse covering a certain number of
        samples depending on the digitizer and machine repetition rates.

        This method may be called with any of its arguments to determine
        the number of samples per pulse, with the first argument not
        identical with None being used in argument order. Calling
        it with no arguments uses the pulse information the component
        was initialized with.

        Args:
            pulse_period (int, optional): Pulse period in PPT units of 4.5 MHz.
            pulse_duration (float, optional): Pulse duration in seconds.
            repetition_rate (float, optional): Pulse repetition rate in Hz.
            pulse_ids (numpy.typing.ArrayLike, optional): Pulse IDs of a
                single train.
            fractional (bool, optional): Whether to round to possible
                EuXFEL repetition rates (default) or return the full
                fractional value.

        Returns:
            samples_per_pulse (int or float): Number of samples per
                pulse, float if `fractional=True`.
        """

        if pulse_period is None:
            # All other forms are converted to pulse period.

            if pulse_duration is not None:
                pulse_period = AdqRawChannel._bunch_repetition_rate \
                    * pulse_duration
            elif repetition_rate is not None:
                pulse_period = AdqRawChannel._bunch_repetition_rate \
                    / repetition_rate
            elif pulse_ids is None and self._pulses is not None:
                pulse_ids = self._pulses.peek_pulse_ids(labelled=False)

            if pulse_ids is not None:
                # May either be passed directly or come from pulses
                # component.

                pulse_period = set(pulse_ids[1:] - pulse_ids[:-1])

                if not pulse_period:
                    raise ValueError('two or more pulses requires to infer '
                                     'pulse period') from None
                elif len(pulse_period) > 1:
                    raise ValueError('more than one period between pulse IDs')\
                        from None

                pulse_period = int(pulse_period.pop())

        if pulse_period is None:
            raise ValueError('must pass either pulse_period, repetition_rate, '
                             'pulse_ids or initialize component with pulse '
                             'information') from None

        if not fractional:
            pulse_period = int(round(pulse_period, 1))

        return self._clock_ratio * pulse_period

    def correct_common_mode(self, data, cm_period, baseline, baselevel=None):
        """Correct common mode.

        For ADQ digitizers, a common mode is present for every Nth
        sample due to the parallel readout architecture, typically with
        N = 8 or N = 16 with interleaving. This method allows to correct
        this behaviour by computing a baseline for every Nth sample up
        to the configured period within the baseline region and substract
        it from every Nth pixel across the entire trace.

        If no additional baselevel is passed, the baseline is always
        pulled to 0 by this method.

        Args:
            data (numpy.typing.ArrayLike): Input data to preprocess.
            cm_period (int): Sample periodicity of the common mode,
                generally 8 or 16 with interleaving.
            baseline (slice or numpy.typing.ArrayLike): Contiguous 1D
                slice of the trace of each train to determine baselevel
                or direct baseline data to use.
            baselevel (float or None, optional): ADU value to pull the
                baseline to, None by default which implicitly pulls the
                baseline to 0.

        Returns:
            out (numpy.ndarray): Corrected input data, same dtype as
                input data if floating otherwise `float32`.
        """

        if cm_period < 1:
            raise ValueError('Common mode must be positive number')

        if not isinstance(data, np.ndarray):
            data = np.asarray(data)

        out = np.zeros_like(data, dtype=data.dtype \
            if np.issubdtype(data.dtype, np.floating) else np.float32)
        self._correct_cm_by_train(data, out, cm_period, baseline, baselevel)

        return out

    def pull_baseline(self, data, baseline, baselevel):
        """Pull baseline to certain level.

        The signal baseline may be at different values than 0 either
        by intention to make optimal use of ADC range or through
        external means. In the absence of common mode correction, this
        method can pull the baseline to any desired level.

        Args:
            data (numpy.typing.ArrayLike): Input data to preprocess,
                will be converted to np.ndarray currently.
            baseline (slice or numpy.typing.ArrayLike): Contiguous 1D
                slice of the trace of each train to determine baselevel
                or direct baseline data to use.
            baselevel (float): ADU value to pull the baseline
                to.

        Returns:
            out (numpy.ndarray): Modified input data.
        """

        if not isinstance(data, np.ndarray):
            data = np.asarray(data)

        out = np.zeros_like(data, dtype=np.float32)
        self._pull_to_baselevel(data, out, baseline, baselevel)

        return out

    def reshape_to_pulses(self, data, first_pulse_offset=None):
        """Reshape train data to pulse data.

        This method performs pulse separation by splitting the trace
        acquired by train into individual traces by pulse based on the
        pulse information the component is initialized with.

        Args:
            data (numpy.typing.ArrayLike): Digitizer trace(s) for one or
                more trains, last axis is assumed to be samples within
                a train.
            first_pulse_offset (int, optional): Sample where the first
                pulse begins, by default the value the component was
                initialized with.

        Returns:
            out (numpy.ndarray): Reshaped pulse traces.
        """

        # TODO: Support data.ndim > 2

        if not isinstance(data, np.ndarray):
            data = np.asarray(data)

        if first_pulse_offset is None:
            first_pulse_offset = self._first_pulse_offset

        pulse_ids = self._pulses.pulse_ids()
        pulse_shape = data.shape[:-2] + (
            len(pulse_ids), self.samples_per_pulse(pulse_ids=pulse_ids))

        out = np.zeros(shape=pulse_shape, dtype=data.dtype)
        self._reshape_flat_pulses(data, out, pulse_ids.to_numpy(),
                                  out.shape[-1])

        return out

    def unstack_pulses(self, data):
        """Unstack pulse axis into train and pulse.

        This method unstacks the pulse axis introduced by
        [pulse_data()][extra.components.AdqRawChannel.pulse_data] into
        separate axis for train and the pulses for each of these trains.

        It is currently limited to 2D data, i.e. expects the first axis
        to exactly represent pulses and the second axis to contain
        samples, and the number of pulses per train have to be constant.

        Args:
            data (numpy.typing.ArrayLike): Data separated by pulse.

        Returns:
            out (numpy.ndarray or xarray.DataArray): Data separated by
                train and pulse. If labelled data with a `pulse`
                index is passed, a labelled result is returned using
                the correspondig coordinates.
        """
        # TODO: Support data.ndim > 2
        # TODO: Support ragged array for differing number of pulses

        pulse_dim = None

        if _isinstance_no_import(data, 'xarray', 'DataArray'):
            if 'pulse' in data.indexes:
                orig_coords = data.coords
                pulse_dim = list(data.indexes['pulse'].names)[-1]

            data = data.values

        train_ids = self._raw_key.train_id_coordinates()
        num_trains = len(train_ids)

        if (data.shape[0] % num_trains) != 0:
            # Error out for now.
            raise ValueError('number of pulses per train not constant')

        unstacked_data = data.reshape(num_trains, -1, data.shape[-1])

        if pulse_dim is None:
            return unstacked_data

        coords = {'trainId': train_ids}

        # TODO: Create this from pulse index
        if pulse_dim == 'pulseId':
            coords['pulseId'] = self._pulses.peek_pulse_ids()
        elif pulse_dim == 'pulseIndex':
            coords['pulseIndex'] = np.arange(unstacked_data.shape[1])
        elif pulse_dim == 'pulseTime':
            pulse_duration = float(
                orig_coords['pulseTime'][1] - orig_coords['pulseTime'][0])
            coords['pulseTime'] = np.arange(
                unstacked_data.shape[1]) * pulse_duration
        else:
            raise ValueError(f'invalid pulse dimension `{pulse_dim}`')

        coords.update(self._build_sample_coords(unstacked_data))

        import xarray as xr
        return xr.DataArray(unstacked_data, coords=coords)

    def find_edges(self, data, edge_func=None, max_edges=50, parallel=None,
                   **edge_kw):
        """Find signal edges.

        In some cases, not the raw data itself may be of interest but
        the location (and amplitude) of certain signals in the raw data.
        One such example is time-of-flight spectroscopy in counting
        mode, where individual charged particles impact a detection
        surface and leave a fast signal on the digitizer trace. Fast
        timing discriminators allow to robustly determine the position
        of such signals.

        By default, it uses the
        [dynamic leading discriminator][extra.signal.dled] from the
        [extra.signal](../signal.md) package, but other from this
        package or entirely custom functions may be used as well. The
        required signatures must include three keyword arguments
        `signal`, `edges` and `amplitudes` corresponding to those from
        [extra.signal.dled][extra.signal.dled]. The default edge finding
        method requires the `threshold` parameter to be passed as
        keyword argument.

        The processing is parallelized via
        [pasha](https://github.com/European-XFEL/pasha).

        Args:
            data (numpy.typing.ArrayLike): Input data to find edges on.
            edge_func (callable, optional): Edge finding method to run
                on each train trace, extra.signal.dled by default.
            max_edges (int, optional): Maximal number of edges per
                train, 50 by default.
            parallel (int or None, optional): Nunmber of parallel
                processes to use, by default 10 or a quarter of all cores
                whichever is lower. Any non-positive value or 1 disable
                parallelization.
            **edge_kw (Any): Any further keyword arguments are passed to
                the edge finding method.

        Returns:
            result (pandas.DataFrame): Edge positions and pulse heights.
        """

        edges, amplitudes = self.find_edge_array(
            data, False, False, edge_func, max_edges, parallel, **edge_kw)

        if data.ndim > 2:
            index = pd.MultiIndex.from_product(
                [np.arange(x) for x in data.shape[:-1]],
                names=[f'dim_{i}' for i in range(data.ndim - 1)])
        else:
            index = pd.Index(np.arange(data.shape[0]), name='dim_0')

        return self._shape_edges(edges, amplitudes, index)

    def find_edge_array(self, data, labelled=True, squeeze_edges=True,
                        edge_func=None, max_edges=50, parallel=None,
                        **edge_kw):
        """Find signal edges as ragged array.

        Alternative method to
        [find_edges()][extra.components.AdqRawChannel.find_edges]
        returning the results as ragged arrays, using `np.nan` as filler
        value.

        Args:
            data (numpy.typing.ArrayLike): Input data to find edges on.
            labelled (bool, optional): Whether data is returned as a
                labelled xarray (default) or unlabelled ndarray.
            squeeze_edges (bool, optional): Whether to minimize the edge
                axis length to the maxinum number of edges found per
                row, True by default.
            edge_func (Callable, optional): Edge finding method to run
                on each train trace, extra.signal.dled by default.
            max_edges (int, optional): Maximal number of edges per
                train, 1/5000 of trace length by default.
            parallel (int or None, optional): Nunmber of parallel
                processes to use, by default 10 or a quarter of all cores
                whichever is lower. Any non-positive value or 1 disable
                parallelization.
            **edge_kw (Any): Any further keyword arguments are passed to
                the edge finding method.

        Returns:
            result (numpy.ndarray, numpy.ndarray): Tuple of edge positions
                and amplitudes, only if not labelled.

            result (xarray.Dataset): Edge positions and pulse heights,
                only if labelled
        """

        # TODO: When given or returning xarray, include digitizer
        # parameters as attributes.

        if _isinstance_no_import(data, 'xarray', 'DataArray'):
            # If input is an xarray, preserve coordinates and use the
            # internal ndarray for processing.
            orig_coords = data.coords
            data = data.values
        else:
            orig_coords = None

        outer_shape = data.shape[:-1] if data.ndim > 1 else tuple()
        data = data.reshape(-1, data.shape[-1])  # Remove all outer axes

        edge_func = self._validate_edge_method(edge_func, edge_kw)

        psh = self._prepare_pasha(parallel)
        edges = psh.alloc(shape=(data.shape[0], max_edges), dtype=data.dtype,
                          fill=np.nan)
        amplitudes = psh.alloc(shape=(data.shape[0], max_edges),
                               dtype=data.dtype, fill=np.nan)

        def digitize_edges(wid, index, trace):
            edge_func(signal=trace, edges=edges[index],
                      amplitudes=amplitudes[index],
                      **edge_kw)

        psh.map(digitize_edges, data)

        if self._sample_dim == 'time':
            edges *= self.sampling_period

        # Add outer axes back in.
        edges = edges.reshape(*outer_shape, edges.shape[-1])
        amplitudes = amplitudes.reshape(*outer_shape, amplitudes.shape[-1])

        return self._shape_edge_array(edges, amplitudes, orig_coords,
                                      labelled, squeeze_edges)

    def train_data(self, labelled=True, roi=(), out=None):
        """Load this channel's raw data by train.

        This method is similar to obtaining the digitized raw traces
        directly via the channel's [KeyData][extra_data.KeyData] object,
        but is optimized to perform the ADQ-specific correction steps
        with minimal CPU and memory impact while reading the data from
        disk. Additionally, it offers labels for the sample
        dimension in either samples (default) or time in addition to the
        train dimension depending on the choice during component
        initialization.

        Args:
            labelled (bool, optional): Whether data is returned as a
                labelled xarray (default) or unlabelled ndarray.
            roi (slice or tuple, optional): Part of the trace of each
                train to read, applied before any preprocessing is
                performed. The entire train trace is read if omitted.
            out (numpy.typing.ArrayLike, optional): Array to read into,
                a new is allocated if omitted.

        Returns:
            data (xarray.DataArray or numpy.ndarray): Digitizer traces.
        """

        if isinstance(roi, slice):
            # The argument was always documented as slice, but actually
            # required a tuple identical to KeyData.ndarray(). Raw trace
            # data can never exceed one dimension, so this is confusing.
            # For comaptibility and and ease of code below, both can be
            # supported implicitly.
            roi = (roi,)

        shape = (self._raw_key.shape[0],) + roi_shape(
            self._raw_key.entry_shape, roi)
        out = self._validate_out(out, shape)

        offset = 0
        for kd in self._raw_key.split_trains(trains_per_part=200):
            num_trains = int(kd.shape[0])  # Beware of uint64
            out_sel = out[offset:offset+num_trains]
            offset += num_trains

            kd.ndarray(roi=roi, out=out_sel)
            self._preprocess(out_sel, out_sel)

        if not labelled:
            return out

        coords = {'trainId': self._raw_key.train_id_coordinates()}
        coords.update(self._build_sample_coords(out))

        import xarray as xr
        return xr.DataArray(out, coords=coords)

    def pulse_data(self, labelled=True, pulse_dim='pulseId', train_roi=(),
                   out=None):
        """Load this channel's raw data by pulse.

        In addition to [AdqRawChannel.train_data], this method also
        separates the data belonging to each pulse into their own trace
        based on the pulse information the component is initialized with.

        This process depends on the `first_pulse_offset` and potentially
        `single_pulse_length` the component was initialized with.

        If the pulse information refers to data beyond the acquired
        traces, it is filled by np.nan for floating data types or
        -1 for integer types.

        Args:
            labelled (bool, optional): Whether data is returned as a
                labelled xarray (default) or unlabelled ndarray.
            pulse_dim ('pulseId' or 'pulseIndex' or 'pulseTime', optional):
                Label for pulse dimension, pulse ID by default.
            train_roi (slice or tuple, optional): Part of the trace of
                each train to read, applied before any preprocessing is
                performed. The entire train trace is read if omitted.
            out (numpy.typing.ArrayLike, optional): Array to read into,
                a new one is allocated if omitted.

        Returns:
            data (xarray.DataArray or numpy.ndarray): Digitizer traces.
        """

        if isinstance(train_roi, slice):
            # See comment in AdqChannel.train_data().
            train_roi = (train_roi,)

        # Drop empty trains for efficient access to train IDs.
        raw_key = self._raw_key.drop_empty_trains()

        # Obtain information about pulse layout.
        pulses, pulse_layout = self._prepare_pulses(raw_key.train_ids)
        pulse_ids = pulses.pulse_ids(labelled=False)

        # Prepare output buffer.
        out_shape = (len(pulse_ids), pulse_layout['length'].max())
        out = self._validate_out(out, out_shape)

        # Temporary buffer for a single iteration.
        tmp = np.zeros((200,) + roi_shape(
            self._raw_key.entry_shape, train_roi), dtype=out.dtype)

        for kd in raw_key.split_trains(trains_per_part=200):
            pulse_sel = np.s_[pulse_layout.loc[kd.train_ids[0]]['first']:
                              pulse_layout.loc[kd.train_ids[-1]]['last']]

            tmp_sel = tmp[:len(kd.train_ids)]
            kd.ndarray(roi=train_roi, out=tmp_sel)

            self._preprocess(tmp_sel, tmp_sel)
            self._reshape_flat_pulses(
                tmp_sel, out[pulse_sel], pulse_ids[pulse_sel],
                pulse_layout['length'].max())

        if not labelled:
            return out

        coords = {'pulse': pulses.build_pulse_index(pulse_dim)}
        coords.update(self._build_sample_coords(out))

        import xarray as xr
        return xr.DataArray(out, coords=coords)

    def train_edges(self, edge_func=None, max_edges=None, parallel=None,
                    **edge_kw):
        """Load data and find signal edges by train.

        This method performs the edge discrimination step while loading
        the data and only returns those results. It is therefore
        significantly more memory efficient than performing these
        operations sequentially from memory.

        Please see [find_edges()][extra.components.AdqRawChannel.find_edges]
        for more details.

        Args:
            edge_func (callable, optional): Edge finding method to run
                on each train trace, extra.signal.dled by default.
            max_edges (int, optional): Maximal number of edges per
                train, 1/5000 of trace length by default.
            parallel (int or None, optional): Nunmber of parallel
                processes to use, by default 10 or a quarter of all cores
                whichever is lower. Any non-positive value or 1 disable
                parallelization.
            **edge_kw (Any): Any further keyword arguments are passed to
                the edge finding method.

        Returns:
            result (pd.DataFrame): Edge positions and pulse heights.
        """

        edges, amplitudes = self.train_edge_array(
            False, False, edge_func, max_edges, parallel, **edge_kw)
        index = pd.Index(self._raw_key.train_id_coordinates(), name='trainId')

        return self._shape_edges(edges, amplitudes, index)

    def train_edge_array(self, labelled=True, squeeze_edges=True,
                         edge_func=None, max_edges=None, parallel=None,
                         **edge_kw):
        """Load data and find signal edges by train as ragged array.

        Alternative method to
        [train_edges()][extra.components.AdqRawChannel.train_edges]
        returning the results as ragged arrays, using `np.nan` as filler
        value.

        Args:
            labelled (bool, optional): Whether data is returned as a
                labelled xarray (default) or unlabelled ndarray.
            squeeze_edges (bool, optional): Whether to minimize the edge
                axis length to the maxinum number of edges found per
                row, True by default.
            edge_func (callable, optional): Edge finding method to run
                on each train trace, extra.signal.dled by default.
            max_edges (int, optional): Maximal number of edges per
                train, 1/5000 of trace length by default.
            parallel (int or None, optional): Nunmber of parallel
                processes to use, by default 10 or a quarter of all cores
                whichever is lower. Any non-positive value or 1 disable
                parallelization.
            **edge_kw (Any): Any further keyword arguments are passed to
                the edge finding method.

        Returns:
            result (numpy.ndarray, numpy.ndarray): Tuple of edge
                positions and amplitudes, only if not labelled.

            result (xarray.Dataset): Edge positions and pulse heights,
                only if labelled.
        """

        edge_func = self._validate_edge_method(edge_func, edge_kw)

        if max_edges is None:
            max_edges = self._raw_key.entry_shape[0] // 5000

        # Set-up pasha for parallel processing.
        psh = self._prepare_pasha(parallel)
        trace_out = np.zeros(self._raw_key.entry_shape, dtype=np.float32)
        edges = psh.alloc(shape=(self._raw_key.shape[0], max_edges),
                          dtype=np.float32, fill=np.nan)
        amplitudes = psh.alloc(like=edges, fill=np.nan)

        def digitize_edges(wid, index, train_id, trace_in):
            edge_func(signal=self._preprocess(trace_in, trace_out),
                      edges=edges[index], amplitudes=amplitudes[index],
                      **edge_kw)

        psh.map(digitize_edges, self._raw_key)

        if self._sample_dim == 'time':
            edges *= self.sampling_period

        return self._shape_edge_array(
            edges, amplitudes,
            {'trainId': self._raw_key.train_id_coordinates()},
            labelled, squeeze_edges)

    def pulse_edges(self, pulse_dim='pulseId', edge_func=None, max_edges=10,
                    parallel=None, **edge_kw):
        """Load data and find signal edges by pulse.

        This method performs the edge discrimination step while loading
        and the data and separating it into pulses, and only returns
        those results. It is therefore significantly more memory
        efficient than performing these operations sequentially from
        memory.

        Please see [pulse_data()][extra.components.AdqRawChannel.pulse_data]
        and [find_edges()][extra.components.AdqRawChannel.find_edges]
        for more details.

        Args:
            pulse_dim ('pulseId' or 'pulseIndex' or 'pulseTime', optional):
                Label for pulse dimension, pulse ID by default.
            edge_func (callable, optional): Edge finding method to run
                on each train trace, extra.signal.dled by default.
            max_edges (int, optional): Maximal number of edges per
                train, 1/100 of trace length per pulse by default.
            parallel (int or None, optional): Nunmber of parallel
                processes to use, by default 10 or a quarter of all cores
                whichever is lower. Any non-positive value or 1 disable
                parallelization.
            **edge_kw (Any): Any further keyword arguments are passed to
                the edge finding method.

        Returns:
            result (pd.DataFrame): Edge positions and pulse heights.
        """

        edges, amplitudes = self.pulse_edge_array(
            False, False, pulse_dim, edge_func, max_edges, parallel, **edge_kw)

        return self._shape_edges(
            edges, amplitudes, self._pulses.build_pulse_index(pulse_dim))

    def pulse_edge_array(self, labelled=True, squeeze_edges=True,
                         pulse_dim='pulseId', edge_func=None, max_edges=10,
                         parallel=None, **edge_kw):
        """Load data and find signal edges by pulse as ragged array.

        Alternative method to
        [pulse_edges][extra.components.AdqRawChannel.pulse_edges]
        returning the results as ragged arrays, using `np.nan` as filler
        value.

        Args:
            labelled (bool, optional): Whether data is returned as a
                labelled xarray (default) or unlabelled ndarray.
            squeeze_edges (bool, optional): Whether to minimize the edge
                axis length to the maxinum number of edges found per
                row, True by default.
            pulse_dim ('pulseId' or 'pulseIndex' or 'pulseTime', optional):
                Label for pulse dimension, pulse ID by default.
            edge_func (callable, optional): Edge finding method to run
                on each train trace, extra.signal.dled by default.
            max_edges (int, optional): Maximal number of edges per
                train, 1/100 of trace length by pulse by default.
            parallel (int or None, optional): Nunmber of parallel
                processes to use, by default 10 or a quarter of all cores
                whichever is lower. Any non-positive value or 1 disable
                parallelization.
            **edge_kw (Any): Any further keyword arguments are passed to
                the edge finding method.

        Returns:
            result (numpy.ndarray, numpy.ndarray): Tuple of edge
                positions and amplitudes, only if not labelled.

            result (xarray.Dataset): Edge positions and pulse heights,
                only if labelled.
        """

        edge_func = self._validate_edge_method(edge_func, edge_kw)

        # Drop empty trains for efficient access to train IDs.
        raw_key = self._raw_key.drop_empty_trains()

        # Obtain information about pulse layout.
        pulses, pulse_layout = self._prepare_pulses(raw_key.train_ids)
        pulse_ids = pulses.pulse_ids(labelled=False)

        if max_edges is None:
            max_edges = pulse_layout['length'].max() // 100

        # Set-up pasha for parallel processing.
        psh = self._prepare_pasha(parallel)
        edges = psh.alloc(shape=(len(pulse_ids), max_edges),
                          dtype=np.float32, fill=np.nan)
        amplitudes = psh.alloc(like=edges, fill=np.nan)

        # These buffers are only used locally and not in shared memory!
        trace_out = np.zeros(raw_key.entry_shape, dtype=np.float32)
        trace_split = np.zeros(
            (pulse_layout['count'].max(), pulse_layout['length'].max()),
            dtype=np.float32)

        def digitize_hits(wid, train_index, train_id, trace_in):
            _, first, last, pulse_len = pulse_layout.iloc[train_index]

            self._reshape_flat_pulses(
                self._preprocess(trace_in, trace_out)[np.newaxis, :],
                trace_split, pulse_ids[first:last], pulse_len)

            # No need to slice trace_split to the actual number of
            # pulses, as first:last limits the loop. It's still
            # important to limit the sample axis to the actual length
            # of pulses in this train, as the reshape above would leave
            # any samples after the actual length untouched from an
            # earlier train with potentially longer pulses.
            for signal, pulse_index in zip(trace_split, range(first, last)):
                edge_func(signal=signal[:pulse_len], edges=edges[pulse_index],
                          amplitudes=amplitudes[pulse_index], **edge_kw)

        psh.map(digitize_hits, raw_key)

        if self._sample_dim == 'time':
            edges *= self.sampling_period

        return self._shape_edge_array(
            edges, amplitudes,
            {'pulse':  pulses.build_pulse_index(pulse_dim)},
            labelled, squeeze_edges)
