
from scipy.constants import speed_of_light
import numpy as np
import pandas as pd

from extra_data import SourceData, KeyData, by_id
from .utils import _identify_instrument, _instrument_to_sase, \
    _select_subcomponent_trains
from .pulses import XrayPulses, OpticalLaserPulses


class OpticalLaserDelay:
    """An interface to pump-probe delay with optical lasers.

    In pump-probe experiments, it is generally critical to sort data by
    the time delay between the X-ray and optical laser pulses. At
    European XFEL, there are generally three sources of laser delay:

    * Electronic trigger delay for the laser system, typically used for
      coarse delays of picoseconds.

    * Stage delay from the motor movement of an optical delay line,
      typically used for delays of few or below picoseconds.

    * Actual time jitter per pulse from the bunch arrival monitor for
      corrections generally of a few femtoseconds or below.

    This component allows to retrieve these sources individually,
    side-by-side or combine them into a single total delay per pulse or
    train. In order to make any of these delay values easier to
    interpret, their measured values can be offsetted. A typical
    procedure is to use the measured values at temporal overlap as
    reference offsets.

    The total delay $\\Delta t$ is then calculated by:

    $$
    \\Delta t = (\\Delta t_{\rm stage} - r_{\rm stage})
      - (\\Delta t_{\rm trigger} - r_{\rm trigger})
      + (\\Delta t_{bam} - r_{\rm bam})
    $$

    When constructed, it will try to find the appropriate delay sources
    automatically for the instrument the data was taken at,
    unless they are explicitly specified or disabled by passing `False`.

    Note that BAM data is only available with pulse pattern information.

    Args:
        data (extra.data.DataCollection): Data to access optical laser
            delay information from.
        instrument (str, optional): Instrument to pick default source
            names, auto-detected from data if any source is not
            specified or disabled.
        pulses (extra.components.pulses.PulsePattern, optional): Pulse
            component to pull pulse information. If omitted and BAM
            delay is enabled, it is constructed automatically from
            data passed.
        unit (str, optional): Time unit symbol to express delays in,
            may be `s` (seconds) or any metric prefix up to `as`
            (attoseconds) with `ps` (picoseconds) by default.
        stage (str or bool, optional): Source name for stage motor, may
            be disabled by passing False or auto-detected if omitted.
        trigger (str or bool, optional): Source name for optical laser
            trigger, may be disabled by passing False or auto-detected
            if omitted.
        bam (str or bool, optional): Source name for bunch arrival
            monitor, may be disabled by passing False or auto-detected
            if omitted.
        ref_trigger (float, optional): Reference offset for trigger
            delay in source units, 0.0 by default.
        ref_stage (float, optional): Reference offset for stage delay
            in source units, 0.0 of by default.
        ref_bam (float, optional): Reference offset for BAM delay in
            source units, 0.0 by default.
        ref_delay (float, optional): Reference offset for total delay
            in chosen time unit (`ps` by default), 0.0 by default.
    """

    _time_scales = {'s': 1.0, 'ms': 1e3, 'us': 1e6, 'μs': 1e6,
                    'ns': 1e9, 'ps': 1e12, 'fs': 1e15, 'as': 1e18}

    _default_sources = {
        'MID': {'trigger': 'MID_LAS_COM/CTRL/PPL_PHASE_SHIFTER',
                'stage': 'MID_LAS_COM/MOTOR/DL800',
                'bam': 'XFEL_SDIAG_BAM/DOOCS/1932M_TL:output'},

        'SQS': {'trigger': 'LA3_LAS_PPL/CTRL/TRGDLY',
                'stage': 'SQS_ILH_LAS/MOTOR/DELAY_AX_800',
                'bam': 'SA3_XTD10_DOOCS/BAM/LCAT1932M_TL:output'}}

    def __init__(self, data, instrument=None, pulses=None, unit='ps',
                 trigger=None, stage=None, bam=None,
                 ref_trigger=0.0, ref_stage=0.0, ref_bam=0.0, ref_delay=0.0):

        if unit not in self._time_scales:
            raise ValueError('unknown time unit, please pass any of: ' +
                             ', '.join(self._time_scales.keys()))

        self._time_unit = unit

        if stage is None or trigger is None or bam is None:
            # Instrument auto-detection is only required if any delay
            # source is unspecified.
            instrument = instrument or _identify_instrument(data)

            if instrument is None:
                raise ValueError('instrument could not be detected '
                                 'automatically, please pass explicitly')
        else:
            instrument = ''  # No instrument actually required

        # Validate passed data sources.

        default_sources = self._default_sources.get(instrument, {})
        trigger = self._try_delay_source(
            data, trigger, default_sources, 'trigger')
        stage = self._try_delay_source(
            data, stage, default_sources, 'stage')
        bam = self._try_delay_source(
            data, bam, default_sources, 'bam')

        # Set of actually used sources.
        used_sources = {stage, trigger, bam} - {False}

        if not used_sources:
            # Must use at least one source.
            raise ValueError('no delay sources passed, enabled or available '
                             'with defaults, please pass explicitly')

        if bam and pulses is None:
            # Pulse information is required, but not passed explicitly.
            pulses = XrayPulses(data, sase=_instrument_to_sase(instrument))
            used_sources.add(pulses.source.source)

        # Select down to trains with data for all used delay sources.
        data = data.select(used_sources, require_all=True)

        if pulses is not None:
            pulses = pulses.select_trains(by_id[data.train_ids])

        # Create source and key objects.
        if trigger:
            self._trigger_src = data[trigger]

            if 'currentOpticalDelay' in self._trigger_src:
                # PplDelayControlML
                self._trigger_delay_key = self._trigger_src[
                    'currentOpticalDelay']
            elif 'actualPosition' in self._trigger_src:
                # DoocsOpticalDelay
                self._trigger_delay_key = self._trigger_src['actualPosition']
            else:
                raise ValueError('unsupported source type for trigger delay')

            if (unit := self._trigger_delay_key.units) not in {None, 'ps'}:
                raise ValueError(f'unexpected trigger delay unit `{unit}`')
        else:
            self._trigger_src = self._trigger_delay_key = None

        if stage:
            self._stage_src = data[stage]
            self._stage_delay_key = self._stage_src['actualPosition']

            if (unit := self._stage_delay_key.units) not in {None, 'mm'}:
                raise ValueError(f'unexpected stage delay unit `{unit}`')
        else:
            self._stage_src = self._stage_delay_key = None

        if bam:
            self._bam_src = data[bam]

            if 'data.absoluteTD' in self._bam_src:
                # Modern key.
                self._bam_delay_key = self._bam_src['data.absoluteTD']
            elif 'data.lowChargeArrivalTime' in self._bam_src:
                # Legacy key.
                self._bam_delay_key = self._bam_src[
                    'data.lowChargeArrivalTime']
            else:
                raise ValueError('unsupported source type for bam delay')

            if (unit := self._bam_delay_key.units) not in {None, 'fs'}:
                raise ValueError(f'unexpected BAM delay unit `{unit}`')
        else:
            self._bam_src = self._bam_delay_key = None

        # Disable BAM again if the *wrong* pulse component is used.
        if bam and isinstance(pulses, OpticalLaserPulses):
            import sys
            print(f'{self.__class__.__name__}.__init__: The passed pulse '
                  f'informations indicate they belong to an optical laser. As '
                  f'the optical laser lacks absolute pulse timing, it cannot '
                  f'be correlated to BAM corrections. Please use pulse '
                  f'informations including FEL pulses, such as those provided '
                  f'by the XrayPulses or PumpProbePulses components.',
                  file=sys.stderr)
            self._bam_src = self._bam_delay_key = None  # Disable BAM.

        self._pulses = pulses
        self._trigger_ref = ref_trigger
        self._stage_ref = ref_stage
        self._bam_ref = ref_bam
        self._delay_ref = ref_delay

    @classmethod
    def _try_delay_source(cls, data, source, defaults, name):
        if source is False:
            # Explicitly disabled.
            return False

        if source is None:
            # Auto-detection.
            if (default := defaults.get(name)) is not None:
                # Existing default.
                source = default
            else:
                # No default, disable.
                return False

        if source not in data.all_sources:
            # Source missing.
            raise ValueError(f'expected source {source} not found in data, '
                             f'please explicit {name} source or disable with '
                             '{name}=False')

        return source

    def _get_train_delays(self, labelled, by_pulse, pulse_dim, kd, ref, scale):
        delays = (kd.ndarray() - ref) * scale

        if by_pulse:
            if self._pulses is None:
                raise ValueError('component is not initialized with pulse '
                                 'information')

            delays = delays.repeat(self._pulses.pulse_counts())

            if labelled:
                # If labelled, build a series now from the ndarray.
                delays = pd.Series(
                    delays, index=self._pulses.build_pulse_index(pulse_dim))

        elif labelled:
            # Labelled series, but not by pulse.
            delays = pd.Series(
                delays,
                index=pd.Index(kd.train_id_coordinates(), name='trainId'))

        return delays

    def __repr__(self):
        use_labels = []

        if self._trigger_src is not None:
            use_labels.append(f'trigger={self._trigger_src.source}'
                              f'[ref={self._trigger_ref}]')

        if self._stage_src is not None:
            use_labels.append(f'stage={self._stage_src.source}'
                              f'[ref={self._stage_ref}]')

        if self._bam_src is not None:
            use_labels.append(f'bam={self._bam_src.source}'
                              f'[ref={self._bam_ref}]')

        if self._pulses is not None:
            use_labels.append(f'pulses={repr(self._pulses)}')

        return '<{} using\n  {}>'.format(
            type(self).__name__, '\n  '.join(use_labels))

    @property
    def _stage_to_time(self) -> float:
        """millimeter to seconds"""
        return 2e-3 / speed_of_light

    @property
    def time_scale(self) -> float:
        """seconds to internal time unit."""
        return self._time_scales[self._time_unit]

    def select_trains(self, trains):
        return _select_subcomponent_trains(self, [
            '_trigger_src', '_trigger_delay_key',
            '_stage_src', '_stage_delay_key',
            '_bam_src', '_bam_delay_key',
            '_pulses'
        ])

    def _trigger_delays(self, labelled=True, by_pulse=False,
                        pulse_dim='pulseId'):
        """Get time delay from electronic trigger.

        Args:
            labelled (bool, optional): Whether a labelled pandas Series
                (default) or unlabelled numpy array is returned.
            by_pulse (bool, optional): Whether data is returned by pulse
                or not (default). As the trigger delay is only recorded
                by train, identical values are returned for every pulse
                if enabled.
            pulse_dim ({pulseId, pulseIndex, pulseTime}, optional):
                Label for pulse dimension, pulse ID by default.

        Returns:
            (pandas.Series or numpy.ndarray): Time delay from electronic
                trigger, indexed by train ID or pulse index if labelled.
        """

        return self._get_train_delays(
            labelled, by_pulse, pulse_dim,
            self._trigger_delay_key, self._trigger_ref, 1e-12*self.time_scale)

    def _stage_delays(self, labelled=True, by_pulse=False, pulse_dim='pulseId'):
        """Get time delay from delay line motor.

        Args:
            labelled (bool, optional): Whether a labelled pandas Series
                (default) or unlabelled numpy array is returned.
            by_pulse (bool, optional): Whether data is returned by pulse
                or not (default). As the stage delay is only recorded
                by train, identical values are returned for every pulse
                if enabled.
            pulse_dim ({pulseId, pulseIndex, pulseTime}, optional):
                Label for pulse dimension, pulse ID by default.

        Returns:
            (pandas.Series or numpy.ndarray): Time delay from delay line
                motor, indexed by train ID or pulse index if labelled.
        """

        return self._get_train_delays(
            labelled, by_pulse, pulse_dim,
            self._stage_delay_key, self._stage_ref,
            self.time_scale * self._stage_to_time)

    def _bam_delays(self, labelled=True, by_pulse=True, pulse_dim='pulseId'):
        """Get time delay from BAM correction.

        Args:
            labelled (bool, optional): Whether a labelled pandas Series
                (default) or unlabelled numpy array is returned.
            by_pulse (bool, optional): Whether data is returned by pulse
                (default) or not. If disabled, the train value is the
                average over all pulses.
            pulse_dim ({pulseId, pulseIndex, pulseTime}, optional):
                Label for pulse dimension, pulse ID by default.

        Returns:
            (pandas.Series or numpy.ndarray): Time delay from BAM
                correction, indexed by train ID or pulse index if
                labelled.
        """

        bam = self._bam_delay_key.ndarray()
        pids = self._pulses.pulse_ids()  # BAM requires pulses.
        delays = np.zeros_like(pids, dtype=np.float32)

        # TODO: Check for 'fel' index and set rows without fel to NaN.

        start = 0
        for (_, train_pids), train_bam in zip(pids.groupby('trainId'), bam):
            stop = start + len(train_pids)
            delays[start:stop] = train_bam[2 * train_pids] - self._bam_ref
            start = stop

        delays *= (1e-15 * self.time_scale)

        if labelled or not by_pulse:
            delays = pd.Series(
                delays, index=self._pulses.build_pulse_index(pulse_dim))

            if not by_pulse:
                delays = delays.groupby('trainId').mean()

            return delays if labelled else delays.to_numpy()

        else:
            return delays

    def total_delays(self, labelled=True, by_pulse=None, pulse_dim='pulseId'):
        """Get total time delay.

        Args:
            labelled (bool, optional): Whether a labelled pandas Series
                (default) or unlabelled numpy array is returned.
            by_pulse (bool, optional): Whether data is returned by pulse
                or not. By default, it is returned by pulse whenever BAM
                data is enabled and by train otherwise.
            pulse_dim ({pulseId, pulseIndex, pulseTime}, optional):
                Label for pulse dimension, pulse ID by default.

        Returns:
            (pandas.Series or numpy.ndarray): Total time delay, indexed
                by train ID or pulse index if labelled.
        """

        delays = -self._delay_ref

        if by_pulse is None:
            by_pulse = self._bam_src is not None

        if self._stage_src is not None:
            delays += self._stage_delays(labelled, by_pulse, pulse_dim)

        if self._trigger_src is not None:
            delays -= self._trigger_delays(labelled, by_pulse, pulse_dim)

        if self._bam_src is not None:
            delays += self._bam_delays(labelled, by_pulse, pulse_dim)

        return delays

    def delays_by_source(self, by_pulse=None, pulse_dim='pulseId'):
        """Get all time delays by source as a dataframe.

        Args:
            by_pulse (bool, optional): Whether data is returned by pulse
                or not. By default, it is returned by pulse whenever BAM
                data is enabled and by train otherwise.
            pulse_dim ({pulseId, pulseIndex, pulseTime}, optional):
                Label for pulse dimension, pulse ID by default.

        Returns:
            (pandas.DataFrame): Time delay by source, indexed by train
                ID or pulse index.
        """

        columns = {}

        if by_pulse is None:
            by_pulse = self._bam_src is not None

        if self._trigger_src is not None:
            columns['trigger'] = self._trigger_delays(True, by_pulse, pulse_dim)

        if self._stage_src is not None:
            columns['stage'] = self._stage_delays(True, by_pulse, pulse_dim)

        if self._bam_src is not None:
            columns['bam'] = self._bam_delays(True, by_pulse, pulse_dim)

        return pd.DataFrame(columns)
