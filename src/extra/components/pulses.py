"""Interface and utilies to work with pulse-resolved data."""


from copy import copy
from typing import Optional
import re

import numpy as np

from euxfel_bunch_pattern import PPL_BITS, is_sase, is_laser
from extra_data import SourceData, KeyData, by_id


try:
    from itertools import pairwise
except ImportError:
    # Compatibility for Python < 3.10
    from itertools import tee

    def pairwise(iterable):
        # pairwise('ABCDEFG') --> AB BC CD DE EF FG
        a, b = tee(iterable)
        next(b, None)
        return zip(a, b)


__all__ = ['XrayPulses', 'OpticalLaserPulses']


class PulsePattern:
    """Abstract interface to pulse patterns.

    This class should not be instantiated directly, but one of its
    implementationsd `XrayPulses` or `OpticalLaserPulses`. It provides
    the shared interface to access pulse patterns encoded in the bunch
    pattern table.
    """

    # All methods are built on top of get_pulse_mask and trains(). Their
    # default implementations require implementation of  _mask_table()
    # and _get_ppdecoder_node().

    # Number of elements in bunch pattern table according to XFEL Timing
    # System Specification, Version 2.2 (2013). The original table may
    # have up to 7222 entries at 9 MHz with the Karabo Timeserver device
    # only forwarding the even places at 4.5 MHz.
    _bunch_pattern_table_len = 3611

    # Timeserver class ID and regular expressions.
    _timeserver_class = 'TimeServer'
    _timeserver_control_re = re.compile(
        r'^\w{3}_(BR|RR)_(UTC|SYS)\/TSYS\/TIMESERVER$')
    _timeserver_pipeline_re = re.compile(r'^{}:outputBunchPattern'.format(
        _timeserver_control_re.pattern[:-1]))

    # Pulse pattern decoder class ID and regular expression.
    _ppdecoder_class = 'PulsePatternDecoder'
    _ppdecoder_re = re.compile(
        r'^\w{3}_(BR|RR)_(UTC|SYS)\/(MDL|TSYS)\/'
        r'(BUNCH|PULSE|PP)\w*_(DECODER|PATTERN)$')

    def __init__(self, data, source=None, sase=None):
        if source is None:
            source = self._find_pulsepattern_source(data)

        self._source = data[source]

        if 'maindump.pulseIds.value' in self._source.keys():
            # PulsePatternDecoder source.
            self._with_timeserver = False

            # TODO: SourceData.train_id_coordinates() would make this
            # redundant.
            self._key = self._source['maindump.pulseIds']
        else:
            # Timeserver source.
            self._with_timeserver = True

            if ':' in source:
                self._key = self._source['data.bunchPatternTable']
            else:
                self._key = self._source['bunchPatternTable']

    @classmethod
    def _find_pulsepattern_source(cls, data):
        """Try to find a pulse pattern source."""

        # Try to go by device class first.
        # By the time the device class was recorded, time servers also
        # were changed to output the bunch pattern via a pipeline.
        timeserver_sources = {
            f'{source}:outputBunchPattern'
            for source in data.control_sources
            if (data[source].device_class == cls._timeserver_class and
                f'{source}:outputBunchPattern' in data.instrument_sources)
        }

        if len(timeserver_sources) > 1:
            raise ValueError('multiple timeserver sources found via device '
                             'class, please pass one explicitly:\n' +
                             ', '.join(sorted(timeserver_sources)))
        elif timeserver_sources:
            return timeserver_sources.pop()

        # Next check for timeserver instrument data.
        for source in data.instrument_sources:
            m = cls._timeserver_pipeline_re.match(source)
            if m is not None:
                timeserver_sources.add(m[0])

        if len(timeserver_sources) > 1:
            raise ValueError('multiple timeserver instrument sources found, '
                             'please pass one explicitly:\n' + ', '.join(
                                sorted(timeserver_sources)))
        elif timeserver_sources:
            return timeserver_sources.pop()

        # Last chance for timeserver control data, likely
        # empty for more recent data after 2020.
        for source in data.control_sources:
            m = cls._timeserver_control_re.match(source)
            if m is not None:
                timeserver_sources.add(m[0])

        if len(timeserver_sources) > 1:
            raise ValueError('multiple timeserver control sources found, '
                             'please pass one explicitly:\n' + ', '.join(
                                sorted(timeserver_sources)))
        elif timeserver_sources:
            return timeserver_sources.pop()

        # Try to go by device class first.
        ppdecoder_sources = {
            source
            for source in data.control_sources
            if data[source].device_class == cls._ppdecoder_class
        }

        if len(ppdecoder_sources) > 1:
            raise ValueError('multiple ppdecoder sources found via device '
                             'class, please pass one explicitly:\n' +
                             ', '.join(sorted(ppdecoder_sources)))
        elif ppdecoder_sources:
            return ppdecoder_sources.pop()

        # Try again by source regexp.
        for source in data.control_sources:
            m = cls._ppdecoder_re.match(source)
            if m is not None:
                ppdecoder_sources.add(m[0])

        if len(ppdecoder_sources) > 1:
            raise ValueError('multiple ppdecoder control sources found, '
                             'please pass one explicitly:\n' + ', '.join(
                                sorted(ppdecoder_sources)))
        elif ppdecoder_sources:
            return ppdecoder_sources.pop()

        raise ValueError('no timeserver or ppdecoder found, please pass '
                         'one explicitly')

    def _make_pulse_index(self, pulse_mask, with_pulse_id=True):
        """Generate multi index for a given pulse mask."""

        pulse_counts = pulse_mask.sum(axis=1)
        train_ids = np.repeat(self._key.train_id_coordinates(), pulse_counts)

        if with_pulse_id:
            train_pulses = [mask.nonzero()[0] for mask in pulse_mask]
            pulses_label = 'pulseId'
        else:
            train_pulses = [np.arange(count) for count in pulse_counts]
            pulses_label = 'pulseNumber'

        import pandas as pd
        return pd.MultiIndex.from_arrays(
            [train_ids, np.concatenate(train_pulses)],
            names=['trainId', pulses_label])

    def _mask_table(self, table):
        """Mask bunch pattern table."""
        raise NotImplementedError('_mask_table')

    def _get_ppdecoder_node(self):
        """Get node in pulse pattern decoder device."""
        raise NotImplementedError('_get_ppdecoder_node')

    @property
    def master_clock(self) -> float:
        """European XFEL timing system master clock in Hz."""
        return 1.3e9

    @property
    def bunch_clock_divider(self) -> int:
        """Divider to generate repetition rate from master clock."""
        return 288

    @property
    def bunch_repetition_rate(self) -> float:
        """European XFEL fundamental bunch repetition rate in Hz.

        Generated from the master clock using a divider of 288,
        resulting in 4.5 MHz.
        """
        return self.master_clock / self.bunch_clock_divider

    @property
    def timeserver(self) -> SourceData:
        """Used timeserver source."""

        if not self._with_timeserver:
            raise ValueError('component is initialized with ppdecoder source, '
                             'timeserver not available')
        return self._source

    @property
    def pulse_pattern_decoder(self) -> SourceData:
        """Used PulsePatternDecoder source."""

        if self._with_timeserver:
            raise ValueError('component is initialized with timeserver '
                             'source, ppdecoder not available')
        return self._source

    @property
    def bunch_pattern_table(self) -> KeyData:
        """Used bunch pattern table key."""

        if not self._with_timeserver:
            raise ValueError('component is initialized with ppdecoder source, '
                             'bunch pattern table not available')

        return self._key

    def select_trains(self, trains):
        """Select a subset of trains in this data.

        This method accepts the same type of arguments as
        [DataCollection.select_trains()][extra_data.DataCollection.select_trains].
        """

        res = copy(self)
        res._source = self._source.select_trains(trains)
        res._key = self._key.select_trains(trains)

        return res

    def get_pulse_mask(self, labelled=True):
        """Get boolean pulse mask.

        The returned mask has the same shape as the full bunch pattern
        table but only contains boolean flags whether a given pulse
        was present in this pattern.

        Args:
            labelled (bool, optional): Whether a labelled xarray
                DataArray (default) or unlabelled numpy array is
                returned.

        Returns:
            (xarray.DataArray or numpy.ndarray):

        Returns:
            (numpy.ndarray):
        """

        if self._with_timeserver:
            mask = self._mask_table(self._key.ndarray())
        else:
            node = self._get_ppdecoder_node()
            mask = np.zeros(
                (len(self._source.train_ids), self._bunch_pattern_table_len),
                dtype=bool)

            for i, (pulse_ids, num_pulses) in enumerate(zip(
                self._source[f'{node}.pulseIds'].ndarray(),
                self._source[f'{node}.nPulses'].ndarray()
            )):
                mask[i, pulse_ids[:num_pulses]] = True

        if labelled:
            import xarray as xr
            return xr.DataArray(
                mask,
                dims=['trainId', 'pulseId'],
                coords={'trainId': self._key.train_id_coordinates(),
                        'pulseId': np.arange(mask.shape[1])})
        else:
            return mask

    def is_constant_pattern(self):
        """Whether pulse IDs are constant in this data.

        Returns:
            (bool): Whether pulse IDs are identical in every train.
        """

        pulse_mask = self.get_pulse_mask(labelled=False)
        return (pulse_mask == pulse_mask[0]).all()

    def get_pulse_counts(self, labelled=True):
        """Get number of pulses per train.

        Args:
            labelled (bool, optional): Whether a labelled pandas Series
                (default) or unlabelled numpy array is returned.

        Returns:
            (pandas.Series or numpy.ndarray): Number of pulses per
                train, indexed by train ID if labelled is True.
        """

        counts = self.get_pulse_mask(labelled=False).sum(axis=1)

        if labelled:
            import pandas as pd
            return pd.Series(
                data=counts, dtype=np.int32,
                index=self._key.train_id_coordinates())
        else:
            return counts

    def peek_pulse_ids(self):
        """Get pulse IDs for the first train.

        This method is a faster alternative to
        `get_pulse_ids()` by only reading the bunch pattern
        table for the very first train of this data.

        Returns:
            (numpy.ndarray): Pulse IDs in the first train of this data.

        """

        first_tid_self = self.select_trains(by_id[
            self._source.drop_empty_trains().select_trains(np.s_[0]).train_ids
        ])

        return first_tid_self.get_pulse_mask(labelled=False)[0].nonzero()[0]

    def get_pulse_ids(self, labelled=True):
        """Get pulse IDs.

        Args:
            labelled (bool, optional): Whether a labelled pandas Series
                (default) or unlabelled numpy array is returned.

        Returns:
            (pandas.Series or numpy.ndarray): Pulse ID indexed by train
                ID and pulse number if labelled is True.
        """

        pulse_mask = self.get_pulse_mask(labelled=False)
        pulse_ids = np.concatenate([mask.nonzero()[0] for mask in pulse_mask])

        if labelled:
            import pandas as pd
            return pd.Series(
                data=pulse_ids, dtype=np.uint32,
                index=self._make_pulse_index(pulse_mask, with_pulse_id=False))
        else:
            return pulse_ids

    def get_pulse_index(self):
        """Get a multi-level index for pulse-resolved data.

        Returns:
            (pandas.MultiIndex): Multi-level index covering train ID and
                pulse ID or pulse number.
        """

        return self._make_pulse_index(self.get_pulse_mask(labelled=False))

    def search_pulse_patterns(self):
        """Search identical pulse patterns in this data.

        Reads the bunch pattern table and gathers contiguous train
        regions of constant pulse pattern. It returns a list of train
        slices and corresponding pulse IDs.

        Returns:
            (list of (slice, ndarray) tuples): List of train regions
                identified by index slices with identical pulse IDs.
        """

        pulse_mask = self.get_pulse_mask(labelled=False)

        # Find the unique patterns and the respective indices for each
        # unique pattern.
        unique_patterns, pattern_indices = np.unique(
            pulse_mask, axis=0, return_inverse=True)

        # Find positions of pattern changes plus beginning and end.
        pattern_changes = np.concatenate([
            [-1],
            (pattern_indices[1:] - pattern_indices[:-1]).nonzero()[0],
            [len(pulse_mask)]])

        tids = self._key.train_id_coordinates()
        one = np.uint64(1)  # Avoid conversion to float64.

        def gen_slice(start, stop):
            return by_id[tids[start]:tids[stop-1]+one]

        # Build list of (train_slice, pattern) tuples.
        patterns = [
            (gen_slice(start+1, stop), pulse_mask[start+1].nonzero()[0])
            for start, stop in pairwise(pattern_changes)]

        return patterns

    def trains(self):
        """Iterate over pulse IDs by train.

        Yields:
            (int, ndarray): Train ID and pulse IDs.
        """

        if self._with_timeserver:
            for train_id, table in self._key.trains():
                yield train_id, self._mask_table(table).nonzero()[0]
        else:
            node = self._get_ppdecoder_node()

            # TODO: SourceData.trains()
            for (train_id, pulse_ids), (_, num_pulses) in zip(
                self._source[f'{node}.pulseIds'].trains(),
                self._source[f'{node}.nPulses'].trains()
            ):
                yield train_id, pulse_ids[:num_pulses]


class XrayPulses(PulsePattern):
    """An interface to X-ray free electron laser pulses.

    The pulse structure of each train at European XFEL is described by
    the bunch pattern table and accesssible in recorded data through the
    timeserver device or in decoded form through pulse pattern decoders.

    This component aids in locating and reading the bunch pattern table,
    as well as providing utility methods to apply the pulse patterns to
    other recorded data. It only considers the X-ray laser pulses
    generated by one of the SASE beamlines and is thus a good choice for
    exclusive use of X-rays or pump-probe experiments with congruent
    optical laser pulses.

    For specific access to pulses from one of the optical laser sources, please
    see the almost corresponding
    [OpticalLaserPulses][extra.components.OpticalLaserPulses] component with the
    same interface.

    This class only deals with X-ray pulses of a particular SASE beamline,
    please see [OpticalLaserPulses][extra.components.OpticalLaserPulses] to
    access pulses of the optical laser sources.

    Args:
        data (extra.data.DataCollection): Data to access bunch pattern
            data from.
        source (str, optional): Source name of a timeserver or pulse
            pattern decoder, only needed if the data includes more than
            one such device or none could not be detected automatically.
        sase (int, optional): SASE beamline to interpret pulses of, only
            needed if the data includes sources from more than one
            beamline or it could not be detected automatically.

    """

    # Source prefixes in use at each SASE.
    _sase_topics = {
        1: {'SA1', 'LA1', 'SPB', 'FXE'},
        2: {'SA2', 'LA2', 'MID', 'HED'},
        3: {'SA3', 'LA3', 'SCS', 'SQS', 'SXP'}
    }

    def __init__(self, data, source=None, sase=None):
        super().__init__(data, source)

        if sase not in {1, 2, 3}:
            sase = self._identify_sase(data)

        self._sase = sase

    def __repr__(self):
        if self._with_timeserver:
            source_type = 'timeserver'
        else:
            source_type = 'ppdecoder'

        return "<{} for SA{} using {}={}>".format(
            type(self).__name__, self._sase, source_type, self._source.source)

    @classmethod
    def _identify_sase(cls, data):
        """Try to identify which SASE this data belongs to."""

        sases = {sase
                 for src in data.all_sources
                 for sase, topics in cls._sase_topics.items()
                 if src[:src.find('_')] in topics}

        if len(sases) == 1:
            return sases.pop()
        elif sases == {1, 3}:
            # SA3 data often contains one or more SA1 sources
            # from shared upstream components.
            return 3
        else:
            raise ValueError('sources from multiple SASE branches {} found, '
                             'please pass the SASE beamline explicitly'.format(
                                ', '.join(sases)))

    def _mask_table(self, table):
        return is_sase(table, sase=self._sase)

    def _get_ppdecoder_node(self):
        return f'sase{self._sase}'

    @property
    def sase(self) -> int:
        """Used SASE beamline."""
        return self._sase


class OpticalLaserPulses(PulsePattern):
    """An interface to optical laser pulses.

    The pump-probe lasers (LAS or PPL) are optical lasers commonly used
    in conjunction with X-ray pulses for pump-probe experiments. There
    are multiple laser sources called seeds distributed across the SASE
    beamlines and instruments with their pulse patterns also contained
    in the bunch pattern table.

    However, an important difference to the FEL pulses is that only the
    number of pulses and their spacing can be inferred from the bunch
    pattern table. Optical laser pulses **always** starts at offset 0
    for technical reasons, even if they temporally overlap with FEL
    pulses by means of optical delay.

    For experiments where all FEL and PPL laser pulses overlap, it is
    recommended to just use the [XrayPulses][extra.components.XrayPulses]
    component.

    Args:
        data (extra.data.DataCollection): Data to access bunch pattern
            data from.
        source (str, optional): Source name of a timeserver or pulse
            pattern decoder, only needed if the data includes more than
            one such device or none could not be detected automatically.
        ppl_seed (extra.components.pulses.PPL_BITS or str, optional):
            PPL seed to interpret pulses of, only needed if the data
            includes sources from more than one instrument or it could
            not be detected automatically. May either be an explicit
            seed value or an instrument as a string.
    """

    # Mapping of instrument names to PPL seeds.
    _instrument_ppl_seeds = {
        'FXE': PPL_BITS.LP_FXE,
        'SPB': PPL_BITS.LP_SPB,
        'MID': PPL_BITS.LP_SASE2,
        'HED': PPL_BITS.LP_SASE2,
        'SCS': PPL_BITS.LP_SCS,
        'SQS': PPL_BITS.LP_SQS
    }

    def __init__(self, data, source=None, ppl_seed=None):
        super().__init__(data, source)

        if not self._with_timeserver:
            # Pulse pattern decoders are configured for a particular
            # PPL seed at runtime.
            native_seed = PPL_BITS[self._source.run_value('laserSource.value')]

        if ppl_seed is None:
            if self._with_timeserver:
                ppl_seed = self._identify_ppl_seed(data)
            else:
                ppl_seed = native_seed
        elif isinstance(ppl_seed, str):
            try:
                ppl_seed = self._instrument_ppl_seeds[ppl_seed.upper()]
            except KeyError:
                raise ValueError(f'no PPL seed known associated to '
                                 f'{ppl_seed}') from None

            if not self._with_timeserver and native_seed != ppl_seed:
                raise ValueError(f'cannot use {PPL_BITS(ppl_seed).name}, '
                                 f'component is initialized with ppdecoder '
                                 f'using {PPL_BITS(native_seed).name}')

        self._ppl_seed = ppl_seed

    def __repr__(self):
        if self._with_timeserver:
            source_type = 'timeserver'
        else:
            source_type = 'ppdecoder'

        return "<{} for {} using {}={}>".format(
                type(self).__name__, self._ppl_seed.name, source_type,
                self._source.source)

    @classmethod
    def _identify_ppl_seed(cls, data):
        """Try to identify which PPL seed this data belongs to."""

        instruments = {instrument for src in data.all_sources
                       if ((instrument := src[:src.find('_')])
                           in cls._instrument_ppl_seeds.keys())}

        if len(instruments) == 1:
            return cls._instrument_ppl_seeds[instruments.pop()]
        elif len(instruments) > 1:
            raise ValueError('sources from multiple instruments {} found, '
                             'please pass the PPL seed explicitly'.format(
                                ', '.join(instruments)))
        else:
            raise ValueError('no source from known instruments {} found, '
                             'please pass the PPL seed explicitly'.format(
                                ', '.join(cls._instrument_ppl_seeds.keys())))

    def _mask_table(self, table):
        return is_laser(table, self._ppl_seed)

    def _get_ppdecoder_node(self):
        return 'laser'

    @property
    def ppl_seed(self) -> Optional[PPL_BITS]:
        """Used laser seed."""
        return self._ppl_seed
