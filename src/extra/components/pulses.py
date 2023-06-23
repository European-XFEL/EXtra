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

    # An implementation of this base class is required to implement the
    # _get_pulse_mask() method.

    # Regular expressions for timeserver control and pipeline data.
    _timeserver_control_re = re.compile(
        r'^[A-Z]{3}_(BR|RR)_(UTC|SYS)/TSYS/TIMESERVER$')
    _timeserver_pipeline_re = re.compile(r'^{}:outputBunchPattern'.format(
        _timeserver_control_re.pattern[:-1]))

    # Class IDs for timeserver devices.
    _timeserver_class = 'TimeServer'

    def __init__(self, data, timeserver=None, sase=None):
        if timeserver is None:
            timeserver = self._find_timeserver(data)

        self._source = data[timeserver]

        if ':' in timeserver:
            self._key = self._source['data.bunchPatternTable']
        else:
            self._key = self._source['bunchPatternTable']

    @classmethod
    def _find_timeserver(cls, data):
        """Try to find a timeserver source."""

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

        raise ValueError('no timeserver found, please pass one explicitly')

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

    def _get_pulse_mask(self, table):
        """Generate pulse mask from a given bunch pattern table."""
        raise NotImplementedError('_get_pulse_mask')

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
        return self._source

    @property
    def bunch_pattern_table(self) -> KeyData:
        """Used bunch pattern table key."""
        return self._key

    def select_trains(self, trains):
        """Select a subset of trains in this data.

        This method accepts the same type of arguments as
        extra_data.DataCollection.select_trains()
        """

        res = copy(self)
        res._source = self._source.select_trains(trains)
        res._key = self._key.select_trains(trains)

        return res

    def is_constant_pattern(self):
        """Whether pulse IDs are constant in this data.

        Args:
            None

        Returns:
            (bool): Whether pulse IDs are identical in every train.
        """

        pulse_mask = self._get_pulse_mask(self._key.ndarray())
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

        counts = self._get_pulse_mask(self._key.ndarray()).sum(axis=1)

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
        PulsePattern.get_pulse_ids by only reading the bunch pattern
        table for the very first train of this data.

        Args:
            None

        Returns:
            (numpy.ndarray): Pulse IDs in the first train of this data.

        """

        return self._get_pulse_mask(self._key[0].ndarray()[0]).nonzero()[0]

    def get_pulse_ids(self, labelled=True):
        """Get pulse IDs.

        Args:
            labelled (bool, optional): Whether a labelled pandas Series
                (default) or unlabelled numpy array is returned.

        Returns:
            (pandas.Series or numpy.ndarray): Pulse ID indexed by train
                ID and pulse number if labelled is True.
        """

        pulse_mask = self._get_pulse_mask(self._key.ndarray())
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

        Args:
            None

        Returns:
            (pandas.MultiIndex): Multi-level index covering train ID and
                pulse ID or pulse number.
        """

        return self._make_pulse_index(
            self._get_pulse_mask(self._key.ndarray()))

    def search_pulse_patterns(self):
        """Search identical pulse patterns in this data.

        Reads the bunch pattern table and gathers contiguous train
        regions of constant pulse pattern. It returns a list of train
        slices and correspondig pulse IDs.

        Args:
            None

        Returns:
            (list of (slice, ndarray) tuples): List of train regions
                identified by index slices with identical pulse IDs.
        """

        pulse_mask = self._get_pulse_mask(self._key.ndarray())

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

        for train_id, table in self._key.trains():
            yield train_id, self._get_pulse_mask(table).nonzero()[0]


class XrayPulses(PulsePattern):
    """An interface to X-ray free electron laser pulses.

    The pulse structure of each train at European XFEL is described by
    the bunch pattern table and accesssible in recorded data through the
    timeserver device.

    This component aids in locating and reading the bunch pattern table,
    as well as providing utility methods to apply the pulse patterns to
    other recorded data. It only considers the X-ray laser pulses
    generated by one of the SASE beamlines and is thus a good choice for
    exclusive use of X-rays or pump-probe experiments with congruent
    optical laser pulses.

    For specific access to pulses from one of the optical laser sources,
    please see the almost correspondig `OpticalLaserPulses` component
    with the same interface.

    This class only deals with X-ray pulses of a particular SASE
    beamline, please see `OpticalLaserPulses` to access pulses of the
    optical laser sources.

    Args:
        data (extra.data.DataCollection): Data to access bunch pattern
            data from.
        timeserver (str, optional): Source name of a timeserver, only
            needed if the data includes more than one timeserver or it
            could not be detected automatically.
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

    def __init__(self, data, timeserver=None, sase=None):
        super().__init__(data, timeserver)

        if sase not in {1, 2, 3}:
            sase = self._identify_sase(data)

        self._sase = sase

    def __repr__(self):
        return "<{} for SA{} using timeserver={}>".format(
            type(self).__name__, self._sase, self._source.source)

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

    def _get_pulse_mask(self, table):
        return is_sase(table, sase=self._sase)

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
    recommended to just use the `XrayPulses` component.

    Args:
        data (extra.data.DataCollection): Data to access bunch pattern
            data from.
        timeserver (str, optional): Source name of a timeserver, only
            needed if the data includes more than one timeserver or it
            could not be detected automatically.
        ppl_seed (extra.components.pulses.PPL_BITS or str, optional):
            PPL seed to interpret pulses of, only needed if the data
            includes sources from more than one instrument or it could
            not be detected automatically. May either be an explicit
            seed value or an instrument as a string.
    """

    _instrument_ppl_seeds = {
        'FXE': PPL_BITS.LP_FXE,
        'SPB': PPL_BITS.LP_SPB,
        'MID': PPL_BITS.LP_SASE2,
        'HED': PPL_BITS.LP_SASE2,
        'SCS': PPL_BITS.LP_SCS,
        'SQS': PPL_BITS.LP_SQS
    }

    def __init__(self, data, timeserver=None, ppl_seed=None):
        super().__init__(data, timeserver)

        if ppl_seed is None:
            ppl_seed = self._identify_ppl_seed(data)
        elif isinstance(ppl_seed, str):
            try:
                ppl_seed = self._instrument_ppl_seeds[ppl_seed]
            except KeyError:
                raise ValueError(f'no PPL seed known associated to '
                                 f'{ppl_seed}') from None

        self._ppl_seed = ppl_seed

    def __repr__(self):
        return "<{} for {} using timeserver={}>".format(
            type(self).__name__, self._ppl_seed.name, self._source.source)

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

    def _get_pulse_mask(self, table):
        return is_laser(table, laser=self._ppl_seed)

    @property
    def ppl_seed(self) -> Optional[PPL_BITS]:
        """Used laser seed."""
        return self._ppl_seed
