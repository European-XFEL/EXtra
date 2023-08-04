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


def _drop_first_level(pd_val):
    """Return first group and drop first level of a pandas multi index."""
    return pd_val.loc[pd_val.head(1).index.values[0][0]]


class PulsePattern:
    """Abstract interface to pulse patterns.

    This class should not be instantiated directly, but one of its
    implementationsd `XrayPulses` or `OpticalLaserPulses`. It provides
    the shared interface to access any pulse pattern.

    Requires to implement _get_pulse_ids().
    """

    # Number of elements in bunch pattern table according to XFEL Timing
    # System Specification, Version 2.2 (2013). The original table may
    # have up to 7222 entries at 9 MHz with the Karabo Timeserver device
    # only forwarding the even places at 4.5 MHz.
    _bunch_pattern_table_len = 3611

    def __init__(self, source: SourceData = None, key: KeyData = None):
        self._source = source
        self._key = key

        self._pulse_ids = None

    def _get_train_ids(self):
        # This method turned out to be the fastest to get just the
        # group labels.
        return self._get_pulse_ids().index.to_frame()['trainId'].unique()

    def _get_pulse_ids(self):
        """Low-level access to pulse IDs.

        This method must be overriden by any implementation of this
        class. It is expected to return a pandas series with one entry
        per pulse ID labelled by a multi index of train ID and pulse
        number. Its result will be cached externally.

        Returns:
            (pd.Series) Pulse IDs labelled by train ID and pulse number.
        """
        raise NotImplementedError('_get_pulse_ids')

    def _get_pulse_mask(self, reduced=False):
        """Default implementation using _get_pulse_ids."""

        pulse_ids = self.get_pulse_ids(copy=False)
        pids_by_train = pulse_ids.groupby(level=0)

        if reduced:
            pid_offset = pulse_ids.min()
            table_len = pulse_ids.max() - pid_offset + 1
        else:
            pid_offset = 0
            table_len = self._bunch_pattern_table_len

        mask = np.zeros((pids_by_train.ngroups, table_len), dtype=bool)

        for i, (_, train_pulses) in enumerate(pids_by_train):
            mask[i, train_pulses - pid_offset] = True

        return mask

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

    def select_trains(self, trains):
        new_train_ids = None
        res = copy(self)

        if self._source is not None:
            res._source = self._source.select_trains(trains)
            new_train_ids = res._source.train_ids

        if self._key is not None:
            res._key = self._key.select_trains(trains)
            new_train_ids = res._key.train_ids

        if self._pulse_ids is not None and new_train_ids is not None:
            res._pulse_ids = self._pulse_ids.loc[new_train_ids]
        else:
            res._pulse_ids = None

        return res

    def get_pulse_ids(self, labelled=True, copy=True):
        """Get pulse IDs.

        Args:
            labelled (bool, optional): Whether a labelled pandas Series
                (default) or unlabelled numpy array is returned.
            copy (bool, optional): Whether a copy is returned (default)
                or potentially a reference to an internal object. In
                the latter case, modifying the returned value will
                likely break other methods.

        Returns:
            (pandas.Series or numpy.ndarray): Pulse ID indexed by train
                ID and pulse number if labelled is True.
        """

        if self._pulse_ids is None:
            self._pulse_ids = self._get_pulse_ids()

        pulse_ids = self._pulse_ids if labelled else self._pulse_ids.to_numpy()
        return pulse_ids.copy() if copy else pulse_ids

    def peek_pulse_ids(self, labelled=True):
        """Get pulse IDs for the first train.

        This method may be significantly faster than to
        `get_pulse_ids()` by only reading the bunch pattern table for
        the very first train of this data.

        Args:
            labelled (bool, optional): Whether a labelled pandas Series
                (default) or unlabelled numpy array is returned.

        Returns:
            (pandas.Series or numpy.ndarray): Pulse ID in the first
                train of this data.

        """

        if self._pulse_ids is not None:
            # Use cached pulse IDs directly if available.
            pulse_ids = self._pulse_ids
        elif self._key is not None or self._source is not None:
            # Load data for the key's or source's first train, if
            # available.
            pulse_ids = self.select_trains(by_id[
                [(self._key or self._source).data_counts().ne(0).idxmax()]
            ]).get_pulse_ids(copy=False)
        else:
            # Just get all pulse IDs.
            pulse_ids = self.get_pulse_ids(copy=False)

        # Drop train ID dimensions.
        pulse_ids = _drop_first_level(pulse_ids)

        return (pulse_ids if labelled else pulse_ids.to_numpy()).copy()

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
            (numpy.ndarray or pandas.Series):
        """

        mask = self._get_pulse_mask()

        if labelled:
            import xarray as xr
            return xr.DataArray(
                mask,
                dims=['trainId', 'pulseId'],
                coords={'trainId': self._get_train_ids(),
                        'pulseId': np.arange(mask.shape[1])})
        else:
            return mask

    def is_constant_pattern(self):
        """Whether pulse IDs are constant in this data.

        Returns:
            (bool): Whether pulse IDs are identical in every train.
        """

        pulse_ids = self.get_pulse_ids(copy=False)

        # This two level check ends up being faster than comparing the
        # sets of pulse IDs for each train including their position.
        return (
            # Do all trains have the same number of pulses?
            pulse_ids.groupby(level=0).count().unique().size == 1 and

            # Are the pulse IDs in each pulse position identical?
            all([len(x) == 1 for x in pulse_ids.groupby(level=1).unique()])
        )

    def get_pulse_counts(self, labelled=True):
        """Get number of pulses per train.

        Args:
            labelled (bool, optional): Whether a labelled pandas Series
                (default) or unlabelled numpy array is returned.

        Returns:
            (pandas.Series or numpy.ndarray): Number of pulses per
                train, indexed by train ID if labelled is True.
        """

        counts = self.get_pulse_ids(copy=False).groupby(level=0).count()
        return counts if labelled else counts.to_numpy()

    def get_pulse_index(self, pulse_dim='pulseId', include_extra_dims=True):
        """Get a multi-level index for pulse-resolved data.

        Args:
            pulse_dim ({pulseId, pulseNumber, time}, optional): Label
                for pulse dimension, pulse ID by default.
            include_extra_dims (bool, optional): Whether to include any
                additional dimensions of this particular implementation
                beyond train ID and pulse dimension.

        Returns:
            (pandas.MultiIndex): Multi-level index covering train ID,
                pulse ID or pulse number and potentially any additonal
                extra index dimensions.
        """

        pulse_ids = self.get_pulse_ids(copy=False)
        index_levels = {'trainId': pulse_ids.index.get_level_values('trainId')}

        if pulse_dim == 'pulseId':
            index_levels[pulse_dim] = pulse_ids.to_numpy().copy()
        elif pulse_dim == 'pulseNumber':
            index_levels[pulse_dim] = pulse_ids.index.get_level_values(
                'pulseNumber')
        elif pulse_dim == 'time':
            index_levels[pulse_dim] = np.concatenate([
                pids - pids.iloc[0] for _, pids
                in pulse_ids.groupby(level=0)]) / self.bunch_repetition_rate
        else:
            raise ValueError('pulse_dim must be one of `pulseId`, '
                             '`pulseNumber`, `time`')

        if include_extra_dims:
            index_levels.update({name: pulse_ids.index.get_level_values(name)
                                 for name in pulse_ids.index.names[2:]})

        import pandas as pd
        return pd.MultiIndex.from_arrays(
            list(index_levels.values()), names=list(index_levels.keys()))

    def search_pulse_patterns(self, labelled=True):
        """Search identical pulse patterns in this data.

        Reads the bunch pattern table and gathers contiguous train
        regions of constant pulse pattern. It returns a list of train
        slices and corresponding pulse IDs.

        Args:
            labelled (bool, optional): Whether a labelled pandas Series
                (default) or unlabelled numpy array is returned.

        Returns:
            (list of (slice, pandas.Series or ndarray) tuples): List of
                train regions identified by index slices with identical
                pulse IDs.
        """

        pulse_mask = self._get_pulse_mask(reduced=True)

        # Find the unique patterns and the respective indices for each
        # unique pattern.
        unique_patterns, pattern_indices = np.unique(
            pulse_mask, axis=0, return_inverse=True)

        # Find positions of pattern changes plus beginning and end.
        pattern_changes = np.concatenate([
            [-1],
            np.flatnonzero(pattern_indices[1:] - pattern_indices[:-1]),
            [len(pulse_mask)]])

        tids = self._get_train_ids()
        one = np.uint64(1)  # Avoid conversion to float64.

        def gen_slice(start, stop):
            return by_id[tids[start]:tids[stop-1]+one]

        pulse_ids = self.get_pulse_ids(copy=False)

        if labelled:
            def gen_pulse_ids(train_idx):
                return pulse_ids.loc[tids[train_idx]].copy()
        else:
            pid_min = pulse_ids.min()

            def gen_pulse_ids(train_idx):
                return pid_min + np.flatnonzero(pulse_mask[train_idx])

        # Build list of (train_slice, pattern) tuples.
        patterns = [
            (gen_slice(start+1, stop), gen_pulse_ids(start+1))
            for start, stop in pairwise(pattern_changes)]

        return patterns

    def trains(self, labelled=True):
        """Iterate over pulse IDs by train.

        Args:
            labelled (bool, optional): Whether a labelled pandas Series
                (default) or unlabelled numpy array is returned.

        Yields:
            (int, pd.Series or ndarray): Train ID and pulse IDs.
        """

        # Generic version implemented on top of get_pulse_ids().
        for train_id, row in self.get_pulse_ids().groupby(level=0):
            yield train_id, \
                _drop_first_level(row) if labelled else row.to_numpy()


class TimeserverPulses(PulsePattern):
    """Abstract interface to timeserver-based based pulse patterns.

    This class should not be instantiated directly, but one of its
    implementations `XrayPulses` or `OpticalLaserPulses`. It provides
    the shared interface to access pulse patterns encoded in the bunch
    pattern table.

    Requires _mask_table() and _get_ppdecoder_node() to be implemented.
    """

    # All methods are built on top of get_pulse_mask and trains(). Their
    # default implementations require implementation of  _mask_table()
    # and _get_ppdecoder_node().

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

        sd = data[source]

        if 'maindump.pulseIds.value' in sd.keys():
            # PulsePatternDecoder source.
            self._with_timeserver = False

            # TODO: SourceData.train_id_coordinates() would make this
            # redundant.
            kd = sd['maindump.pulseIds']
        else:
            # Timeserver source.
            self._with_timeserver = True

            if ':' in source:
                kd = sd['data.bunchPatternTable']
            else:
                kd = sd['bunchPatternTable']

        super().__init__(sd, kd)

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

    def _get_train_ids(self):
        # Faster version reading INDEX data directly.
        return self._key.train_id_coordinates()

    def _get_pulse_ids(self):
        if self._with_timeserver:
            pids_by_train = [np.flatnonzero(mask) for mask
                             in self._mask_table(self._key.ndarray())]
        else:
            node = self._get_ppdecoder_node()
            pids_by_train = [
                pulse_ids[:num_pulses] for pulse_ids, num_pulses in zip(
                    self._source[f'{node}.pulseIds'].ndarray(),
                    self._source[f'{node}.nPulses'].ndarray())]

        counts = [len(pids) for pids in pids_by_train]

        import pandas as pd
        index = pd.MultiIndex.from_arrays([
            np.repeat(self._key.train_id_coordinates(), counts),
            np.concatenate([np.arange(count) for count in counts])
        ], names=['trainId', 'pulseNumber'])

        return pd.Series(data=np.concatenate(pids_by_train),
                         index=index, dtype=np.int32)

    def _get_pulse_mask(self, reduced=False):
        if not self._with_timeserver:
            return super()._get_pulse_mask(reduced)

        # Optimized version in the case of timeserver device.

        if not reduced:
            return self._mask_table(self._key.ndarray())

        if self._pulse_ids is not None:
            # If pulse IDs are already loaded, cut the table already
            # during readout.
            roi = np.s_[self._pulse_ids.min():self._pulse_ids.max()+1]
            return self._mask_table(self._key.ndarray(roi=roi))
        else:
            # If no pulse IDs are available, load the entire table
            # and slice afterwards.
            mask = self._mask_table(self._key.ndarray())
            row_slice = np.s_[
                mask.argmax(axis=1).min():
                mask.shape[1] - mask[:, ::-1].argmax(axis=1).min()]
            return mask[:, row_slice]

    def _mask_table(self, table):
        """Mask bunch pattern table."""
        raise NotImplementedError('_mask_table')

    def _get_ppdecoder_node(self):
        """Get node in pulse pattern decoder device."""
        raise NotImplementedError('_get_ppdecoder_node')

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


class XrayPulses(TimeserverPulses):
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

    For specific access to pulses from one of the optical laser sources,
    please see the almost corresponding
    [OpticalLaserPulses][extra.components.OpticalLaserPulses] component
    with the same interface.

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


class OpticalLaserPulses(TimeserverPulses):
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
