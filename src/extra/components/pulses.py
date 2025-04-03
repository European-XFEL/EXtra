"""Interface and utilies to work with pulse-resolved data."""


from __future__ import annotations  # to defer evaulation of annotations
from copy import copy
from itertools import pairwise
from functools import wraps, lru_cache
from typing import Optional
from warnings import warn
import re

import numpy as np

from euxfel_bunch_pattern import is_sase, is_laser, \
    PPL_BITS, DESTINATION_TLD, DESTINATION_T4D, DESTINATION_T5D, \
    PHOTON_LINE_DEFLECTION
from extra_data import SourceData, KeyData, by_id

from .utils import identify_sase, _instrument_to_sase


__all__ = ['XrayPulses', 'OpticalLaserPulses', 'MachinePulses',
           'PumpProbePulses', 'DldPulses']


def _drop_first_level(pd_val):
    """Return first group and drop first level of a pandas multi index."""
    return pd_val.loc[pd_val.head(1).index.values[0][0]]


class PulsePattern:
    """Abstract interface to pulse patterns.

    This class should not be instantiated directly, but one of its
    implementationsd `XrayPulses` or `OpticalLaserPulses`. It provides
    the shared interface to access any pulse pattern.

    Instances of this class are assumed to be immutable, which is
    exploited in various methods for caching purposes. An implementation
    should not allow altering its internal object state in a way that
    causes return values of public APIs to change.

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
        """Low-level access to train IDs.

        This method may be overriden by any implementation of this class
        for either performance or if the underlying data may contain
        train IDs that have no pulse IDs associated with it. The default
        implementation draws train IDs from the series of pulse IDs, and
        thus cannot contain trains without pulses. This is particularly
        relevant when counting pulses.

        Returns:
            (np.ndarray) Train IDs, expected to be in order.
        """

        # This method turned out to be the fastest to get just the
        # group labels.
        return self._get_pulse_ids().index.to_frame()['trainId'].unique()

    def _get_pulse_ids(self):
        """Low-level access to pulse IDs.

        This method must be overriden by any implementation of this
        class. It is expected to return a pandas series with one entry
        per pulse ID labelled by a multi index of train ID and pulse
        index. Its result will be cached externally.

        Returns:
            (pd.Series) Pulse IDs labelled by train ID and pulse index.
        """
        raise NotImplementedError('_get_pulse_ids')

    def _get_pulse_mask(self, reduced=False):
        """Default implementation based on train and pulse IDs."""

        train_ids = self._get_train_ids()
        pulse_ids = self.pulse_ids(copy=False)

        if reduced and pulse_ids.empty:
            pid_offset = 0
            table_len = 0
        elif reduced:
            pid_offset = pulse_ids.min()
            table_len = pulse_ids.max() - pid_offset + 1
        else:
            pid_offset = 0
            table_len = self._bunch_pattern_table_len

        num_trains = len(train_ids)
        mask = np.zeros((num_trains, table_len), dtype=bool)
        train_idx = 0

        for train_id, train_pulses in pulse_ids.groupby(level=0):
            # Search the index for this train ID, starting from where
            # the previous iteration left off.
            for i in range(train_idx, num_trains):
                if train_ids[i] == train_id:
                    train_idx = i
                    break

            mask[train_idx, train_pulses - pid_offset] = True

        return mask

    def _extend_all_trains(self, act_entries, fill_value=None):
        """Extend pd.Series to all trains."""

        train_ids = self._get_train_ids()

        if len(act_entries) == len(train_ids):
            # Assume trains are always a subset of internal trains and
            # thus are equal if length matches.
            return act_entries

        import pandas as pd
        ext_entries = pd.Series(
            np.zeros_like(train_ids, dtype=act_entries.dtype),
            index=pd.Index(train_ids, name='trainId'))

        if fill_value is not None:
            ext_entries[:] = fill_value

        ext_entries[act_entries.index] = act_entries

        return ext_entries

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
    def source(self) -> SourceData:
        """Source used for pulse information."""
        return self._source

    def select_trains(self, trains):
        """Select a subset of trains.

        The argument can be any object accepted by
        [DataCollection.select_trains()][extra_data.DataCollection.select_trains].
        """
        new_train_ids = None
        res = copy(self)

        if self._source is not None:
            res._source = self._source.select_trains(trains)
            new_train_ids = res._source.train_ids

        if self._key is not None:
            res._key = self._key.select_trains(trains)
            new_train_ids = res._key.train_ids

        if self._pulse_ids is not None and new_train_ids is not None:
            # Extract those entries from cached pulse IDs that intersect
            # with the newly selected train IDs.
            # Note that this does NOT slice the underlying index's data!
            # Direct access to it via .index.levels[0] will reveal the
            # original train IDs.
            res._pulse_ids = self._pulse_ids.loc[np.intersect1d(
                new_train_ids, self._pulse_ids.index.to_frame()['trainId'])]
        else:
            res._pulse_ids = None

        return res

    def pulse_ids(self, labelled=True, copy=True):
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
                ID and pulse index if labelled is True.
        """

        if self._pulse_ids is None:
            self._pulse_ids = self._get_pulse_ids()

        pulse_ids = self._pulse_ids if labelled else self._pulse_ids.to_numpy()
        return pulse_ids.copy() if copy else pulse_ids

    def get_pulse_ids(self, *args, **kwargs):
        warn("Use pulse_ids() instead of get_pulse_ids()",
             DeprecationWarning, stacklevel=2)
        return self.pulse_ids(*args, **kwargs)

    def peek_pulse_ids(self, labelled=True):
        """Get pulse IDs for the first train.

        This method may be significantly faster than to
        `pulse_ids()` by only reading data for the very first train
        of data.

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
            ]).pulse_ids(copy=False)
        else:
            # Just get all pulse IDs.
            pulse_ids = self.pulse_ids(copy=False)

        if not pulse_ids.empty:
            # Drop train ID dimensions.
            pulse_ids = _drop_first_level(pulse_ids)
        else:
            import pandas as pd
            pulse_ids = pd.Series([], dtype=np.int32)

        return (pulse_ids if labelled else pulse_ids.to_numpy()).copy()

    def pulse_mask(self, labelled=True):
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

    def get_pulse_mask(self, *args, **kwargs):
        warn("Use pulse_mask() instead of get_pulse_mask()",
             DeprecationWarning, stacklevel=2)
        return self.pulse_mask(*args, **kwargs)

    def is_constant_pattern(self, *, include_empty_trains=False):
        """Whether pulse IDs are constant in this data.

        Args:
            include_empty_trains (bool, optional): Whether trains
                without any pulses cause a pulse pattern to not be
                considered constant, False by default.

        Returns:
            (bool): Whether pulse IDs are identical in every train.
        """

        pulse_ids = self.pulse_ids(copy=False)

        # These checks end up being faster than comparing the sets of
        # pulse IDs for each train including their position.

        # Do all trains have the same number of pulses?
        pids_by_train = pulse_ids.groupby(level=0)
        if pids_by_train.count().unique().size > 1:
            return False

        # Is the pulse data (ID and any other index flags) in each pulse
        # position identical?
        # To do this, move any indices beyond train and pulse into
        # columns, and then group by pulse.
        pulse_data = pulse_ids.reset_index(pulse_ids.index.names[2:])
        data_by_pulse = pulse_data.groupby(level=1)
        if any([len(unique_rows) > 1
                for column in pulse_data.columns
                for unique_rows in data_by_pulse[column].unique()]):
            return False

        if include_empty_trains:
            # If empty trains are considered, are the pulses' train IDs
            # identical to the global train IDs?
            return np.array_equal(pids_by_train.first().index,
                                  self._get_train_ids())
        else:
            return True


    def is_interleaved_with(self, other: PulsePattern) -> Optional[bool]:
        """Check whether pulses are interleaved with other pattern.

        This method returns True if and only if it can safely determine
        that the pulse pattern described by this component and the
        passed component are interleaved, and False if and only if it is
        clear they are not interleaved. In all other cases, the pulses
        may or may not be interleaved and None is returned.

        Returns:
            (bool or None) Whether pulses are interleaved or None if the
                pulse patterns do not allow the determination.
        """

        for (_, self_pids), (_, other_pids) in zip(
            self.pulse_ids(copy=False).groupby(level=0),
            other.pulse_ids(copy=False).groupby(level=0)
        ):
            self_num = len(self_pids)
            other_num = len(other_pids)

            if self_num == 0 or other_num == 0:
                # No statement can be drawn from missing pulses.
                continue

            self_first_pulse = self_pids.min()
            other_first_pulse = other_pids.min()

            if self_num > 1 and other_num > 1:
                # Multiple pulses in both self and other.

                if self_first_pulse < other_first_pulse:
                    # self early, interleaved <=> other starts within self
                    return self_pids.max() > other_first_pulse
                else:
                    # other early, interleaved <=> self starts within other
                    return other_pids.max() > self_first_pulse

            elif self_num == 1 and other_num > 1:
                # Single pulse in self, but multiple in other.
                if other_first_pulse < self_first_pulse:
                    # other early, interleaved <=> self is within other
                    return other_pids.max() > self_first_pulse

            elif self_num > 1 and other_num == 1:
                # Multiple pulses in self, but single in other.
                if self_first_pulse < other_first_pulse:
                    # self early, interleaved <=> other is within self
                    return self_pids.max() > other_first_pulse

        return  # Reached in case of *maybe*

    def pulse_counts(self, labelled=True):
        """Get number of pulses per train.

        Args:
            labelled (bool, optional): Whether a labelled pandas Series
                (default) or unlabelled numpy array is returned.

        Returns:
            (pandas.Series or numpy.ndarray): Number of pulses per
                train, indexed by train ID if labelled is True.
        """

        counts = self._extend_all_trains(
            self.pulse_ids(copy=False).groupby(level=0).count())

        return counts if labelled else counts.to_numpy()

    def pulse_periods(self, labelled=True, single_pulse_value=None,
                      no_pulse_value=None):
        """Get pulse periods per train.

        The pulse period is expressed in units of the fundamental bunch
        repetition rate of 4.5 MHz. Its exact value can be accessed via
        PulsePattern.bunch_repetition_rate.

        In each train, the smallest period between consecutive pulses is
        considered if the pulses do not have equal distances. For trains
        with a single pulse, the value can be specified or is filled by
        the largest value encountered across other trains with more
        pulses.

        The resulting series is cast to integers unless it contains
        non-finite values as a result of fill values.

        Args:
            labelled (bool, optional): Whether a labelled pandas Series
                (default) or unlabelled numpy array is returned.
            single_pulse_value (number, optional): Fill value for trains
                with a single pulse. If omitted the largest period
                encountered in the data is used or an exception raised
                if there are no trains with more than one pulse.
            no_pulse_value (number, optional): Fill value for trains
                without any pulse, 0 if omitted.

        Returns:
            (pandas.Series or numpy.ndarray): Pulse period per train
                train, indexed by train ID if labelled is True.
        """

        # Minimum of difference between pulses in a train by train.
        act_periods = (self.pulse_ids(copy=False)
            .groupby(level=0).diff()  # Periods within each train.
            .groupby(level=0).min()  # Minimal period in each train.
        )

        # Fill any NaN value with a fill value.
        single_pulse_periods = act_periods.isna()

        if single_pulse_periods.any():
            if single_pulse_value is None:
                if single_pulse_periods.all():
                    raise ValueError('data contains no trains with more than '
                                     'one pulse, explicit single_pulse_value '
                                     'required')

                single_pulse_value = int(act_periods.max())

            act_periods.fillna(single_pulse_value, inplace=True)

        # Extend to all trains, filling with 0(.0) or the passed value.
        periods = self._extend_all_trains(act_periods, no_pulse_value)

        # Cast to integer, if possible.
        if np.isfinite(periods).all():
            periods = periods.astype(int)

        return periods if labelled else periods.to_numpy()

    def pulse_repetition_rates(self, labelled=True):
        """Get pulse repetition rate per train in Hz.

        In each train, the highest repetition rate is used if the pulses
        do not have equal distances.

        For trains with a single pulse, 0.0 is returned while trains
        without any pulse return NaN. If different fill values are
        desired, PulsePattern.bunch_repetition_rate can be divided
        manually by the result from PulsePattern.pulse_periods().

        Args:
            labelled (bool, optional): Whether a labelled pandas Series
                (default) or unlabelled numpy array is returned.

        Returns:
            (pandas.Series or numpy.ndarray): Pulse repetition rate per
                train, indexed by train ID if labelled is True.
        """

        return self.bunch_repetition_rate \
            / self.pulse_periods(labelled, np.inf, np.nan)

    def train_durations(self, labelled=True):
        """Get durations of each train in seconds.

        The duration is the time difference between the first and last
        pulse in each train. For trains with no pulses, NaN is returned.

        Args:
            labelled (bool, optional): Whether a labelled pandas Series
                (default) or unlabelled numpy array is returned.

        Returns:
            (pandas.Series or numpy.ndarray): Train duration in seconds
                per train, indexed by train ID if labelled is True.
        """

        pids_by_train = self.pulse_ids(copy=False).groupby(level=0)
        durations = self._extend_all_trains(
            (pids_by_train.max() - pids_by_train.min())
            / self.bunch_repetition_rate, np.nan)

        return durations if labelled else durations.to_numpy()

    def get_pulse_counts(self, *args, **kwargs):
        warn("Use pulse_counts() instead of get_pulse_counts()",
             DeprecationWarning, stacklevel=2)
        return self.pulse_counts(*args, **kwargs)

    def build_pulse_index(self, pulse_dim='pulseId', include_extra_dims=True,
                          pulse_dtype=None):
        """Get a multi-level index for pulse-resolved data.

        Args:
            pulse_dim ({pulseId, pulseIndex, pulseTime}, optional):
                Label for pulse dimension, pulse ID by default.
            include_extra_dims (bool, optional): Whether to include any
                additional dimensions of this particular implementation
                beyond train ID and pulse dimension.
            pulse_dtype (numpy.dtype, optional): If specified, force
                casts the pulse dimension to the specified dtype.

        Returns:
            (pandas.MultiIndex): Multi-level index covering train ID,
                pulse ID or pulse index and potentially any additonal
                extra index dimensions.
        """

        pulse_ids = self.pulse_ids(copy=False)
        index_levels = {'trainId': pulse_ids.index.get_level_values('trainId')}

        if pulse_dim == 'pulseId':
            index_levels[pulse_dim] = pulse_ids.to_numpy().copy()
        elif pulse_dim == 'pulseIndex':
            index_levels[pulse_dim] = pulse_ids.index.get_level_values(
                'pulseIndex')
        elif pulse_dim in {'time', 'pulseTime'}:
            if pulse_dim == 'time':
                warn('Use `pulseTime` instead of `time`',
                     DeprecationWarning, stacklevel=2)

            index_levels[pulse_dim] = np.concatenate([
                pids - pids.iloc[0] for _, pids
                in pulse_ids.groupby(level=0)]) / self.bunch_repetition_rate
        else:
            raise ValueError('pulse_dim must be one of `pulseId`, '
                             '`pulseIndex`, `pulseTime`')

        if include_extra_dims:
            index_levels.update({name: pulse_ids.index.get_level_values(name)
                                 for name in pulse_ids.index.names[2:]})

        if pulse_dtype is not None:
            index_levels[pulse_dim] = index_levels[pulse_dim].astype(pulse_dtype)

        import pandas as pd
        return pd.MultiIndex.from_arrays(
            list(index_levels.values()), names=list(index_levels.keys()))

    def get_pulse_index(self, *args, **kwargs):
        warn("Use build_pulse_index() instead of get_pulse_index()",
             DeprecationWarning, stacklevel=2)
        return self.build_pulse_index(*args, **kwargs)

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

        pulse_ids = self.pulse_ids(copy=False)

        if labelled:
            import pandas as pd
            def gen_pulse_ids(train_idx):
                try:
                    return pulse_ids.loc[tids[train_idx]].copy()
                except KeyError:
                    return pd.Series([], dtype=np.int32)
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

        # Generic version implemented on top of pulse_ids().
        for train_id, row in self.pulse_ids().groupby(level=0):
            yield train_id, \
                _drop_first_level(row) if labelled else row.to_numpy()


class TimeserverPulses(PulsePattern):
    """Abstract interface to timeserver-based based pulse patterns.

    This class should not be instantiated directly, but one of its
    implementations `XrayPulses` or `OpticalLaserPulses`. It provides
    the shared interface to access pulse patterns encoded in the bunch
    pattern table.

    Requires _mask_table() and _get_ppdecoder_nodes() to be implemented.
    """

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

        # Also support SourceData objects for internal use.
        sd = data[source] if not isinstance(source, SourceData) else source

        if 'maindump.pulseIds.value' in sd.keys():
            # PulsePatternDecoder source.
            self._with_timeserver = False

            # TODO: SourceData.train_id_coordinates() would make this
            # redundant.
            kd = sd['maindump.pulseIds']
        else:
            # Timeserver source.
            self._with_timeserver = True

            if ':' in sd.source:
                kd = sd['data.bunchPatternTable']
            else:
                kd = sd['bunchPatternTable']

        super().__init__(sd, kd)

    def __repr__(self, location=None):
        if self._with_timeserver:
            source_type = 'timeserver'
        else:
            source_type = 'ppdecoder'

        loc_str = ' ' + location if location is not None else ''

        return "<{}{} using {}={}>".format(
            type(self).__name__, loc_str, source_type, self._source.source)

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
        return self._key.train_id_coordinates()

    def _get_pulse_ids(self):
        if self._with_timeserver:
            pids_by_train = [np.flatnonzero(mask) for mask
                             in self._mask_table(self._key.ndarray())]
        else:
            nodes = self._get_ppdecoder_nodes()

            if len(nodes) == 1:
                # Optimization when using only a single node, which is
                # the most common case and the general path below is up
                # to 70% slower.
                node = nodes[0]
                pids_by_train = [
                    pulse_ids[:num_pulses] for pulse_ids, num_pulses in zip(
                        self._source[f'{node}.pulseIds'].ndarray(),
                        self._source[f'{node}.nPulses'].ndarray())]
            else:
                # Generalization of the comprehension above to combine
                # multiple nodes into a single ordered list.
                # The outer comprehension first loads and collects the
                # pulseIds and nPulses datasets for each node, combining
                # them into a single iterable by train.
                # In its body, it then uses nPulses to index into the
                # pulseIds dataset for every node, concatenates the
                # resulting pulse IDs and finally sorts by train.
                pids_by_train = [np.sort(np.concatenate([
                    pulse_ids[:num_pulses] for pulse_ids, num_pulses
                    in zip(per_train[0::2], per_train[1::2])]
                )) for per_train in zip(*sum([
                    [self._source[f'{node}.pulseIds'].ndarray(),
                     self._source[f'{node}.nPulses'].ndarray()]
                    for node in nodes], []))]

        counts = [len(pids) for pids in pids_by_train]

        import pandas as pd

        if not counts:
            # Immediately return an empty series if there is no data.
            return pd.Series([], dtype=np.int32)

        index = pd.MultiIndex.from_arrays([
            np.repeat(self._key.train_id_coordinates(), counts),
            np.concatenate([np.arange(count) for count in counts])
        ], names=['trainId', 'pulseIndex'])

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

    def _get_ppdecoder_nodes(self):
        """Get nodes in pulse pattern decoder device."""
        raise NotImplementedError('_get_ppdecoder_nodes')

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

    def machine_pulses(self, **kwargs) -> MachinePulses:
        """Get MachinePulses component for the same data.

        Any additional keyword arguments are passed to the MachinePulses
        initializer.

        Returns:
            (MachinePulses) Corresponding component using the same data
                as this component.
        """

        return MachinePulses(None, self._source, **kwargs)

    @lru_cache
    def machine_repetition_rate(self) -> float:
        """Actual machine repetition rate."""
        return self.machine_pulses().pulse_repetition_rates().min()

    @lru_cache
    def is_sa1_interleaved_with_sa3(self) -> Optional[bool]:
        """Check whether SASE1 and SASE3 pulses are interleaved.

        Due to their unique geometry, the pulses of the SASE1 and SASE3
        beamlines can either occur after each other on the pulse train
        or be interleaved. Here, each beamline generally receives their
        first pulse directly after each other, with their relative pulse
        patterns following afterwards in a back-and-forth fashion.

        This method returns True if and only if it can be safely
        determined the SA1 and SA3 pulses are intereavled, and False if
        and only if it is clear they are not interleaved. In all other
        cases, the pulses may or may not be interleaved and None is
        returned.

        Returns:
            (bool or None) Whether SA1 and SA3 are interleaved or None
                if the pulse patterns do not allow the determination.
        """

        sa1_pulses = XrayPulses(None, self._source, sase=1) \
            if self._sase != 1 else self
        sa3_pulses = XrayPulses(None, self._source, sase=3) \
            if self._sase != 3 else self

        return sa1_pulses.is_interleaved_with(sa3_pulses)


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

    This class only deals with X-ray pulses of a particular SASE
    beamline. Please see
    [OpticalLaserPulses][extra.components.OpticalLaserPulses] to access
    pulses of the optical laser sources or
    [PumpProbePulses][extra.components.PumpProbePulses] to combine both
    into a single pattern.

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

    def __init__(self, data, source=None, sase=None):
        super().__init__(data, source)

        if sase not in {1, 2, 3}:
            sase = identify_sase(data)

        self._sase = sase

    def __repr__(self):
        return super().__repr__(f'for SA{self._sase}')

    def _mask_table(self, table):
        return is_sase(table, sase=self._sase)

    def _get_ppdecoder_nodes(self):
        return [f'sase{self._sase}']

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
    component. In the case of more complex or non-overlapping patterns,
    the [PumpProbePulses][extra.components.PumpProbePulses] allows to
    combine both into a single pulse pattern.

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
        return super().__repr__(self._ppl_seed.name)

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

    def _get_ppdecoder_nodes(self):
        return ['laser']

    @property
    def ppl_seed(self) -> Optional[PPL_BITS]:
        """Used laser seed."""
        return self._ppl_seed


class MachinePulses(TimeserverPulses):
    """An interface to machine pulses and other bunch pattern data.

    The bunch pattern table contains much more information than just the
    pulses of each SASE beamline or the optical laser systems, in
    particular about the upstream electron systems.

    This component is configurable to extract any information from the
    bunch pattern table beyond the X-ray or optical laser pulses. By
    default, it considers all generated electron bunches by combining
    the various electron dumps along the accelerator. Please consult the
    XFEL Timing System Specification for the various mask bits used.

    Note that the full range of bunch pattern table information is only
    available when used with timeserver data. The pulse pattern decoder
    data only contains information about those pulses in the main dump
    and SASE dumps.

    Args:
        data (extra.data.DataCollection): Data to access bunch pattern
            data from.
        source (str, optional): Source name of a timeserver or pulse
            pattern decoder, only needed if the data includes more than
            one such device or none could not be detected automatically.
        mask (int, optional): Custom mask into the bunch pattern table,
            by default all electron dumps are used. When initialized
            with ppdecoder data, only the main and SASE dumps are
            supported.
        require_all_bits (bool, optional): Whether to consider entries
            having any or all of the bits specified by mask, False by
            default. Only supported when initialized with timeserver data.
    """

    #                main              SA1/SA3           SA2
    _all_dumps = DESTINATION_TLD | DESTINATION_T4D | DESTINATION_T5D

    #                                     SA3 kicker
    _ppdecoder_allowed = _all_dumps | PHOTON_LINE_DEFLECTION

    def __init__(self, data, source=None, mask=_all_dumps,
                 require_all_bits=False):
        super().__init__(data, source)

        self._require_all_bits = require_all_bits

        if not self._with_timeserver:
            # Support with ppdecoder data is limited.

            if self._require_all_bits:
                # TimeserverPulses hardcodes `or` for now.
                raise ValueError('require_all_bits may be not be used with '
                                 'ppdecoder data')

            elif (mask | self._ppdecoder_allowed) != self._ppdecoder_allowed:
                # Data only contains certain bits.
                raise ValueError('mask may only contain main or SASE dumps'
                                 'when initialized with ppdecoder data')

        if isinstance(mask, np.integer):
            # Avoid any of the numpy integer types, as some like uint64
            # can do unexpected things.
            mask = int(mask)
        elif not isinstance(mask, int):
            raise TypeError('mask must be an integer')

        self._mask = mask

    def __repr__(self):
        active_bits = np.flatnonzero([(self._mask & (1 << i)) > 0
                                      for i in range(32)])

        if len(active_bits) == 1:
            return super().__repr__(f'for bit {active_bits[0]}')
        else:
            return super().__repr__('for bits ' + (
                '&' if self._require_all_bits else '|'
            ).join(active_bits.astype(str)))

    def _mask_table(self, table):
        if not self._require_all_bits:
            return (table & self._mask) > 0
        else:
            return (table & self._mask) == self._mask

    def _get_ppdecoder_nodes(self):
        nodes = []

        if (self._mask & DESTINATION_TLD) == DESTINATION_TLD:
            nodes.append('maindump')

        if (self._mask & DESTINATION_T4D) == DESTINATION_T4D:
            nodes.append('sase1')
            nodes.append('sase3')

        if (self._mask & DESTINATION_T5D) == DESTINATION_T5D:
            nodes.append('sase2')

        return nodes


class PumpProbePulses(XrayPulses, OpticalLaserPulses):
    """An interface to combined FEL and PPL pulses.

    This component offers support for arbitrary pulse relations between
    X-ray FEL and optical laser pulses (PPL) in pump-probe experiments.
    As the PPL pulse information in the bunch pattern table always
    starts at offset 0 for technical reasons irrespective of its actual
    temporal relation, it is corrected during initialization by exactly
    one of three methods:

    1) Offset all PPL pulses to a fixed bunch table position.
    2) Offset all PPL pulses relative to the first FEL pulse in units of
       the bunch pattern table.
    3) Offset all PPL pulses relative to the first FEL pulse in units of
       FEL pulses.

    In cases where there are no FEL pulses (for method 2) or too few
    (for method 3) to determine this offset, it may be extrapolated from
    a previous train if enabled. If extrapolation is disabled, an
    exception is raised.

    Unlike [XrayPulses][extra.components.XrayPulses] and
    [OpticalLaserPulses][extra.components.OpticalLaserPulses], this
    component adds additional levels to the pulse index indicated
    whether an FEL or PPL pulse is present in any particular position.
    It will only consider a pattern equal if it is equal for both FEL
    and PPL.

    For experiments where all FEL and PPL laser pulses overlap, it is
    recommended to just use the [XrayPulses][extra.components.XrayPulses]
    component.

    Args:
        data (extra.data.DataCollection): Data to access bunch pattern
            data from.
        source (str, optional): Source name of a timeserver or pulse
            pattern decoder, only needed if the data includes more than
            one such device or none could not be detected automatically.
        instrument (src or tuple, optional): Instrument to interpret FEL
            and PPL pulses of, only needed if the data includes sources
            from more than one instrument or it could not be detected
            automatically. May also be a tuple of (sase, ppl_seed)
            corresponding to arguments for
            [XrayPulses][extra.components.XrayPulses] and
            [OpticalLaserPulses][extra.components.OpticalLaserPulses].
        bunch_table_position (int, optional): Absolute bunch table
            position or pulse ID for the PPL pulse.
        bunch_table_offset (int, optional): Offset to the first FEL
            pulse in bunch table positions or pulse IDs.
        pulse_offset (number, optional): Offset to the first FEL pulse
            in units of the spacing between the first two FEL pulses,
            i.e. in units of FEL pulses.
        extrapolate (bool, optional): Whether FEL pulse IDs may be
            extrapolated from past trains if missing, true by default.
            An exception is raised if disabled and the PPL anchoring
            method is missing the minimum number of required FEL pulses.
    """

    # This class inherits from two classes which both have the same
    # parent in turn, thus forming a diamond-shaped inheritance diagram:
    #
    #               PulsePattern
    #                    |
    #             TimeserverPulses
    #               /          \
    #         XrayPulses   OpticalLaserPulses
    #               \          /
    #             PumpProbePulses
    #
    # To avoid ambiguities, all calls to super classes are thus explicit
    # via their respective class object. The use of super() should be
    # avoided!

    def __init__(self, data, source=None, instrument=None, *,
                 bunch_table_position=None, bunch_table_offset=None,
                 pulse_offset=None, extrapolate=True):
        self._bunch_table_position = None
        self._bunch_table_offset = None
        self._pulse_offset = None  # Allowed to be float!
        self._extrapolate = extrapolate

        # Logic should always check these modes in exactly this order,
        # see below.
        if bunch_table_position is not None:
            self._bunch_table_position = int(bunch_table_position)
        elif bunch_table_offset is not None:
            self._bunch_table_offset = int(bunch_table_offset)
        elif pulse_offset is not None:
            self._pulse_offset = pulse_offset  # Allowed to be float!
        else:
            raise ValueError('must specify one of bunch_table_position, '
                             'bunch_table_offset, pulse_offset')

        if self._pulse_offset == 0:
            # Implement this case via the bunch_table_offset mechanism
            # to enable it to work even if there is only a single FEL
            # pulse in any given train.
            # These modes are always checked in the same order as above,
            # while __repr__ accounts for this special case.
            self._bunch_table_offset = 0

        if instrument is None:
            sase = None
            ppl_seed = None
        elif isinstance(instrument, tuple) and len(instrument) == 2:
            sase = int(instrument[0])
            ppl_seed = instrument[1]
        elif isinstance(instrument, str):
            sase = _instrument_to_sase(instrument.upper())
            ppl_seed = instrument
        else:
            raise TypeError('instrument must be str, 2-tuple or None')

        # Run the OpticalLaserPulses initializer to handle constraints
        # with pulse pattern decoder.
        OpticalLaserPulses.__init__(self, data, source, ppl_seed=ppl_seed)

        # Run missing initialization for XrayPulses.
        if sase is None:
            sase = identify_sase(data)

        self._sase = sase

    def __repr__(self):
        if self._bunch_table_position is not None:
            offset_str = f'@{self._bunch_table_position}b'
        elif self._pulse_offset is not None:
            # This branch is intentionally ahead of buch_table_offset to
            # still represent the case pulse_offset == 0 correctly even
            # if it is implemented by bunch_table_offset = 0 internally.
            offset_str = f'@SA{self._sase}{self._pulse_offset:+d}p'
        elif self._bunch_table_offset is not None:
            offset_str = f'@SA{self._sase}{self._bunch_table_offset:+d}b'

        if self._with_timeserver:
            source_type = 'timeserver'
        else:
            source_type = 'ppdecoder'

        return "<{} for SA{} / {}{} using {}={}>".format(
                type(self).__name__, self._sase, self._ppl_seed.name,
                offset_str, source_type, self._source.source)

    def _get_ppl_offset(self, fel):
        if self._bunch_table_position is not None:
            return self._bunch_table_position
        elif self._bunch_table_offset is not None:
            return fel[0] + self._bunch_table_offset
        elif self._pulse_offset is not None:
            return fel[0] + int((fel[1] - fel[0]) * self._pulse_offset)

    def _iter_timeserver_pids(self):
        for row in self._key.ndarray():
            yield np.flatnonzero(XrayPulses._mask_table(self, row)), \
                np.flatnonzero(OpticalLaserPulses._mask_table(self, row))

    def _iter_ppdecoder_pids(self):
        # XrayPulses and OpticalLaserPulses always return a single node,
        # and there is no use case at the moment to correlate laser and
        # machine pulses.
        fel_node = XrayPulses._get_ppdecoder_nodes(self)[0]
        ppl_node = OpticalLaserPulses._get_ppdecoder_nodes(self)[0]

        for (_, fel_ids), (_, fel_num), (_, ppl_ids), (_, ppl_num) in zip(
            self._source[f'{fel_node}.pulseIds'].trains(),
            self._source[f'{fel_node}.nPulses'].trains(),
            self._source[f'{ppl_node}.pulseIds'].trains(),
            self._source[f'{ppl_node}.nPulses'].trains()
        ):
            yield fel_ids[:fel_num], ppl_ids[:ppl_num]

    def _get_pulse_ids(self):
        iter_pulse_ids = self._iter_timeserver_pids() \
            if self._with_timeserver else self._iter_ppdecoder_pids()

        pids_by_train = []
        fel_by_train = []
        ppl_by_train = []
        counts = []

        train_ids = self._key.train_id_coordinates()
        prev_fel_pids = None

        for train_id, (fel_pids, ppl_pids) in zip(train_ids, iter_pulse_ids):
            try:
                ppl_pids += self._get_ppl_offset(fel_pids)
            except IndexError:
                # The IndexError can occur with trains having exactly
                # one FEL pulse (in pulse_offset mode) or no FEL pulses
                # (in bunch_table_offset and pulse_offset mode).

                if not self._extrapolate or prev_fel_pids is None:
                    if len(fel_pids) == 0:
                        msg = f'missing any FEL pulses in train {train_id} ' \
                              f'to determine absolute pulse position'
                    else:
                        msg = f'missing second FEL pulse in train ' \
                              f'{train_id} to determine pulse repetition rate'

                    if self._extrapolate:
                        msg += ', no train encountered yet to extrapolate from'

                    raise ValueError(msg) from None

                # Otherwise extrapolation is enabled and prior valid
                # patterns exist.

                ppl_pids += self._get_ppl_offset(prev_fel_pids)
            else:
                prev_fel_pids = fel_pids

            pids = np.union1d(fel_pids, ppl_pids)

            pids_by_train.append(pids)
            counts.append(len(pids))
            fel_by_train.append(np.isin(pids, fel_pids))
            ppl_by_train.append(np.isin(pids, ppl_pids))

        import pandas as pd
        index = pd.MultiIndex.from_arrays([
            np.repeat(train_ids, counts),
            np.concatenate([np.arange(count) for count in counts]),
            np.concatenate(fel_by_train), np.concatenate(ppl_by_train)
        ], names=['trainId', 'pulseIndex', 'fel', 'ppl'])

        return pd.Series(data=np.concatenate(pids_by_train),
                         index=index, dtype=np.int32)

    def _get_pulse_mask(self, reduced=False):
        # Actually returns flags instead of a mask.

        train_ids = self._get_train_ids()
        pulse_ids = self.pulse_ids(copy=False)
        pids_by_train = pulse_ids.groupby(level=0)

        if reduced:
            pid_offset = pulse_ids.min()
            table_len = pulse_ids.max() - pid_offset + 1
        else:
            pid_offset = 0
            table_len = self._bunch_pattern_table_len

        num_trains = len(train_ids)
        flags = np.zeros((num_trains, table_len), dtype=np.int8)
        train_idx = 0

        for train_id, train_pids in pulse_ids.groupby(level=0):
            # See PulsePattern._get_pulse_mask.
            for i in range(train_idx, num_trains):
                if train_ids[i] == train_id:
                    train_idx = i
                    break

            try:
                flags[train_idx, train_pids.loc[:, :, True, :] - pid_offset] |= 1
            except KeyError:
                # No FEL pulses in this train.
                pass

            try:
                flags[train_idx, train_pids.loc[:, :, :, True] - pid_offset] |= 2
            except KeyError:
                # No PPL pulses in this train.
                pass

        return flags

    def pulse_mask(self, labelled=True, field=None):
        """Get boolean pulse mask.

        The returned mask has the same shape as the full bunch pattern
        table but only contains boolean flags whether a given pulse
        was present in this pattern.

        For instance, to see whether each FEL pulse has a corresponding pump
        laser pulse in a consistent pulse pattern:

        ```python
        ppl_on = ppp.select_trains(0).pulse_mask(field='ppl').squeeze()
        fel_on = ppp.select_trains(0).pulse_mask(field='fel').squeeze()

        fel_pulse_is_pumped = ppl_on[fel_on]
        ```

        Args:
            labelled (bool, optional): Whether a labelled xarray
                DataArray (default) or unlabelled numpy array is
                returned.
            field (str, optional): 'fel'/'ppl' to mark only FEL or only pump
                laser pulses in the returned array. By default, pulses are True
                whether either an FEL pulse or a PPL pulse occured.

        Returns:
            (xarray.DataArray or numpy.ndarray):
        """
        bitfield = TimeserverPulses.pulse_mask(self, labelled=labelled)
        if field is None:
            return bitfield.astype(bool)
        elif field == 'fel':
            return (bitfield & 1).astype(bool)
        elif field == 'ppl':
            return (bitfield & 2).astype(bool)
        else:
            raise ValueError(f"{field=!r} parameter was not 'fel'/'ppl'/None")


class DldPulses(PulsePattern):
    """An interface to pulses from DLD reconstruction.

    The facility-provided event reconstruction for delay line detectors
    records its own pulse pattern information as part of its output in
    the `raw.triggers` key. This class exposes the same pulse pattern
    interface based on this information, and is primarily meant to be
    used alongside such data. Note that it is influenced by parameters
    set at the time of reconstruction, and hence may (incorrectly!)
    differ from timeserver data.

    For data processed before October 2022, this data may not contain
    flags for FEL/PPL pulses and may also be based on analog trigger
    signals. It is also lacking the true pulse IDs, which are in this
    case estimated by this component based on trigger positions.

    Args:
        detector (SourceData): Instrument source of reconstructed event
            data to retrieve trigger information from.
        clock_ratio (int, optional): Ratio between bunch repetition rate
            and digitizer sampling rate, only used in case of missing
            pulse ID information in data and 196 by default
            (non-interleaved ADQ412-3G).
        first_pulse_id (int, optional): Pulse ID for the first pulse,
            only used in case of missing pulse ID information in data
            and 0 by default.
        negative_ppl_indices (bool, optional): Whether to enumerate
            PPL-only pulses with negative indices (while keeping
            non-negative indices for FEL-only or pumped pulses) or
            interleave all pulses with positive indices (default).
    """

    def __init__(self, detector, *, clock_ratio=None, first_pulse_id=None,
                 negative_ppl_indices=False):
        super().__init__(detector, detector['raw.triggers'])

        self._clock_ratio = clock_ratio
        self._first_pulse_id = first_pulse_id
        self._negative_ppl_indices = negative_ppl_indices

    def _get_train_ids(self):
        return np.unique(self._key.train_id_coordinates())

    def _get_pulse_ids(self):
        triggers = self._key.ndarray()
        train_ids = self._key.train_id_coordinates()

        if self._negative_ppl_indices:
            if 'ppl' not in triggers.dtype.fields:
                raise ValueError('negative PPL indices only available '
                                 'with PPL information')

            pulse_indices = np.zeros(len(triggers), dtype=int)
            prev_tid = -1

            for i, tid, row in zip(range(len(triggers)), train_ids, triggers):
                if tid != prev_tid:
                    prev_tid = tid
                    only_ppl_index = -1
                    with_fel_index = 0

                if row['ppl'] and not row['fel']:
                    pulse_indices[i] = only_ppl_index
                    only_ppl_index -= 1
                else:
                    pulse_indices[i] = with_fel_index
                    with_fel_index += 1

        else:
            pulse_indices = np.concatenate([
                np.arange(count, dtype=np.int32) for count
                in self._key.data_counts(labelled=False)])

        index_levels = {
            'trainId': train_ids,
            'pulseIndex': pulse_indices,
        }

        if 'fel' in triggers.dtype.fields:
            index_levels['fel'] = triggers['fel'].copy()
            index_levels['ppl'] = triggers['ppl'].copy()

        import pandas as pd
        index = pd.MultiIndex.from_arrays(
            list(index_levels.values()), names=list(index_levels.keys()))

        if 'pulse' in triggers.dtype.fields:
            pulse_ids = triggers['pulse'].copy()
        else:
            # Try to guess pulse IDs from trigger positions.
            import sys
            print(f'{self.__class__.__name__}._get_pulse_ids(): No actual '
                  f'pulse IDs available in data, estimating from trigger '
                  f'positions. See documentation for more details.',
                  file=sys.stderr)

            pulse_ids = (triggers['start'] - triggers['start'][0]) \
                // (self._clock_ratio or 196) + (self._first_pulse_id or 0)

        return pd.Series(data=pulse_ids, index=index, dtype=np.int32)

    def triggers(self, labelled=True):
        """Get trigger information.

        Returns:
            (pd.Series or ndarray): Trigger fields start, stop, offset
        """

        from numpy.lib.recfunctions import drop_fields
        triggers = drop_fields(self._key.ndarray(), ['pulse', 'fel', 'ppl'])

        if labelled:
            import pandas as pd
            return pd.DataFrame(data=triggers, index=self.build_pulse_index())
        else:
            return triggers

    def get_triggers(self, *args, **kwargs):
        warn("Use triggers() instead of get_triggers()",
             DeprecationWarning, stacklevel=2)
        return self.triggers(*args, **kwargs)
