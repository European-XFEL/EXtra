
from copy import copy
import re

import numpy as np
import pandas as pd


class DelayLineDetector:
    """Interface for processed delay line detector data.

    Args:
        data (extra_data.DataCollection): Data to access DLD data from.
        detector (str, optional): Source name of the detector, only
            needed if the data includes more than one.
        pulses (extra.components.pulses.PulsePattern, optional): Pulse
            component to pull pulse information. If omitted, it is
            constructed from the internal trigger information.
    """

    _instrument_re = re.compile(r'^(\w{3}_\w+_DLD\d\/DET\/\w+):output?$')

    def __init__(self, data, detector=None, pulses=None):
        if detector is None:
            # Try to find detector automatically.
            detector = self._find_detector(data)
        elif detector.endswith(':output'):
            # Strip output pipeline if passed
            detector = detector[:-7]

        self._detector_name = detector

        if detector in data.control_sources:
            # Some very early data may not have the control source yet.
            self._control_src = data[detector]
        else:
            self._control_src = None

        self._instrument_src = data[f'{detector}:output']
        self._pulses = pulses

        if self._pulses is None:
            self._pulses = self.pulses()

    def __repr__(self):
        return "<{} {}>".format(
            type(self).__name__, self._detector_name)

    @classmethod
    def _find_detector(cls, data):
        """Try to find detector source."""

        detectors = set()

        for source in data.instrument_sources:
            m = cls._instrument_re.match(source)
            if m is not None:
                detectors.add(m[1])

        if len(detectors) > 1:
            raise ValueError('multiple detectors found, please pass one '
                             'explicitly:\n' + ', '.join(sorted(detectors)))
        elif detectors:
            return detectors.pop()

        raise ValueError('no detector found, please pass one explicitly')

    def _align_pulse_index(self, kd, pulse_dim):
        """Align pulse index to KeyData.

        Args:
            kd (extra_data.KeyData): KeyData object to align to.
            pulse_dim ({pulseId, pulseIndex, time}, optional): Label
                for pulse dimension, pulse ID by default.

        Returns:
            (pandas.Index or None): Index from internal pulses component
                aligned to KeyData object in terms of trains.
        """

        if (data_counts := kd.data_counts()).sum() == 0:
            return

        pulse_counts = self._pulses.pulse_counts()

        # Merge both counts in a dataframe joined over train ID.
        entry_counts = pd.merge(
            pulse_counts.rename('by_pulses'),
            data_counts.rename('by_data'),
            how='outer', left_index=True, right_index=True
        ).fillna(-1).astype(int)

        # Only consider rows with data counts.
        entry_counts = entry_counts[entry_counts['by_data'] > 0]

        if (entry_counts['by_pulses'] == -1).any():
            # Missing one or more trains in the pulse information.
            raise ValueError('missing pulse information for one or more '
                             'trains with data')

        if len(entry_counts) < min(len(data_counts), len(pulse_counts)):
            # Missing counts in one or more trains in data, select data
            # and pulses down.
            from extra_data import by_id
            train_sel = by_id[entry_counts.index.to_numpy()]

            pulses = self._pulses.select_trains(train_sel)
            kd = kd.select_trains(train_sel)
        else:
            pulses = self._pulses

        return pulses.build_pulse_index(pulse_dim)

    def _build_reduced_pd(self, data, index, entry_level):
        """Get variable-length data as series or dataframe.

        Reduce data with a variable number of actual entries among a
        fixed number of potential entries to build a linear pandas
        Series (for scalar floating types) or DataFrame (for structured
        data containing at least one floating type). The floating type
        is necessary for NaN determination of valid entries.

        The data must be 2-dimensional, with the first axis is assumed
        to be an entry ("pulse") dimension and the second to only
        contain actual data up to a variable number for each entry.

        Args:
            data (ndarray): 2D data array.
            index (pd.Index or None): Index to apply to reduced data,
                or None to indicate no valid entries.
            entry_level (str or None): Additional index level inserted
                for each entry or omitted if None.

        Returns:
            (pandas.Series or pandas.DataFrame): Series objects are
                returned for scalar data and DataFrame objects for
                structured data.
        """

        num_rows = data.shape[1]
        raw = data.ravel()

        # Obtain mask which entries are actually filled.
        if data.dtype.fields is not None:
            pd_cls = pd.DataFrame
            field = {name for name, (dtype, offset)
                    in data.dtype.fields.items()
                    if np.issubdtype(dtype, np.floating)}.pop()

            finite_mask = np.isfinite(raw[field])
        elif np.issubdtype(data.dtype, np.floating):
            pd_cls = pd.Series
            finite_mask = np.isfinite(raw)
        else:
            raise TypeError(data.dtype)

        if data.size == 0 or index is None:
            return pd_cls(np.zeros(0, dtype=data.dtype))

        # Obtain pulse index and repeat each row by the number of
        # actual data in each entry, then convert to a dataframe.
        index_df = index.repeat(
            finite_mask.reshape(-1, num_rows).sum(axis=1)).to_frame()

        # Insert additional index level enumerating actual entries.
        if entry_level is not None:
            index_df[entry_level] = np.flatnonzero(finite_mask) % num_rows

        return pd_cls(raw[finite_mask], pd.MultiIndex.from_frame(index_df))

    @property
    def detector_name(self):
        return self._detector_name

    @property
    def control_source(self):
        """Control source."""

        if self._control_src is None:
            raise ValueError('component is initialized with earlier data '
                             'lacking a control source')

        return self._control_src

    @property
    def instrument_source(self):
        """Instrument source."""
        return self._instrument_src

    @property
    def rec_params(self):
        """Reconstruction parameters."""
        return {key.removesuffix('.value'): value
                for key, value
                in self.control_source.run_values().items()
                if not key.endswith('.timestamp')}

    def select_trains(self, trains):
        new_self = copy(self)

        if self._control_src is not None:
            new_self._control_src = self._control_src.select_trains(trains)
        new_self._instrument_src = self._instrument_src.select_trains(trains)
        new_self._pulses = self._pulses.select_trains(trains)

        return new_self

    def pulses(self):
        """Get pulse object based on internal triggers.

        Returns:
            (extra.components.DldPulses): Pulse object based on
                constructed trigger information.
        """

        from .pulses import DldPulses

        if isinstance(self._pulses, DldPulses):
            return self._pulses

        return DldPulses(self._instrument_src)

    def triggers(self):
        """Get triggers as dataframe.

        Returns:
            (pandas.DataFrame): Constructed trigger information as
                dataframe.
        """

        return self.pulses().triggers()

    def edges(self, channel_index=True, pulse_dim='pulseId'):
        """Get raw edges as series or dataframe.

        Args:
            channel_index (bool, optional): Whether to insert the edge
                channel as index level and return Series object
                (default), or as column and return DataFrame object.
            pulse_dim ({pulseId, pulseIndex, time}, optional): Label
                for pulse dimension, pulse ID by default.

        Returns:
            (pd.Series or pd.DataFrame): Raw edge positions as Series
                (indexed by channel) or DataFrame (with channel as
                column) object.
        """

        kd = self._instrument_src['raw.edges']
        index = self._align_pulse_index(kd, pulse_dim)
        data = kd.ndarray()
        raw_edges = [self._build_reduced_pd(data[:, i, :], index, None)
                     for i in range(data.shape[1])]

        index = pd.MultiIndex.from_frame(
            pd.concat([e.index.to_frame() for e in raw_edges]))
        positions = np.concatenate([e.to_numpy() for e in raw_edges])
        channels = np.arange(len(raw_edges)).repeat(
            [len(e) for e in raw_edges])

        edges = pd.DataFrame(dict(position=positions, channel=channels),
                             index=index)
        edges.sort_values(by='position', inplace=True)
        edges.sort_index(inplace=True)

        if channel_index:
            edges.set_index('channel', drop=True, append=True, inplace=True)
            return edges['position']
        else:
            return edges

    def signals(self, pulse_dim='pulseId'):
        """Get reconstructed signals as dataframe.

        Args:
            pulse_dim ({pulseId, pulseIndex, time}, optional): Label
                for pulse dimension, pulse ID by default.

        Returns:
            (pandas.DataFrame) Detector signals.
        """

        return self._build_reduced_pd(
            (kd := self._instrument_src['rec.signals']).ndarray(),
            self._align_pulse_index(kd, pulse_dim), 'signalIndex')

    def hits(self, pulse_dim='pulseId'):
        """Get reconstructed hits as dataframe.

        Args:
            pulse_dim ({pulseId, pulseIndex, time}, optional): Label
                for pulse dimension, pulse ID by default.

        Returns:
            (pandas.DataFrame) Detector hits.
        """

        return self._build_reduced_pd(
            (kd := self._instrument_src['rec.hits']).ndarray(),
            self._align_pulse_index(kd, pulse_dim), 'hitIndex')
