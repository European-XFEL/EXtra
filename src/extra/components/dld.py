
from copy import copy
import re

import numpy as np
import pandas as pd

from extra.data import KeyData
from .utils import _isinstance_no_import


class DelayLineDetector:
    """Interface for processed delay line detector data.

    Raw analog data from quad and hex delay line detectors acquired with
    GHz digitizers can be reconstructed into hits by the European XFEL
    offline processing machinery. This component allows convenient
    access to the resulting sparse data as pulse-labeled pandas series
    and dataframes.

    Note that this component is not able to access data saved by the
    proprietary SurfaceConcepts TDC integration.

    Args:
        data (extra_data.DataCollection): Data to access DLD data from.
        detector (str, optional): Source name of the detector, only
            needed if the data includes more than one.
        pulses (extra.components.pulses.PulsePattern, optional): Pulse
            component to pull pulse information. If omitted, it is
            constructed from the internal trigger information using any
            remaining additional keyword argument.
    """

    _instrument_re = re.compile(r'^(\w{3}_\w+_DLD\d\/DET\/\w+):output?$')

    def __init__(self, data, detector=None, pulses=None, **kwargs):
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
            self._pulses = self.pulses(**kwargs)

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
            pulse_dim ({pulseId, pulseIndex, pulseTime}, optional):
                Label for pulse dimension, pulse ID by default.

        Returns:
            Index (pandas.Index or None): Internal pulses component
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

        if len(entry_counts) < len(pulse_counts):
            # Missing one or more trains in the actual data compared
            # to the pulse information, select pulses down.
            from extra_data import by_id
            train_sel = by_id[entry_counts.index.to_numpy()]

            pulses = self._pulses.select_trains(train_sel)
        else:
            pulses = self._pulses

        return pulses.build_pulse_index(pulse_dim)

    def _build_reduced_pd(self, data, index, entry_level=None, mask_func=None):
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
            data (numpy.typing.ArrayLike): 2D data array.
            index (pd.Index or None): Index to apply to reduced data,
                or None to indicate no valid entries.
            entry_level (str or None): Additional index level inserted
                for each entry or omitted if None.
            mask_func (callable or None): Additional mask applied to
                data before reduction, must be a callable taking the
                raveled input data as an argument and return an equal-
                length boolean array.

        Returns:
            result (pandas.Series or pandas.DataFrame): Series objects
                are returned for scalar data and DataFrame objects for
                structured data.
        """

        num_rows = data.shape[-1]
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

        if mask_func is not None:
            finite_mask &= mask_func(raw)

        if data.size == 0 or index is None:
            return pd_cls(np.zeros(0, dtype=data.dtype))

        # Obtain pulse index and repeat each row by the number of
        # actual data in each entry, then convert to a dataframe.
        index_df = index.repeat(
            finite_mask.reshape(-1, num_rows).sum(axis=1)).to_frame()

        # Insert additional index level enumerating actual entries.
        if entry_level is not None:
            index_df[entry_level] = np.flatnonzero(finite_mask) % num_rows

        return pd_cls(np.ascontiguousarray(raw[finite_mask]),
                      pd.MultiIndex.from_frame(index_df))

    @staticmethod
    def insert_aligned_columns(df, columns):
        """Add pulse-indexed data to reduced dataframe.

        Args:
            df (pandas.DataFrame): Frame to insert columns into.
            columns (dict): Mapping of column name to labeled 1D data to
                insert, may be pandas series, xarray DataArray or
                KeyData. Must be re-indexable by internal train or pulse
                index and data is repeated accordingly.

        Returns:
            (NoneType): None
        """

        # Compute these lazily.
        num_per_pulse = None
        num_per_train = None

        for name, data in columns.items():
            if isinstance(data, KeyData):
                data = data.series()

            elif _isinstance_no_import(data, 'xarray', 'DataArray'):
                data = data.to_series()

            elif not isinstance(data, pd.Series):
                raise ValueError('columns must be given as KeyData or '
                                 'xarray.DataArray or pandas.Series')

            if data.ndim > 1:
                raise ValueError('only 1D data can be aligned to frame')


            shared_index = [data_level for data_level, df_level
                            in zip(data.index.names, df.index.names)
                            if data_level == df_level]

            if shared_index[:2] == df.index.names[:2]:
                # Same pulse dimensions as the dataframe.
                if num_per_pulse is None:
                    num_per_pulse = df.groupby(
                        level=df.index.names[:-1]).size()

                align = num_per_pulse

            elif shared_index == ['trainId']:
                # Same train ID dimension as the dataframe.
                if num_per_train is None:
                    num_per_train = df.groupby(level=df.index.names[0]).size()

                align = num_per_train

            else:
                raise ValueError('index incompatible with dataframe')

            df[name] = data.reindex(align.index).repeat(align).to_numpy()

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

    def pulses(self, **kwargs):
        """Get pulse object based on internal triggers.

        Args:
            **kwargs (Any): Any keyword arguments are passed to the
                underlying [DldPulses][extra.components.DldPulses].

        Returns:
            pulses (extra.components.DldPulses): Pulse object based on
                constructed trigger information.
        """

        from .pulses import DldPulses

        if isinstance(self._pulses, DldPulses) and not kwargs:
            return self._pulses

        return DldPulses(self._instrument_src, **kwargs)

    def triggers(self):
        """Get triggers as dataframe.

        Returns:
            Triggers (pandas.DataFrame): Constructed trigger information
                as dataframe.
        """

        return self.pulses().triggers()

    def edges(self, channel_index=True, pulse_dim='pulseId'):
        """Get raw edges as series or dataframe.

        Args:
            channel_index (bool, optional): Whether to insert the edge
                channel as index level and return Series object
                (default), or as column and return DataFrame object.
            pulse_dim ('pulseId' or 'pulseIndex' or 'pulseTime', optional):
                Label for pulse dimension, pulse ID by default.

        Returns:
            edges (pd.Series or pd.DataFrame): Raw edge positions as
                series (indexed by channel) or dataframe (with channel
                as column) object.
        """

        kd = self._instrument_src['raw.edges']
        index = self._align_pulse_index(kd, pulse_dim)
        data = kd.ndarray()
        raw_edges = [self._build_reduced_pd(data[:, i, :], index)
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

    def signals(self, pulse_dim='pulseId', extra_columns={}, max_method=None):
        """Get reconstructed signals as dataframe.

        This data is primarily for detector diagnostics purposes and
        should generally not be used for scientific data analysis,
        please refer instead to reconstructed x, y, t data obtained from
        [hits()][extra.components.DelayLineDetector.hits].

        Args:
            pulse_dim ('pulseId' or 'pulseIndex' or 'pulseTime', optional):
                Label for pulse dimension, pulse ID by default.
            extra_columns (dict): Mapping of column name to labeled 1D
                data to insert, may be pandas series, xarray DataArray
                or KeyData. Must be re-indexable by internal train or
                pulse index and data is repeated accordingly.
            max_method (int or None, optional): Maximal reconstruction
                method to include in the result, by default all hits are
                included. Generally methods up to and including 10 can be
                considered safe and > 14 should be treated as risky,
                please consult processing reports for more details.

        Returns:
            Signals (pandas.DataFrame): Detector signals after
                reconstruction.
        """

        if max_method is not None:
            # rec.hits is needed to obtain the method mask.
            hits_raw = self._instrument_src['rec.hits'].ndarray().ravel()
            mask_func = lambda _: hits_raw['m'] <= max_method
        else:
            mask_func = None

        df = self._build_reduced_pd(
            (kd := self._instrument_src['rec.signals']).ndarray(),
            self._align_pulse_index(kd, pulse_dim), 'signalIndex',
            mask_func)

        if extra_columns:
            self.insert_aligned_columns(df, extra_columns)

        return df

    def hits(self, pulse_dim='pulseId', extra_columns={}, max_method=None):
        """Get reconstructed hits as dataframe.

        By default, this method only includes non-risky hit
        reconstructions in the returned dataset. Please refer to the
        `max_method` argument and the correspondig processing reports
        for more information

        Args:
            pulse_dim ('pulseId' or 'pulseIndex' or 'pulseTime', optional):
                Label for pulse dimension, pulse ID by default.
            extra_columns (dict): Mapping of column name to labeled 1D
                data to insert, may be pandas series, xarray DataArray
                or KeyData. Must be re-indexable by internal train or
                pulse index and data is repeated accordingly.
            max_method (int or None, optional): Maximal reconstruction
                method to include in the result, by default all hits are
                included. Generally methods up to and including 10 can
                be considered safe and > 14 should be treated as risky,
                please consult processing reports for more details.

        Returns:
            Hits (pandas.DataFrame): Detector hits after reconstruction.
        """

        if max_method is not None:
            max_method = int(max_method)
            mask_func = lambda rows: rows['m'] <= max_method
        else:
            mask_func = None

        df = self._build_reduced_pd(
            (kd := self._instrument_src['rec.hits']).ndarray(),
            self._align_pulse_index(kd, pulse_dim), 'hitIndex',
            mask_func)

        if extra_columns:
            self.insert_aligned_columns(df, extra_columns)

        return df
