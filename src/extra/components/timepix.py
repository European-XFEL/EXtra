
from collections import defaultdict
from io import IOBase
from os import PathLike
import re

import numpy as np
import pandas as pd

from extra_data import by_id


# Pulse indices used internally for virtual leading and trailing pulses.
TRAILING_PULSE_INDEX = 2**31 - 1
LEADING_PULSE_INDEX = -TRAILING_PULSE_INDEX - 1


class Timepix3:
    """Interface to Timepix3 event data.

    Timepix3 is an event-based pixel detectors, generating a stream of
    events of when and how long a singular pixel is detected. This
    component gives convenient [pd.dataframe][pandas.DataFrame]-based
    access to directly acquired raw pixel events, as well as centroided
    events available after automatic processing. These centroids attempt
    to group events in time and space that belong to the same particle
    impact to a single event.

    The component can be initialized with either the raw data source,
    the virtual calibrated source containing the centroided results or
    both. Pulse pattern information are necessary in both cases to sort
    events into pulses, as they are originally recorded by train.

    Args:
        data (extra_data.DataCollection): Data to access Timepix3 data from.
        detector (str or tuple, optional): Name of the detector, which
            may be the domain (first part of the source name up to the
            first slash) or a tuple with the explicit raw and centroided
            source name. If omitted, an attempt is made to detect them
            automatically.
        pulses (extra.components.pulses.PulsePattern, optional): Pulse
            component to pull pulse information. If omitted, an
            [XrayPulses][extra.components.XrayPulses] object is
            constructed from the data.
    """

    # Only support single-chip detectors for now.
    _instrument_re = re.compile(
        r'^(\w{3,6}_\w+_TIMEPIX)\/(CAM|DET|CAL)\/\w+:daqOutput.chip0$')

    def __init__(self, data, detector=None, pulses=None, **kwargs):
        # Always run detection to potentially find the raw and
        # centroided data sources depending on passed prefix.
        self._detector_name, sources = self._find_detector(
            data, detector or '')

        self._raw_control_src = None
        self._raw_instrument_src = None
        self._centroids_control_src = None
        self._centroids_instrument_src = None

        for source in sources:
            if '/DET/' in source or '/CAM/' in source:
                self._raw_instrument_src = data[source]

                if (s := source[:source.rfind(':')]) in data.control_sources:
                    self._raw_control_src = data[s]
            elif '/CAL/' in source:
                self._centroids_instrument_src = data[source]

                if (s := source[:source.rfind(':')]) in data.control_sources:
                    self._centroids_control_src = data[s]

        if self._raw_instrument_src is not None:
            # For raw data, several keys must be accessed as well as
            # potentially correlated with centroid data. For efficient
            # access with pasha, we keep around a minimal DataCollection
            # selected down to those trains with raw data.

            selection = {self._raw_instrument_src.source: {
                'data.x', 'data.y', 'data.toa', 'data.tot'}}

            if self._has_centroid_labels():
                selection[self._centroids_instrument_src.source] = {
                    'data.labels'}

            raw_train_ids = self.raw_size_key.drop_empty_trains().train_ids
            self._raw_instrument_dc = data \
                .select(selection).select_trains(by_id[raw_train_ids])

        if pulses is None:
            from . import XrayPulses
            pulses = XrayPulses(data)

        # This pulse component is not yet train-aligned, as raw and
        # centroided data may cover different trains.
        self._pulses = pulses

    def __repr__(self):
        data_labels = []

        if self._raw_instrument_src is not None:
            data_labels.append('raw')

        if self._centroids_instrument_src is not None:
            data_labels.append('centroids')

        return "<{} {}: {}>".format(type(self).__name__, self._detector_name,
                                    ', '.join(data_labels))

    def _has_centroid_labels(self):
        """Whether centroid labels are available."""
        return (self._centroids_instrument_src is not None and
                'data.labels' in self._centroids_instrument_src)

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

    @classmethod
    def _find_detector(cls, data, prefix_or_source):
        """Try to find detector source."""

        if isinstance(prefix_or_source, tuple) and len(prefix_or_source) == 2:
            # Explicit tuple of source names.

            def _find_sources(data, source):
                sd = data[source]
                domain = source.partition('/')[0]

                fast_source = f'{source}:daqOutput.chip0'
                if sd.is_control and fast_source in data.all_sources:
                    return domain, fast_source
                elif sd.is_instrument:
                    return domain, source

            raw_source, centroided_source = prefix_or_source

            if not raw_source and not centroided_source:
                raise ValueError('tuple of source names may not be all empty')

            found_sources = []

            if raw_source:
                domain, source = _find_sources(data, raw_source)
                found_sources.append(source)

            if centroided_source:
                domain, source = _find_sources(data, centroided_source)
                found_sources.append(source)

            return domain, found_sources

        elif isinstance(prefix_or_source, str):
            # Detector domain.

            detectors = defaultdict(list)

            for source in data.instrument_sources:
                m = cls._instrument_re.match(source)
                if m is not None and m[1].startswith(prefix_or_source):
                    detectors[m[1]].append(source)

            if len(detectors) > 1:
                raise ValueError('multiple detector domains found, please '
                                 'pass one explicitly via the `detector` '
                                 'argument:\n' + ', '.join(sorted(detectors)))
            elif detectors:
                return next(iter(detectors.items()))

            if prefix_or_source:
                raise ValueError(f'no detector sources found for '
                                 f'{prefix_or_source}, please pass explicit '
                                 f'source name(s)')
            else:
                raise ValueError('no detector detected, please narrow the '
                                 'search with the `detector` argument')

        raise TypeError(
            'detector may be a string, tuple of two strings or empty')

    @staticmethod
    def _sort_timepix_data(train_id, pids, toa_offset, rep_rate, timewalk_lut,
                           toa_in, tot_in, toa, tof, pulse_indices, pos):
        """Sort Timepix data and bin into pulses."""

        # Convert input ToA to μs and apply custom offset.
        toa_in *= 1e6
        toa_in -= toa_offset

        # Sort input ToA and load into output.
        pos[:] = np.argsort(toa_in)
        toa[:] = toa_in[pos]

        # Apply raw timewalk correction if applicable.
        if timewalk_lut is not None:
            toa -= timewalk_lut[(tot_in[pos] // 25) - 1] * 1e6

        # Determine pulse boundaries.
        try:
            train_pids = pids.loc[train_id].to_numpy()
        except KeyError:
            # No pulses present in this train, move all rows to trailing.
            pulse_starts = np.array([-1])
            last_pulse_end = -1
        else:
            # Start time of all pulses.
            pulse_starts = (train_pids - train_pids[0]) / rep_rate

            # End time of the last pulse.
            if len(pulse_starts) > 1:
                # Add the largest pulse difference to the last start.
                last_pulse_end = pulse_starts[-1] + np.diff(pulse_starts).max()
            else:
                # Take largest ToA as pulse end if there is only a
                # single pulse in the train.
                last_pulse_end = toa.max()

        # Sort ToA into pulse bins, except for the last pulse.
        indices = pd.cut(pd.Series(toa), pulse_starts, labels=False,
                         include_lowest=True)

        # Any negative ToA is before the first pulse, assign temporarily
        # to the first pulse for ToF computation below as it uses the
        # same offset.
        indices[toa < 0.0] = 0.0

        # Assign magic pulse index to rows are after the last pulse.
        indices[toa > last_pulse_end] = TRAILING_PULSE_INDEX

        # Anything still NaN is IN the last pulse, as pd.cut above only
        # covered pulse beginnings.
        indices.fillna(len(pulse_starts) - 1, inplace=True)

        # Assign result to output (may cause an int cast) and generate
        # mask for pulses we can offset with pulse_starts above, i.e.
        # the actual pulses and virtual leading pulse and not the
        # virtual trailing pulse).
        pulse_indices[:] = indices
        mask = pulse_indices < TRAILING_PULSE_INDEX

        # Subtract pulse starts for ToF of non-trailing hits, and
        # subtract the last pulse's end from trailing hits.
        tof[:] = toa
        tof[mask] -= pulse_starts[pulse_indices[mask]]
        tof[~mask] -= last_pulse_end

        # Finally correct virtual leading pulse, those with negative
        # ToA/ToF, to their own magic pulse index.
        pulse_indices[toa < 0.0] = LEADING_PULSE_INDEX

        return pos

    @staticmethod
    def _build_pulse_index(train_ids, pulses, pulse_dim, rows_per_train,
                           row_pidx, out_of_pulse_events):
        """Build pulse index aligned to Timepix hits or centroids."""

        # Determine additional pulse indices and values required for
        # leading and trailing hits, i.e. before and after "pulses", or
        # create a mask to exclude them.
        # The indices are used internally and denote those cases with
        # -1 and INT32_MAX, respectively. The returned index however
        # is forced to float using -np.inf and +np.inf.
        # If either virtual pulses are disabled, they are masked out.
        if out_of_pulse_events:
            mask = np.s_[:]
            extra_pulse_idx = [LEADING_PULSE_INDEX, TRAILING_PULSE_INDEX]
            extra_pulse_vals = [-np.inf, +np.inf]
        else:
            mask = (row_pidx != LEADING_PULSE_INDEX) & \
                (row_pidx != TRAILING_PULSE_INDEX)
            extra_pulse_idx = []
            extra_pulse_vals = []

        # Build the initial pulse index with the pulse pattern
        # component, which is by-pulse. If either virtual pulses are
        # enabled, force the pulse dimension to be float to account for
        # its special values.
        pulse_index = pulses.build_pulse_index(
            pulse_dim, pulse_dtype=np.float64 if extra_pulse_vals else None)

        if extra_pulse_vals:
            # For now, only the generic pulse pattern and their custom
            # fields are supported. When needed, add default values for
            # the leading/trailing pseudo-pulses.
            if set(pulse_index.names[2:]) - {'fel', 'ppl'}:
                raise ValueError('custom pulse index labels are not supported '
                                 'when including out of pulse events')

            custom_pulse_vals = []
            if 'fel' in pulse_index.names:
                custom_pulse_vals.append([False])

            if 'ppl' in pulse_index.names:
                custom_pulse_vals.append([False])

            orig_names = pulse_index.names
            pulse_index = pulse_index.append(pd.MultiIndex.from_product(
                [train_ids, extra_pulse_vals, *custom_pulse_vals]
            )).sort_values()
            pulse_index.names = orig_names  # Restore names after append.

        # Count the number of hits per train and pulse that actually
        # occured - this may be missing trains and pulses without any
        # hits.
        actual_rows_per_pulse = pd.DataFrame(data={
            'trainId': np.repeat(train_ids, rows_per_train)[mask],
            'pulseIndex': row_pidx[mask]
        }).groupby(['trainId', 'pulseIndex']).size()

        # Build an index for each possibly occuring train and pulse.
        # Beginning from the pulse ID index from the pulse pattern
        # component, drop any custom levels, append the extra values for
        # leading/trailing pulses above and sort again.
        pulse_ids = pulses.pulse_ids()
        counts_index = pulse_ids.index \
            .droplevel(list(range(2, pulse_ids.index.nlevels))) \
            .append(pd.MultiIndex.from_product([train_ids, extra_pulse_idx])) \
            .sort_values()

        # Use the data frame with train/pulse data from above to count
        # hits by pulse for any possibly occuring pulse - this is
        # important as there may be pulses in the pulse index without
        # any hits!
        rows_per_pulse = pd.Series(np.zeros(len(counts_index), dtype=int),
                                   index=counts_index)
        rows_per_pulse[actual_rows_per_pulse.index] = actual_rows_per_pulse

        # Repeat the initial data index (which is by pulse) accordingly
        # for each hit, and add in a new level for a hit index.
        index_df = pulse_index.repeat(rows_per_pulse).to_frame(index=False)

        return pd.MultiIndex.from_frame(index_df), mask

    @property
    def detector_name(self):
        return self._detector_name

    @property
    def raw_control_src(self):
        if self._raw_control_src is None:
            raise ValueError('raw control data is not available as component '
                             'was initialized with data not containing it')

        return self._raw_control_src

    @property
    def raw_instrument_src(self):
        if self._raw_instrument_src is None:
            raise ValueError('raw pixel events are not available as component '
                             'was initialized with data not containing it')

        return self._raw_instrument_src

    @property
    def raw_x_key(self):
        return self.raw_instrument_src['data.x']

    @property
    def raw_y_key(self):
        return self.raw_instrument_src['data.y']

    @property
    def raw_toa_key(self):
        return self.raw_instrument_src['data.toa']

    @property
    def raw_tot_key(self):
        return self.raw_instrument_src['data.tot']

    @property
    def raw_size_key(self):
        return self.raw_instrument_src['data.size']

    @property
    def centroids_control_src(self):
        if self._centroids_control_src is None:
            raise ValueError('centroid control data is not available as '
                             'component was initialized with data not '
                             'containing it')

        return self._centroids_control_src

    @property
    def centroids_instrument_src(self):
        if self._centroids_instrument_src is None:
            raise ValueError('centroids are not available as component was '
                             'initialized with data not containing it')

        return self._centroids_instrument_src

    @property
    def centroids_key(self):
        return self.centroids_instrument_src['data.centroids']

    @property
    def centroid_labels_key(self):
        return self.centroids_instrument_src['data.labels']

    @staticmethod
    def spatial_bins(min_pos=0, max_pos=255, bins_per_px=1):
        """Build suitable bin edges for Timepix position data.

        The bins are centered around pixel position to avoid binning
        artifacts otherwise commonly appearing with Timepix data.

        Args:
            min_pos (int, optional): Lower spatial boundary,
                0 by default.
            max_pos (int, optional): Upper spatial boundary,
                255 by default.
            bins_per_px (int, optional): Number of bins per pixel,
                1 by default

        Returns:
            bins (numpy.ndarray): Spatial bin edges.
        """

        return np.arange(min_pos, max_pos + 2 / bins_per_px, 1 / bins_per_px) \
            - 0.5 / bins_per_px

    def pixel_events(self, pulse_dim='pulseId', toa_offset=0.0,
                     timewalk_lut=None, out_of_pulse_events=False,
                     extended_columns=False, parallel=None):
        """Get raw pixel events as dataframe.

        Loads the original raw pixel events as a dataframe and sorts
        them into pulses using the initialized pulse pattern component.

        By default, only events the pulse windows are included. The
        `out_of_pulse_events` arguments allows to extend this to events
        before the first and after the last pulse.

        An event is considered part of a particular pulse when its
        time-of-arrival is between the beginning of that and the next
        pulse For the last pulse, the single largest spacing between
        pulses is chosen while in case of a single pulse in the entire
        train, all pulses will be assigned to it. The `toa_offset`
        argument denotes the time-of-arrival at the start of the first
        pulse of the train.

        Args:
            pulse_dim ('pulseId' or 'pulseIndex' or 'pulseTime', optional):
                Which pulse coordinates to use in the index, pulse ID by
                default.
            toa_offset (float, optional): Time-of-arrival offset applied
                by train in microseconds, 0.0 by default.
            timewalk_lut (numpy.typing.ArrayLike or os.PathLike or None, optional):
                Timewalk LUT to correct ToA or path to a file
                compatible with np.load, none by default.
            out_of_pulse_events (bool, optional): Whether to include
                hits before the first pulse and after the last pulse in
                virtual pulses labelled -np.inf and np.inf, False by
                default. If enabled, the pulse index dimension will be
                forced to float.
            extended_columns (bool, optional): Whether to include the
                original time-of-arrival, readout position and centroid
                labels for each pixel event, False by default. Labels
                require centroiding data processed after Feburary 2024
                to be present.
            parallel (int or None, optional): Nunmber of parallel
                processes to use, by default 10 or a quarter of all cores
                whichever is lower. Any non-positive value or 1 disable
                parallelization.

        Returns:
            hits (pandas.DataFrame): Raw detector hits.
        """

        # Prepare aligned data sources.
        tids = self._raw_instrument_dc.train_ids
        pulses = self._pulses.select_trains(by_id[tids])
        pids = pulses.pulse_ids()
        machine_rep_rate = pulses.bunch_repetition_rate * 1e-6

        # Obtain hits per train and compute total number of hits.
        hits_per_train = self.raw_size_key.ndarray()
        num_hits = int(hits_per_train.sum())

        if (hits_per_train > self.raw_x_key.entry_shape[0]).any():
            # data.size may actually record more pixel events than can
            # be placed into array buffer. The device is supposed to
            # prevent this, but there is some commissioning data
            # affected by it.

            diff = hits_per_train - self.raw_x_key.entry_shape[0]
            max_deviation = diff.max()
            num_deviations = (diff > 0).sum()

            from warnings import warn
            warn(f'reported number of pixel events exceed buffer size in '
                 f'{num_deviations} trains by up to {max_deviation}',
                 category=RuntimeWarning, stacklevel=2)

        # Offset of each train in the by-hit buffers below.
        hit_train_offsets = np.zeros_like(hits_per_train)
        hit_train_offsets[1:] = np.cumsum(hits_per_train[:-1])

        # Buffers by hit for X, Y, ToA, ToF, ToT, pulse index, (original)
        # position and optionally centroid labels.
        psh = self._prepare_pasha(parallel)
        hits_x = psh.alloc(shape=num_hits, dtype=self.raw_x_key.dtype)
        hits_y = psh.alloc(shape=num_hits, dtype=self.raw_y_key.dtype)
        hits_toa = psh.alloc(shape=num_hits, dtype=self.raw_toa_key.dtype)
        hits_tof = psh.alloc(shape=num_hits, dtype=self.raw_toa_key.dtype)
        hits_tot = psh.alloc(shape=num_hits, dtype=self.raw_tot_key.dtype)
        hits_pidx = psh.alloc(shape=num_hits, dtype=np.int32)
        hits_pos = psh.alloc(shape=num_hits, dtype=np.int32)

        if extended_columns and self._has_centroid_labels():
            hits_label = psh.alloc(shape=num_hits, dtype=np.int32)
        else:
            hits_label = None

        if isinstance(timewalk_lut, (str, PathLike, IOBase)):
            timewalk_lut = np.load(timewalk_lut)

        def load_tpx_raw(wid, index, train_id, data):
            raw = data[self._raw_instrument_src.source]

            # Retrieve the number of pixel events in this train, making
            # sure to not exceed the buffer shape.
            count = min(hits_per_train[index], raw['data.x'].shape[0])

            if count == 0:
                return

            offset = hit_train_offsets[index]
            dest_slice = np.s_[offset:offset+count]

            sorted_idx = self._sort_timepix_data(
                train_id, pids, toa_offset, machine_rep_rate, timewalk_lut,
                raw['data.toa'][:count], raw['data.tot'][:count],
                hits_toa[dest_slice], hits_tof[dest_slice],
                hits_pidx[dest_slice], hits_pos[dest_slice])

            hits_x[dest_slice] = raw['data.x'][sorted_idx]
            hits_y[dest_slice] = raw['data.y'][sorted_idx]
            hits_tot[dest_slice] = raw['data.tot'][sorted_idx]

            if hits_label is not None:
                centroids = data[self._centroids_instrument_src.source]
                hits_label[dest_slice] = centroids['data.labels'][sorted_idx]

        # Load data and bin hits into pulses.
        psh.map(load_tpx_raw, self._raw_instrument_dc)

        # Build a properly aligned pulse index.
        index, mask = self._build_pulse_index(
            tids, pulses, pulse_dim, hits_per_train,
            hits_pidx, out_of_pulse_events)

        # Build the data frame with detector data and the prepared index.
        frame_data = {'x': hits_x[mask], 'y': hits_y[mask],
                      't': hits_tof[mask], 'tot': hits_tot[mask]}

        if extended_columns:
            frame_data['toa'] = hits_toa[mask]
            frame_data['pos'] = hits_pos[mask]

            if hits_label is not None:
                frame_data['label'] = hits_label[mask]

        df = pd.DataFrame(data=frame_data, index=index)

        return df

    def centroid_events(self, pulse_dim='pulseId', toa_offset=0.0,
                        out_of_pulse_events=False, extended_columns=False,
                        parallel=None):
        """Get centroided events as dataframe.

        Loads the processed centroided pixel events as a dataframe and
        sorts them into pulses using the initialized pulse pattern
        component. The arguments and behaviour are analogous to
        [pixel_events()][extra.components.Timepix3.pixel_events].

        Args:
            pulse_dim ('pulseId' or 'pulseIndex' or 'pulseTime', optional):
                Which pulse coordinates to use in the index, pulse ID by
                default.
            toa_offset (float, optional): Time-of-arrival offset applied
                by train in microseconds, 0.0 by default.
            out_of_pulse_events (bool, optional): Whether to include
                centroids before the first pulse and after the last
                pulse in virtual pulses labelled -np.inf and np.inf,
                False by default. If enabled, the pulse index dimension
                will be forced to float.
            extended_columns (bool, optional): Whether to include the
                average and maximal time-over-threshold as well
                as original time-of-arrival and labels for each
                centroid, False by default.
            parallel (int or None, optional): Nunmber of parallel
                processes to use, by default 10 or a quarter of all cores
                whichever is lower. Any non-positive value or 1 disable
                parallelization.

        Returns:
            centroids (pandas.DataFrame): Centroids.
        """

        # Prepare aligned data sources.
        kd = self.centroids_key.drop_empty_trains()
        tids = kd.train_ids
        pulses = self._pulses.select_trains(by_id[tids])
        pids = pulses.pulse_ids()
        machine_rep_rate = pulses.bunch_repetition_rate * 1e-6

        # Centroids (unfortunately) do not have a size key, so load
        # all of them into memory to obtain the centroid per train
        # from the data via finite checks
        centroids = kd.ndarray()

        # Obtain centroid per train and compute total number of centroids.
        rows_per_train = centroids.shape[1]
        centroids_per_train = np.isfinite(centroids['x']).sum(axis=1)
        num_centroids = int(centroids_per_train.sum())

        # Offset of each train in the by-centroid buffers below.
        centroids_train_offsets = np.zeros_like(centroids_per_train)
        centroids_train_offsets[1:] = np.cumsum(centroids_per_train[:-1])

        # Buffers by centroid for pulse index, row (index) and label.
        psh = self._prepare_pasha(parallel)
        centroids_toa = psh.alloc(
            shape=num_centroids, dtype=self.centroids_key.dtype['toa'])
        centroids_tof = psh.alloc(
            shape=num_centroids, dtype=self.centroids_key.dtype['toa'])
        centroids_pidx = psh.alloc(shape=num_centroids, dtype=np.int32)
        centroids_rows = psh.alloc(shape=num_centroids, dtype=np.int32)
        centroids_label = psh.alloc(shape=num_centroids, dtype=np.int32)

        def process_tpx_centroids(wid, index, centroids):
            count = centroids_per_train[index]

            if count == 0:
                return

            offset = centroids_train_offsets[index]
            dest_slice = np.s_[offset:offset+count]

            sorted_idx = self._sort_timepix_data(
                tids[index], pids, toa_offset, machine_rep_rate, None,
                centroids['toa'][:count], None,  # ToT not required
                centroids_toa[dest_slice], centroids_tof[dest_slice],
                centroids_pidx[dest_slice], centroids_label[dest_slice])

            centroids_rows[dest_slice] = index * rows_per_train + sorted_idx

        # Process centroids to sort by ToA and bin into pulses.
        psh.map(process_tpx_centroids, centroids)

        # Build a properly aligned pulse index.
        index, mask = self._build_pulse_index(
            tids, pulses, pulse_dim, centroids_per_train,
            centroids_pidx, out_of_pulse_events)

        # Build the data frame with centroids and the prepared index.
        df = pd.DataFrame.from_records(
            centroids.ravel()[centroids_rows][mask], index=index,
            exclude=(['tot_avg', 'tot_max', 'toa']
                     if not extended_columns else []))
        df.insert(2, 't', centroids_tof[mask])
        df.rename(columns={'size': 'centroid_size'}, inplace=True)

        if extended_columns:
            # Re-order centroid size.
            df.insert(4, 'centroid_size', df.pop('centroid_size'))

            df['toa'] = centroids_toa[mask]  # Overwrite with modified data.
            df.insert(7, 'toa', df.pop('toa'))  # Re-order ToA
            df.insert(3, 'tot', df.pop('tot'))  # Re-order ToT.
            df['label'] = centroids_label[mask]

        return df
