"""Fast ADC component"""

from functools import cached_property
import re

import numpy as np
import pandas as pd
import xarray as xr

from extra_data import by_id, DataCollection
from extra_data.read_machinery import roi_shape
from .pulses import XrayPulses
from .utils import _isinstance_no_import


class AdcRawChannel:
    """A high-level interface for the (raw) output of fast ADC channels.
    """

    # The maximum rate of 4.5 MHz (see extra.components.pulses)
    _bunch_repetition_divider = 288
    _fast_adc_divider = 12
    _bunch_repetition_rate = 1.3e9 / _bunch_repetition_divider
    _fast_adc_clock_rate = 1.3e9 / _fast_adc_divider
    _clock_ratio = _bunch_repetition_divider / _fast_adc_divider
    _adc_regex = re.compile(r'^.*\/ADC\/[0-9]+:channel_([0-9]+).output$')

    def __init__(
        self,
        data: DataCollection,
        adc_channel: str | int, *,
        pulses: bool | None = None,  # In the future, + manual pulses
        first_pulse_offset: None | int = None,
        drop_empty_trains: bool = False,
    ):

        self._adc_channel = adc_channel
        self._first_pulse_offset = first_pulse_offset

        self._instrument_sources = data.instrument_sources
        self._control_sources = data.control_sources
        self._inst_source_name = self._validate_inst_source()
        self._ctrl_source_name = self._validate_ctrl_source()
        self._inst_sourcedata = data[self._inst_source_name]
        self._ctrl_sourcedata = data[self._ctrl_source_name]

        # Would be better to do this lazily but here we are... for example,
        # the .train_data method does not require it.
        self._pulses = self._validate_pulses_kwarg(data, pulses)

        # Check that the required keys are present and get keydata
        self._inst_keydata = self._get_inst_keydata()

        # Get two useful control source properties
        self._sample_first_bunch = self._get_sample_first_bunch()
        self._number_of_samples = self._get_number_of_samples()

    @property
    def samples_per_pulse(self):
        """Blah blah blah"""

        if self._pulses is None:
            raise RuntimeError("This was not initialized with a pulse pattern "
                               ". Please ...")
        pulse_period = self._pulses.pulse_periods().unique()

        if len(pulse_period) > 1:
            raise ValueError("There more than one pulse period between "
                             "selection. To proceed, split the trains "
                             "into groups with a common period and try "
                             "again.") from None

        return int(pulse_period[0])

    @property
    def _channel_number(self):
        """Extract the channel number from the instrument source name."""

        # Could also do AdcRawChannel._adc_regex or define a @classmethod
        # if this is deemed unclear
        result = self._adc_regex.search(self._inst_source_name)

        if result:
            return int(result.group(1))

        raise ValueError("Couldn't extract the channel number from "
                         "the source name.")

    def _get_inst_keydata(self):
        """Make sure the instrument raw data key is present and if so set
        raw."""

        key = 'data.rawData'
        if key not in self._inst_sourcedata:
            raise KeyError(f"Source {self._inst_source_name} does not "
                           f"contain a '{key}' key.")

        return self._inst_sourcedata[key]

    def _get_sample_first_bunch(self):
        """Check if an explicit offset was passed otherwise extract it."""

        if isinstance(self._first_pulse_offset, int):
            value = self._first_pulse_offset
        elif self._first_pulse_offset is None:
            value = self._get_unique_value(
                self._ctrl_sourcedata, 'sampleFirstBunch.value')
        else:
            raise TypeError("The `first_pulse_index` argument should be "
                            "an integer or None, you passed an object of "
                            "type {type(first_pulse_index)}.")

        return value

    def _get_number_of_samples(self):
        return self._get_unique_value(
            self._ctrl_sourcedata, 'numberRawSamples.value')

    @staticmethod
    def _get_unique_value(sourcedata, key):
        """General method that takes a key and returns a unique
        value in the KeyData.ndarray()."""

        source_type = None
        if sourcedata.is_control:
            source_type = 'control'
        elif sourcedata.is_instrument:
            source_type = 'instrument'
        else:
            raise ValueError(
                f"The SourceData I received, {sourcedata.canonical_name} "
                "is not a control or instrument source.")

        if key not in sourcedata:
            raise KeyError(f"The {key} key of the {source_type} source "
                           f"{sourcedata.canonical_name} is missing.")

        array = sourcedata[key].ndarray()
        array = np.unique(array)

        if len(array) > 1:
            raise ValueError("It looks like the `sampleFirstBunch.value` key "
                             "changed during the run. Please specify a value "
                             f"explicitly. I found {array}.")
        return int(array[0])

    def _validate_pulses_kwarg(self, data, pulses):
        """Offloads pulses validation logic from __init__"""

        if pulses is False:
            pulses = None
        else:
            # Write unit tests: no BPT, BPT_DECODER, full BPT, & both
            try:
                pulses = XrayPulses(data)
            except ValueError:
                raise ValueError("Only runs with a timeserver or pulse "
                                 "pattern decoder source are supported "
                                 "at the moment.") from None

        return pulses

    @property
    def _control_sourcedata(self):
        inst_src_name = self._inst_sourcedata.canonical_name
        ctrl_src_name = inst_src_name.split(':')[0]
        if ctrl_src_name not in self._control_sources:
            raise AttributeError(f"The control source {ctrl_src_name} "
                                 "does not exist.")
        return self._data[ctrl_src_name]

    @property
    def pulses(self):
        return self._pulses

    @property
    def pulse_counts(self):
        return self.pulses.pulse_counts()

    @cached_property
    def pulse_ids(self):
        return self.pulses.pulse_ids()

    @cached_property
    def pulse_periods(self):
        return self.pulses.pulse_periods()

    def _validate_inst_source(self):
        """Check that source is in instrument_sources"""

        found = set(filter(self._adc_regex.search,
                           self._instrument_sources))

        adc_channel = self._adc_channel

        candidates = set()
        for channel in found:
            if adc_channel in channel:
                candidates.add(channel)

        len_ = len(candidates)
        if len_ == 0:
            raise AttributeError(
                    f"The source you provided, {adc_channel}, cannot "
                    f"be matched to one of instrument sources.")
        if len_ > 1:
            raise AttributeError(
                    f"There are too many sources that match the "
                    f"source you provided, {adc_channel}. Namely "
                    f"{candidates}. Please be more specific.")

        return candidates.pop()

    def _validate_ctrl_source(self):
        """Check that the corresponding control source is present"""

        ctrl_name = self._inst_source_name.split(':')[0]

        if ctrl_name not in self._control_sources:
            raise KeyError(
                "The corresponding control source to the instrument "
                f"{self._inst_source_name}, which should be called "
                f"{ctrl_name}, is missing from `.control_sources`.")

        return ctrl_name

    def train_data(self, labelled=True, out=None):
        """Return train data"""

        out = np.empty((self._inst_keydata.shape))

        offset = 0
        for chunk in self._inst_keydata.split_trains(trains_per_part=200):
            # In AdqRawChannel, there is a BEWARE about uint64, what is this
            # about? Doing `type(chunk.shape[0])` gives int... trainIds are
            # uint64 but...
            num_trains = chunk.shape[0]
            this_slice = slice(offset, offset + num_trains)
            offset += num_trains
            out[this_slice] = chunk.ndarray()

        if not labelled:
            return out

        coords = {'trainId': self._inst_keydata.train_id_coordinates()}
        coords.update({'sample': np.arange(out.shape[1])})
        return xr.DataArray(out, coords=coords)

    def pulse_data(self, labelled=True, out=None):
        """Identify all the pulses and return an array containing them."""

        samples = self.pulse_counts * self.pulse_periods
        samples *= AdcRawChannel._base_samples_per_bunch

        samples += self._sample_first_bunch

        width = int(samples.max())
        roi_ = slice(None, width)
        out = np.empty((self._inst_keydata.shape[0], width))

        offset = 0
        for chunk in self._inst_keydata.split_trains(trains_per_part=200):
            # In AdqRawChannel, there is a BEWARE about uint64, what is this
            # about? Doing `type(chunk.shape[0])` gives int... trainIds are 
            # uint64 but...
            num_trains = int(chunk.shape[0])
            this_slice = slice(offset, offset + num_trains)
            offset += num_trains
            out[this_slice] = chunk.ndarray(roi=roi_)

        if not labelled:
            return out

        coords = {'trainId': self._inst_keydata.train_id_coordinates()}
        coords.update({'sample': np.arange(width)})
        return xr.DataArray(out, coords=coords)

    def find_peaks(self, labelled=True):
        """Find the peak heights and locations in the trace"""

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
