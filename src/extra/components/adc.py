"""Fast ADC component"""

from typing import Literal
import re

import numpy as np
from xarray import DataArray

from extra_data import by_id, DataCollection, KeyData, SourceData
from extra_data.read_machinery import roi_shape
from .pulses import XrayPulses, PulsePattern
from .utils import _isinstance_no_import


class AdcRawChannel:
    """A high-level interface for the (raw) output of fast ADC channels.
    """

    # The maximum rate of 4.5 MHz (see extra.components.pulses)
    _bunch_repetition_divider = 288
    # The Fast ADC boards run at 108.3333 MHz (see ...)
    _fast_adc_divider = 12
    _bunch_repetition_rate = 1.3e9 / _bunch_repetition_divider
    _fast_adc_clock_rate = 1.3e9 / _fast_adc_divider
    _clock_ratio = _bunch_repetition_divider // _fast_adc_divider
    _adc_regex = re.compile(r'^.*\/ADC\/[0-9]+:channel_([0-9]+).output$')

    def __init__(
        self,
        data: DataCollection,
        adc_channel: str | int,
        *,
        pulses: bool = True,  # In the future, + manual pulses
        first_pulse_offset: None | int = None,
    ):

        self._adc_channel = str(adc_channel)
        self._first_pulse_offset = first_pulse_offset

        self._instrument_sources = data.instrument_sources
        self._control_sources = data.control_sources

        # Check that the required instrument and control sources are present
        self._inst_source_name = self._validate_inst_source()
        self._ctrl_source_name = self._validate_ctrl_source()

        # If all is in order, assign the SourceData objects
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
    def pulses(self) -> PulsePattern | Literal[False]:
        """The pulse pattern (XrayPulses) if present, False if not."""

        return self._pulses

    @property
    def pulse_period(self) -> int:
        """Check that there is one unique pulse period and extract it"""

        self._check_pulses_not_false()

        try:
            pulse_period = self._pulses.pulse_periods().unique()
        except ValueError:
            # PulsePattern.pulse_period raises ValueError when there is only
            # one pulse per train, if so, catch it and set pulse period to
            # to a reasonable value, 4 say
            pulse_period = 4

        # TODO: Probably should do this in the ._validate_pulses_kwarg method
        if len(pulse_period) > 1:
            raise ValueError(
                "There is more than one pulse period in this selection. To "
                "proceed, split the trains into selections with a common "
                "period and try again.")

        pulse_period = int(pulse_period[0])

        return pulse_period

    def _check_pulses_not_false(self) -> None:
        """Raises ValueError if ._pulses is False."""

        if self.pulses is False:
            raise ValueError(
                "This component was not initialized with a "
                "pulse pattern.") from None

    @property
    def samples_per_pulse(self) -> int:
        """Compute the number of samples per pulse."""

        # Raise a ValueError if user invokes this method when pulses = False
        self._check_pulses_not_false()

        return self.pulse_period * self._clock_ratio

    @property
    def pulses_per_train(self) -> int:
        """The number of pulses per train."""

        # Raise a ValueError if user invokes this method when pulses = False
        self._check_pulses_not_false()

        try:
            ppt, = self.pulses.pulse_counts().unique()
        except ValueError:
            raise ValueError(
                "The pulse pattern changed within this selection."
            ) from None

        return int(ppt)

    @property
    def trace_length(self) -> int | None:
        """Compute the trace length based on the pulse pattern"""

        # Raise a ValueError if user invokes this method when pulses = False
        self._check_pulses_not_false()

        trace_len = self.samples_per_pulse * self.pulses_per_train
        trace_len += self._sample_first_bunch

        # Make sure trace_len <= self._number_of_samples.
        # TODO: Issue warning if not? This would mean that the device was not
        # configured correctly before the run...
        trace_len = min(trace_len, self._number_of_samples)

        # Technically, wrapping in int is not necessary as all values involved
        # are of type int
        return int(trace_len)

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
            raise TypeError(
                "The `first_pulse_offset` argument should be an integer "
                "or None, you passed an object of type "
                f"{type(self._first_pulse_offset)}.") from None

        return value

    def _get_number_of_samples(self):
        return self._get_unique_value(
            self._ctrl_sourcedata, 'numberRawSamples.value')

    @staticmethod
    def _get_unique_value(source: SourceData, key: str) -> np.typing.ArrayLike:
        """General method that takes a key and returns a unique
        value in the KeyData.ndarray()."""

        if key not in source:
            raise KeyError(
                f"The SourceData object you passed, {source}, does not "
                f"contain the {key} key. Please provide a valid key."
            )

        array = source[key].ndarray()
        array = np.unique(array)

        if len(array) > 1:
            # 
            raise ValueError(
                f"It looks like the {key} key changed during the run. "
                f"Please specify a value explicitly. I found {array}."
            )

        return int(array[0])

    def _validate_pulses_kwarg(
            self,
            data: KeyData,
            pulses: bool,
    ) -> PulsePattern | Literal[False]:
        """Offloads pulses validation logic from the __init__ method.

        The validation needs to check for the following conditions:
          1. If pulses is not explicitly set to False, then a timeserver or
             a pulse pattern decoder source needs to be present.
          2. The pulse pattern needs to be constant.
        """

        # Do nothing if instantiated with `pulses=False`
        if pulses is False:
            return pulses

        # Write unit tests: no BPT, BPT_DECODER, full BPT, & both
        try:
            pulses_ = XrayPulses(data)
        except ValueError:
            raise ValueError(
                "No valid timeserver or pulse pattern decoder found. "
                "Please explicitly disable this feature by setting "
                "`pulses=False`.") from None

        # Make sure that the pulse pattern if constant
        if not pulses_.is_constant_pattern():
            raise RuntimeError(
                "The pulse pattern is not constant. Please split the "
                "current selection into sub-selections with constant patterns."
            )

        return pulses_

    # @property
    # def _control_sourcedata(self):
    #     inst_src_name = self._inst_sourcedata.canonical_name
    #     ctrl_src_name = inst_src_name.split(':')[0]
    #     if ctrl_src_name not in self._control_sources:
    #         raise AttributeError(f"The control source {ctrl_src_name} "
    #                              "does not exist.")
    #     return self._data[ctrl_src_name]

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
            raise ValueError(
                    f"The source you provided, {adc_channel}, cannot "
                    f"be matched to one of instrument sources.")
        if len_ > 1:
            raise ValueError(
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

    def train_data(
            self,
            labelled: bool = True,
            train_roi: slice = slice(None),
    ) -> np.ndarray | DataAray:
        """Return train data"""

        # Validate the roi parameter
        if not isinstance(train_roi, slice):
            raise ValueError("The roi parameter must be a slice object.")

        shape = self._inst_keydata[:, train_roi].shape
        out = np.empty(shape)

        offset = 0
        for chunk in self._inst_keydata.split_trains(trains_per_part=200):
            # In AdqRawChannel, there is a BEWARE about uint64, what is this
            # about? Doing `type(chunk.shape[0])` gives int... trainIds are
            # uint64 but...
            num_trains = chunk.shape[0]
            this_slice = slice(offset, offset + num_trains)
            offset += num_trains
            out[this_slice] = chunk.ndarray()

        if labelled:
            return out

        coords = {'trainId': self._inst_keydata.train_id_coordinates()}
        coords.update({'sample': np.arange(out.shape[1])})

        return DataArray(out, coords=coords)

    def pulse_data(
        self,
        labelled: bool = True,
        train_roi: slice = slice(None),
        auto_trim_trace: bool = False,
    ):
        """Identify all the pulses and return an array containing them."""

        if auto_trim_trace and self.trace_length is not None:
            train_roi = slice(None, self.trace_length)

        keydata = self._inst_keydata.drop_empty_trains()

        width = train_roi.stop - train_roi.start

        quotient, remainder = divmod()
        # Check that the train_roi can be split into integer pulses
        assert width % self.samples_per_pulse == 0
        out = np.empty((keydata.shape[0], width))

        offset = 0
        for chunk in keydata.split_trains(trains_per_part=200):
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

        return DataArray(out, coords=coords)

    def find_peaks(self, labelled=True):
        """Find the peak heights and locations in the trace"""

    def _prepare_pulses(self, train_ids):
        """Prepare pulse information."""

        self._check_pulses_not_false()

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
        # try:
        #     # Beware, pulse_period is not aligned to train_ids here!
        #     pulse_period = int(pulse_ids.groupby(level=0).diff().min())
        # except ValueError:
        #     samples_per_pulse = self._single_pulse_length
        # else:
        #     samples_per_pulse = self.samples_per_pulse(
        #         pulse_period=pulse_period)

        # Generate offsets of first pulse and last pulse of each
        # train relative to all pulses.
        pulse_last = num_pulses.cumsum()
        pulse_first = pulse_last - num_pulses

        # Combine pulse layout into a single dataframe.
        # TODO: samples_per_pulses is currently assumed to be constant
        pulse_layout = pd.DataFrame({
            'count': num_pulses,
            'first': pulse_first,
            'last': pulse_last,
            'length': self.samples_per_pulse})

        return aligned_pulses, pulse_layout
