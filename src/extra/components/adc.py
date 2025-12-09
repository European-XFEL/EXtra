"""Fast ADC component"""

from typing import Literal
import re

import numpy as np
from xarray import DataArray

from extra_data import by_id, DataCollection, KeyData, SourceData
from extra_data.exceptions import SourceNameError
from extra_data.read_machinery import roi_shape
from .pulses import XrayPulses, PulsePattern
from .utils import _isinstance_no_import


class AdcRawChannel:
    r"""A high-level interface for the (raw) output of fast ADC channels.


    ![](../images/pulses.svg)

    Args:
        data (DataCollection): The object returned by `extra.data.RunDirectory`
            or `extra.data.OpenRun`.
        adc_channel (str | int): either a channel number if only one FastADC
            digitizer is present or enough of the digitizer name to uniquely
            identify it and the channel of interest (see examples below).
        pulses (PulsePattern | None, optional): An instance of [XrayPulses]
            [extra.components.XrayPulses]. Defaults to `None`.


    Warning:
        Instantiation will fail if the pulse pattern changed during teh run.

    Examples:
        >>> from extra.components import AdcRawChannel
        >>> from extra.data import open_run
        >>> run = open_run(700004, 19)
        >>> adc1_8 = AdcRawChannel(run, 8)
        ...
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
        pulses: PulsePattern | None = True,  # In the future, + manual pulses
        first_pulse_offset: None | int = None,
    ):

        self._adc_channel = str(adc_channel)
        self._first_pulse_offset = first_pulse_offset

        # TODO: digitizer parameter if multiple are present.
        # NOTE: the adc_channel parameter already accomplishes this. No need
        #       for an additional parameter.

        # Check that the required instrument and control sources are present
        # and have the correct keys.
        inst_name, ctrl_name = self._validate_sources(data)

        # If all is in order, assign the SourceData objects
        self._inst_sourcedata = data[inst_name]
        self._ctrl_sourcedata = data[ctrl_name]

        # Would be better to do this lazily but here we are... for example,
        # the .train_data method does not require it.
        self._pulses = self._validate_pulses_kwarg(data, pulses)

        # Check that the required keys are present and get keydata
        # self._inst_keydata = self._get_inst_keydata()
        # extra.data will raise KeyError if key is not found
        self._inst_keydata = self._inst_sourcedata['data.rawData']

        # Get two useful control source properties
        self._sample_first_bunch = self._get_sample_first_bunch()
        self._number_of_samples = self._get_number_of_samples()

    # TODO: may have to change logic elsewhere since now return None
    @property
    def pulses(self) -> PulsePattern | None:
        """The pulse pattern (XrayPulses) if present, False if not."""

        return self._pulses

    @property
    def pulse_period(self) -> int:
        """Check that there is one unique pulse period and extract it"""

        if self._check_pulses_not_none():
            pulse_period = self.pulses.pulse_periods().unique()

        # TODO: Probably should do this in the ._validate_pulses_kwarg method
        if len(pulse_period) > 1:
            raise ValueError(
                "There is more than one pulse period in this selection. To "
                "proceed, split the trains into selections with a common "
                "period and try again.")

        pulse_period = int(pulse_period[0])

        return pulse_period

    def _check_pulses_not_none(self) -> Literal[True]:
        """Raises ValueError if ._pulses is None else return True."""

        if self.pulses is None:
            raise ValueError(
                "This component was not initialized with a "
                "pulse pattern.") from None

        return True

    @property
    def samples_per_pulse(self) -> int:
        """Compute the number of samples per pulse."""

        # Raise a ValueError if user invokes this method when pulses = False
        self._check_pulses_not_none()
        # TODO: minimum length if 1 pulse per train
        return self.pulse_period * self._clock_ratio

    @property
    def pulses_per_train(self) -> int:
        """The number of pulses per train."""

        # Raise a ValueError if user invokes this method when pulses = False
        self._check_pulses_not_none()

        try:
            ppt, = self.pulses.pulse_counts().unique()
        except ValueError:
            raise ValueError(
                "The pulse pattern changed within this selection."
            ) from None

        return int(ppt)

    @property
    def trace_length(self) -> int:
        """Compute the trace length based on the pulse pattern"""

        # Raise a ValueError if user invokes this method when pulses = False
        self._check_pulses_not_none()

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

    # TODO: remove this method
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
            # `.as_single_value` raises a ValueError if it cannot reduce to a single value
            value = self._ctrl_sourcedata['sampleFirstBunch.value'].as_single_value()
        else:
            raise TypeError(
                "The `first_pulse_offset` argument should be an integer "
                "or None, you passed an object of type "
                f"{type(self._first_pulse_offset)}.") from None

        return int(value)

    def _get_number_of_samples(self):
        return (self._ctrl_sourcedata['numberRawSamples.value']
                .as_single_value()
               )

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
            pulses: PulsePattern | None,
    ) -> PulsePattern | Literal[False]:
        """Offloads pulses validation logic from the __init__ method.

        The validation needs to check for the following conditions:
          1. If pulses is not explicitly set to False, then a timeserver or
             a pulse pattern decoder source needs to be present.
          2. The pulse pattern needs to be constant.
        """

        # Do nothing if instantiated with `pulses=False`
        if pulses is None:
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

    def _validate_sources(
            self,
            data: DataCollection
    ) -> tuple[str, str]:
        """Check that instrument and control sources are present and valid.
        """

        found = set(filter(self._adc_regex.search,
                           data.instrument_sources))

        adc_channel = self._adc_channel

        candidates = set()
        for channel in found:
            if adc_channel in channel:
                candidates.add(channel)

        len_ = len(candidates)
        if len_ == 0:
            message = (
                    f"The source you provided, {adc_channel}, could not be "
                    "be matched to any of the instrument sources in the "
                    "data. See data.instrument_sources."
            )
            raise SourceNameError(custom_message=message)

        if len_ > 1:
            raise ValueError(
                    f"There are too many instrument sources in the data that "
                    f"match the source you provided, {adc_channel}. Namely "
                    f"{candidates}. Please be more specific.")

        inst_name = candidates.pop()
        ctrl_name = inst_name.split(':')[0]

        if ctrl_name not in data.control_sources:
            message = (
                "The corresponding control source to the instrument "
                f"{self._inst_keydata.source}, which should be called "
                f"{ctrl_name}, is missing from `.control_sources`."
            )
            raise SourceNameError(custom_message=message)

        return inst_name, ctrl_name

    def train_data(
        self,
        labelled: bool = True,
        train_roi: slice | None = None,
        auto_trim_trace: bool = False,
        # num_cores_to_use: None | int = None,
    ) -> np.ndarray | DataArray:
        """Return train data"""

        # Validate the roi parameter
        # if not isinstance(train_roi, slice):
        #    raise ValueError("The roi parameter must be a slice object.")

        # Trim the trace to avoid carrying useless information around
        if auto_trim_trace and train_roi is None:
            start = train_roi.start
            if train_roi.stop is not None:
                stop = min(train_roi.stop, self.trace_length)
            else:
                stop = self.trace_length
            train_roi = slice(start, stop)

        if train_roi is None:
            train_roi = slice(None)

        temp = roi_shape(self._inst_keydata.entry_shape, (train_roi,))
        shape = (self._inst_keydata.shape[0],) \
                 + temp
        out = np.empty(shape)

        offset = 0
        for chunk in self._inst_keydata.split_trains(trains_per_part=200):
            # In AdqRawChannel, there is a BEWARE about uint64, what is this
            # about? Doing `type(chunk.shape[0])` gives int... trainIds are
            # uint64 but...
            num_trains = chunk.shape[0]
            this_slice = slice(offset, offset + num_trains)
            offset += num_trains
            chunk.ndarray(roi=train_roi, out=out[this_slice])

        if labelled:
            return out

        coords = {
            'trainId': self._inst_keydata.train_id_coordinates(),
            'sample': np.arange(out.shape[1]),
        }

        return DataArray(out, coords=coords)

    def pulse_data(
        self,
        labelled: bool = True,
        # train_roi: slice = slice(None),
        # auto_trim_trace: bool = False,
    ) -> np.ndarray | DataArray:
        """Identify all the pulses and return an array containing them.

        See [XrayPulses.pulse_periods]
        [extra.components.XrayPulses.pulse_periods] for details about the 
        pulse component.
        """

        keydata = self._inst_keydata.drop_empty_trains()

        # pulse_trace = self.trace_length - self._sample_first_bunch

        # The quotient gives the number of full pulses supported by a trace
        # of this length. The remainder can be used for diagnosing 
        # quotient, remainder = divmod(pulse_trace, self.samples_per_pulse)
        quotient = (self.trace_length - self._sample_first_bunch)
        quotient //= self.samples_per_pulse

        pulse_trace_len = quotient * self.samples_per_pulse
        pulse_trace_roi = slice(
            self._sample_first_bunch,
            self.trace_length
        )
        out = np.empty((keydata.shape[0], pulse_trace_len))

        offset = 0
        for chunk in keydata.split_trains(trains_per_part=200):
            # In AdqRawChannel, there is a BEWARE about uint64, what is this
            # about? Doing `type(chunk.shape[0])` gives int... trainIds are 
            # uint64 but...
            num_trains = int(chunk.shape[0])
            this_slice = slice(offset, offset + num_trains)
            offset += num_trains
            # out[this_slice] = chunk.ndarray(roi=pulse_trace_roi)
            chunk.ndarray(roi=pulse_trace_roi, out=out[this_slice])

        out = out.reshape((keydata.shape[0], quotient, self.samples_per_pulse))

        if not labelled:
            return out

        pulse_ids = self.pulses.peek_pulse_ids(labelled=False)
        pulse_ids = pulse_ids[:quotient]

        coords = {
            'trainId': self._inst_keydata.train_id_coordinates(),
            'pulseId': pulse_ids,
            'sample': np.arange(self.samples_per_pulse)
        }
        return DataArray(data=out, coords=coords)

    def find_peaks(self, labelled=True):
        """Find the peak heights and locations in the trace"""
