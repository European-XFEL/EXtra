"""Fast ADC component"""

from typing import Literal, TypeAlias
import re

import numpy as np
from xarray import DataArray

from extra_data import by_id, DataCollection, KeyData, SourceData
from extra_data.exceptions import SourceNameError
from extra_data.read_machinery import roi_shape
from .pulses import XrayPulses, PulsePattern
from .utils import _isinstance_no_import

PulsesOrFalse: TypeAlias = PulsePattern | Literal[False] | None
"""An optional PulsePattern type or `False` to explicitly disable it."""


class AdcRawChannel:
    r"""A high-level interface for the (raw) output of fast ADC channels.

    ![](../images/pulses.svg)

    Args:
        data (DataCollection): The object returned by `extra.data.RunDirectory`
            or `extra.data.OpenRun`.
        adc_channel (str | int): either a channel number if only one FastADC
            digitizer is present or enough of the digitizer name to uniquely
            identify it and the channel of interest (see examples below).
        pulses (PulsesOrFalse, optional): An
            `extra.components.PulsePattern` instance. For example,
            [XrayPulses][extra.components.XrayPulses] or
            [ManualPulses][extra.components.ManualPulses].
            Defaults to `None`, in which case the pulse pattern is extracted
            from the run if possible. Otherwise, it remains


    Warning:
        Instantiation will fail if the pulse pattern changed during teh run.

    Examples:
        >>> from extra.components import AdcRawChannel
        >>> from extra.data import open_run
        >>> run = open_run(700004, 19)
        >>> adc2_8 = AdcRawChannel(run, 8)
        Traceback (most recent call last):
            ...
        ValueError: More than one instrument source matched 
            match the source you provided, 8. Namely {
            'LA3_LAS_PPL/ADC/2:channel_8.output',
        'LA3_LAS_PPL/ADC/3:channel_8.output'}. Please be more specific.

        This failed because this run has two different FastADCs, both with a
        channel 8. To instantiate either one, we need to provide more of the
        name to uniquely identify the digitizer channel we're interested in.

        >>> adc2_8 = AdcRawChannel(run, '2:channel_8')


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
        pulses: PulsesOrFalse = None,
        first_pulse_offset: int | None  = None,
    ):

        self._adc_channel = str(adc_channel)
        self._first_pulse_offset = first_pulse_offset

        # Check that the required instrument and control sources are present
        # and have the correct keys.
        # extra.data will raise KeyError if a key is not found
        inst_name, ctrl_name = self._validate_sources(data)
        self._selection = data.select({
            inst_name: {'data.rawData'},
            ctrl_name: {'sampleFirstBunch.value', 'numberRawSamples.value'},
        }, require_all=True)

        self._ctrl_sourcedata = self._selection[ctrl_name]
        self._inst_sourcedata = self._selection[inst_name]

        self._inst_keydata = data[inst_name, 'data.rawData']

        self._pulses = self._validate_pulses(data, pulses)

        # If all is in order, assign the SourceData objects

        # Get two useful control source properties
        self._sample_first_bunch = self._get_sample_first_bunch()

        if first_pulse_offset is not None:
            self._sample_first_pulse = \
                data[ctrl_name, 'sampleFirstBunch.value'].as_single_value()
            value = self._ctrl_sourcedata['sampleFirstBunch.value'] \
                    .as_single_value()
        self._number_of_samples = self._ctrl_sourcedata['numberRawSamples.value'] \
            .as_single_value()

    @property
    def pulses(self) -> PulsePattern | None:
        """The pulse pattern (XrayPulses) if present, False if not."""

        return self._pulses

    @property
    def pulse_period(self) -> int | None:
        """Get a unique pulse period from the data.

        If a pulse pattern was successfully detected or a `PulsePattern` object
        was explicitly passed to the pulses parameter, extract the pulse
        period.

        Returns:
            The common pulse period for all trains in the data.

        """

        # If the condition evaluates to True, then .is_constant_pattern
        # also evaluated to True and we can safely assume that there is a
        # unique pulse_period etc.
        if isinstance(self.pulses, PulsePattern):
            # Take a reasonable guess, 10 say, for the pulse_period if the
            # trains contain a single pulse. This will additionally prevent
            # .pulse_periods from raising a ValueError.
            pulse_period, = self.pulses.pulse_periods(
                    single_pulse_value=10).unique()
            return int(pulse_period)

        return None

    def _pulses_not_none_or_false(self) -> Literal[True]:
        """Raises ValueError if ._pulses is None else return True."""

        if self.pulses is None:
            raise ValueError(
                "This component was not initialized with a "
                "pulse pattern.") from None

        return True

    @property
    def samples_per_pulse(self) -> int | None:
        """Compute the number of samples per pulse."""

        if isinstance(self.pulses, PulsePattern):
            samples = self.pulse_period * self._clock_ratio
            return samples

        return None

    @property
    def pulses_per_train(self) -> int | None:
        """The number of pulses per train."""

        # Raise a ValueError, also if .is_constant_pattern() is False
        if isinstance(self.pulses, PulsePattern):
            ppt, = self.pulses.pulse_counts().unique()
            return int(ppt)

        return None

    @property
    def trace_length(self) -> int | None:
        """Compute the trace length based on the pulse pattern"""

        if isinstance(self.pulses, PulsePattern):
            trace_len = self.samples_per_pulse * self.pulses_per_train \
                    + self._sample_first_bunch

            # Make sure trace_len <= self._number_of_samples.
            # TODO: Issue warning if not? This would mean that the device was not
            # configured correctly before the run...
            trace_len = min(trace_len, self._number_of_samples)

            return int(trace_len)

        return None

    @property
    def _channel_number(self):
        """Extract the channel number from the instrument source name."""

        # Could also do AdcRawChannel._adc_regex or define a @classmethod
        # if this is deemed unclear
        result = self._adc_regex.search(self._inst_keydata.source)

        if result:
            return int(result.group(1))

        raise ValueError("Couldn't extract the channel number from "
                         "the source name.")

    def _get_sample_first_bunch(self):
        """Check if an explicit offset was passed otherwise extract it."""

        if isinstance(self._first_pulse_offset, int):
            value = self._first_pulse_offset
        elif self._first_pulse_offset is None:
            # `.as_single_value` raises a ValueError if it cannot reduce data
            # to a single value
            value = self._ctrl_sourcedata['sampleFirstBunch.value'] \
                    .as_single_value()
        else:
            raise TypeError(
                "The `first_pulse_offset` argument should be an integer "
                "or None, you passed an object of type "
                f"{type(self._first_pulse_offset)}.") from None

        return int(value)

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

    def _validate_pulses(
            self,
            data: KeyData,
            pulses: PulsePattern | Literal[False] | None,
    ) -> PulsePattern | None:
        """Offloads pulses validation logic from the __init__ method.

        The validation needs to check for the following conditions:
          1. If pulses is not explicitly set to False, then a timeserver or
             a pulse pattern decoder source needs to be present.
          2. The pulse pattern needs to be constant.
        """

        if pulses is False:
            return None

        if isinstance(pulses, PulsePattern):
            return pulses

        if pulses is None:
            # If none, try to auto-detect XrayPulses for this data.
            try:
                pulses = XrayPulses(data)
            except ValueError:
                # Probably better to re-raise the ValueError from the pulses
                # component because it's much more fine-grained?
                raise ValueError(
                    'Could not auto-detect pulse information, please pass a '
                    'PulsePattern object to the pulses parameter. Otherwise, '
                    'please explicitly disable it with pulses=False.'
                ) from None

        # Make sure that the pulse pattern if constant
        if not pulses.is_constant_pattern():
            raise RuntimeError(
                "The pulse pattern is not constant. Please split the "
                "current selection into sub-selections with constant patterns."
            )

        return pulses

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

        if not labelled:
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
