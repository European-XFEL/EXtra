import random
import re
import string
from itertools import product

import numpy as np
import xarray
from extra.data import PropertyNameError


def mangle_device_id_underscore(device_id):
    """Converts deviceId to property notation (underscore version)."""
    return ''.join(c if c.isalnum() else '_' for c in device_id)


def mangle_device_id_camelcase(device_id):
    """Converts deviceId to property notation (camelcase version)."""
    # Replace non alpha-numeric chars with blanks
    mangled_id = ''.join(c if c.isalnum() else ' ' for c in device_id)
    # Make camelCase
    mangled_id = ''.join(mangled_id.title().split())
    if mangled_id:
        mangled_id = mangled_id[0].lower() + mangled_id[1:]
    return mangled_id


def mangle_pattern(pattern, mangle, **args):
    """Applies notation conversion to the format pattern."""
    mangled = mangle(pattern.format(**args))
    for field, placeholder in args.items():
        mangled = mangled.replace(mangle(placeholder), "{" + field + "}")
    return mangled


def guess_device_id(mangled_id, underscores=2):
    """Tries to guess original id from mangled id."""
    device_id = ''
    for c in mangled_id:
        u = c.upper()
        if underscores > 0 and c == u:
            device_id += '_'
            underscores -= 1
        device_id += u
    return device_id


class DetectorMotors:
    """Base class to access motors of detectors movable groups."""
    _position_key = "actualPosition"

    def __init__(self, dc, detector_id, src_ptrn, key_ptrn, **dims):
        """
        Args:
            dc (extra_data.DataCollection):
                The data
            detector_id (str):
                The detector ID, e.g. SPB_IRU_AGIPD1M or SPB_IRU_AGIPD1M
            src_ptrn (str):
                Format string pattern for motor sources
            key_ptrn (str):
                Format string pattern for motor position keys
            **dims (lists):
                Agruments to expand patterns and generate all motor data keys
        Raises:
            SourceNameError:
                If any motor source is not found
            PropertyNameError:
                If any motor position key is not found
        """
        self.detector_id = detector_id
        self.src_ptrn = src_ptrn
        self.key_ptrn = key_ptrn
        self.dims = dims

        names, coordinates = zip(*self.dims.items())
        self.motor_ids = [(
                src_ptrn.format(**dict(zip(names, values))),
                key_ptrn.format(**dict(zip(names, values)))
            ) for values in product(*coordinates)]
        self.motor_labels = [
            ''.join(f"{n.upper()}{v}" for n, v in zip(names, values))
            for values in product(*coordinates)]

        self.shape = tuple(len(c) for c in coordinates)
        self.num_sources = np.prod(self.shape)
        self.dc = dc.select(self.motor_ids)

        self.num_trains = len(self.dc.train_ids)

        # check sources
        self.keys = []
        for src, key in self.motor_ids:
            self.keys.append(self.dc[src, key])

    @property
    def train_ids(self):
        """The list of train IDs."""
        return self.dc.train_ids

    def positions(self, labelled=True):
        """Returns the motor positions for all trains.

        Args:
            labelled (bool):
                If True, returns the xarray with labelled dimensions,
                overwise returns numpy.ndarray

        Returns:
            positions (numpy.ndarray or xarray.DataArray):
                The motor positions
        """
        if not hasattr(self, "_positions"):
            values = np.zeros((self.num_trains, self.num_sources), dtype=float)
            for source_no, key_data in enumerate(self.keys):
                values[:, source_no] = key_data.ndarray()
            values = values.reshape(-1, *self.shape)
            self._positions = values
        else:
            values = self._positions

        if labelled:
            dims = ["trainId"] + list(self.dims.keys())
            coords = {"trainId": self.train_ids}
            coords.update({name: values for name, values in self.dims.items()})
            return xarray.DataArray(
                values, dims=dims, coords=coords, name=self._position_key)
        else:
            return values

    def _get_unique_pos(self):
        """Returns unique motor positions."""
        if not hasattr(self, "_unique_positions"):
            values, index, counts = np.unique(
                self.positions(labelled=False),
                return_index=True, return_counts=True, axis=0)
            trainId = np.asarray(self.train_ids)[index]
            self._unique_positions = trainId, values, counts

        return self._unique_positions

    def unique_pos(self, labelled=True):
        """Returns the unique motor positions and corresponding train IDs.

        Args:
            labelled (bool):
                If True, returns the xarray with labelled dimensions,
                overwise returns tuple of numpy.ndarrays

        Returns:
            positions (tuple of two numpy.ndarray or xarray.DataArray):
                The motor positions
        """
        trainId, values, _ = self._get_unique_pos()
        if labelled:
            dims = ["trainId"] + list(self.dims.keys())
            coords = {"trainId": trainId}
            coords.update({name: values for name, values in self.dims.items()})
            return xarray.DataArray(
                values, dims=dims, coords=coords, name=self._position_key)
        else:
            return trainId, values

    def positions_at(self, tid):
        """Returns motor positions at given train.

        Args:
            tid (int):
                Train ID

        Returns:
            postions (numpy.ndarray):
                The motor positions
        """
        trainId, values, _ = self._get_unique_pos()
        i = np.searchsorted(trainId, tid)
        return values[i if i < len(trainId) else -1]

    @property
    def most_frequent_positions(self):
        """Returns most frequent motor positions."""
        trainId, values, counts = self._get_unique_pos()
        return values[np.argmax(counts)]

    @property
    def first(self):
        """Returns the first motor positions."""
        tid = self.train_ids[0]
        return self.positions_at(tid)

    @property
    def last(self):
        """Returns the last motor positions."""
        tid = self.train_ids[-1]
        return self.positions_at(tid)

    def __repr__(self):
        if not hasattr(self, "_ts"):
            self._ts = self.dc.train_timestamps()[0]

        return (f"<{self.__class__.__name__}{self.shape}"
                f"for {self.detector_id} at {self._ts}>")


def count_sources(collection, pattern, **dims):
    """Counts strings expanded from pattern in collection."""
    names, coordinates = zip(*dims.items())
    collection = set(collection)
    sources = (pattern.format(**dict(zip(names, values)))
               for values in product(*coordinates))
    return len(set(src for src in sources if src in collection))


def _make_motor_placeholders(**dims):
    placeholders = {}
    re_args = {}
    frm_args = {}
    num_motors = 1
    for name, coordinates in dims.items():
        frm_args[name] = "{" + name + "}"
        num_motors *= len(coordinates)
        # guess type
        numbers = all(
            isinstance(a, int) or (isinstance(a, str) and a.isdigit())
            for a in coordinates)

        if numbers:
            re_args[name] = r"\d+"
            placeholders[name] = ''.join(random.sample(string.digits, 10))
        else:
            re_args[name] = r"\w+"
            placeholders[name] = ''.join(
                random.sample(string.ascii_uppercase, 10))

    return num_motors, frm_args, re_args, placeholders


def find_detectors_and_motors(dc, pattern, position_key, **dims):
    """Looks for motors in data collection."""
    num_motors, frm_args, re_args, placeholders = (
        _make_motor_placeholders(**dims))

    placeholders["detector_id"] = "DETPLCHLDR"
    pattern_camelcase = mangle_pattern(
        pattern, mangle_device_id_camelcase, **placeholders)
    pattern_underscore = mangle_pattern(
        pattern, mangle_device_id_underscore, **placeholders)

    re_args["detector_id"] = r"(?P<detector_id>\w+)"
    re_ptrn = re.compile(pattern.format(**re_args))
    re_ptrn_cc = re.compile(pattern_camelcase.format(**re_args))
    re_ptrn_us = re.compile(pattern_underscore.format(**re_args))

    matches = (re_ptrn.match(src) for src in dc.control_sources)
    detectors = {}
    for detector_id in (m["detector_id"] for m in matches if m is not None):
        src_ptrn = pattern.format(detector_id=detector_id, **frm_args)
        num_sources = count_sources(dc.control_sources, src_ptrn, **dims)
        if num_sources == num_motors:
            detectors[detector_id] = (src_ptrn, position_key)

    data_selectors = []
    for src in dc.control_sources:
        try:
            class_id = dc.get_run_value(src, "classId")
            if class_id == "SlowDataSelector":
                data_selectors.append(src)
        except PropertyNameError:
            # class id is unknown, skip source
            pass

    suffix = f".{position_key}.value"
    for data_selector_id in data_selectors:
        keys = set(
            key.partition('.')[0] for key in dc[data_selector_id].keys()
            if key.endswith(suffix))
        matches = (re_ptrn_cc.match(src) for src in keys)
        unique_detectors = set(
            m["detector_id"] for m in matches if m is not None)
        if not unique_detectors:
            matches = (re_ptrn_us.match(src) for src in keys)
            unique_detectors = set(
                m["detector_id"] for m in matches if m is not None)
            unique_detectors = {
                detector_id: pattern_underscore.format(
                    detector_id=detector_id, **frm_args)
                for detector_id in unique_detectors
            }
        else:
            unique_detectors = {
                guess_device_id(detector_id):
                pattern_camelcase.format(detector_id=detector_id, **frm_args)
                for detector_id in unique_detectors
            }

        for detector_id, src_ptrn in unique_detectors.items():
            if detector_id in detectors:
                continue

            num_sources = count_sources(keys, src_ptrn, **dims)
            if num_sources == num_motors:
                detectors[detector_id] = (
                    data_selector_id, src_ptrn + '.' + position_key)

    return detectors


def find_motors(dc, pattern, position_key, **dims):
    num_motors, frm_args, _, placeholders = _make_motor_placeholders(**dims)

    pattern_camelcase = mangle_pattern(
        pattern, mangle_device_id_camelcase, **placeholders)
    pattern_underscore = mangle_pattern(
        pattern, mangle_device_id_underscore, **placeholders)

    src_ptrn = pattern.format(**frm_args)
    num_sources = count_sources(dc.control_sources, src_ptrn, **dims)
    if num_sources == num_motors:
        return src_ptrn, position_key

    data_selectors = []
    for src in dc.control_sources:
        try:
            class_id = dc.get_run_value(src, "classId")
            if class_id == "SlowDataSelector":
                data_selectors.append(src)
        except PropertyNameError:
            # class id is unknown, skip source
            pass

    suffix = f".{position_key}.value"
    for data_selector_id in data_selectors:
        keys = set(
            key.partition('.')[0] for key in dc[data_selector_id].keys()
            if key.endswith(suffix))
        src_ptrn = pattern_camelcase.format(**frm_args)
        num_sources = count_sources(keys, src_ptrn, **dims)
        if num_sources == num_motors:
            return data_selector_id, src_ptrn + '.' + position_key

        src_ptrn = pattern_underscore.format(**frm_args)
        num_sources = count_sources(keys, src_ptrn, **dims)
        if num_sources == num_motors:
            return data_selector_id, src_ptrn + '.' + position_key

    raise ValueError("Detector motors are not found")


class AGIPD1MQuadrantMotors(DetectorMotors):
    """Interface to AGIPD quadrant motors.

    Example usage in a Jupyter notebook:
    ```python
            -----------------------------------------------------------
    In [1]: |motors = AGIPD1MQuadrantMotors(run)                      |
            |motors                                                   |
            -----------------------------------------------------------
    Out[1]: <AGIPD1MQuadrantMotors(4, 2) for SPB_IRU_AGIPD1M at
            2023-04-04T17:44:46.844869000>
    ```
    """
    # SPB
    # SPB_IRU_AGIPD1M/MOTOR/Q{q+1}M{m+1}
    # SPB_IRU_AGIPD1M/MOTOR/Z_STEPPER

    # MID
    # MID_EXP_AGIPD1M/MOTOR//Q{q+1}M{m+1}
    # MID_EXP_UPP/MOTOR/T4
    def __init__(self, dc, detector_id=None):
        """
        Args:
            dc (extra_data.DataCollection):
                The data
            detector_id (str):
                The detector ID, e.g. SPB_IRU_AGIPD1M or SPB_IRU_AGIPD1M
        Raises:
            ValueError:
                If motors are not found or multiple motor groups are found
        """
        pattern = "{detector_id}/MOTOR/Q{q}M{m}"

        num_groups = 4
        num_motors = 2
        groups = list(range(1, num_groups + 1))
        motors = list(range(1, num_motors + 1))

        if detector_id is None:
            detectors = find_detectors_and_motors(
                dc, pattern, self._position_key, q=groups, m=motors)
            num_detectors = len(detectors)
            if num_detectors == 0:
                ValueError("Detector motors are not found")
            elif num_detectors > 1:
                detector_ids = ", ".join(detectors.keys())
                raise ValueError(f"Multiple detectors found: {detector_ids}. "
                                 f"Use 'detector_id' argument to choose one.")
            device_id, (device_ptrn, key_ptrn) = detectors.popitem()
        else:
            pattern = pattern.format(detector_id=detector_id, q="{q}", m="{m}")
            device_ptrn, key_ptrn = find_motors(
                dc, pattern, self._position_key, q=groups, m=motors)

        super().__init__(dc, device_id, device_ptrn, key_ptrn,
                         q=groups, m=motors)
