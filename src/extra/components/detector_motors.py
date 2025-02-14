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


class DetectorMotors:
    """Base class to access motors of detectors movable groups."""
    _position_key = "actualPosition"

    def __init__(self, dc, detector_id, motors, **coordinates):
        """
        Args:
            dc (extra_data.DataCollection):
                The data
            detector_id (str):
                The detector ID, e.g. SPB_IRU_AGIPD1M or SPB_IRU_AGIPD1M
            motors (dict):
                The motors data sources in the dictionary, where keys are
                stings labelling the motors and values are tuples of
                source and key names in data collection
            **coordinates (dict):
                Agruments to expand patterns and generate all motor labels

        Raises:
            PropertyNameError:
                If any motor position key is not found
            SourceNameError:
                If any motor source is not found
            ValueError:
                If the source is not specified for one of the motors
        """
        self.detector_id = detector_id
        self.coordinates = coordinates
        self.motors = motors

        self.shape = tuple(len(v) for v in coordinates.values())
        self.num_sources = np.prod(self.shape)
        self.dc = dc.select(motors.values(), require_all=True)

        self.num_trains = len(self.dc.train_ids)
        self.train_ids = self.dc.train_ids
        self.train_id_coordinates = np.asarray(self.train_ids)

        # check sources
        self.keys = []
        for motor_id in product(*coordinates.values()):
            label = ''.join(f"{n}{v}" for n, v in
                            zip(coordinates.keys(), motor_id))
            try:
                src, key = self.motors[label]
            except IndexError:
                raise ValueError(
                    f"The source for motor {label} is not specified.")

            self.keys.append(self.dc[src, key])

    def positions(self, labelled=True, compressed=False):
        """Returns the motor positions for all trains.

        Args:
            labelled (bool):
                If True, returns the xarray with labelled dimensions,
                overwise returns numpy.ndarray
            compressed (bool):
                If True, returns positions only when they change,
                overwise positions in all recorded trains

        Returns:
            positions (tuple of two numpy.ndarray or xarray.DataArray):
                The motor positions
        """
        if compressed:
            train_ids, pos, counts = self._compress_positions()
        else:
            pos = self._read_positions()
            train_ids = self.train_id_coordinates

        if labelled:
            dims = ["trainId"] + list(self.coordinates.keys())
            coords = {"trainId": train_ids}
            coords.update(self.coordinates)
            return xarray.DataArray(
                pos, dims=dims, coords=coords, name=self._position_key)
        else:
            return train_ids, pos

    def _read_positions(self):
        """Reads motor positions."""
        if hasattr(self, "_positions"):
            return self._positions

        # read positions
        pos = np.zeros((self.num_trains, self.num_sources), dtype=float)
        for source_no, key_data in enumerate(self.keys):
            pos[:, source_no] = key_data.ndarray()
        self._positions = pos.reshape(-1, *self.shape)
        return self._positions

    def _compress_positions(self):
        """Compresses motor positions."""
        if hasattr(self, "_compressed_pos"):
            return self._compressed_pos

        pos = self._read_positions()
        # compress
        axes = tuple(range(1, pos.ndim))
        ix = np.flatnonzero(
            np.insert(np.any(np.diff(pos, axis=0), axis=axes), 0, True))

        # train ids when positions are changed
        train_ids = np.asarray(self.dc.train_ids)[ix]
        # count of trains at every position
        counts = np.diff(np.append(train_ids.astype(int),
                                   int(self.dc.train_ids[-1]) + 1))

        self._compressed_pos = train_ids, pos[ix], counts
        return self._compressed_pos

    def positions_at(self, tid):
        """Returns motor positions at a given train.

        Args:
            tid (int):
                Train ID

        Returns:
            postions (numpy.ndarray):
                The motor positions

        Raises:
            ValueError:
                If train is not found
        """
        i = np.searchsorted(self.train_id_coordinates, tid)
        if (i >= self.num_trains) or (self.train_id_coordinates[i] != tid):
            raise ValueError(
                f"The train Id ({tid}) is outside of data collection")
        return self._read_positions()[i]

    def most_frequent_positions(self):
        """Returns most frequent motor positions."""
        _, values, counts = self._compress_positions()
        return values[np.argmax(counts)]

    def __repr__(self):
        if not hasattr(self, "_ts"):
            self._ts = self.dc.train_timestamps()[0]

        return (f"<{self.__class__.__name__}{self.shape}"
                f"for {self.detector_id} at {self._ts}>")


def sources_by_class(dc, class_id="SlowDataSelector"):
    """Returns control sources with a given classId."""
    sources = {}
    for src_name in dc.control_sources:
        try:
            src = dc[src_name]
            source_class_id = src.run_value("classId")
            if class_id == source_class_id:
                sources[src_name] = src.keys(inc_timestamps=False)
        except PropertyNameError:
            # class id is unknown, skip source
            pass

    return sources


def find_motors(dc, pattern, position_key, data_selectors=None, **coordinates):
    """Searches for motor sources according to a given pattern."""
    if data_selectors is None:
        data_selectors = sources_by_class(dc)

    motors = {}
    dims = coordinates.keys()
    for point in product(*coordinates.values()):
        args = dict(zip(dims, point))
        src = pattern.format(**args)
        label = ''.join(f"{n}{v}" for n, v in args.items())

        if (
            src in dc.control_sources and
            position_key in dc[src]
        ):
            motors[label] = (src, position_key)
            continue

        cc_key = mangle_device_id_camelcase(src) + '.' + position_key
        us_key = mangle_device_id_underscore(src) + '.' + position_key
        for ds_src, ds_keys in data_selectors.items():
            if cc_key in ds_keys:
                motors[label] = (ds_src, cc_key)
                break
            elif us_key in ds_keys:
                motors[label] = (ds_src, us_key)
                break

        if label not in motors:
            return None

    return motors


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
    KNOWN_DETECTORS = ["SPB_IRU_AGIPD1M", "MID_EXP_AGIPD1M"]

    def __init__(self, dc, detector_id=None):
        """
        Args:
            dc (extra_data.DataCollection):
                The data
            detector_id (str):
                The detector ID, e.g. SPB_IRU_AGIPD1M or MID_EXP_AGIPD1M

        Raises:
            ValueError:
                If motors are not found or multiple motor groups are found
        """
        pattern = "{detector_id}/MOTOR/Q{q}M{m}"

        num_groups = 4
        num_motors = 2
        groups = list(range(1, num_groups + 1))
        motors = list(range(1, num_motors + 1))

        data_selectors = sources_by_class(dc)

        all_motors = {}
        detectors = (
            self.KNOWN_DETECTORS if detector_id is None else [detector_id])
        for det_id in detectors:
            pattern = det_id + "/MOTOR/Q{q}M{m}"
            det_motors = find_motors(dc, pattern, self._position_key,
                                     data_selectors, q=groups, m=motors)
            if det_motors:
                all_motors[det_id] = det_motors

        if len(all_motors) == 0:
            raise ValueError("Motors are not found")
        elif len(all_motors) > 1:
            raise ValueError(
                "Many detector found: {', '.join(det_motors.keys())}. "
                "Please specify 'detector_id'")

        detector_id, detector_motors = all_motors.popitem()
        super().__init__(dc, detector_id, detector_motors, q=groups, m=motors)


class JF4MHalfMotors(DetectorMotors):
    """Interface to Jungfrau 4M half motors.

    Example usage in a Jupyter notebook:
    ```python
            -----------------------------------------------------------
    In [1]: |motors = JF4MHalfMotors(run)                             |
            |motors                                                   |
            -----------------------------------------------------------
    Out[1]: <JF4MHalfMotors(2, 1)for SPB_IRDA_JF4M at
            2024-08-21T19:38:47.690044000>
    ```
    """
    # SPB
    # SPB_IRDA_JF4M/MOTOR/X{q+1}
    # SPB_IRDA_JF4M/MOTOR/Z

    KNOWN_DETECTORS = ["SPB_IRDA_JF4M"]

    def __init__(self, dc, detector_id=None):
        """
        Args:
            dc (extra_data.DataCollection):
                The data
            detector_id (str):
                The detector ID, e.g. SPB_IRDA_JF4M

        Raises:
            ValueError:
                If motors are not found or multiple motor groups are found
        """
        pattern = "{detector_id}/MOTOR/X{q}"

        num_groups = 2
        num_motors = 1
        groups = list(range(1, num_groups + 1))
        motors = list(range(1, num_motors + 1))

        data_selectors = sources_by_class(dc)

        all_motors = {}
        detectors = (
            self.KNOWN_DETECTORS if detector_id is None else [detector_id])
        for det_id in detectors:
            pattern = det_id + "/MOTOR/X{q}"
            det_motors = find_motors(dc, pattern, self._position_key,
                                     data_selectors, q=groups, m=motors)
            if det_motors:
                all_motors[det_id] = det_motors

        if len(all_motors) == 0:
            raise ValueError("Motors are not found")
        elif len(all_motors) > 1:
            raise ValueError(
                "Many detector found: {', '.join(det_motors.keys())}.\n"
                "Please specify 'detector_id'")

        detector_id, detector_motors = all_motors.popitem()
        super().__init__(dc, detector_id, detector_motors, q=groups, m=motors)
