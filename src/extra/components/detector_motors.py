import numpy as np
import xarray


def mangle_device_id_underscore(device_id):
    return ''.join(c if c.isalnum() else '_' for c in device_id)


def mangle_device_id_camelcase(device_id):
    # Replace non alpha-numeric chars with blanks
    mangled_id = ''.join(c if c.isalnum() else ' ' for c in device_id)
    # Make camelCase
    mangled_id = ''.join(mangled_id.title().split())
    if mangled_id:
        mangled_id = mangled_id[0].lower() + mangled_id[1:]
    return mangled_id


def check_data_selector_nodes(source, keys):
    nodes = set((
        key.partition('.')[0] for key in source.keys()
    ))
    return all(k in nodes for k in keys)


class DetectorMotors:
    """Interface to detector quadrant motors.

    Example usage in a Jupyter notebook:
    ```python
            -----------------------------------------------------------
    In [1]: |motors = DetectorMotors(run)                             |
            |xgm                                                      |
            -----------------------------------------------------------
    Out[1]: <DetectorMotors SPB_IRU_AGIPD1M/MOTOR/Q{1..4}M{1..2} at
            2023-04-04T17:44:46.844869000>
    ```
    """

    # SPB
    # SPB_IRU_AGIPD1M/MOTOR/Q{q+1}M{m+1}
    # SPB_IRU_AGIPD1M/MOTOR/Z_STEPPER

    # MID
    # MID_EXP_AGIPD1M/MOTOR/Q1M1
    # MID_EXP_UPP/MOTOR/T4
    _position_key = "actualPosition"

    DETECTORS = {
        "SPB_IRU_AGIPD1M": {
            "num_groups": 4,
            "num_motors": 2,
            "motor_pattern": "{detector_id}/MOTOR/Q{q}M{m}",
            "data_selector": "SPB_IRU_AGIPD1M1/MDL/DATA_SELECTOR",
        },
        "MID_EXP_AGIPD1M": {
            "num_groups": 4,
            "num_motors": 2,
            "motor_pattern": "{detector_id}/MOTOR/Q{q}M{m}",
            "data_selector": "MID_EXP_AGIPD1M1/MDL/DATA_SELECTOR",
        },
    }
    ARGS = ("num_groups", "num_motors", "motor_pattern", "data_selector")

    def __init__(self, dc, detector_id="", groups=None, motors_per_group=None,
                 motor_pattern=None, data_selector=None):
        """
        Parameters
        ----------
        dc: extra_data.DataCollection
            The data
        detector_id: str
            The detector ID, e.g. SPB_IRU_AGIPD1M or SPB_IRU_AGIPD1M
        groups: int
            The number of movable groups
        motors_per_group: int
            The number of motors per movable group
        motor_patter: str
            The pattern to generate motor IDs, expected to have fields:
            detector_id, q, m
        data_selector: str
            The data selector ID
        """
        self.detector_id = detector_id

        detector_param = self.DETECTORS.get(detector_id, {})
        for key in self.ARGS:
            value = locals().get(key)
            value = detector_param.get(key, value)
            setattr(self, key, value)

        args = (self.num_groups, self.num_motors, self.motor_pattern)
        if any(arg is None for arg in args):
            raise ValueError(
                "You need specify all parameters for custom detector")

        self.num_sources = self.num_groups * self.num_motors

        try:
            self.dc = dc.select(
                [(source_id, "*") for source_id in self.motor_ids],
                require_all=True
            )
            self.get_key = self._get_key_from_dc
        except ValueError:
            if self.data_selector is None:
                raise ValueError("The motors are not found")

            self.dc = dc.select(self.data_selector)
            self.get_key = self._get_key_from_data_selector

            # check the keys of data selector
            data_selector_src = self.dc[self.data_selector]
            camelcase = [mangle_device_id_camelcase(source_id)
                         for source_id in self.motor_ids]
            underscore = [mangle_device_id_underscore(source_id)
                          for source_id in self.motor_ids]

            if check_data_selector_nodes(data_selector_src, camelcase):
                self._mangle_id = mangle_device_id_camelcase
            elif check_data_selector_nodes(data_selector_src, underscore):
                self._mangle_id = mangle_device_id_underscore
            else:
                raise ValueError("The motors are not found")

        self.num_trains = len(self.train_ids)

    def _get_key_from_dc(self, source_id, key):
        return self.dc[source_id, key]

    def _get_key_from_data_selector(self, source_id, key):
        data_selector_key = f"{self._mangle_id(source_id)}.{key}"
        return self.dc[self.data_selector, data_selector_key]

    @property
    def motor_ids(self):
        """The list of motor device IDs."""
        if not hasattr(self, "_motor_ids"):
            self._motor_ids = [
                self.motor_pattern.format(
                    detector_id=self.detector_id,
                    q=i // self.num_motors + 1,
                    m=i % self.num_motors + 1,
                )
                for i in range(self.num_sources)
            ]
        return self._motor_ids

    @property
    def train_ids(self):
        """The list of train IDs."""
        return self.dc.train_ids

    def train_id_coordinates(self):
        """Returns the array of train IDs."""
        return np.asarray(self.dc.train_ids)

    @property
    def motor_labels(self):
        """The motor labels."""
        return [f"Q{i // self.num_motors + 1}M{i % self.num_motors + 1}"
                for i in range(self.num_sources)]

    def positions(self, labelled=True):
        """Returns the motor positions for all trains.

        Parameters
        ----------
        labelled: bool
            If True, returns the xarray with labelled dimensions,
            overwise returns numpy.ndarray

        Returns
        -------
        positions: numpy.ndarray or xarray.DataArray
            The motor positions
        """
        if not hasattr(self, "_positions"):
            values = np.zeros((self.num_trains, self.num_sources), dtype=float)
            for source_no, source_id in enumerate(self.motor_ids):
                values[:, source_no] = self.get_key(
                    source_id, self._position_key).ndarray()
            values = values.reshape(-1, self.num_groups, self.num_motors)
            self._positions = values
        else:
            values = self._positions

        if labelled:
            dims = ["trainId", "groupId", "motorId"]
            coords = {
                "trainId": self.train_id_coordinates(),
                "groupId": range(1, self.num_groups + 1),
                "motorId": range(1, self.num_motors + 1),
            }
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

        Parameters
        ----------
        labelled: bool
            If True, returns the xarray with labelled dimensions,
            overwise returns tuple of numpy.ndarrays

        Returns
        -------
        positions: tuple of two numpy.ndarray or xarray.DataArray
            The motor positions
        """
        trainId, values, _ = self._get_unique_pos()
        if labelled:
            dims = ["trainId", "groupId", "motorId"]
            coords = {
                "trainId": trainId,
                "groupId": range(1, self.num_groups + 1),
                "motorId": range(1, self.num_motors + 1),
            }
            return xarray.DataArray(
                values, dims=dims, coords=coords, name=self._position_key)
        else:
            return trainId, values

    def positions_at(self, tid):
        """Returns motor positions at given train.

        Parameters
        ----------
        tid: int
            Train ID

        Returns
        -------
        postions: numpy.ndarray
            Then motor positions
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

        motor_id = self.motor_pattern.format(
            detector_id=self.detector_id,
            q=f"{{1..{self.num_groups}}}",
            m=f"{{1..{self.num_motors}}}"
        )
        return f"<{self.__class__.__name__} {motor_id} at {self._ts}>"
