import ast
from warnings import warn

from extra_data import SourceData

class Scantool:
    """Interface for the European XFEL scantool (Karabacon).

    Note that the [repr][] function for this class uses
    [Scantool.format()][extra.components.Scantool.format]
    internally, so evaluating a `Scantool` object in a Jupyter notebook
    cell will print the scantool configuration:

    ```python
            -----------------------------------------------------------
    In [1]: |scantool = Scantool(run)                                 |
            |scantool                                                 |
            -----------------------------------------------------------
    Out[1]: Scantool (MID_RR_SYS/MDL/KARABACON) configuration:
              Scan type: dscan
              Acquisition time: 1.0s

            Motors:
              DET2_TX (MID_EXP_DES/MOTOR/DET2_TX): -0.05 -> 0.05, 100 steps
    ```

    See the [scantool
    documentation](https://rtd.xfel.eu/docs/scantool/en/latest/index.html) for
    more information about the device itself.
    """

    def __init__(self, run, src=None):
        """
        Args:
            run (extra_data.DataCollection): A run containing the scantool.
            src (str): The device name of the scantool. If this is not passed the class
                will try to find the right device automatically.
        """
        if src is None:
            possible_devices = [x for x in run.control_sources if "KARABACON" in x]
            if len(possible_devices) == 0:
                raise RuntimeError("Could not find a KARABACON device in the run, please pass an explicit source name with the `src` argument'")
            elif len(possible_devices) == 1:
                src = possible_devices[0]
            else:
                raise RuntimeError(f"Found multiple possible scantools, please pass one explicitly with the `src` argument: {', '.join(possible_devices)}")

        values = run.get_run_values(src)

        def get_first_value(keys):
            for key in keys:
                if key in values:
                    return values[key]

            raise KeyError(f"Could not find any of these RUN section keys: {', '.join(keys)}")

        # These are a list of possible property names for different versions of
        # the scantool. So far we've only seen the names being different, the
        # values are the same.
        acquisition_time_keys = ["deviceEnv.acquisitionTime.value", "acquisitionTime.value"]
        active_motors_keys = ["deviceEnv.activeMotors.value", "activeMotors.value"]

        # Get scan metadata and list of motors
        self._source_name = src
        self._source = run[src]
        self._active = self.source["isMoving"].ndarray().any()
        self._scan_type = values["scanEnv.scanType.value"]
        self._acquisition_time = get_first_value(acquisition_time_keys)
        self._motors = [x.decode() for x in get_first_value(active_motors_keys) if len(x) > 0]

        # The deviceEnv.activeMotors property stores the motor aliases,
        # but we can try to get the actual device names from the
        # actualConfiguration property.
        self._motor_devices = None
        motors_line = [x for x in values["actualConfiguration.value"].split("---") if "Motors:" in x]
        device_names_warning = "Couldn't extract the Karabo device names for the active motors."
        if len(motors_line) == 1:
            try:
                motors_list = motors_line[0].strip().removeprefix("Motors: ")
                motors_list = [x.split(":")[0] for x in ast.literal_eval(motors_list)]
                self._motor_devices = dict(zip(self._motors, motors_list))
            except Exception:
                warn(device_names_warning)
        else:
            warn(device_names_warning)

        # Get the number of steps and start/stop positions for each motor
        n_motors = len(self.motors)
        self._steps = dict(zip(self.motors,
                               values["scanEnv.steps.value"][:n_motors]))
        self._start_positions = dict(zip(self.motors,
                                         values["scanEnv.startPoints.value"][:n_motors]))
        self._stop_positions = dict(zip(self.motors,
                                        values["scanEnv.stopPoints.value"][:n_motors]))

    @property
    def source_name(self) -> str:
        """The name of the scantool device."""
        return self._source_name

    @property
    def source(self) -> SourceData:
        """`SourceData` object for the device."""
        return self._source

    @property
    def active(self) -> bool:
        """Boolean to indicate whether the scantool was used during the run."""
        return self._active

    @property
    def scan_type(self) -> str:
        """The type of scan configured (ascan, dscan, mesh, etc)."""
        return self._scan_type

    @property
    def acquisition_time(self) -> float:
        """Acquisition time in seconds."""
        return self._acquisition_time

    @property
    def motors(self) -> list:
        """List of aliases of the motors being moved.

        Note that these are scantool-specific aliases, not [EXtra-data
        aliases](https://extra-data.readthedocs.io/en/latest/reading_files.html#using-aliases).
        """
        return self._motors

    @property
    def motor_devices(self) -> dict:
        """A dictionary mapping motor aliases to their actual device names.

        Warning:
            This property is obtained by parsing a configuration string, which may
            not be compatible with previous versions of the scantool. If it was not
            possible to get the device names then a warning will be printed when
            initializing the class, and this property will be ``None``.
        """
        return self._motor_devices

    @property
    def steps(self) -> dict:
        """A dictionary mapping motor aliases to the number of steps they were
        scanned over."""
        return self._steps

    @property
    def start_positions(self) -> dict:
        """A dictionary mapping motor aliases to their start positions."""
        return self._start_positions

    @property
    def stop_positions(self) -> dict:
        """A dictionary mapping motor aliases to their stop positions."""
        return self._stop_positions

    def _motor_fmt(self, name, compact=True):
        """Helper function to format a single motor"""
        motion_info = f"{self.start_positions[name]} -> {self.stop_positions[name]}, {self.steps[name]} steps"

        if compact:
            return f"{name} ({motion_info})"
        elif not compact:
            if self.motor_devices is None:
                return f"{name}: {motion_info}"
            else:
                return f"{name} ({self.motor_devices[name]}): {motion_info}"

    def format(self, compact=True):
        """Format information about the scantool as a string.

        Args:
            compact (bool): Whether to print the information in a compact 1-line format or a
                multi-line format.
        """
        if compact and not self.active:
            return f"Scantool ({self.source_name}) not active."
        else:
            if compact:
                motor_info = [self._motor_fmt(name, compact=True) for name in self.motors]
                return f"{self.scan_type} {self.acquisition_time}s: {', '.join(motor_info)}"
            else:
                info = [f"Scantool ({self.source_name}) configuration:",
                        f"  Scan type: {self.scan_type}",
                        f"  Acquisition time: {self.acquisition_time}s",
                        "",
                        "Motors:"]

                info.extend(["  " + self._motor_fmt(name, compact=False)
                             for name in self.motors])

                if not self.active:
                    info = ["Note: the scantool was not active for this run!",
                            ""] + info

                return "\n".join(info)

    def __repr__(self):
        return self.format(compact=False)

    def __str__(self):
        return self.format(compact=True)
