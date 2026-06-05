
from collections.abc import Iterable
from dataclasses import dataclass, fields, is_dataclass, MISSING
from typing import Optional

import numpy as np

from ..data import PropertyNameError
from .detector import DetectorData


def any_is_none(*values):
    for value in values:
        if value is None:
            return True

    return False


class AutoConditionsError(ValueError):
    """Used when detector conditions could not be inferred from data."""

    def __init__(self, missing, sources):
        self.missing = missing
        self.sources = sources

    def __str__(self):
        msg = 'required parameters could not be inferred from data: ' + \
            ', '.join(sorted(self.missing))

        if self.sources:
            msg += '\n\navailable sources to infer parameters:\n'

            for name, value in self.sources.items():
                if isinstance(value, Iterable) and not isinstance(value, str):
                    if len(value) > 1:
                        value = f'{sorted(value)[0]}, ...[{len(value) - 1} more]'

                    elif len(value) == 1:
                        value = str(next(iter(value)))

                    elif not value:
                        value = None

                msg += f'- {name}: {value}\n'

        return msg


class ConditionsBase:
    calibration_types = {}  # For subclasses: {calibration: [parameter names]}

    @classmethod
    def from_data(cls, params, **sources):
        if is_dataclass(cls):
            # Try to give a nicer error message if the conditions type
            # is a dataclass.
            required_fields = {f.name for f in fields(cls) if
                               bool(f.init and f.default is MISSING and
                                    f.default_factory is MISSING)}

            if (missing := required_fields - params.keys()):
                raise AutoConditionsError(missing, sources)

        return cls(**params)

    def make_dict(self, parameters) -> dict:
        d = dict()

        for db_name in parameters:
            value = getattr(self, db_name.lower().replace(" ", "_"))
            if isinstance(value, str):
                d[db_name] = value
            elif value is not None:
                d[db_name] = float(value)

        return d

    def _repr_markdown_(self):
        attr_names = [f.name for f in fields(self)]
        items = []
        for n in attr_names:
            if (value := getattr(self, n)) is not None:
                items.append(f"- {n.replace('_', ' ').capitalize()}: {value}")
        return '\n'.join(items)

    @staticmethod
    def _purge_missing_sources(data, *sources):
        result = []

        for src in sources:
            if isinstance(src, str):
                result.append(src if src in data.all_sources else None)
            else:
                result.append([s for s in src if s in data.all_sources])

        return result


@dataclass
class AGIPDConditions(ConditionsBase):
    """Conditions for AGIPD detectors"""
    sensor_bias_voltage: float
    memory_cells: int
    acquisition_rate: float
    gain_setting: Optional[int]
    gain_mode: Optional[int]
    source_energy: float = 9.2
    integration_time: int = 12
    pixels_x: int = 512
    pixels_y: int = 128

    _base_parameters = [
        "Sensor Bias Voltage",
        "Pixels X",
        "Pixels Y",
        "Memory cells",
        "Acquisition rate",
        "Gain setting"
    ]

    _dark_parameters = _base_parameters + ["Gain mode", "Integration time"]
    _gain_parameters = _base_parameters + ["Integration time"]
    _illuminated_parameters = _base_parameters + ["Integration time", "Source energy"]

    calibration_types = {
        "Offset": _dark_parameters,
        "Noise": _dark_parameters,
        "ThresholdsDark": _dark_parameters,
        "BadPixelsDark": _dark_parameters,
        "BadPixelsPC": _gain_parameters,
        "SlopesPC": _gain_parameters,
        "BadPixelsFF": _illuminated_parameters,
        "SlopesFF": _illuminated_parameters,
        "BadPixelsCS": _base_parameters,
        "SlopesCS": _base_parameters
    }

    def make_dict(self, parameters):
        cond = super().make_dict(parameters)

        # Fix-up some database quirks.
        if int(cond.get("Gain mode", -1)) == 0:
            del cond["Gain mode"]

        if int(cond.get("Integration time", -1)) == 12:
            del cond["Integration time"]

        return cond

    @classmethod
    def from_data(cls, data, detector,
                  fpga_comp=None, mpod=None, fpga_control=None, xtdf=None,
                  client=None, **params):
        # Uncaught exceptions that may be thrown in here:
        # - NoDataError

        if any_is_none(fpga_comp, mpod, fpga_control, xtdf):
            if not isinstance(detector, DetectorData):
                detector = DetectorData.from_identifier(
                    detector, pdu_snapshot_at=data, client=client)

            control_domain = detector.karabo_control_domain

            fpga_comp = fpga_comp or f'{control_domain}/MDL/FPGA_COMP'
            mpod = mpod or f'{control_domain[:-1]}/PSC/HV'

            if xtdf is None:
                xtdf = detector.source_names

            if fpga_control is None:
                fpga_control = [f'{control_domain}/FPGA/M_{i}'
                                for i in range(len(xtdf))]

        fpga_comp, mpod, fpga_control, xtdf = cls._purge_missing_sources(
            data, fpga_comp, mpod, fpga_control, xtdf)

        if fpga_comp is not None:
            # Prefer control data from FPGA composite device.
            sd = data[fpga_comp]

            if 'memory_cells' not in params:
                params['memory_cells'] = cls.memory_cells_from_comp(sd)

            if 'acquisition_rate' not in params:
                try:
                    val = cls.acquisition_rate_from_comp(sd)
                except PropertyNameError:
                    pass  # Not present in older data
                else:
                    params['acquisition_rate'] = val

            if 'gain_setting' not in params:
                params['gain_setting'] = cls.gain_setting_from_comp(sd)

            if 'gain_mode' not in params:
                params['gain_mode'] = cls.gain_mode_from_comp(sd)

            if 'integration_time' not in params:
                try:
                    val = cls.integration_time_from_comp(sd)
                except PropertyNameError:
                    # More recent feature of comp device, fallback to
                    # prior defalt.
                    val = 12

                params['integration_time'] = val

        if xtdf:
            # Fallback to estimate some parameters from XTDF data.
            if 'memory_cells' not in params:
                for src in xtdf:
                    if (val := cls.memory_cells_from_xtdf(data[src])) > 0:
                        break

                params['memory_cells'] = val

            if 'acquisition_rate' not in params:
                for src in xtdf:
                    if (val := cls.acquisition_rate_from_xtdf(data[src])) > 0:
                        break

                params['acquisition_rate'] = val

        if 'sensor_bias_voltage' not in params:
            if mpod is not None:
                # AGIPD Gen1
                params['sensor_bias_voltage'] = cls.bias_voltage_from_mpod(
                    data[mpod])

            elif fpga_control:
                # AGIPD Gen2
                for src in fpga_control:
                    if (val := cls.bias_voltage_from_control(data[src])) > 0:
                        break

                params['sensor_bias_voltage'] = val

        return super().from_data(params,
                                 fpga_comp=fpga_comp, mpod=mpod,
                                 fpga_control=fpga_control, xtdf=xtdf)

    @staticmethod
    def bias_voltage_from_control(sd):
        # These device used to suffer from switching to excessive values
        # randomly, so cut off any unreasonable values.
        values = sd['highVoltage.target'].ndarray()
        values = values[values < 1000.0]
        return int(np.median(values))

    @staticmethod
    def bias_voltage_from_mpod(sd, modules=None):
        if modules is not None:
            keys = [f'channels.U{modno}.voltage' for modno in modules]
        else:
            keys = sd.select_keys('channels.U*.voltage').keys(False)

        for key in keys:
            if (val := sd[key].as_single_value(atol=1, rtol=1)) > 0:
                return int(val)
        else:
            return 0  # All modules are powered off.

    @staticmethod
    def memory_cells_from_comp(sd):
        return int(sd['bunchStructure.nPulses']
            .as_single_value(reduce_by='max'))

    @staticmethod
    def memory_cells_from_xtdf(sd):
        # Only look at one train?
        cell_ids = sd['image.cellId'].drop_empty_trains().ndarray().squeeze()
        options = np.array([4, 32, 64, 76, 128, 176, 202, 250, 352])
        return int(options[np.flatnonzero(options > np.max(cell_ids)).min()])

    @staticmethod
    def acquisition_rate_from_comp(sd):
        return round(float(sd['bunchStructure.repetitionRate']
            .as_single_value()), 1)

    @staticmethod
    def acquisition_rate_from_xtdf(sd):
        pulse_ids = sd['image.pulseId'].drop_empty_trains().ndarray().squeeze()
        return round(np.floor(45 / np.diff(pulse_ids).min()) / 10, 1)

    @staticmethod
    def gain_setting_from_comp(sd):
        if 'gain' in sd:
            return int(sd['gain'].as_single_value(atol=0))

        # Legacy method for older versions of composite device.

        setupr = sd['setupr'].as_single_value()
        pattern_type_idx = sd['patternTypeIndex'].as_single_value()

        if (setupr == 0 and pattern_type_idx < 4):
            return 0
        elif (setupr == 32 and pattern_type_idx == 4):
            return 0
        elif (setupr == 8 and pattern_type_idx < 4):
            return 1
        elif (setupr == 40 and pattern_type_idx == 4):
            return 1

        raise ValueError('unexpected setupr and patternTypeIndex values to '
                         'determine CDS mode')

    @staticmethod
    def gain_mode_from_comp(sd):
        return int(sd['gainModeIndex'].as_single_value(atol=0))

    @staticmethod
    def integration_time_from_comp(sd):
        return int(sd['integrationTime'].as_single_value())


@dataclass
class LPDConditions(ConditionsBase):
    sensor_bias_voltage: float = 250.0
    memory_cells: int = 512
    memory_cell_order: Optional[str] = None
    feedback_capacitor: float = 5.0
    source_energy: float = 9.3
    category: int = 0
    pixels_x: int = 256
    pixels_y: int = 256
    parallel_gain: bool = False

    _base_params = [
        "Sensor Bias Voltage",
        "Memory cells",
        "Pixels X",
        "Pixels Y",
        "Feedback capacitor",
    ]
    _dark_parameters = _base_params + [
        "Memory cell order", "Parallel gain"
    ]
    _illuminated_parameters = _base_params + ["Source Energy", "category"]

    calibration_types = {
        "Offset": _dark_parameters,
        "Noise": _dark_parameters,
        "BadPixelsDark": _dark_parameters,
        "RelativeGain": _illuminated_parameters,
        "GainAmpMap": _illuminated_parameters,
        "FFMap": _illuminated_parameters,
        "BadPixelsFF": _illuminated_parameters,
    }

    def make_dict(self, parameters):
        cond = super().make_dict(parameters)

        # Legacy value for no parallel gain not injected for backwards
        # compatibility with prior calibration data.
        if int(cond.get("Parallel gain", -1)) == 0:
            del cond["Parallel gain"]

        return cond

    @classmethod
    def from_data(cls, data, detector, fem_comp=None, xtdf=None,
                  validate_memcell_order=False, use_memcell_order='auto',
                  client=None, **params):
        if any_is_none(fem_comp, xtdf):
            if not isinstance(detector, DetectorData):
                detector = DetectorData.from_identifier(
                    detector, pdu_snapshot_at=data, client=client)

            fem_comp = fem_comp or (
                detector.karabo_control_domain + '/COMP/FEM_MDL_COMP')

            if xtdf is None:
                xtdf = data.instrument_sources.intersection(
                    detector.source_names)

        fem_comp, xtdf = cls._purge_missing_sources(data, fem_comp, xtdf)

        if fem_comp is not None and 'parallel_gain' not in params:
            try:
                val = cls.parallel_gain_from_fem_comp(data[fem_comp])
            except PropertyNameError:
                # Added to the FEM comp with introduction of parallel
                # gain support.
                val = False

            params['parallel_gain'] = val

        if xtdf and 'memory_cell_order' not in params:
            prev_val = None  # used with validate_memory_order

            for src in xtdf:
                if (val := cls.memory_cell_order_from_xtdf(data[src])).size:
                    if validate_memcell_order:
                        if prev_val is not None and (prev_val != val).any():
                            raise ValueError('inconsistent memory order '
                                             'across modules')

                        prev_val = val
                    else:
                        break

            if use_memcell_order == 'auto':
                use = len(val) > 2 and (np.diff(val.astype(np.int32)) < 0).any()
            else:
                use = use_memcell_order == 'always'

            params['memory_cell_order'] = '{},'.format(
                ','.join([str(c) for c in val])) if use else None

        return super().from_data(params, fem_comp=fem_comp, xtdf=xtdf)

    @staticmethod
    def gain_mode_from_fem_comp(sd):
        # Not actually used in the condition at the moment.
        return int(sd.run_value('femAsicGain'))

    @staticmethod
    def parallel_gain_from_fem_comp(sd):
        return bool(sd.run_value('femAsicGainOverride'))

    @staticmethod
    def memory_cell_order_from_xtdf(sd):
        return sd['image.cellId'].drop_empty_trains()[0].ndarray().flatten()


@dataclass
class DSSCConditions(ConditionsBase):
    """Conditions for DSSC detectors"""
    sensor_bias_voltage: float
    memory_cells: int
    pulse_id_checksum: Optional[float] = None
    acquisition_rate: Optional[float] = None
    target_gain: Optional[int] = None
    encoded_gain: Optional[int] = None
    pixels_x: int = 512
    pixels_y: int = 128

    _params = [
        "Sensor Bias Voltage",
        "Memory cells",
        "Pixels X",
        "Pixels Y",
        "Pulse id checksum",
        "Acquisition rate",
        "Target gain",
        "Encoded gain",
    ]
    calibration_types = {
        "Offset": _params,
        "Noise": _params,
    }


@dataclass
class JUNGFRAUConditions(ConditionsBase):
    """Conditions for JUNGFRAU detectors"""
    sensor_bias_voltage: float
    memory_cells: int
    integration_time: float
    gain_setting: int
    gain_mode: Optional[int] = None
    exposure_timeout: int = 25
    sensor_temperature: float = 291
    pixels_x: int = 1024
    pixels_y: int = 512

    _params = [
        "Sensor Bias Voltage",
        "Memory Cells",
        "Pixels X",
        "Pixels Y",
        "Integration Time",
        "Sensor temperature",
        "Gain Setting",
        "Gain mode",
    ]
    _dark_params = _params + ["Exposure timeout"]

    calibration_types = {
        "Offset10Hz": _dark_params,
        "Noise10Hz": _dark_params,
        "BadPixelsDark10Hz": _dark_params,
        "RelativeGain10Hz": _params,
        "BadPixelsFF10Hz": _params,
    }

    # Before 2022, the settings key indicated both gain
    # mode (as in adaptive vs fixed gain) as well as gain
    # setting (as in high CDS or not). Since then, there
    # is a dedicated gainMode key and settings only
    # indicates high CDS.
    # See karaboDevices/slsDetectors@4433ae9c00edcca3309bec8b7515e0938f5f502c
    legacy_settings = {
        # old setting:  new settings, new gainMode
        'dynamicgain': ('gain0', 'dynamic'),
        'dynamichg0':  ('highgain0', 'dynamic'),
        'fixgain1': ('gain0', 'fixg1'),
        'fixgain2': ('gain0', 'fixg2'),
        'forceswitchg1': ('gain0', 'forceswitchg1'),
        'forceswitchg2': ('gain0', 'forceswitchg2'),
    }

    gain_mode_labels = {
        'dynamic': 0,
        'fixg0': 1,
        'fixg1': 2,
        'fixg2': 3,

        # forceswitchg1, forceswitchg2 may only be used for
        # darks and are not equivalent to their fixed gain
        # equivalents.
    }

    def make_dict(self, parameters):
        cond = super().make_dict(parameters)

        # Fix-up some database quirks.
        if int(cond.get("Gain mode", -1)) == 0:
            del cond["Gain mode"]

        # Fix-up some database quirks.
        if int(cond.get("Exposure timeout", -1)) == 25:
            del cond["Exposure timeout"]

        return cond

    @classmethod
    def from_data(cls, data, detector, control=None, client=None, **params):
        if control is None:
            detector = DetectorData.from_identifier(detector, client=client,
                                                    pdu_snapshot_at=data)

            control = control or '{}/DET/CONTROL'.format(
                detector.karabo_control_domain)

        control, = cls._purge_missing_sources(data, control)

        if control is not None:
            sd = data[control]

            if 'sensor_bias_voltage' not in params:
                params['sensor_bias_voltage'] = cls.sensor_bias_voltage_from_control(sd)

            if 'memory_cells' not in params:
                params['memory_cells'] = cls.memory_cells_from_control(sd)

            if 'integration_time' not in params:
                params['integration_time'] = cls.integration_time_from_control(sd)

            if 'exposure_timeout' not in params:
                params['exposure_timeout'] = cls.exposure_timeout_from_control(sd)

            if 'gain_setting' not in params:
                params['gain_setting'] = cls.gain_setting_from_control(sd)

            if 'gain_mode' not in params:
                params['gain_mode'] = cls.gain_mode_from_control(sd)

        return super().from_data(params, control=control)

    @staticmethod
    def sensor_bias_voltage_from_control(sd):
        for key in ['highVoltage', 'vHighVoltage']:
            if key not in sd:
                continue

            return int(sd.run_value(key)[0])

        raise PropertyNameError('highVoltage or vHighVoltage', sd.source)

    @staticmethod
    def memory_cells_from_control(sd):
        return int(sd.run_value('storageCells')) + 1

    @staticmethod
    def memory_cell_start_from_control(sd):
        # Not used in condition, but relevant for dark characterization.
        return int(sd.run_value('storageCellStart'))

    @staticmethod
    def integration_time_from_control(sd):
        return 1e6 * float(sd.run_value('exposureTime'))

    @staticmethod
    def exposure_timeout_from_control(sd):
        return int(sd.run_value('exposureTimeout'))

    @classmethod
    def gain_setting_from_control(cls, sd, raw=False):
        val = sd.run_value('settings')

        if 'gainMode' not in sd:
            # Convert from legacy value.
            val = cls.legacy_settings[val][0]

        if raw:
            return val

        return int(val == 'highgain0')

    @classmethod
    def gain_mode_from_control(cls, sd, raw=False):
        try:
            val = sd.run_value('gainMode')
        except PropertyNameError:
            val = cls.legacy_settings[sd.run_value('settings')][1]

        if raw:
            return val

        try:
            return cls.gain_mode_labels[val]
        except KeyError:
            raise ValueError(f'invalid gain mode {val!s} encountered') from None


@dataclass
class ShimadzuHPVX2Conditions(ConditionsBase):
    burst_frame_count: float

    calibration_types = {
        'Offset': ['Burst Frame Count'],
        'DynamicFF': ['Burst Frame Count'],
    }


detector_cond_cls = {
    'AGIPD-Type': AGIPDConditions,
    'LPD-Type': LPDConditions,
    'jungfrau-Type': JUNGFRAUConditions
}
