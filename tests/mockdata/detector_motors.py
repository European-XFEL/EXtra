import numpy as np
from itertools import product

from extra_data.tests.mockdata.base import DeviceBase
from extra_data.tests.mockdata.motor import Motor
from extra.components.detector_motors import mangle_device_id_camelcase


def get_motor_sourcenames(detector_id):
    groups = range(1, 5)
    motors = range(1, 3)
    return [f"{detector_id}/MOTOR/Q{q}M{m}"
            for q, m in product(groups, motors)]


def get_motor_sources(detector_id):
    return [Motor(name)
            for name in get_motor_sourcenames(detector_id)]


def write_motor_positions(f, detector_id, data_selector=None):
    quad_pos = np.array([(-525, 625), (-550, -10), (520, -160), (542.5, 475)])
    groups = range(1, 5)
    motors = range(1, 3)
    nparts = 3
    for q, m in product(groups, motors):
        if data_selector is None:
            motor_id = f"{detector_id}/MOTOR/Q{q}M{m}"
            ds = f[f"CONTROL/{motor_id}/actualPosition/value"]
        else:
            motor_id = mangle_device_id_camelcase(
                f"{detector_id}/MOTOR/Q{q}M{m}")
            ds = f[f"CONTROL/{data_selector}/{motor_id}/actualPosition/value"]
        ntrains = ds.shape[0]
        parts = [slice(ntrains * i // nparts, ntrains * (i + 1) // nparts)
                 for i in range(nparts)]

        p0 = quad_pos[q - 1, m - 1]
        for i, s in enumerate(parts):
            ds[s] = p0 + i * 2


class DetectorMotorDataSelector(DeviceBase):
    extra_run_values = [
        ('classId', None, 'SlowDataSelector'),
    ]

    def __init__(self, device_id, detector_id, nsamples=None):
        self.detector_id = detector_id
        self.control_keys = [
            (mangle_device_id_camelcase(src_name) + "/actualPosition", "f4", ())  # noqa
            for src_name in get_motor_sourcenames(detector_id)
        ]
        super().__init__(device_id, nsamples=nsamples)

    def write_control(self, f):
        super().write_control(f)
        write_motor_positions(f, self.detector_id, self.device_id)
