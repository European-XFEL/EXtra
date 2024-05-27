import numpy as np
from itertools import product

from extra_data.tests.mockdata.motor import Motor


def get_motor_sources(detector_id):
    groups = range(1, 5)
    motors = range(1, 3)
    return [Motor(f"{detector_id}/MOTOR/Q{q}M{m}")
            for q, m in product(groups, motors)]


def write_motor_positions(f, detector_id):
    quad_pos = np.array([(-525, 625), (-550, -10), (520, -160), (542.5, 475)])
    groups = range(1, 5)
    motors = range(1, 3)
    nparts = 3
    for q, m in product(groups, motors):
        motor_id = f"{detector_id}/MOTOR/Q{q}M{m}"
        ds = f[f"CONTROL/{motor_id}/actualPosition/value"]
        ntrains = ds.shape[0]
        parts = [slice(ntrains * i // nparts, ntrains * (i + 1) // nparts)
                 for i in range(nparts)]

        p0 = quad_pos[q - 1, m - 1]
        for i, s in enumerate(parts):
            ds[s] = p0 + i * 2
