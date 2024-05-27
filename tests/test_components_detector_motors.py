import numpy as np
import xarray as xa
from extra.components import AGIPD1MQuadrantMotors


def test_detector_motors(mock_spb_aux_run):
    nparts = 3
    motors = AGIPD1MQuadrantMotors(mock_spb_aux_run)

    ntrains = len(mock_spb_aux_run.train_ids)
    trains = np.array(mock_spb_aux_run.train_ids)
    first_ix = np.array([ntrains * i // nparts for i in range(nparts)], int)
    first = trains[first_ix]

    quad_pos = np.array([(-525, 625), (-550, -10), (520, -160), (542.5, 475)])
    unique_pos = np.zeros((nparts,) + quad_pos.shape, quad_pos.dtype)
    for i in range(nparts):
        unique_pos[i] = quad_pos + i * 2.0

    assert motors.detector_id == "SPB_IRU_AGIPD1M"
    assert motors.src_ptrn == "SPB_IRU_AGIPD1M/MOTOR/Q{q}M{m}"
    assert motors.key_ptrn == "actualPosition"

    p = motors.positions()
    assert isinstance(p, xa.DataArray)
    assert len(p) == ntrains
    assert p.dims == ('trainId', 'q', 'm')

    p = motors.positions(labelled=False)
    assert isinstance(p, np.ndarray)
    assert len(p) == ntrains

    p = motors.compressed_positions()
    assert isinstance(p, xa.DataArray)
    assert len(p) == nparts
    assert p.dims == ('trainId', 'q', 'm')
    assert np.array_equal(p.trainId.values, first)
    assert np.array_equal(p.q.values, range(1, 5))
    assert np.array_equal(p.m.values, range(1, 3))
    assert np.array_equal(p.values, unique_pos)

    t, p = motors.compressed_positions(labelled=False)
    assert isinstance(p, np.ndarray)
    assert len(p) == nparts
    assert np.array_equal(t, first)
    assert np.array_equal(p, unique_pos)

    one = np.uint64(1)
    p = motors.positions_at(trains[0] - one)
    assert np.array_equal(p, unique_pos[0])
    p = motors.positions_at(trains[0])
    assert np.array_equal(p, unique_pos[0])

    p = motors.positions_at(first[1] - one)
    assert np.array_equal(p, unique_pos[0])
    p = motors.positions_at(first[1])
    assert np.array_equal(p, unique_pos[1])
    p = motors.positions_at(first[1] + one)
    assert np.array_equal(p, unique_pos[1])

    p = motors.positions_at(trains[-1])
    assert np.array_equal(p, unique_pos[-1])
    p = motors.positions_at(trains[-1] + one)
    assert np.array_equal(p, unique_pos[-1])

    assert np.array_equal(motors.first, unique_pos[0])
    assert np.array_equal(motors.last, unique_pos[-1])
    assert np.array_equal(motors.most_frequent_positions, unique_pos[-1])
