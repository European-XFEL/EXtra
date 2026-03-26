import numpy as np
import pytest
import xarray as xa
from extra.components import AGIPD1MQuadrantMotors, JF4MHalfMotors

testdata = [
    ("SPB_IRU_AGIPD1M", "SPB_IRU_AGIPD1M/MOTOR/Q{q}M{m}", "actualPosition"),
    ("SPB_EXP_AGIPD1M2", "SPB_EXP_AGIPD1M2/DS", "spbExpAgipd1M2MotorQ{q}M{m}.actualPosition"),  # noqa
]


@pytest.mark.parametrize(["detector_id", "src_ptrn", "key_ptrn"], testdata,
                         ids=["motors", "data_selector"])
def test_agipd1m_motors(mock_spb_aux_run, detector_id, src_ptrn, key_ptrn):
    nparts = 3

    with pytest.raises(ValueError):
        AGIPD1MQuadrantMotors(mock_spb_aux_run, "DETECTOR_WITHOUT_MOTORS")

    motors = AGIPD1MQuadrantMotors(mock_spb_aux_run, detector_id)

    ntrains = len(mock_spb_aux_run.train_ids)
    trains = np.array(mock_spb_aux_run.train_ids)
    first_ix = np.array([ntrains * i // nparts for i in range(nparts)], int)
    first = trains[first_ix]

    quad_pos = np.array([(-525, 625), (-550, -10), (520, -160), (542.5, 475)])
    unique_pos = np.zeros((nparts,) + quad_pos.shape, quad_pos.dtype)
    for i in range(nparts):
        unique_pos[i] = quad_pos + i * 2.0

    assert motors.detector_id == detector_id

    p = motors.positions()
    assert isinstance(p, xa.DataArray)
    assert len(p) == ntrains
    assert p.dims == ('trainId', 'q', 'm')

    t, p = motors.positions(labelled=False)
    assert isinstance(p, np.ndarray)
    assert len(p) == ntrains
    assert np.array_equal(motors.train_ids, t)

    p = motors.positions(compressed=True)
    assert isinstance(p, xa.DataArray)
    assert len(p) == nparts
    assert p.dims == ('trainId', 'q', 'm')
    assert np.array_equal(p.trainId.values, first)
    assert np.array_equal(p.q.values, range(1, 5))
    assert np.array_equal(p.m.values, range(1, 3))
    assert np.array_equal(p.values, unique_pos)

    t, p = motors.positions(labelled=False, compressed=True)
    assert isinstance(p, np.ndarray)
    assert len(p) == nparts
    assert np.array_equal(t, first)
    assert np.array_equal(p, unique_pos)

    one = np.uint64(1)
    with pytest.raises(ValueError):
        p = motors.positions_at(trains[0] - one)
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
    with pytest.raises(ValueError):
        p = motors.positions_at(trains[-1] + one)

    assert np.array_equal(motors.most_frequent_positions(), unique_pos[-1])


def test_jf4_motors(mock_spb_aux_run):
    motors = JF4MHalfMotors(mock_spb_aux_run)

    ntrains = len(mock_spb_aux_run.train_ids)
    trains = np.array(mock_spb_aux_run.train_ids)

    px = motors.positions()
    assert isinstance(px, xa.DataArray)
    assert len(px) == ntrains
    assert px.dims == ('trainId', 'q', 'm')

    t, p = motors.positions(labelled=False)
    assert isinstance(p, np.ndarray)
    assert len(p) == ntrains
    assert np.array_equal(motors.train_ids, t)

    assert np.array_equal(px.values, p)
    assert p.shape == (ntrains, 2, 1)

    a = np.zeros([ntrains, 2, 1])
    a[:, 0, 0] = 10
    a[:10, 1, 0] = 5
    a[10:, 1, 0] = 6
    assert np.array_equal(p, a)
