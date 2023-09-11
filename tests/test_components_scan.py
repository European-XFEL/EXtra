from extra.components import Scan

import pytest
import numpy as np
import xarray as xr


def test_scan(mock_spb_aux_run):
    motor = mock_spb_aux_run["MOTOR/MCMOTORYFACE"]

    # Test passing a SourceData object
    s = Scan(motor)
    assert s._name == "MOTOR/MCMOTORYFACE.actualPosition"

    # And a KeyData object
    s = Scan(motor["targetPosition"])
    assert s._name == "MOTOR/MCMOTORYFACE.targetPosition"

    # And a named DataArray
    s = Scan(motor["actualPosition"].xarray())
    assert s._name == "MOTOR/MCMOTORYFACE.actualPosition"

    # And an unsupported type
    with pytest.raises(TypeError):
        Scan(motor["actualPosition"].ndarray())

    # Create fake scan to test detection
    s, steps = Scan._mkscan(20, step_length_rnd=0.5)
    actual_positions = [step[0] for step in steps]

    # Test that we detected the right number of steps
    assert len(s.positions) == len(steps)

    # And the right positions were detected
    assert all([detected_pos == step[0] for detected_pos, step
                in zip(s.positions, steps)])

    # And that each detected step is the right length
    assert all([len(detected_step) == len(step) for detected_step, step
                in zip(s.positions_train_ids, steps)])

    # Test behaviour with a motor that isn't moving
    motor = s._input_pos
    motor[...] = 1
    motor.name = "foo"
    not_a_scan = Scan(motor)
    assert len(not_a_scan.positions) == 0

    # Test behaviour with a noisy motor, which will initially be detected as
    # having a single step, and that single step should be filtered out such
    # that no steps are detected.
    motor += np.random.rand(len(motor)) * 0.1
    assert len(Scan(motor).steps) == 0

    # Smoke tests
    s.plot()
    s._plot_resolution_data()
    repr(s)
    s.format()
    s.info()
