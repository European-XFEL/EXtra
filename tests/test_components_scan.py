from extra.components import Scan

import numpy as np
import xarray as xr


def test_scan():
    s, steps = Scan._mkscan(20, step_length_rnd=0.5)
    actual_positions = [step[0] for step in steps]

    # Smoke tests
    s.plot()
    s._plot_resolution_data()
    str(s)
    repr(s)

    # Test that we detected the right number of steps
    assert len(s.positions) == len(steps)

    # And the right positions were detected
    assert all([detected_pos == step[0] for detected_pos, step
                in zip(s.positions, steps)])

    # And that each detected step is the right length
    assert all([len(detected_step) == len(step) for detected_step, step
                in zip(s.positions_tids, steps)])

    # Test behaviour with a motor that isn't moving
    motor = s._input_pos
    motor[...] = 1
    motor.name = "foo"
    s = Scan(motor)
    assert len(s.positions) == 0

    assert s._name == motor.name
