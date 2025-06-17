from extra.components import Scan

import pytest
import numpy as np
import xarray as xr

import extra_data


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

    # Test splitting a DataCollection by scan steps
    steps_data = s.split_by_steps(mock_spb_aux_run)
    assert len(s.steps) == 10
    assert len(steps_data) == len(s.steps)
    assert all(isinstance(dc, extra_data.DataCollection) for dc in steps_data)
    assert {len(dc.train_ids) for dc in steps_data} == {9, 10}

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

    # Test an edge case with a motor that's jittering between two values (see
    # the ZeroDivisionError check in Scan._guess_resolution()). This is somewhat
    # finicky to test because we have to make sure that the *sum* of a bunch of
    # floats is exactly 0, which is tricky because of floating point error. We
    # force this to happen by summing only a few values to limit the error.
    motor_slice = motor[:9]
    # The length of the motor array must be odd, because then we'll have an even
    # number of diffs to sum (to 0).
    assert len(motor_slice) == 9
    motor_slice[1::2] = -1
    motor_slice[::2] = 1
    # Make sure that the fake motor has the right values
    assert np.sum(np.diff(motor_slice)) == 0
    # This should not throw, and should not detect any steps
    assert len(Scan(motor_slice).steps) == 0

    # Test scan binning with some fake data
    s, _ = Scan._mkscan(20)
    data = xr.DataArray(np.random.rand(len(s._input_pos.trainId)),
                        name="fake-data",
                        dims=("trainId",),
                        coords=dict(trainId=s._input_pos.trainId))

    binned_data = s.bin_by_steps(data, uncertainty_method="stderr")
    assert len(binned_data) == 20
    assert set(binned_data.coords.keys()) == {"position", "uncertainty", "counts"}
    assert binned_data.name == data.name
    assert binned_data.attrs["motor"] == s.name

    # Only 'std' and 'stderr' should be accepted as uncertainty methods
    with pytest.raises(ValueError):
        s.bin_by_steps(data, uncertainty_method="foo")

    # Remove some trains from `data` to make sure they aren't used while binning
    data_sel = data.sel(trainId=data.trainId[:-5])
    # This should not throw an exception
    s.bin_by_steps(data_sel)

    # Smoke tests
    s.plot()
    s._plot_resolution_data()
    repr(s)
    s.format()
    s.info()
    s.plot_bin_by_steps(data)


def test_scan_bin_multidimensional(mock_spb_aux_run):
    s = Scan(mock_spb_aux_run["MOTOR/MCMOTORYFACE"])

    # Test scan binning with data with additional dimensionsn
    xgm_intensity = mock_spb_aux_run[
        "SPB_XTD9_XGM/DOOCS/MAIN:output", "data.intensityTD"
    ].xarray(extra_dims=["pulse"])

    binned = s.bin_by_steps(xgm_intensity)
    assert binned.dims == ("position", "pulse")
    # pulse is present as a dimension, but doesn't have coordinates
    assert set(binned.coords.keys()) == {"position", "uncertainty", "counts"}
    assert binned.shape == (10, 1000)

    # Smoke test
    s.plot_bin_by_steps(xgm_intensity)
