from unittest.mock import MagicMock

from extra.components import Scantool

import pytest
import numpy as np

def test_scantool():
    mock_name = "CRISPY/KARABACON"
    mock_source = MagicMock()

    # Values taken from p3334, run 167
    mock_run_values = {
        "deviceEnv.acquisitionTime.value": 1.0,
        "deviceEnv.activeMotors.value": np.array([b'polarizer_theta', b'Theta_ana', b'', b'']),
        "scanEnv.scanType.value": "dscan",
        "scanEnv.steps.value": np.array([40, 1, 1, 1]),
        "scanEnv.startPoints.value": np.array([-0.0032, -0.002, 1, 1]),
        "scanEnv.stopPoints.value": np.array([0.0032, 0.002, 1, 1]),
        "actualConfiguration.value": "--- Motors: ['MID_AUXT1_UPP/MOTOR/R1:default', 'MID_EXP_UPP/MOTOR/T7:default']--- Data Sources: ['MID-EXP-UPP/CTRL/KEITHLEY-1:value', 'MID-EXP-UPP/CTRL/KEITHLEY-2:value']--- Triggers: ['MID_RR_SYS/MDL/TRIGGER']"
    }

    mock_run = MagicMock()
    mock_run.__getitem__ = lambda self, _: mock_source
    mock_run.get_run_values.return_value = mock_run_values

    # Test a run without a scantool
    mock_run.control_sources = { "foo" }
    with pytest.raises(RuntimeError) as e:
        Scantool(mock_run)
    assert "please pass an explicit source name" in str(e)

    # Test a run with multiple scantools
    mock_run.control_sources = { mock_name, f"{mock_name}2" }
    with pytest.raises(RuntimeError) as e:
        Scantool(mock_run)
    assert "multiple possible scantools" in str(e)

    mock_run.control_sources = { mock_name }

    # Test a run where the scantool is not active
    mock_source["isMoving"].ndarray.return_value = np.zeros(10, dtype=np.uint8)
    scantool = Scantool(mock_run)
    assert not scantool.active
    assert "not active" in str(scantool)
    assert "not active" in repr(scantool)

    # Now with a run where it is active
    mock_source["isMoving"].ndarray.return_value = np.ones(10, dtype=np.uint8)
    scantool = Scantool(mock_run)
    assert scantool.active
    assert scantool.motors == ["polarizer_theta", "Theta_ana"]
    assert scantool.motor_devices is not None
    assert scantool.scan_type in str(scantool)
    assert scantool.scan_type in repr(scantool)

    # And with a run with an unsupported actualConfiguration format
    mock_run_values["actualConfiguration.value"] = "--- Motors:['MID_AUXT1_UPP/MOTOR/R1:default', 'MID_EXP_UPP/MOTOR/T7:default']"
    with pytest.warns():
        scantool = Scantool(mock_run)
    assert scantool.scan_type in str(scantool)
    assert scantool.scan_type in repr(scantool)

    # Test compatibility with Karabacon 3.0.7-2.16.2, which is an older version
    # still used at FXE. In this version the activeMotors and acquisitionTime
    # are stored as first-level properties rather than being under deviceEnv.
    mock_run_values["acquisitionTime.value"] = mock_run_values["deviceEnv.acquisitionTime.value"]
    mock_run_values["activeMotors.value"] = mock_run_values["deviceEnv.activeMotors.value"]
    del mock_run_values["deviceEnv.acquisitionTime.value"]
    del mock_run_values["deviceEnv.activeMotors.value"]

    # This shouldn't throw an exception
    scantool = Scantool(mock_run)
