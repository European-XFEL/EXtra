from unittest.mock import patch

import pint
import pytest
import numpy as np
import pandas as pd
import xarray as xr
import os

from extra.data import open_run

from extra.recipes import Grating2DCalibration, Grating1DCalibration
from extra.components import Scan

def test_create_grating_2d_calibration():
    # instantiates it without doing any calibration, only to check for syntax errors
    cal = Grating2DCalibration()

def test_create_grating_1d_calibration():
    # instantiates it without doing any calibration, only to check for syntax errors
    cal = Grating1DCalibration()

def test_grating_1d_fit():
    # instantiates it without doing any calibration, only to check for syntax errors
    cal = Grating1DCalibration()

    # fake calibration
    energies = np.linspace(990, 1010, 10+1)
    pix = np.linspace(0, 1000, 1000+1)
    true_offset = 990
    true_slope = 20/1000
    true_pix_to_e = lambda p: true_offset + p*true_slope
    true_e_to_pix = lambda e: (e - true_offset)/true_slope
    data = np.exp(-0.5*(pix[None, :] - true_e_to_pix(energies)[:,None])**2)
    cal.calibration_energies = energies
    cal.calibration_data = data
    cal.calibration_mask = np.ones(data.shape[0], dtype=bool)
    cal.calibration_unc = np.zeros_like(data)

    cal.fit()

    assert np.isclose(cal.e0, true_offset, atol=1e-2, rtol=1e-2)
    assert np.isclose(cal.slope, true_slope, atol=1e-2, rtol=1e-2)

def test_grating_2d_fit():
    # instantiates it without doing any calibration, only to check for syntax errors
    cal = Grating2DCalibration()

    # fake calibration
    energies = np.linspace(990, 1010, 10+1)
    pix = np.linspace(0, 1000, 1000+1)
    true_offset = 990
    true_slope = 20/1000
    true_pix_to_e = lambda p: true_offset + p*true_slope
    true_e_to_pix = lambda e: (e - true_offset)/true_slope
    data = np.exp(-0.5*(pix[None, :] - true_e_to_pix(energies)[:,None])**2)
    cal.calibration_energies = energies
    cal.calibration_data = data
    cal.calibration_mask = np.ones(data.shape[0], dtype=bool)
    cal.calibration_unc = np.zeros_like(data)
    cal.calibration_motor = np.zeros((data.shape[0]))

    cal.fit()

    assert np.isclose(cal.e0, true_offset, atol=1e-2, rtol=1e-2)
    assert np.isclose(cal.slope, true_slope, atol=1e-2, rtol=1e-2)

def test_reading_grating2d(mock_sqs_grating_calibration_run, tmp_path):
    monochromator_energy = "SA3_XTD10_MONO/MDL/PHOTON_ENERGY"
    final_photon_spectrometer = "SQS_DIAG3_BIU/CAM/CAM_6:daqOutput"

    monochromator_scan = Scan(mock_sqs_grating_calibration_run[monochromator_energy, "actualEnergy"], resolution=1)
    grating_calibration = Grating2DCalibration(angle=0.0)
    grating_calibration.setup(mock_sqs_grating_calibration_run[final_photon_spectrometer, "data.image.pixels"],
                              monochromator_scan,
                             )
    d = tmp_path / "data"
    d.mkdir()
    fpath = str(d / "grating2d_test.h5")
    grating_calibration.to_file(fpath)
    grating_calibration = Grating2DCalibration.from_file(fpath)

    calibrated = grating_calibration.apply(mock_sqs_grating_calibration_run.select_trains(np.s_[10:20]))

    assert np.isclose(grating_calibration.e0, 990.0, atol=1e-2, rtol=1e-2)
    assert np.isclose(grating_calibration.slope, 20.0/1000.0, atol=1e-2, rtol=1e-2)

def test_reading_grating1d(mock_sqs_grating_calibration_run, tmp_path):
    monochromator_energy = "SA3_XTD10_MONO/MDL/PHOTON_ENERGY"
    final_photon_spectrometer = "SQS_EXP_GH2-2/CORR/RECEIVER:daqOutput"

    monochromator_scan = Scan(mock_sqs_grating_calibration_run[monochromator_energy, "actualEnergy"], resolution=1)
    grating_calibration = Grating1DCalibration(min_pixel=0, max_pixel=1000)
    grating_calibration.setup(mock_sqs_grating_calibration_run[final_photon_spectrometer, "data.image.pixels"],
                              monochromator_scan,
                             )
    d = tmp_path / "data"
    d.mkdir()
    fpath = str(d / "grating1d_test.h5")
    grating_calibration.to_file(fpath)
    grating_calibration = Grating1DCalibration.from_file(fpath)

    calibrated = grating_calibration.apply(mock_sqs_grating_calibration_run.select_trains(np.s_[10:20]))

    assert np.isclose(grating_calibration.e0, 990.0, atol=1e-2, rtol=1e-2)
    assert np.isclose(grating_calibration.slope, 20.0/1000.0, atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(not os.path.isdir("/gpfs/exfel/d"), reason="GPFS not available")
@pytest.mark.vcr
def test_full_grating2d_calibration_from_data():
    # when a mono-chromator is used, this data source provides the information of which
    # energy the monochromator was set in
    monochromator_energy = "SA3_XTD10_MONO/MDL/PHOTON_ENERGY"

    # the data from the DIAG3 grating spectrometer after the cookie box is available for some runs
    final_photon_spectrometer = "SQS_DIAG3_BIU/CAM/CAM_6:daqOutput"

    # open run
    background_run = open_run(proposal=8697, run=23)
    monochromator_runs = range(173, 185)
    monochromator_runs = [open_run(proposal=8697, run=r) for r in monochromator_runs]
    monochromator_runs = monochromator_runs[0].union(*monochromator_runs[1:])

    monochromator_scan = Scan(monochromator_runs[monochromator_energy, "actualEnergy"], resolution=1)
    grating_calibration = Grating2DCalibration(angle=-10.5)
    grating_calibration.setup(monochromator_runs[final_photon_spectrometer, "data.image.pixels"],
                              monochromator_scan,
                              grating_bkg=background_run[final_photon_spectrometer, "data.image.pixels"],
                              grating_motor=monochromator_runs['SQS_DIAG3_SCAM/MOTOR/ST_AXIS_X', 'encoderPosition.value']
                             )
    grating_calibration.to_file('grating2d_calib.h5')
    grating_calibration = Grating2DCalibration.from_file('grating2d_calib.h5')

    calibrated = grating_calibration.apply(monochromator_runs.select_trains(np.s_[:10]))

    assert np.isclose(grating_calibration.e0, 990.1325, atol=1e-2, rtol=1e-2)
    assert np.isclose(grating_calibration.slope, 0.013908700693988751, atol=1e-2, rtol=1e-2)
    assert np.isclose(grating_calibration.slope_motor, -7.6386088778649395, atol=1e-2, rtol=1e-2)

    print(f"Test successful.")

