from unittest.mock import patch

import pint
import pytest
import numpy as np
import pandas as pd
import xarray as xr
import os

from extra.data import open_run

from extra.recipes import Grating2DCalibration, Grating1DCalibration
from extra.components import Scan, XrayPulses

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
    grating_calibration.setup(mock_sqs_grating_calibration_run[final_photon_spectrometer, "data.adc"],
                              monochromator_scan,
                              pulses=XrayPulses(mock_sqs_grating_calibration_run),
                             )
    d = tmp_path / "data"
    d.mkdir()
    fpath = str(d / "grating1d_test.h5")
    grating_calibration.to_file(fpath)
    grating_calibration = Grating1DCalibration.from_file(fpath)

    calibrated = grating_calibration.apply(mock_sqs_grating_calibration_run.select_trains(np.s_[10:20]))

    assert np.isclose(grating_calibration.e0, 990.0, atol=1e-2, rtol=1e-2)
    assert np.isclose(grating_calibration.slope, 20.0/1000.0, atol=1e-2, rtol=1e-2)


