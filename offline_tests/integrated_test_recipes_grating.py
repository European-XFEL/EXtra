import numpy as np
import pandas as pd
import xarray as xr

from extra.recipes import Grating2DCalibration
from extra_data import open_run

def test_grating_2d_calibration():
    # calibrate a given run

    # setup calibration runs
    bkg_run = open_run(proposal=8697, run=23)
    runs = range(173, 185)
    mono_run = [open_run(proposal=8697, run=r) for r in runs]
    mono_run = mono_run[0].union(*mono_run[1:])

    calib = Grating2DCalibration(angle=-10.5)
    # do calibration
    calib.setup(mono_run["SQS_DIAG3_BIU/CAM/CAM_6:daqOutput", "data.image.pixels"],
                bkg_run["SQS_DIAG3_BIU/CAM/CAM_6:daqOutput", "data.image.pixels"],
                Scan(mono_run["SA3_XTD10_MONO/MDL/PHOTON_ENERGY", "actualEnergy"], resolution=1))

    calib.to_file('grating_calib.h5')
    cal_read = Grating2DCalibration.from_file('grating_calib.h5')

    r182 = open_run(proposal=8697, run=182).select_trains(np.s_[:5])
    r182_cal = cal_read.apply(r182)

    print(r182_cal)

    #old_r182_cal = xr.load_data("data/grating2d.h5")
    #xr.testing.assert_allclose(r182_cal, old_r182_cal)

if __name__ == '__main__':
    test_grating_2d_calibration()
