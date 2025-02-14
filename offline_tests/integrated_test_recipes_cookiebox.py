import numpy as np
import pandas as pd
import xarray as xr

from extra.recipes import CookieboxCalibration

def test_cookiebox_calibration():
    # calibrate a given run
    pes1 = 'SQS_DIGITIZER_UTC4/ADC/1:network'
    pes2 = 'SQS_DIGITIZER_UTC5/ADC/1:network'
    tof_settings = {
           0:(pes1, "1_A"),
           1:(pes1, "1_C"),
           2:(pes1, "2_A"),
           3:(pes1, "2_C"),
           4:(pes1, "3_A"),
           5:(pes1, "3_C"),
           6:(pes1, "4_A"),
           7:(pes1, "4_C"),
           8:(pes2, "1_A"),
           9:(pes2, "1_C"),
           10:(pes2, "2_A"),
           11:(pes2, "2_C"),
           12:(pes2, "3_A"),
           13:(pes2, "3_C"),
           14:(pes2, "4_A"),
           15:(pes2, "4_C"),
           }

    # setup calibration runs
    calib_run = [open_run(proposal=8697, run=r) for r in range(173, 185)]
    calib_run = calib_run[0].union(*calib_run[1:])

    energy_axis = np.linspace(968, 1026, 160)

    cal = CookieboxCalibration(
                    # these were chosen by eye
                    # if set to None, automatic discovery is used
                    # but it may fail
                    first_pulse_offset=1300,
                    auger_start_roi=150,
                    start_roi=200,
                    stop_roi=320,
                    xgm_source="SQS_DIAG1_XGMD/XGM/DOOCS",
                    interleaved=True,
    )

    # do calibration
    cal.setup(run=calib_run, energy_axis=energy_axis, tof_settings=tof_settings)

    cal.to_file('cookiebox_calib.h5')
    cal_read = CookieboxCalibration.from_file('cookiebox_calib.h5')

    r188 = open_run(proposal=8697, run=188).select(cal_read.sources, require_all=True).select_trains(np.s_[:5])
    r188_cal = cal_read.apply(r188)

    print(r188_cal)

    #old_r188_cal = xr.load_data("data/cookiebox.h5")
    #xr.testing.assert_allclose(r188_cal, old_r188_cal)

if __name__ == '__main__':
    test_cookiebox_calibration()

