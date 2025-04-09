import numpy as np
import pandas as pd
import xarray as xr

from extra.recipes import TOFAnalogResponse
from extra_data import open_run

def test_response():
    tof_id = 4
    channel = "3_A"
    digitizer = "SQS_DIGITIZER_UTC4/ADC/1:network"
    energy_source = "SA3_XTD10_MONO/MDL/PHOTON_ENERGY"
    calib_run = open_run(proposal=900485, run=320)
    scan = Scan(calib_run[energy_source, "actualEnergy.value"],
                resolution=2, intra_step_filtering=1)
    calib_run = calib_run.select([digitizer, energy_source], require_all=True)
    tof = AdqRawChannel(calib_run,
                        channel=channel,
                        digitizer=digitizer,
                        first_pulse_offset=23300,
                        single_pulse_length=600,
                        interleaved=True,
                        baseline=np.s_[:20000],
                        )
    response = TOFAnalogResponse(roi=np.s_[75:])
    response.setup(tof, scan)
    response.to_file("tof4.h5")
    h = response.get_response()


if __name__ == '__main__':
    test_response()

