import numpy as np
import h5py

from extra_data import open_run, by_id
from extra.components import AdqRawChannel, Scan, XGM
from extra.applications import CookieboxCalibration


def produce_calibration(filename):
    """
    Calibrate an example run and dump calibration constants to H5 file.
    """

    # the time server device is used to identify each pulse and train
    pulse_timing = "SQS_RR_UTC/TSYS/TIMESERVER:outputBunchPattern"

    # there are two eTOF digitizers providing their data in DIAG3
    time_of_flight_group_1 = 'SQS_DIGITIZER_UTC4/ADC/1:network'
    time_of_flight_group_2 = 'SQS_DIGITIZER_UTC5/ADC/1:network'

    # the pulse energy can be read from these sources
    pulse_energy = "SQS_DIAG1_XGMD/XGM/DOOCS"
    pulse_energy_output = "SQS_DIAG1_XGMD/XGM/DOOCS:output"

    # when a mono-chromator is used, this data source provides the information of which
    # energy the monochromator was set in
    monochromator_energy = "SA3_XTD10_MONO/MDL/PHOTON_ENERGY"

    # the data from the DIAG3 grating spectrometer after the cookie box is available for some runs
    final_photon_spectrometer = "SQS_DIAG3_BIU/CAM/CAM_6:daqOutput"

    # open run
    calibration_run = open_run(proposal=900485, run=348).select_trains(by_id[2253934513:])
    calibration_run = calibration_run.select([pulse_timing,
                                  time_of_flight_group_1, time_of_flight_group_2,
                                  pulse_energy, pulse_energy_output,
                                  monochromator_energy], require_all=True)

    channel_names = ["1_A", "1_C",
                    "2_A", "2_C",
                    "3_A", "3_C",
                    "4_A", "4_C",
                    "1_A", "1_C",
                    "2_A", "2_C",
                    "3_A", "3_C",
                    "4_A", "4_C"]
    tof_ids = list(range(16))

    def create_channel(time_of_flight_group, channel_name):
        return AdqRawChannel(calibration_run,
                             channel_name,
                             digitizer=time_of_flight_group,
                             first_pulse_offset=23300,
                             single_pulse_length=400,
                             interleaved=True,
                             baseline=np.s_[:20000],
                             extra_cm_period=[440])

    tof_channel = {}
    for i in range(8):
        tof_channel[i] = create_channel(time_of_flight_group_1, channel_names[i])
    for i in range(8, 16):
        tof_channel[i] = create_channel(time_of_flight_group_2, channel_names[i])

    scan = Scan(calibration_run[monochromator_energy, "actualEnergy"], resolution=2)

    energy_axis = np.linspace(965, 1070, 160)
    xgm = XGM(calibration_run, pulse_energy)
    cal = CookieboxCalibration(
                    auger_start_roi=1,
                    start_roi=75,
                    stop_roi=320,
    )
    cal.setup(run=calibration_run, energy_axis=energy_axis, tof_settings=tof_channel,
              xgm=xgm,
              scan=scan)

    cal.mask[0] = False
    cal.mask[1] = False
    cal.mask[3] = False
    cal.mask[8] = False
    cal.mask[11] = False

    cal.to_file(filename)

def dump(filename):
    with h5py.File(filename, "r") as fid:
        energies = fid["calibration_energies"]
        print(f"energies = [{', '.join('%0.3f' % x for x in energies)}]")
        print("data = {")
        for tof_id in range(16):
            data = fid[f"calibration_data/{tof_id}"][:, :200]
            b = '],\n'.join(', '.join('%0.3f' %x for x in y) for y in data)
            print(f"{tof_id}: [")
            print(b)
            print(f"],")
        print("}")
        for tof_id in range(16):
            A = fid[f"tof_fit_result/{tof_id}/A"]
            Aa = fid[f"tof_fit_result/{tof_id}/Aa"]
            e = fid[f"tof_fit_result/{tof_id}/energy"]
            mu = fid[f"tof_fit_result/{tof_id}/mu"]
            mu_auger = fid[f"tof_fit_result/{tof_id}/mu_auger"]
            offset = fid[f"tof_fit_result/{tof_id}/offset"]
            sigma = fid[f"tof_fit_result/{tof_id}/sigma"]
            print(f"tof_fit_result[{tof_id}] = TofFitResult(energy=np.array([{', '.join('%0.3f' % x for x in e)}]),\n"
                  f"                                        mu=np.array([{', '.join('%0.3f' % x for x in mu)}]),\n"
                  f"                                        sigma=np.array([{', '.join('%0.3f' % x for x in sigma)}]),\n"
                  f"                                        A=np.array([{', '.join('%0.3f' % x for x in A)}]),\n"
                  f"                                        Aa=np.array([{', '.join('%0.3f' % x for x in Aa)}]),\n"
                  f"                                        offset=np.array([{', '.join('%0.3f' % x for x in offset)}]),\n"
                  f"                                        mu_auger=np.array([{', '.join('%0.3f' % x for x in mu_auger)}]),\n"
                  f"                                        )")
            xgm = fid[f"calibration_mean_xgm/{tof_id}"]
            print(f"calibration_mean_xgm[{tof_id}] = np.array([{', '.join('%0.3f' % x for x in xgm)}])")

filename = "example_calibration_p900485_r348.h5"
produce_calibration(filename)
dump(filename)


