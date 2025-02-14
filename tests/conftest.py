
from pathlib import Path
from shutil import copytree
from tempfile import TemporaryDirectory

import h5py
import numpy as np
import pytest
from extra_data import RunDirectory
from extra_data.tests.mockdata import write_file

from extra_data.tests.mockdata.motor import Motor
from .mockdata.adq import AdqDigitizer
from .mockdata.detector_motors import (DetectorMotorDataSelector,
                                       get_motor_sources,
                                       write_motor_positions)
from .mockdata.dld import ReconstructedDld
from .mockdata.timepix import Timepix3Receiver, Timepix3Centroids
from .mockdata.timeserver import PulsePatternDecoder, Timeserver
from .mockdata.xgm import XGM, XGMD, XGMReduced


@pytest.fixture(scope='session')
def mock_spb_aux_directory():
    """Mock run directory with SPB auxiliary sources.

    Pulse pattern per train:
        - 0:5, no pulses
        - SA1
            - 10:50, 50 pulses at 1000:1300:6
            - 50:100, 25 pulses at 1000:1300:12
        - SA2
            - 10:100, 62 pulses at 1500:2000:8
        - SA3
            - 10:100, 1 pulse at 200
        - LP_SPB
            - 5:100, 50 pulses at 0:300:6
    """

    sources = [
        Timeserver('SPB_RR_SYS/TSYS/TIMESERVER'),
        PulsePatternDecoder('SPB_RR_SYS/MDL/BUNCH_PATTERN'),
        Timeserver('ODD_TIMESERVER_NAME'),
        PulsePatternDecoder('TRAIN_LESS_DECODER', no_ctrl_data=True),
        Timeserver('TRAIN_LESS_TIMESERVER', no_ctrl_data=True, nsamples=0),
        Timeserver('PULSE_LESS_TIMESERVER', no_pulses=True),
        XGM('SPB_XTD9_XGM/DOOCS/MAIN'),
        Motor("MOTOR/MCMOTORYFACE"),
        DetectorMotorDataSelector("SPB_IRU_AGIPD1M/DS", "SPB_IRU_AGIPD1M"),
        DetectorMotorDataSelector("SPB_EXP_AGIPD1M2/DS", "SPB_EXP_AGIPD1M2"),
        Motor('SPB_IRDA_JF4M/MOTOR/X1'),
        Motor('SPB_IRDA_JF4M/MOTOR/X2'),
    ]
    sources += get_motor_sources("SPB_IRU_AGIPD1M")

    with TemporaryDirectory() as td:
        path = Path(td) / 'RAW-R0001-DA01-S00000.h5'
        write_file(path, sources, 100)
        with h5py.File(path, 'a') as f:
            motor_ds = f['CONTROL/MOTOR/MCMOTORYFACE/actualPosition/value']
            # Simulate a scan of 10 steps, with intermediate positions for
            # 1 train at each transition between steps.
            motor_ds[:] = np.repeat(np.arange(10), 10)
            motor_ds[10::10] = np.arange(9) + 0.5
            # write agipd quadrand motor positions
            write_motor_positions(f, "SPB_IRU_AGIPD1M")
            # write jf4m halves motor positions
            jfx1 = f['CONTROL/SPB_IRDA_JF4M/MOTOR/X1/actualPosition/value']
            jfx1[:] = 10
            jfx2 = f['CONTROL/SPB_IRDA_JF4M/MOTOR/X2/actualPosition/value']
            jfx2[:10] = 5
            jfx2[10:] = 6

        yield td


@pytest.fixture(scope='function')
def mock_spb_aux_run(mock_spb_aux_directory):
    yield RunDirectory(mock_spb_aux_directory)


@pytest.fixture(scope="session")
def multi_xgm_directory():
    sources = [
        XGM("SA2_XTD1_XGM/XGM/DOOCS"),
        XGMD("SPB_XTD9_XGM/XGM/DOOCS"),
        XGMReduced("SQS_DIAG1_XGMD/XGM/DOOCS"),
        XGM("NON_ACTUALIZED_XGM/XGM/DOOCS", main_nbunches_property="numberOfBunches"),
        XGM("HOBBIT_XGM/XGM/DOOCS", main_nbunches_property="nummberOfBrunches")
    ]

    with TemporaryDirectory() as td:
        # We need format version 1.1 for the XGM tests, because without the full
        # run metadata we can't get RUN values after a .union() or .select().
        write_file(Path(td) / 'RAW-R0001-DA01-S00000.h5', sources, 100, format_version="1.1")
        yield td


@pytest.fixture(scope="function")
def multi_xgm_run(multi_xgm_directory):
    aliases = {"sa2-xgm": "SA2_XTD1_XGM/XGM/DOOCS"}
    yield RunDirectory(multi_xgm_directory).with_aliases(aliases)


@pytest.fixture(scope='session')
def mock_sqs_remi_directory():
    sources = [
        Timeserver('SQS_RR_UTC/TSYS/TIMESERVER'),
        PulsePatternDecoder('SQS_RR_UTC/TSYS/PP_DECODER'),
        XGM('SA3_XTD10_XGM/XGM/DOOCS'),
        ReconstructedDld('SQS_REMI_DLD6/DET/TOP'),
        ReconstructedDld('SQS_REMI_DLD6/DET/BOTTOM'),
        Motor('SQS_ILH_LAS/MOTOR/DELAY_AX_800'),
        AdqDigitizer('SQS_DIGITIZER_UTC1/ADC/1', channels_per_board=2 * [4]),
        AdqDigitizer('SQS_DIGITIZER_UTC2/ADC/1', channels_per_board=4 * [4],
                     data_channels={(0, 0), (2, 1)})]

    with TemporaryDirectory() as td:
        write_file(Path(td) / 'RAW-R0001-DA01-S00000.h5', sources, 100)
        yield td


@pytest.fixture(scope='function')
def mock_sqs_remi_run(mock_sqs_remi_directory):
    yield RunDirectory(mock_sqs_remi_directory)


@pytest.fixture(scope='session')
def mock_sqs_timepix_directory():
    sources = [
        Timeserver('SQS_RR_UTC/TSYS/TIMESERVER'),
        Timepix3Receiver('SQS_EXTRA_TIMEPIX/DET/TIMEPIX3'),
        Timepix3Receiver('SQS_EXP_TIMEPIX/DET/TIMEPIX3'),
        Timepix3Centroids('SQS_EXP_TIMEPIX/CAL/TIMEPIX3')]

    with TemporaryDirectory() as td:
        write_file(Path(td) / 'RAW-R0001-DA01-S00000.h5', sources, 100)
        yield td


@pytest.fixture(scope='function')
def mock_sqs_timepix_run(mock_sqs_timepix_directory):
    yield RunDirectory(mock_sqs_timepix_directory)


@pytest.fixture(scope='function')
def mock_timepix_exceeded_buffer_run(mock_sqs_timepix_directory):
    with TemporaryDirectory() as td:
        copytree(mock_sqs_timepix_directory, td, dirs_exist_ok=True)

        with h5py.File(next(Path(td).glob('*.h5')), 'r+') as f:
            tpx_root = f['INSTRUMENT/SQS_EXP_TIMEPIX/DET/TIMEPIX3'
                         ':daqOutput.chip0']

            size_dset = tpx_root['data/size']
            size_dset[np.argmax(size_dset)] += tpx_root['data/x'].shape[1]

        yield RunDirectory(td).deselect('SQS_EXTRA*')
