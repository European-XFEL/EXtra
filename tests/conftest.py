
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from extra_data import RunDirectory
from extra_data.tests.mockdata import write_file
from extra_data.tests.mockdata.xgm import XGM
from extra_data.tests.mockdata.motor import Motor

from .mockdata.timeserver import Timeserver, PulsePatternDecoder


@pytest.fixture(scope='session')
def mock_spb_aux_directory():
    sources = [
        Timeserver('SPB_RR_SYS/TSYS/TIMESERVER'),
        PulsePatternDecoder('SPB_RR_SYS/MDL/BUNCH_PATTERN'),
        Timeserver('ODD_TIMESERVER_NAME'),
        PulsePatternDecoder('TRAIN_LESS_DECODER', no_ctrl_data=True),
        Timeserver('TRAIN_LESS_TIMESERVER', no_ctrl_data=True, nsamples=0),
        Timeserver('PULSE_LESS_TIMESERVER', no_pulses=True),
        XGM('SPB_XTD9_XGM/DOOCS/MAIN'),
        Motor("MOTOR/MCMOTORYFACE")]

    with TemporaryDirectory() as td:
        write_file(Path(td) / 'RAW-R0001-DA01-S00000.h5', sources, 100)
        yield td


@pytest.fixture(scope='function')
def mock_spb_aux_run(mock_spb_aux_directory):
    yield RunDirectory(mock_spb_aux_directory)
