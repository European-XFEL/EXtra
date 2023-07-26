
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from extra_data.tests.mockdata import write_file
from extra_data.tests.mockdata.xgm import XGM

from .mockdata.timeserver import Timeserver, PulsePatternDecoder


@pytest.fixture(scope='session')
def mock_spb_aux_run():
    sources = [
        Timeserver('SPB_RR_SYS/TSYS/TIMESERVER'),
        PulsePatternDecoder('SPB_RR_SYS/MDL/BUNCH_PATTERN'),
        Timeserver('ODD_TIMESERVER_NAME'),
        XGM('SPB_XTD9_XGM/DOOCS/MAIN')]

    with TemporaryDirectory() as td:
        write_file(Path(td) / 'RAW-R0001-DA01-S00000.h5', sources, 100)
        yield td
