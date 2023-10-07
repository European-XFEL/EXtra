
from pathlib import Path
from tempfile import TemporaryDirectory

import h5py
import numpy as np
import pytest
from extra_data import RunDirectory
from extra_data.tests.mockdata import write_file
from extra_data.tests.mockdata.base import DeviceBase
from extra_data.tests.mockdata.motor import Motor
from extra_data.tests.mockdata.xgm import XGM

from .mockdata.timeserver import PulsePatternDecoder, Timeserver


class PPU(DeviceBase):
    control_keys = [
        ('trainTrigger.numberOfTrains', 'i4', ()),
        ('trainTrigger.sequenceStart', 'i4', ()),
    ]
    extra_run_values = [
        ('classId', None, 'PulsePickerTrainTrigger'),
    ]


@pytest.fixture(scope='session')
def mock_spb_aux_run():
    sources = [
        Timeserver('SPB_RR_SYS/TSYS/TIMESERVER'),
        PulsePatternDecoder('SPB_RR_SYS/MDL/BUNCH_PATTERN'),
        Timeserver('ODD_TIMESERVER_NAME'),
        XGM('SPB_XTD9_XGM/DOOCS/MAIN'),
        Motor("MOTOR/MCMOTORYFACE")]

    with TemporaryDirectory() as td:
        write_file(Path(td) / 'RAW-R0001-DA01-S00000.h5', sources, 100)
        yield RunDirectory(td)


@pytest.fixture(scope='session')
def ppu_run():
    sources = [
        PPU('HED_XTD6_PPU/MDL/PPU_TRIGGER'),
        PPU('HED_DIPOLE_PPU/MDL/PPU_TRIGGER'),
        Timeserver('HED_RR_SYS/TSYS/TIMESERVER'),
    ]

    with TemporaryDirectory() as td:
        fpath = Path(td) / 'RAW-R0001-DA01-S00000.h5'
        write_file(fpath, sources, 100, firsttrain=10000, format_version='1.3')

        with h5py.File(fpath, 'r+') as f:
            f['/CONTROL/HED_XTD6_PPU/MDL/PPU_TRIGGER/trainTrigger/numberOfTrains'] = np.array([10] * 100, dtype=np.int64)
            f['/CONTROL/HED_XTD6_PPU/MDL/PPU_TRIGGER/trainTrigger/sequenceStart'] = np.repeat([9000, 10080], 50)
            f['/CONTROL/HED_DIPOLE_PPU/MDL/PPU_TRIGGER/trainTrigger/numberOfTrains'] = np.array([1] * 100, dtype=np.int64)
            f['/CONTROL/HED_DIPOLE_PPU/MDL/PPU_TRIGGER/trainTrigger/sequenceStart'] = np.repeat([9985, 10015, 10045, 10075], 25)

        aliases = {'ppu-hed': 'HED_XTD6_PPU/MDL/PPU_TRIGGER',
                   'ppu-dipole': 'HED_DIPOLE_PPU/MDL/PPU_TRIGGER'}
        run = RunDirectory(td)
        yield run.with_aliases(aliases)
