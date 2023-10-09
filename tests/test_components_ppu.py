import pandas as pd
import pytest

from extra_data.reader import DataCollection
from extra.components import PPU
from extra.components.ppu import _find_ppu


def test_find_ppu(ppu_run):
    source = _find_ppu(ppu_run, ppu_run['HED_DIPOLE_PPU/MDL/PPU_TRIGGER'])
    assert source.source == 'HED_DIPOLE_PPU/MDL/PPU_TRIGGER'

    source = _find_ppu(ppu_run, ppu_run['HED_DIPOLE_PPU/MDL/PPU_TRIGGER', 'trainTrigger.sequenceStart'])
    assert source.source == 'HED_DIPOLE_PPU/MDL/PPU_TRIGGER'

    source = _find_ppu(ppu_run, 'HED_DIPOLE_PPU/MDL/PPU_TRIGGER')
    assert source.source == 'HED_DIPOLE_PPU/MDL/PPU_TRIGGER'

    source = _find_ppu(ppu_run, 'ppu-hed')
    assert source.source == 'HED_XTD6_PPU/MDL/PPU_TRIGGER'

    source = _find_ppu(ppu_run, 'XTD6')
    assert source.source == 'HED_XTD6_PPU/MDL/PPU_TRIGGER'

    source = _find_ppu(ppu_run.select('HED_XTD6_PPU*'))
    assert source.source == 'HED_XTD6_PPU/MDL/PPU_TRIGGER'

    # fails with multiple PPUs
    with pytest.raises(KeyError) as excinfo:
        _find_ppu(ppu_run)
    assert 'Multiple PPU' in str(excinfo.value)

    # fails with invalid device type
    with pytest.raises(KeyError) as excinfo:
        _find_ppu(ppu_run, 1)
    assert 'not int' in str(excinfo.value)

    # fails with 0 PPUs
    with pytest.raises(KeyError) as excinfo:
        _find_ppu(ppu_run.select('*TIMESERVER'))
    assert 'Could not find a PPU' in str(excinfo.value)

    # too many match
    with pytest.raises(KeyError) as excinfo:
        _find_ppu(ppu_run, 'PPU')
    assert 'Multiple PPUs found matching' in str(excinfo.value)

    # no match
    with pytest.raises(KeyError) as excinfo:
        _find_ppu(ppu_run, 'PPU2')
    assert 'Couldn\'t identify a PPU' in str(excinfo.value)


def test_train_ids(ppu_run):
    # single trigger sequence
    ppu = PPU(ppu_run, 'ppu-hed')
    train_ids = ppu.train_ids()
    assert isinstance(train_ids, list)
    assert len(train_ids) == 10
    train_ids = ppu.train_ids(labelled=True)
    assert isinstance(train_ids, pd.Series)
    assert train_ids.size == 10  # 10 trains in total
    assert train_ids.index.unique().size == 1  # single trigger sequence

    # multiple trigger sequences
    ppu = PPU(ppu_run, 'ppu-dipole')
    train_ids = ppu.train_ids()
    assert isinstance(train_ids, list)
    assert len(train_ids) == 3
    train_ids = ppu.train_ids(labelled=True)
    assert isinstance(train_ids, pd.Series)
    assert train_ids.index.unique().size == 3  # 3 trigger sequence
    assert train_ids.size == 3  # 1 train per sequence


def test_trains(ppu_run):
    ppu = PPU(ppu_run, 'ppu-dipole')
    reduced_run = ppu.trains()
    assert isinstance(reduced_run, DataCollection)
    assert reduced_run.train_ids == [10015, 10045, 10075]

    # split per sequence
    reduced_run = ppu.trains(split_sequence=True)
    assert isinstance(reduced_run, list)
    assert len(reduced_run) == 3
    assert reduced_run[0].train_ids == [10015]
