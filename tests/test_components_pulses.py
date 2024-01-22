
from unittest.mock import MagicMock

import pytest
import numpy as np

import extra

from euxfel_bunch_pattern import PPL_BITS
from extra.data import RunDirectory, SourceData, KeyData, by_id
from extra.components import XrayPulses, OpticalLaserPulses, PumpProbePulses, DldPulses


pattern_sources = dict(
    argvalues=['SPB_RR_SYS/TSYS/TIMESERVER',
               'SPB_RR_SYS/TSYS/TIMESERVER:outputBunchPattern',
               'SPB_RR_SYS/MDL/BUNCH_PATTERN'],
    ids=['timeserver-control', 'timeserver-instrument', 'ppdecoder']
)


def assert_equal_sourcedata(sd1, sd2):
    assert isinstance(sd1, SourceData)
    assert isinstance(sd2, SourceData)
    assert sd1.source == sd2.source
    assert sd1.train_ids == sd2.train_ids


def assert_equal_keydata(kd1, kd2):
    assert isinstance(kd1, KeyData)
    assert isinstance(kd2, KeyData)
    assert kd1.source == kd2.source
    assert kd1.key == kd2.key
    assert kd1.train_ids == kd2.train_ids


def test_definitions(mock_spb_aux_run):
    pulses = XrayPulses(mock_spb_aux_run.select('SPB*'))
    pulses.master_clock
    pulses.bunch_clock_divider
    pulses.bunch_repetition_rate


def test_init(mock_spb_aux_run):
    # First only select "regular" sources.
    run = mock_spb_aux_run.select('SPB*')

    pulses = XrayPulses(run)
    assert_equal_sourcedata(
        pulses.timeserver,
        run['SPB_RR_SYS/TSYS/TIMESERVER:outputBunchPattern'])
    assert_equal_keydata(
        pulses.bunch_pattern_table,
        pulses.timeserver['data.bunchPatternTable'])
    assert pulses.sase == 1

    # Trim down to only control source.
    pulses = XrayPulses(run.select('*TSYS/TIMESERVER'))
    assert_equal_sourcedata(
        pulses.timeserver,
        run['SPB_RR_SYS/TSYS/TIMESERVER'])
    assert_equal_keydata(
        pulses.bunch_pattern_table,
        pulses.timeserver['bunchPatternTable'])
    assert pulses.sase == 1

    # Overwrite SASE.
    pulses = XrayPulses(run, sase=2)
    assert pulses.sase == 2

    # Now take the full run, should have two timeservers and raise
    # exception.
    run = mock_spb_aux_run

    with pytest.raises(ValueError):
        XrayPulses(run)

    # Pick one specifically.
    pulses = XrayPulses(run, source='ODD_TIMESERVER_NAME')
    assert_equal_sourcedata(
        pulses.timeserver,
        run['ODD_TIMESERVER_NAME'])
    assert_equal_keydata(
        pulses.bunch_pattern_table,
        pulses.timeserver['bunchPatternTable'])
    assert pulses.sase == 1

    # Remove all timeservers, should always raise exception.
    run_no_ts = run.select('*XGM*')

    with pytest.raises(ValueError):
        XrayPulses(run_no_ts)

    with pytest.raises(ValueError):
        XrayPulses(run_no_ts, sase=1)


@pytest.mark.parametrize(
    'source',
    argvalues=['TRAIN_LESS_TIMESERVER',
               'TRAIN_LESS_TIMESERVER:outputBunchPattern',
               'TRAIN_LESS_DECODER'],
    ids=['timeserver-control', 'timeserver-instrument', 'ppdecoder']
)
def test_no_trains(mock_spb_aux_run, source):
    # Test with entirely empty data.
    pulses = XrayPulses(mock_spb_aux_run, source)
    assert pulses.pulse_ids().empty
    assert pulses.pulse_counts().empty


def test_select_trains(mock_spb_aux_run):
    run = mock_spb_aux_run
    pulses = XrayPulses(run.select('SPB*'))

    subrun = run.select_trains(np.s_[:20])
    subpulses = pulses.select_trains(np.s_[:20])

    assert_equal_sourcedata(
        subpulses.timeserver,
        subrun['SPB_RR_SYS/TSYS/TIMESERVER:outputBunchPattern'])
    assert_equal_keydata(
        subpulses.bunch_pattern_table,
        subpulses.timeserver['data.bunchPatternTable'])

    # Select down to trains with no pulses.
    pulses = XrayPulses(run, 'PULSE_LESS_TIMESERVER')
    pulses.select_trains(np.s_[:20])


@pytest.mark.parametrize('source', **pattern_sources)
def test_pulse_mask(mock_spb_aux_run, source):
    run = mock_spb_aux_run
    pulses = XrayPulses(run, source=source)

    mask = XrayPulses(run, source=source).pulse_mask()
    assert mask.dims == ('trainId', 'pulseId')
    assert (mask.coords['trainId'] == run.train_ids).all()
    assert (mask.coords['pulseId'] == np.arange(mask.shape[1])).all()
    assert not mask[:10, :].any()
    assert mask[10:50, 1000:1300:6].all()
    assert mask[10:, 1000:1300:12].all()

    mask = XrayPulses(run, source=source, sase=2).pulse_mask()
    assert mask[10:, 1500:2000:8].all()

    assert XrayPulses(run, source=source, sase=2).pulse_mask(
        labelled=False)[10:, 1500:2000:8].all()

    # Test deprecated method
    with pytest.warns():
        assert pulses.get_pulse_mask().equals(pulses.pulse_mask())


@pytest.mark.parametrize('source', **pattern_sources)
def test_is_constant_pattern(mock_spb_aux_run, source):
    run = mock_spb_aux_run
    pulses = XrayPulses(run, source=source)

    assert not pulses.is_constant_pattern()
    assert pulses.select_trains(np.s_[:50]).is_constant_pattern()


@pytest.mark.parametrize('source', **pattern_sources)
def test_pulse_counts(mock_spb_aux_run, source):
    run = mock_spb_aux_run
    pulses = XrayPulses(run, source=source)

    # Test labelled.
    counts = pulses.pulse_counts(labelled=True)
    assert (counts.index == run.train_ids).all()
    assert (counts.iloc[:10] == 0).all()
    assert (counts.iloc[10:50] == 50).all()
    assert (counts.iloc[50:] == 25).all()

    # Test unlabelled.
    np.testing.assert_equal(pulses.pulse_counts(labelled=False), counts)

    # Test different SASE.
    counts = XrayPulses(run, source=source, sase=2).pulse_counts()
    assert (counts.index == run.train_ids).all()
    assert (counts[:10] == 0).all()
    assert (counts[10:] == 63).all()

    # Test deprecated method
    with pytest.warns():
        assert pulses.get_pulse_counts().equals(pulses.pulse_counts())


@pytest.mark.parametrize('source', **pattern_sources)
def test_peek_pulse_ids(mock_spb_aux_run, source):
    run_full = mock_spb_aux_run.select('SPB*')
    run_beam = mock_spb_aux_run.select('SPB*').select_trains(np.s_[10:])

    # Test full run with empty pulses in the beginning.
    pulses = XrayPulses(run_full, source=source)
    assert pulses.peek_pulse_ids().empty
    assert pulses.peek_pulse_ids(labelled=False).size == 0

    # Test run starting at train with pulses.
    pulses = XrayPulses(run_beam, source=source)
    pids = pulses.peek_pulse_ids()
    np.testing.assert_equal(pids.index, np.arange(len(pids)))
    np.testing.assert_equal(pids, np.r_[1000:1300:6])
    np.testing.assert_equal(pulses.peek_pulse_ids(labelled=False), pids)

    # Test different SASE.
    np.testing.assert_equal(
        XrayPulses(run_beam, source=source, sase=2).peek_pulse_ids(),
        np.r_[1500:2000:8])


@pytest.mark.parametrize('source', **pattern_sources)
def test_pulse_ids(mock_spb_aux_run, source):
    run = mock_spb_aux_run
    pulses = XrayPulses(run, source=source)

    # Test labelled.
    pids = pulses.pulse_ids()
    assert len(pids) == 3250
    np.testing.assert_equal(pids[:, 0].index, np.array(run.train_ids[10:]))
    np.testing.assert_equal(pids[run.train_ids[10], :].index, np.r_[:50])
    np.testing.assert_equal(pids[run.train_ids[10], :], np.r_[1000:1300:6])
    np.testing.assert_equal(pids[run.train_ids[50], :], np.r_[1000:1300:12])

    # Test unlabelled.
    np.testing.assert_equal(pulses.pulse_ids(labelled=False), pids)

    # Test deprecated method
    with pytest.warns():
        assert pulses.get_pulse_ids().equals(pulses.pulse_ids())


@pytest.mark.parametrize('source', **pattern_sources)
def test_build_pulse_index(mock_spb_aux_run, source):
    run = mock_spb_aux_run
    pulses = XrayPulses(run, source=source)

    index = pulses.build_pulse_index()
    assert index.names == ['trainId', 'pulseId']

    train_ids = index.get_level_values(0)
    np.testing.assert_equal(
        train_ids[:2000], np.repeat(run.train_ids[10:50], 50))
    np.testing.assert_equal(
        train_ids[2000:], np.repeat(run.train_ids[50:], 25))

    pulse_ids = index.get_level_values(1)
    np.testing.assert_equal(
        pulse_ids[:2000], np.tile(np.r_[1000:1300:6], 40))
    np.testing.assert_equal(
        pulse_ids[2000:], np.tile(np.r_[1000:1300:12], 50))

    pulse_indices = pulses.build_pulse_index('pulseIndex').get_level_values(1)
    np.testing.assert_equal(pulse_indices[:2000], np.tile(np.r_[:50], 40))
    np.testing.assert_equal(pulse_indices[2000:], np.tile(np.r_[:25], 50))

    times = pulses.build_pulse_index('time').get_level_values(1)
    rate = pulses.bunch_repetition_rate
    np.testing.assert_allclose(
        times[:2000], np.tile((np.r_[1000:1300:6] - 1000) / rate, 40))
    np.testing.assert_allclose(
        times[2000:], np.tile((np.r_[1000:1300:12] - 1000) / rate, 50))

    # Test deprecated methods
    with pytest.warns():
        assert pulses.get_pulse_index().equals(pulses.build_pulse_index())


@pytest.mark.parametrize('source', **pattern_sources)
def test_search_pulse_patterns(mock_spb_aux_run, source):
    run = mock_spb_aux_run
    pulses = XrayPulses(run, source=source)

    for labelled in [True, False]:
        patterns = pulses.search_pulse_patterns(labelled=labelled)
        assert len(patterns) == 3
        assert patterns[0][0].value == by_id[10000:10009].value
        assert len(patterns[0][1]) == 0
        assert patterns[1][0].value == by_id[10010:10049].value
        np.testing.assert_equal(patterns[1][1], np.r_[1000:1300:6])
        assert patterns[2][0].value == by_id[10050:10100].value
        np.testing.assert_equal(patterns[2][1], np.r_[1000:1300:12])

        if labelled:
            assert patterns[0][1].index.empty
            np.testing.assert_equal(patterns[1][1].index, np.arange(50))
            np.testing.assert_equal(patterns[2][1].index, np.arange(25))


@pytest.mark.parametrize('source', **pattern_sources)
def test_trains(mock_spb_aux_run, source):
    run = mock_spb_aux_run
    pulses = XrayPulses(run, source=source)

    for ref_tid, (tid_l, pids_l), (tid_ul, pids_ul) in zip(
        run.train_ids[10:], pulses.trains(), pulses.trains(labelled=False)
    ):
        assert ref_tid == tid_l == tid_ul

        if tid_l < run.train_ids[50]:
            np.testing.assert_equal(pids_l.index, np.r_[:50])
            np.testing.assert_equal(pids_l, np.r_[1000:1300:6])
            np.testing.assert_equal(pids_l, pids_ul)
        else:
            np.testing.assert_equal(pids_l.index, np.r_[:25])
            np.testing.assert_equal(pids_l, np.r_[1000:1300:12])
            np.testing.assert_equal(pids_l, pids_ul)


@pytest.mark.parametrize('source', **pattern_sources)
def test_optical_laser_basic(mock_spb_aux_run, source):
    run = mock_spb_aux_run
    pulses = OpticalLaserPulses(run.select_trains(np.s_[10:]), source=source)

    assert pulses.ppl_seed == PPL_BITS.LP_SPB
    assert (pulses.pulse_counts() == 50).all()
    assert (pulses.pulse_ids(labelled=False)[:50] == np.r_[0:300:6]).all()


def test_optical_laser_specials(mock_spb_aux_run):
    run = mock_spb_aux_run.select('SPB*')

    # Different laser seed by enum
    pulses = OpticalLaserPulses(run, ppl_seed=PPL_BITS.LP_SQS)
    assert pulses.ppl_seed == PPL_BITS.LP_SQS
    assert (pulses.pulse_counts() == 0).all()

    # Different laser seed by string.
    pulses = OpticalLaserPulses(run, ppl_seed='MID')
    assert pulses.ppl_seed == PPL_BITS.LP_SASE2
    assert (pulses.pulse_counts() == 0).all()

    # Full run with two timeservers
    run = mock_spb_aux_run

    with pytest.raises(ValueError):
        OpticalLaserPulses(run)

    OpticalLaserPulses(run, source='ODD_TIMESERVER_NAME')

    # Explicit PPL seeds with pulse pattern decoder.
    OpticalLaserPulses(run, source='SPB_RR_SYS/MDL/BUNCH_PATTERN',
                       ppl_seed='SPB')

    with pytest.raises(ValueError):
        OpticalLaserPulses(run, source='SPB_RR_SYS/MDL/BUNCH_PATTERN',
                           ppl_seed='FXE')

    # Only odd timeserver now
    run = run.select('ODD_TIMESERVER_NAME*')

    with pytest.raises(ValueError):
        OpticalLaserPulses(run)

    OpticalLaserPulses(run, ppl_seed=PPL_BITS.LP_SPB)


def test_ppdecoder(mock_spb_aux_run):
    run = mock_spb_aux_run
    pulses = XrayPulses(run, source='SPB_RR_SYS/MDL/BUNCH_PATTERN')

    assert_equal_sourcedata(pulses.pulse_pattern_decoder,
                            run['SPB_RR_SYS/MDL/BUNCH_PATTERN'])

    with pytest.raises(ValueError):
        pulses.timeserver

    with pytest.raises(ValueError):
        pulses.bunch_pattern_table


def test_dld_pulses(capsys):
    from numpy.lib.recfunctions import drop_fields

    trigger_dt = np.dtype([('start', np.int32), ('stop', np.int32),
                           ('offset', np.float64), ('pulse', np.int16),
                           ('fel', np.bool_), ('ppl', np.bool_)])

    triggers = np.zeros(10, dtype=trigger_dt)
    triggers['start'] = 10 + np.arange(10) * 1000
    triggers['stop'] = 794 + np.arange(10) * 1000
    triggers['offset'] = 0.0
    triggers['pulse'] = 100 + np.arange(10) * 4
    triggers['fel'] = True
    triggers['ppl'] = False

    mock_key = MagicMock()
    mock_key.train_id_coordinates.return_value = np.repeat(1000, 10)
    mock_key.data_counts.return_value = np.array([10])
    mock_key.ndarray.return_value = triggers

    mock_source = MagicMock()
    mock_source.__getitem__ = lambda self, _: mock_key

    # Test regular.
    pulses = DldPulses(mock_source)
    pulse_ids = pulses.pulse_ids()
    assert pulse_ids.index.names == ['trainId', 'pulseIndex', 'fel', 'ppl']
    np.testing.assert_equal(pulse_ids, triggers['pulse'])

    counts = pulses.pulse_counts()
    np.testing.assert_equal(counts, np.array([10]))

    triggers_ = pulses.triggers()
    assert triggers_.index.names == ['trainId', 'pulseId', 'fel', 'ppl']
    np.testing.assert_equal(triggers_['start'],  triggers['start'])
    np.testing.assert_equal(triggers_['stop'],  triggers['stop'])
    np.testing.assert_equal(triggers_['offset'],  triggers['offset'])

    # Test without fel/ppl fields.
    mock_key.ndarray.return_value = drop_fields(triggers, ['fel', 'ppl'])
    assert DldPulses(mock_source).pulse_ids().index.names == [
        'trainId', 'pulseIndex']

    # Test without pulse field and default settings.
    mock_key.ndarray.return_value = drop_fields(triggers, ['pulse'])
    np.testing.assert_equal(DldPulses(mock_source).pulse_ids(),
                            np.r_[:50:5])
    captured = capsys.readouterr()
    assert 'No actual pulse IDs available in data' in captured.err

    # Test without pulse field and custom settings.
    mock_key.ndarray.return_value = drop_fields(triggers, ['pulse'])
    pulses = DldPulses(mock_source, clock_ratio=196/1.97, first_pulse_id=100)
    np.testing.assert_equal(pulses.pulse_ids(), np.r_[100:200:10])
    captured = capsys.readouterr()
    assert 'No actual pulse IDs available in data' in captured.err

    # Test the deprecated method
    with pytest.warns():
        assert pulses.get_triggers().equals(pulses.triggers())


@pytest.mark.parametrize('source', **pattern_sources)
def test_pump_probe_basic(mock_spb_aux_run, source):
    run = mock_spb_aux_run
    pulses = PumpProbePulses(run, source=source, pulse_offset=1)

    assert pulses.sase == 1
    assert pulses.ppl_seed == PPL_BITS.LP_SPB

    with pytest.raises(ValueError):
        # Cannot extrapolate due to missing pulses in the beginning.
        pids = pulses.pulse_ids()

    pulses = PumpProbePulses(run.select_trains(np.s_[10:]),
                             source=source, pulse_offset=1)

    # Pulse IDs
    pids = pulses.pulse_ids()
    assert pids.index.names == ['trainId', 'pulseIndex', 'fel', 'ppl']
    np.testing.assert_equal(pids[run.train_ids[10]], np.r_[1000:1306:6])

    fel = pids.index.get_level_values('fel')
    assert fel[:50].all()
    assert not fel[50]

    ppl = pids.index.get_level_values('ppl')
    assert not ppl[0]
    assert ppl[1:51].all()

    # Pulse mask
    assert pulses.pulse_mask(labelled=False)[0, 1000:1306:6].all()

    # Is constant pattern?
    assert not pulses.is_constant_pattern()
    assert pulses.select_trains(np.s_[:1]).is_constant_pattern()

    # Search patterns
    patterns = pulses.search_pulse_patterns()
    assert len(patterns) == 2
    assert patterns[0][0].value == by_id[10010:10049].value
    np.testing.assert_equal(patterns[0][1], np.r_[1000:1306:6])
    assert patterns[1][0].value == by_id[10050:10100].value
    assert patterns[1][1].iloc[0] == 1000
    np.testing.assert_equal(patterns[1][1][1:], np.r_[1012:1312:6])

    with pytest.raises(ValueError):
        # Should fail with any relation keyword.
        pulses = PumpProbePulses(run, source=source)


def test_pump_probe_defaults(mock_spb_aux_run):
    run = mock_spb_aux_run.select('SPB*').select_trains(np.s_[10:])
    np.testing.assert_equal(
        PumpProbePulses(run, pulse_offset=1).pulse_ids()[run.train_ids[0]],
        np.r_[1000:1306:6])
