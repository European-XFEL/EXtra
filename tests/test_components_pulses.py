
from unittest.mock import MagicMock

import pytest
import numpy as np
import pandas as pd

import extra

from euxfel_bunch_pattern import PPL_BITS, DESTINATION_TLD, DESTINATION_T5D, \
    PHOTON_LINE_DEFLECTION
from extra.data import RunDirectory, SourceData, KeyData, by_id
from extra.components import XrayPulses, OpticalLaserPulses, MachinePulses, \
    PumpProbePulses, DldPulses

from .mockdata import assert_equal_sourcedata, assert_equal_keydata


pattern_sources = dict(
    argvalues=['SPB_RR_SYS/TSYS/TIMESERVER',
               'SPB_RR_SYS/TSYS/TIMESERVER:outputBunchPattern',
               'SPB_RR_SYS/MDL/BUNCH_PATTERN'],
    ids=['timeserver-control', 'timeserver-instrument', 'ppdecoder']
)


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

    # Construct from SourceData.
    pulses = XrayPulses(None, source=run['ODD_TIMESERVER_NAME'], sase=1)
    assert_equal_sourcedata(
        pulses.timeserver,
        run['ODD_TIMESERVER_NAME'])

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

    # Test deprecated method.
    with pytest.warns():
        assert pulses.get_pulse_mask().equals(pulses.pulse_mask())


@pytest.mark.parametrize('source', **pattern_sources)
def test_is_constant_pattern(mock_spb_aux_run, source):
    pulses = XrayPulses(mock_spb_aux_run, source=source)

    # Default arguments (empty trains are ignored).
    assert not pulses.is_constant_pattern()
    assert pulses.select_trains(np.s_[:50]).is_constant_pattern()

    # Explicit handling of empty trains.
    assert not pulses.is_constant_pattern(
        include_empty_trains=False)
    assert pulses.select_trains(np.s_[:50]).is_constant_pattern(
        include_empty_trains=False)

    assert not pulses.is_constant_pattern(
        include_empty_trains=True)
    assert not pulses.select_trains(np.s_[:50]).is_constant_pattern(
        include_empty_trains=True)
    assert pulses.select_trains(np.s_[10:50]).is_constant_pattern(
        include_empty_trains=True)

    with pytest.raises(TypeError):
        pulses.is_constant_pattern(True)


@pytest.mark.parametrize('source', **pattern_sources)
def test_is_interleaved(mock_spb_aux_run, source):
    pulses = XrayPulses(mock_spb_aux_run, source=source, sase=1)

    assert pulses.is_interleaved_with(pulses.machine_pulses())
    assert not pulses.is_interleaved_with(
        XrayPulses(mock_spb_aux_run, source=source, sase=2))
    assert not pulses.is_sa1_interleaved_with_sa3()


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

    # Test deprecated method.
    with pytest.warns():
        assert pulses.get_pulse_counts().equals(pulses.pulse_counts())


@pytest.mark.parametrize('source', **pattern_sources)
def test_pulse_periods(mock_spb_aux_run, source):
    run = mock_spb_aux_run
    pulses = XrayPulses(run, source=source)

    # Test labelled.
    periods = pulses.pulse_periods(labelled=True)
    assert (periods.index == run.train_ids).all()
    assert np.issubdtype(periods.dtype, np.integer)
    assert (periods.iloc[:10] == 0).all()
    assert (periods.iloc[10:50] == 6).all()
    assert (periods.iloc[50:] == 12).all()

    # Test unlabelled.
    np.testing.assert_equal(pulses.pulse_periods(labelled=False), periods)

    # Test different SASE with only one pulse.
    pulses = XrayPulses(run, source=source, sase=3)

    with pytest.raises(ValueError):
        # Fails without explicit single_pulse_value.
        periods = pulses.pulse_periods()

    # Explicit (finite) single_pulse_value.
    periods = pulses.pulse_periods(single_pulse_value=-1)
    assert (periods.index == run.train_ids).all()
    assert np.issubdtype(periods.dtype, np.integer)
    assert (periods.iloc[:10] == 0).all()  # Should still be 0.
    assert (periods.iloc[10:] == -1).all()

    # Explicit (finite) single_pulse_value and no_pulse_value.
    periods = pulses.pulse_periods(single_pulse_value=-1, no_pulse_value=-2)
    assert (periods.index == run.train_ids).all()
    assert np.issubdtype(periods.dtype, np.integer)
    assert (periods.iloc[:10] == -2).all()
    assert (periods.iloc[10:] == -1).all()

    # Explicit (not finite) single_pulse_value and no_pulse_value.
    periods = pulses.pulse_periods(
        single_pulse_value=np.inf, no_pulse_value=42.2)
    assert (periods.index == run.train_ids).all()
    assert np.issubdtype(periods.dtype, np.floating)
    assert (periods.iloc[:10] == 42.2).all()
    assert (periods.iloc[10:] == np.inf).all()



@pytest.mark.parametrize('source', **pattern_sources)
def test_pulse_repetition_rates(mock_spb_aux_run, source):
    run = mock_spb_aux_run
    pulses = XrayPulses(run, source=source)

    # Test labelled.
    rates = pulses.pulse_repetition_rates(labelled=True)
    assert (rates.index == run.train_ids).all()
    assert rates.iloc[:10].isna().all()
    np.testing.assert_allclose(rates.iloc[10:50], (1.3e9 / 288) / 6)
    np.testing.assert_allclose(rates.iloc[50:], (1.3e9 / 288) / 12)

    # Test unlabelled.
    np.testing.assert_equal(
        pulses.pulse_repetition_rates(labelled=False), rates)

    # Test different SASE with only one pulse.
    rates = XrayPulses(run, source=source, sase=3).pulse_repetition_rates()
    assert (rates.index == run.train_ids).all()
    assert rates[:10].isna().all()
    np.testing.assert_allclose(rates[10:], 0.0)

    # Test special method for machine repetition rate, should be 2.2 MHz.
    assert np.isclose(pulses.machine_repetition_rate(), 2.2e6, atol=1e5)


@pytest.mark.parametrize('source', **pattern_sources)
def test_train_durations(mock_spb_aux_run, source):
    run = mock_spb_aux_run
    pulses = XrayPulses(run, source=source)

    # Test labelled.
    times = pulses.train_durations(labelled=True)
    assert (times.index == run.train_ids).all()
    assert times.iloc[:10].isna().all()
    np.testing.assert_allclose(times.iloc[10:50], 294 / (1.3e9 / 288))
    np.testing.assert_allclose(times.iloc[50:], 288 / (1.3e9 / 288))

    # Test unlabelled.
    np.testing.assert_equal(pulses.train_durations(labelled=False), times)

    # Test different SASE with only one pulse.
    times = XrayPulses(run, source=source, sase=3).train_durations()
    assert (times.index == run.train_ids).all()
    assert times[:10].isna().all()
    np.testing.assert_allclose(times[10:], 0.0)


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

    # Test deprecated method.
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

    times = pulses.build_pulse_index('pulseTime').get_level_values(1)
    rate = pulses.bunch_repetition_rate
    np.testing.assert_allclose(
        times[:2000], np.tile((np.r_[1000:1300:6] - 1000) / rate, 40))
    np.testing.assert_allclose(
        times[2000:], np.tile((np.r_[1000:1300:12] - 1000) / rate, 50))

    # Test deprecated methods.
    with pytest.warns():
        assert pulses.get_pulse_index().equals(pulses.build_pulse_index())

    # Force dtype.
    assert pulses.build_pulse_index(
        'pulseIndex', pulse_dtype=np.float32
    ).dtypes['pulseIndex'] == np.float32


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

    # Different laser seed by enum.
    pulses = OpticalLaserPulses(run, ppl_seed=PPL_BITS.LP_SQS)
    assert pulses.ppl_seed == PPL_BITS.LP_SQS
    assert (pulses.pulse_counts() == 0).all()

    # Different laser seed by string.
    pulses = OpticalLaserPulses(run, ppl_seed='MID')
    assert pulses.ppl_seed == PPL_BITS.LP_SASE2
    assert (pulses.pulse_counts() == 0).all()

    # Full run with two timeservers.
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

    # Only odd timeserver now.
    run = run.select('ODD_TIMESERVER_NAME*')

    with pytest.raises(ValueError):
        OpticalLaserPulses(run)

    OpticalLaserPulses(run, ppl_seed=PPL_BITS.LP_SPB)


@pytest.mark.parametrize('source', **pattern_sources)
def test_machine_pulses_default(mock_spb_aux_run, source):
    pulses = MachinePulses(mock_spb_aux_run, source=source)
    np.testing.assert_equal(pulses.pulse_ids().iloc[:1350], np.r_[:2700:2])

    pulse_counts = pulses.pulse_counts()
    assert not pulse_counts[:10].any()
    assert (pulse_counts[10:] == 1350).all()

    # Get from other pulse components.
    indirect_pulses = XrayPulses(
        mock_spb_aux_run, source=source).machine_pulses()
    pd.testing.assert_series_equal(
        pulses.pulse_ids(), indirect_pulses.pulse_ids())


def test_machine_pulses_specials(mock_spb_aux_run):
    timeserver_run = mock_spb_aux_run.select('*TSYS/TIMESERVER*')

    # Make an odd combination of bits, here SASE2 or soft kick.
    odd_pulses = MachinePulses(timeserver_run,
                               mask=DESTINATION_T5D | PHOTON_LINE_DEFLECTION)
    pids = odd_pulses.pulse_ids()
    assert pids.iloc[0] == 200  # The one SASE3 pulse.
    np.testing.assert_equal(pids.iloc[1:64], np.r_[1500:2000:8])  # SASE2.

    # Main dump and SPB laser seed.
    odd_pulses = MachinePulses(timeserver_run, require_all_bits=True,
                               mask=DESTINATION_TLD | PPL_BITS.LP_SPB)
    np.testing.assert_equal(odd_pulses.pulse_ids().iloc[:50], np.r_[:300:6])

    # Main dump and SA2.
    odd_pulses = MachinePulses(timeserver_run, require_all_bits=True,
                               mask=DESTINATION_TLD | DESTINATION_T5D)
    assert odd_pulses.pulse_ids().empty

    # Works with problematic numpy integer types.
    assert (MachinePulses(
        timeserver_run, mask=np.uint64(PHOTON_LINE_DEFLECTION)
    ).pulse_ids() == 200).all()

    # Does not work with non-int types
    with pytest.raises(TypeError):
        MachinePulses(timeserver_run, mask='0')

    # ppdecoder limitations
    ppdec_run = mock_spb_aux_run.select('SPB_RR_SYS/MDL/BUNCH_PATTERN')

    with pytest.raises(ValueError):
        MachinePulses(ppdec_run, require_all_bits=True)

    with pytest.raises(ValueError):
        MachinePulses(ppdec_run, mask=1)


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

    # Test the deprecated method.
    with pytest.warns():
        assert pulses.get_triggers().equals(pulses.triggers())

    # Test with negative PPL indices.
    mock_key.ndarray.return_value = triggers
    mock_key.ndarray.return_value[:2]['ppl'] = True
    mock_key.ndarray.return_value[:2]['fel'] = False
    mock_key.ndarray.return_value[-2:]['ppl'] = True
    mock_key.ndarray.return_value[-2:]['fel'] = False

    pids = DldPulses(mock_source, negative_ppl_indices=True).pulse_ids()
    np.testing.assert_equal(
        pids.index.get_level_values('pulseIndex'),
        np.array([-1, -2, 0, 1, 2, 3, 4, 5, -3, -4]))


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

    # Pulse IDs.
    pids = pulses.pulse_ids()
    assert pids.index.names == ['trainId', 'pulseIndex', 'fel', 'ppl']
    np.testing.assert_equal(pids[run.train_ids[10]], np.r_[1000:1306:6])

    fel = pids.index.get_level_values('fel')
    assert fel[:50].all()
    assert not fel[50]

    ppl = pids.index.get_level_values('ppl')
    assert not ppl[0]
    assert ppl[1:51].all()

    # Pulse mask.
    assert pulses.pulse_mask(labelled=False)[0, 1000:1306:6].all()

    # Obtain a pulse mask for the entire run, including trains without
    # any pulses and trains without FEL pulses.
    # Requires use of bunch_table_position to avoid extrapolating FEL
    # pulses at the beginning of the run.
    pulses2 = PumpProbePulses(
        run, source=source, bunch_table_position=1001
    )
    mask = pulses2.pulse_mask(labelled=False)
    fel_mask = pulses2.pulse_mask(labelled=False, field='fel')
    ppl_mask = pulses2.pulse_mask(labelled=False, field='ppl')

    assert not mask[:5].any()  # No pulses at all.
    assert not fel_mask[:5].any()
    assert not ppl_mask[:5].any()

    # No FEL pulses but PPL pulses.
    assert not mask[5:10, 1000:1300:6].any() and mask[5:10, 1001:1301:6].all()
    assert not fel_mask[5:10, 1000:1301].any()
    assert ppl_mask[5:10, 1001:1301:6].all()

    # FEL and PPL pulses.
    assert mask[10, 1000:1300:6].all() and mask[10, 1001:1301:6].all()
    assert fel_mask[10, 1000:1300:6].all() and not fel_mask[10, 1001:1301:6].any()
    assert not ppl_mask[10, 1000:1300:6].any() and ppl_mask[10, 1001:1301:6].all()

    # Is constant pattern?
    assert not pulses.is_constant_pattern()
    assert pulses.select_trains(np.s_[:1]).is_constant_pattern()

    # Search patterns.
    patterns = pulses.search_pulse_patterns()
    assert len(patterns) == 2
    assert patterns[0][0].value == by_id[10010:10049].value
    np.testing.assert_equal(patterns[0][1], np.r_[1000:1306:6])
    assert patterns[1][0].value == by_id[10050:10100].value
    assert patterns[1][1].iloc[0] == 1000
    np.testing.assert_equal(patterns[1][1][1:], np.r_[1012:1312:6])

    with pytest.raises(ValueError):
        # Should fail without any relation keyword.
        pulses = PumpProbePulses(run, source=source)

    try:
        pulses = PumpProbePulses(run, source=source, instrument='SQS',
                                 pulse_offset=0)
    except ValueError as e:
        # Will fail for ppdecoder, but not relevant here.
        msg = str(e)
        assert 'LP_SPB' in msg and 'LP_SQS' in msg
    else:
        assert pulses.sase == 3
        assert pulses.ppl_seed == PPL_BITS.LP_SQS

    with pytest.raises(ValueError):
        PumpProbePulses(run, source=source, instrument='123', pulse_offset=0)


def test_pump_probe_specials(mock_spb_aux_run, mock_sqs_remi_run):
    # Test full pulse IDs using default arguments.
    run = mock_spb_aux_run.select('SPB*').select_trains(np.s_[10:])
    np.testing.assert_equal(
        PumpProbePulses(run, pulse_offset=1).pulse_ids()[run.train_ids[0]],
        np.r_[1000:1306:6])

    # Test single pulse case with pulse_offset == 0 (working through
    # bunch_table_offset = 0 instead).
    run = mock_sqs_remi_run.select_trains(np.s_[10:])
    assert PumpProbePulses(run, pulse_offset=0).pulse_ids().any()

    # Test single pulse case with pulse_offset != 0 (fails).
    with pytest.raises(ValueError):
        PumpProbePulses(run, pulse_offset=1).pulse_ids()

    # Test a pattern constant in pulse IDs, but having different pump
    # probe flags (e.g. an alternating laser on/off pattern).
    pulses = PumpProbePulses(mock_sqs_remi_run[10:], pulse_offset=0)

    # Inject custom cached pulse ID, but make sure to use the same trains.
    train_ids = pulses._get_train_ids()
    pulse_indices = np.arange(20)
    pulses._pulse_ids = pd.Series(
        np.tile(1000 + pulse_indices * 8, len(train_ids)),
        pd.MultiIndex.from_tuples(
            [(train_id, pulse_idx, True, (train_id % 2) == 0)
            for train_id in train_ids for pulse_idx in pulse_indices],
            names=('trainId', 'pulseIndex', 'fel', 'ppl')))

    assert not pulses.is_constant_pattern()
