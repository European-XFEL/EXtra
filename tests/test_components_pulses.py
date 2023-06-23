
import pytest
import numpy as np

import extra

from euxfel_bunch_pattern import PPL_BITS
from extra.data import RunDirectory, SourceData, KeyData, by_id
from extra.components import XrayPulses, OpticalLaserPulses


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
    pulses = XrayPulses(RunDirectory(mock_spb_aux_run).select('SPB*'))
    pulses.master_clock
    pulses.bunch_clock_divider
    pulses.bunch_repetition_rate


def test_init(mock_spb_aux_run):
    # First only select "regular" sources.
    run = RunDirectory(mock_spb_aux_run).select('SPB*')

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
    run = RunDirectory(mock_spb_aux_run)

    with pytest.raises(ValueError):
        XrayPulses(run)

    # Pick one specifically.
    pulses = XrayPulses(run, timeserver='ODD_TIMESERVER_NAME')
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


def test_select_trains(mock_spb_aux_run):
    run = RunDirectory(mock_spb_aux_run).select('SPB*')
    pulses = XrayPulses(run)

    subrun = run.select_trains(np.s_[:20])
    subpulses = pulses.select_trains(np.s_[:20])

    assert_equal_sourcedata(
        subpulses.timeserver,
        subrun['SPB_RR_SYS/TSYS/TIMESERVER:outputBunchPattern'])
    assert_equal_keydata(
        subpulses.bunch_pattern_table,
        subpulses.timeserver['data.bunchPatternTable'])


def test_is_constant_pattern(mock_spb_aux_run):
    run = RunDirectory(mock_spb_aux_run).select('SPB*')
    pulses = XrayPulses(run)

    assert not pulses.is_constant_pattern()
    assert pulses.select_trains(np.s_[:50]).is_constant_pattern()


def test_get_pulse_counts(mock_spb_aux_run):
    run = RunDirectory(mock_spb_aux_run).select('SPB*')
    pulses = XrayPulses(run)

    # Test labelled.
    counts = pulses.get_pulse_counts(labelled=True)
    assert (counts.index == run.train_ids).all()
    assert (counts.iloc[:50] == 50).all()
    assert (counts.iloc[50:] == 25).all()

    # Test unlabelled.
    np.testing.assert_equal(pulses.get_pulse_counts(labelled=False), counts)

    # Check different SASE.
    counts = XrayPulses(run, sase=2).get_pulse_counts()
    assert (counts.index == run.train_ids).all()
    assert (counts == 63).all()


def test_peek_pulse_ids(mock_spb_aux_run):
    run = RunDirectory(mock_spb_aux_run).select('SPB*')

    np.testing.assert_equal(XrayPulses(run).peek_pulse_ids(),
                            np.r_[1000:1300:6])
    np.testing.assert_equal(XrayPulses(run, sase=2).peek_pulse_ids(),
                            np.r_[1500:2000:8])


def test_get_pulse_ids(mock_spb_aux_run):
    run = RunDirectory(mock_spb_aux_run).select('SPB*')
    pulses = XrayPulses(run)

    # Test labelled.
    pids = pulses.get_pulse_ids()
    assert len(pids) == 3750
    np.testing.assert_equal(pids[:, 0].index, np.array(run.train_ids))
    np.testing.assert_equal(pids[run.train_ids[0], :].index, np.r_[:50])
    np.testing.assert_equal(pids[run.train_ids[0], :], np.r_[1000:1300:6])
    np.testing.assert_equal(pids[run.train_ids[50], :], np.r_[1000:1300:12])

    # Test unlabelled.
    np.testing.assert_equal(pulses.get_pulse_ids(labelled=False), pids)


def test_search_pulse_patterns(mock_spb_aux_run):
    run = RunDirectory(mock_spb_aux_run).select('SPB*')
    pulses = XrayPulses(run)

    patterns = pulses.search_pulse_patterns()
    assert len(patterns) == 2
    assert patterns[0][0].value == by_id[10000:10049].value
    np.testing.assert_equal(patterns[0][1], np.r_[1000:1300:6])
    assert patterns[1][0].value == by_id[10050:10100].value
    np.testing.assert_equal(patterns[1][1], np.r_[1000:1300:12])


def test_trains(mock_spb_aux_run):
    run = RunDirectory(mock_spb_aux_run).select('SPB*')
    pulses = XrayPulses(run)

    for ref_tid, (tid, pids) in zip(run.train_ids, pulses.trains()):
        assert ref_tid == tid

        if tid < run.train_ids[50]:
            np.testing.assert_equal(pids, np.r_[1000:1300:6])
        else:
            np.testing.assert_equal(pids, np.r_[1000:1300:12])


def test_optical_laser(mock_spb_aux_run):
    run = RunDirectory(mock_spb_aux_run).select('SPB*')

    # Regular usage.
    pulses = OpticalLaserPulses(run)
    assert pulses.ppl_seed == PPL_BITS.LP_SPB
    assert (pulses.get_pulse_counts() == 50).all()
    assert (pulses.get_pulse_ids(labelled=False)[:50] == np.r_[0:300:6]).all()

    # Different laser seed by enum
    pulses = OpticalLaserPulses(run, ppl_seed=PPL_BITS.LP_SQS)
    assert pulses.ppl_seed == PPL_BITS.LP_SQS
    assert (pulses.get_pulse_counts() == 0).all()

    # Different laser seed by string.
    pulses = OpticalLaserPulses(run, ppl_seed='MID')
    assert pulses.ppl_seed == PPL_BITS.LP_SASE2
    assert (pulses.get_pulse_counts() == 0).all()


    # Full run with two timeservers
    run = RunDirectory(mock_spb_aux_run)

    with pytest.raises(ValueError):
        OpticalLaserPulses(run)

    OpticalLaserPulses(run, timeserver='ODD_TIMESERVER_NAME')

    # Only odd timeserver now
    run = run.select('ODD_TIMESERVER_NAME*')

    with pytest.raises(ValueError):
        OpticalLaserPulses(run)

    OpticalLaserPulses(run, ppl_seed=PPL_BITS.LP_SPB)