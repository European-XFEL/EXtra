
import numpy as np
import pandas as pd
import pytest

from extra.components import DelayLineDetector, XrayPulses

from .mockdata import assert_equal_sourcedata


def test_dld_init(mock_sqs_remi_run):
    run = mock_sqs_remi_run

    # Auto-detect with a single detector.
    dld = DelayLineDetector(run.select('*/DET/TOP*'))
    assert dld.detector_name == 'SQS_REMI_DLD6/DET/TOP'
    assert_equal_sourcedata(
        dld.control_source, run['SQS_REMI_DLD6/DET/TOP'])
    assert_equal_sourcedata(
        dld.instrument_source, run['SQS_REMI_DLD6/DET/TOP:output'])

    # Auto-detect with two detectors.
    with pytest.raises(ValueError):
        dld = DelayLineDetector(run)

    # Explicit with two detectors.
    dld = DelayLineDetector(run, 'SQS_REMI_DLD6/DET/BOTTOM')
    assert dld.detector_name == 'SQS_REMI_DLD6/DET/BOTTOM'
    assert_equal_sourcedata(
        dld.control_source, run['SQS_REMI_DLD6/DET/BOTTOM'])
    assert_equal_sourcedata(
        dld.instrument_source, run['SQS_REMI_DLD6/DET/BOTTOM:output'])

    # Explicit pulse object.
    pulses = XrayPulses(run)
    dld = DelayLineDetector(run, 'SQS_REMI_DLD6/DET/TOP', pulses)

    # Reconstruction parameters.
    assert 'digitizer.baseline_region' in dld.rec_params


@pytest.mark.parametrize('pulse_dim', ['pulseId', 'pulseIndex', 'time'])
@pytest.mark.parametrize('channel_index', [True, False],
                         ids=['channelIndex', 'channelColumn'])
def test_dld_edges(mock_sqs_remi_run, pulse_dim, channel_index):
    dld = DelayLineDetector(mock_sqs_remi_run, 'SQS_REMI_DLD6/DET/TOP')
    edges = dld.edges(channel_index=channel_index, pulse_dim=pulse_dim)

    # There should be 28 edges per pulse.
    assert np.all(edges.groupby(['trainId', pulse_dim]).count() == 28)

    # Check channels for the first pulse, should be in descending order
    # and decreasing number of edges.
    if channel_index:
        actual_channels = edges.index.get_level_values('channel')
    else:
        actual_channels = edges['channel']

    expected_channels = np.repeat(np.arange(7), np.arange(7)+1)[::-1]
    np.testing.assert_equal(actual_channels[:28], expected_channels)


@pytest.mark.parametrize('pulse_dim', ['pulseId', 'pulseIndex', 'time'])
@pytest.mark.parametrize('key', ['signal', 'hit'])
def test_dld_df(mock_sqs_remi_run, key, pulse_dim):
    from .mockdata import dld as dld_mockdata
    dtype = getattr(dld_mockdata, f'{key}_dt')

    dld = DelayLineDetector(mock_sqs_remi_run, 'SQS_REMI_DLD6/DET/TOP')
    df = getattr(dld, f'{key}s')(pulse_dim)

    assert (df.columns == list(dtype.names)).all()
    assert df.index.names == [
        'trainId', pulse_dim, 'fel', 'ppl', f'{key}Index']

    # Check counts per pulse, should be a repeating pattern of 1-4.
    np.testing.assert_equal(
        df.groupby(['trainId', pulse_dim]).count()[dtype.names[0]],
        np.tile([1, 2, 3, 4], 990))


def test_dld_pulse_align(mock_sqs_remi_run):
    run = mock_sqs_remi_run.select('*/DET/TOP*')
    dld = DelayLineDetector(run)
    pulses = dld.pulses()
    all_hits = dld.hits()

    # Only trains with no pulses.
    dld = DelayLineDetector(run.select_trains(np.s_[:1]))
    assert dld.hits().empty
    assert dld.edges().empty

    # Less trains for pulse information.
    dld = DelayLineDetector(run, pulses=pulses.select_trains(np.s_[:50]))
    with pytest.raises(ValueError):
        dld.hits()

    # Less trains for detector data.
    dld = DelayLineDetector(run.select_trains(np.s_[1:50]), pulses=pulses)
    hits = dld.hits()
    pd.testing.assert_frame_equal(hits, all_hits.loc[np.r_[10002:10050], :])


def test_dld_extra_columns(mock_sqs_remi_run):
    dld = DelayLineDetector(mock_sqs_remi_run.select('*TOP*'))
    index = dld.pulses().build_pulse_index()

    kd = mock_sqs_remi_run['SQS_RR_UTC/TSYS/PP_DECODER', 'laser.nPulses']

    # Get hits with additional extra column of different kinds.
    hits = dld.hits(extra_columns={
        'per_pulse': pd.Series(np.arange(len(index)), index=index),
        'per_train': pd.Series(kd.ndarray(), index=pd.Index(
            kd.train_id_coordinates(), name='trainId')),
        'xarray': kd.xarray(),
        'keydata': kd})

    assert np.all(
        hits['per_pulse'][:12] == [1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 6, 7])
    pd.testing.assert_series_equal(hits['per_train'], hits['xarray'],
                                   check_names=False)
    pd.testing.assert_series_equal(hits['per_train'], hits['keydata'],
                                   check_names=False)

    # Try with wrong index.
    index = dld.pulses().build_pulse_index(pulse_dim='pulseIndex')
    extra = pd.Series(np.arange(len(index)), index=index)

    with pytest.raises(ValueError):
        dld.hits(extra_columns={'foo': extra})


def test_dld_max_method(mock_sqs_remi_run):
    dld = DelayLineDetector(mock_sqs_remi_run.select('*TOP*'))
    all_hits = dld.hits(max_method=None)
    all_signals = dld.signals(max_method=None)

    for m in range(20):
        mask = all_hits['m'] <= m
        pd.testing.assert_frame_equal(
            all_hits[mask], dld.hits(max_method=m))
        pd.testing.assert_frame_equal(
            all_signals[mask], dld.signals(max_method=m))
