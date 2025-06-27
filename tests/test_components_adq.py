
from itertools import product

import pytest
import numpy as np
import xarray as xr
import pandas as pd

from extra.components import AdqRawChannel, XrayPulses

from .mockdata import assert_equal_sourcedata, assert_equal_keydata


@pytest.mark.parametrize('channel', ['1_C', '1C'])
def test_adq_init(mock_sqs_remi_run, channel):
    ctrl_sd = mock_sqs_remi_run['SQS_DIGITIZER_UTC2/ADC/1']
    instr_sd = mock_sqs_remi_run['SQS_DIGITIZER_UTC2/ADC/1:network']

    # Run contains UTC1 and UTC2, first test regular operation with
    # specific digitizer.
    ch = AdqRawChannel(
        mock_sqs_remi_run, channel, digitizer='SQS_DIGITIZER_UTC2')
    assert_equal_sourcedata(ch.control_source, ctrl_sd)
    assert_equal_sourcedata(ch.instrument_source, instr_sd)
    assert_equal_keydata(ch.raw_samples_key,
                         instr_sd['digitizers.channel_1_C.raw.samples'])
    assert_equal_keydata(ch.channel_key('raw.samples'),
                         instr_sd['digitizers.channel_1_C.raw.samples'])
    assert ch._pulses.sase == 3

    # Full auto-detection should fail on the entire run with two
    # digitizers.
    with pytest.raises(ValueError):
        ch = AdqRawChannel(mock_sqs_remi_run, channel)

    # Full auto-detection should work after selection.
    subrun = mock_sqs_remi_run.deselect('SQS_DIGITIZER_UTC1*')
    ch = AdqRawChannel(subrun, channel)
    assert_equal_sourcedata(
        ch.instrument_source, subrun['SQS_DIGITIZER_UTC2/ADC/1:network'])

    # Can also pass explicit (instrument!) source name.
    ch = AdqRawChannel(mock_sqs_remi_run, channel,
                       digitizer='SQS_DIGITIZER_UTC2/ADC/1:network')
    assert_equal_sourcedata(
        ch.instrument_source, subrun['SQS_DIGITIZER_UTC2/ADC/1:network'])

    # Remove control data.
    ctrlless_run = subrun.deselect('SQS_DIGITIZER_UTC2/ADC/1')

    # Will fail creation without further keywords.
    with pytest.raises(ValueError):
        AdqRawChannel(ctrlless_run, channel)

    # But works with passing interleaving flag explicitly.
    ch = AdqRawChannel(ctrlless_run, channel, interleaved=False)
    assert_equal_sourcedata(
        ch.instrument_source, ctrlless_run['SQS_DIGITIZER_UTC2/ADC/1:network'])

    # Remove timeserver information
    timeless_run = subrun.deselect('SQS_RR_UTC/*')

    # Simply creating a channel should fail now.
    with pytest.raises(ValueError):
        AdqRawChannel(timeless_run, channel)

    # But it still works with pulse information disabled.
    ch = AdqRawChannel(timeless_run, channel, pulses=False)
    assert_equal_sourcedata(
        ch.instrument_source, timeless_run['SQS_DIGITIZER_UTC2/ADC/1:network'])


def test_adq_properties(mock_sqs_remi_run):
    ch = AdqRawChannel(mock_sqs_remi_run, '1C', digitizer='SQS_DIGITIZER_UTC1')
    assert ch.board == 1
    assert ch.letter == 'C'
    assert ch.number == 3
    assert ch.name == '1_C'

    assert not ch.interleaved
    assert ch.clock_ratio == 440
    assert np.isclose(ch.sampling_rate, 2.0e9, rtol=1e-2)
    assert np.isclose(ch.sampling_period, 0.5e-9)
    assert ch.trace_shape == 50000
    assert np.isclose(ch.trace_duration, 25e-6, rtol=1e-2)

    assert ch.board_parameters == {
        'enable': True, 'enable_raw': True, 'interleavedMode': False}
    assert ch.channel_parameters == {
        'enable': True, 'offset': 0, 'enable_raw': True,
        'interleavedMode': False}


    # Overwrite interleaving.
    ch = AdqRawChannel(mock_sqs_remi_run, '1C', digitizer='SQS_DIGITIZER_UTC1',
                       interleaved=True)
    assert ch.clock_ratio == 880
    assert np.isclose(ch.sampling_rate, 4.0e9, rtol=1e-2)
    assert np.isclose(ch.sampling_period, 0.25e-9)
    assert np.isclose(ch.trace_duration, 12.5e-6, rtol=1e-2)

    # One of the special 3G boards.
    ch = AdqRawChannel(mock_sqs_remi_run, '1C', digitizer='SQS_DIGITIZER_UTC2')
    assert ch.clock_ratio == 392
    assert np.isclose(ch.sampling_rate, 1.76e9, rtol=1e-2)
    assert np.isclose(ch.sampling_period, 0.57e-9)
    assert np.isclose(ch.trace_duration, 28e-6, rtol=1e-2)


def test_adq_samples_per_pulse(mock_sqs_remi_run):
    # Skip early trains with no or too many pulses.
    run = mock_sqs_remi_run.select_trains(np.s_[50:])

    # Use SAS for pulse information, since SA3 has only a single one.
    ch = AdqRawChannel(run, '1C', digitizer='SQS_DIGITIZER_UTC1',
                       pulses=XrayPulses(run, sase=1))

    assert ch.samples_per_pulse() == 5280
    assert ch.samples_per_pulse(pulse_period=14) == 6160
    assert ch.samples_per_pulse(pulse_duration=3.1e-6) == 6160
    assert ch.samples_per_pulse(repetition_rate=320e3) == 6160
    assert ch.samples_per_pulse(pulse_ids=np.array([1000, 1014, 1028])) == 6160

    # These can give different results with fractional enabled.
    assert np.isclose(
        ch.samples_per_pulse(pulse_duration=3.1e-6, fractional=True), 6156.94)
    assert np.isclose(
        ch.samples_per_pulse(repetition_rate=320e3, fractional=True), 6206.6)


@pytest.mark.parametrize('in_dtype', [np.int16, np.float32, np.float64])
def test_adq_correct_common_mode(mock_sqs_remi_run, in_dtype):
    ch = AdqRawChannel(mock_sqs_remi_run, '1C', digitizer='SQS_DIGITIZER_UTC1')
    expected_dtype = in_dtype if np.issubdtype(in_dtype, np.floating) \
        else np.float32

    # Construct a trace with extreme common mode and tile it.
    traces = np.zeros((2, 3, 500), dtype=in_dtype)

    for offset in range(5):
        traces[:, :, offset::5] = offset

    # Default baselevel (i.e. 0).
    out = ch.correct_common_mode(traces, 5, np.s_[:50])
    np.testing.assert_allclose(out, 0.0)
    assert out.shape == traces.shape
    assert out.dtype == expected_dtype

    # Custom baselevel.
    out = ch.correct_common_mode(traces, 5, np.s_[:50], 17.14)
    np.testing.assert_allclose(out, 17.14)
    assert out.shape == traces.shape
    assert out.dtype == expected_dtype

    # Also test the raveled array.
    out = ch.correct_common_mode(traces.ravel(), 5, np.s_[:50])
    np.testing.assert_allclose(out, 0.0)
    assert out.shape == (traces.size,)
    assert out.dtype == expected_dtype


@pytest.mark.parametrize('in_dtype', [np.int16, np.float32, np.float64])
def test_pull_baseline(mock_sqs_remi_run, in_dtype):
    ch = AdqRawChannel(mock_sqs_remi_run, '1C', digitizer='SQS_DIGITIZER_UTC1')
    expected_dtype = in_dtype if np.issubdtype(in_dtype, np.floating) \
        else np.float32

    # Test a single trace first.
    single_trace = np.arange(100, dtype=in_dtype)
    out = ch.pull_baseline(single_trace, np.s_[:50], 0)
    np.testing.assert_allclose(out, single_trace - 24.5)

    # Now tile it multiple times.
    traces = np.tile(single_trace, [2, 3, 1])
    out = ch.pull_baseline(traces, np.s_[:50], 0)

    for i, j in product(range(2), range(3)):
        np.testing.assert_allclose(out[i, j], single_trace - 24.5)


def test_train_data(mock_sqs_remi_run):
    ch = AdqRawChannel(mock_sqs_remi_run, '3B', digitizer='SQS_DIGITIZER_UTC2')
    assert ch.raw_samples_key[0].ndarray()[0, 0] == -100

    xr_data = ch.train_data(roi=(np.s_[:30000],))
    assert isinstance(xr_data, xr.DataArray)
    assert xr_data.dims == ('trainId', 'sample')

    # Test unlabelled and non-tuple roi.
    np_data = ch.train_data(labelled=False, roi=np.s_[:30000])
    assert isinstance(np_data, np.ndarray)

    for data in [xr_data.data, np_data]:
        assert data.shape == (100, 30000)
        assert data.dtype == np.float32

        assert np.isclose(data[0, 0], 0.0)
        assert np.isclose(data.min(), 0.0)
        assert np.isclose(data.max(), 5.0)
        np.testing.assert_array_equal(data.argmax(axis=1), 8974)


def test_pulse_data(mock_sqs_remi_run):
    pulses = XrayPulses(mock_sqs_remi_run, sase=1)
    ch = AdqRawChannel(mock_sqs_remi_run, '3B', digitizer='SQS_DIGITIZER_UTC2',
                       pulses=pulses, first_pulse_offset=2000,
                       baseline=np.s_[-10000:])

    xr_data = ch.pulse_data(train_roi=(np.s_[:-100],))
    assert isinstance(xr_data, xr.DataArray)
    assert xr_data.dims == ('pulse', 'sample')

    np_data = ch.pulse_data(labelled=False, train_roi=np.s_[:-100])
    assert isinstance(np_data, np.ndarray)

    for data in [xr_data.data, np_data]:
        assert data.shape == (pulses.pulse_counts().sum(),
                              ch.samples_per_pulse())
        assert data.dtype == np.float32

        max_by_pulse = data.max(axis=1)

        # Test whether the peak was put in the right pulse.
        assert np.isclose(
            max_by_pulse[:20],
            [0.0, 0.0, 5.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).all()

        # Test whether all pulses after the trace are nan.
        assert np.isnan(max_by_pulse[21:50]).all()

    # Use pulse information with not all trains available.
    ch = AdqRawChannel(
        mock_sqs_remi_run, '3B', digitizer='SQS_DIGITIZER_UTC2',
        pulses=XrayPulses(mock_sqs_remi_run.select_trains(np.s_[:-10])))

    with pytest.raises(ValueError):
        ch.pulse_data()

    # Use pulse information with more trains available.
    ch = AdqRawChannel(
        mock_sqs_remi_run.select_trains(np.s_[:-10]), '3B',
        digitizer='SQS_DIGITIZER_UTC2', pulses=pulses)
    data = ch.pulse_data()

    assert data.shape[0] < pulses.pulse_counts().sum()


def test_train_edge_array(mock_sqs_remi_run):
    ch = AdqRawChannel(mock_sqs_remi_run, '3B', digitizer='SQS_DIGITIZER_UTC2')

    ds = ch.train_edge_array(threshold=1.0)
    assert isinstance(ds, xr.Dataset)
    assert ds.edges.shape == (100, 1)
    assert ds.amplitudes.shape == (100, 1)
    assert list(ds.coords.keys()) == ['trainId', 'edge']

    np_edges, np_amplitudes = ch.train_edge_array(
        threshold=1.0, labelled=False)
    assert isinstance(np_edges, np.ndarray)
    assert isinstance(np_amplitudes, np.ndarray)
    assert np_edges.shape == (100, 1)
    assert np_amplitudes.shape == (100, 1)

    for edges, amplitudes in [
        (ds.edges.data, ds.amplitudes.data),
        (np_edges, np_amplitudes)
    ]:
        finite_mask = np.isfinite(edges)
        np.testing.assert_allclose(edges[finite_mask], 8944.0)
        np.testing.assert_allclose(amplitudes[finite_mask], 5.0)


def test_train_edges(mock_sqs_remi_run):
    ch = AdqRawChannel(mock_sqs_remi_run, '3B', digitizer='SQS_DIGITIZER_UTC2')
    df = ch.train_edges(threshold=1.0)

    assert isinstance(df, pd.DataFrame)
    np.testing.assert_array_equal(df.columns, ['edge', 'amplitude'])
    np.testing.assert_array_equal(df.index.names, ['trainId', 'edgeIndex'])
    assert len(df) == 100

    np.testing.assert_allclose(df['edge'], 8944.0)
    np.testing.assert_allclose(df['amplitude'], 5.0)


def test_pulse_edge_array(mock_sqs_remi_run):
    pulses = XrayPulses(mock_sqs_remi_run, sase=1)
    ch = AdqRawChannel(mock_sqs_remi_run, '3B', digitizer='SQS_DIGITIZER_UTC2',
                       pulses=pulses, first_pulse_offset=1000,
                       baseline=np.s_[-10000:])
    ds = ch.pulse_edge_array(threshold=0.5)

    assert isinstance(ds, xr.Dataset)
    assert ds.edges.shape == (pulses.pulse_counts().sum(), 1)
    assert ds.amplitudes.shape == (pulses.pulse_counts().sum(), 1)
    assert list(ds.coords.keys()) == ['pulse', 'trainId', 'pulseId', 'edge']

    finite_mask = np.isfinite(ds.edges.data)
    np.testing.assert_allclose(ds.edges.data[finite_mask], 888.0)
    np.testing.assert_allclose(ds.amplitudes.data[finite_mask], 5.0)


def test_pulse_edges(mock_sqs_remi_run):
    pulses = XrayPulses(mock_sqs_remi_run, sase=1)
    ch = AdqRawChannel(mock_sqs_remi_run, '3B', digitizer='SQS_DIGITIZER_UTC2',
                       pulses=pulses, first_pulse_offset=1000,
                       baseline=np.s_[-10000:])
    df = ch.pulse_edges(threshold=0.5)

    assert isinstance(df, pd.DataFrame)
    np.testing.assert_array_equal(df.columns, ['edge', 'amplitude'])
    np.testing.assert_array_equal(
        df.index.names, ['trainId', 'pulseId', 'edgeIndex'])
    assert len(df) == 40

    np.testing.assert_array_equal(
        df.index.get_level_values('trainId'), np.r_[10010:10050])
    np.testing.assert_array_equal(df.index.get_level_values('pulseId'), 1018)
    np.testing.assert_array_equal(
        df.groupby(['trainId', 'pulseId']).count(), 1)
    np.testing.assert_allclose(df['edge'], 888.0)
    np.testing.assert_allclose(df['amplitude'], 5.0)
