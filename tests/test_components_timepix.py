
from io import BytesIO

import pytest
import numpy as np
import pandas as pd

from extra.components import Timepix3, XrayPulses, PumpProbePulses
from .mockdata import assert_equal_sourcedata, assert_equal_keydata


def test_timepix3_init(mock_sqs_timepix_run):
    run = mock_sqs_timepix_run.deselect('SQS_EXTRA*')

    tpx = Timepix3(run)
    assert tpx.detector_name == 'SQS_EXP_TIMEPIX'
    assert_equal_sourcedata(
        tpx.raw_control_src,
        run['SQS_EXP_TIMEPIX/DET/TIMEPIX3'])
    assert_equal_sourcedata(
        tpx.raw_instrument_src,
        run['SQS_EXP_TIMEPIX/DET/TIMEPIX3:daqOutput.chip0'])
    assert_equal_keydata(
        tpx.raw_x_key,
        run['SQS_EXP_TIMEPIX/DET/TIMEPIX3:daqOutput.chip0', 'data.x'])
    assert_equal_sourcedata(
        tpx.centroids_control_src,
        run['SQS_EXP_TIMEPIX/CAL/TIMEPIX3'])
    assert_equal_sourcedata(
        tpx.centroids_instrument_src,
        run['SQS_EXP_TIMEPIX/CAL/TIMEPIX3:daqOutput.chip0'])
    assert 'SQS_EXP_TIMEPIX' in repr(tpx)
    assert 'raw, centroids' in repr(tpx)

    # Whole run contains more than one detector.
    with pytest.raises(ValueError):
        tpx = Timepix3(mock_sqs_timepix_run)

    # But works with explicit naming
    tpx = Timepix3(mock_sqs_timepix_run, 'SQS_EXP_TIMEPIX')
    assert_equal_sourcedata(
        tpx.raw_control_src,
        mock_sqs_timepix_run['SQS_EXP_TIMEPIX/DET/TIMEPIX3'])

    # Also for the other detector with caveats.
    tpx = Timepix3(mock_sqs_timepix_run, 'SQS_EXTRA_TIMEPIX')
    assert_equal_sourcedata(
        tpx.raw_control_src,
        mock_sqs_timepix_run['SQS_EXTRA_TIMEPIX/DET/TIMEPIX3'])
    assert_equal_sourcedata(
        tpx.raw_instrument_src,
        mock_sqs_timepix_run['SQS_EXTRA_TIMEPIX/DET/TIMEPIX3:daqOutput.chip0'])

    with pytest.raises(ValueError):
        tpx.centroids_control_src

    with pytest.raises(ValueError):
        tpx.centroids_instrument_src


@pytest.mark.parametrize('method', ['pixel_events', 'centroid_events'])
def test_timepix3_data(mock_sqs_timepix_run, method):
    run = mock_sqs_timepix_run.deselect('SQS_EXTRA*')

    # Default initialization will use SA3 with a single pulse.
    tpx = Timepix3(run)
    data = getattr(tpx, method)()

    np.testing.assert_array_equal(data.columns, ['x', 'y', 't', 'tot'])
    assert data.index.names == ['trainId', 'pulseId']

    np.testing.assert_equal(
        data.index.get_level_values('trainId').value_counts().sort_index(),
        np.arange(10, 100))
    assert (data.index.get_level_values('pulseId') == 200).all()

    for _, d in data.groupby(['trainId', 'pulseId']):
        N = len(d)
        np.testing.assert_equal(
            d['x'], np.linspace(0, 256, N, dtype=data.dtypes['x']))
        np.testing.assert_equal(
            d['y'], np.linspace(256, 0, N, dtype=data.dtypes['y']))
        np.testing.assert_allclose(d['t'], np.linspace(0.0, 5.0, N))
        np.testing.assert_equal(
            d['tot'], np.linspace(200, 800, N, dtype=data.dtypes['tot']))

    # Try without parallelization once.
    pd.testing.assert_frame_equal(data, getattr(tpx, method)(parallel=False))

    # Try different pulse dimension.
    assert getattr(tpx, method)(
        pulse_dim='pulseIndex').index.names[1] == 'pulseIndex'

    # Apply time-of-arrival offset.
    data = getattr(tpx, method)(toa_offset=-1.0)
    for _, d in data.groupby(['trainId', 'pulseId']):
        np.testing.assert_allclose(d['t'], np.linspace(1.0, 6.0, len(d)))

    # Initialize for SA1 for hits across multiple pulses and with
    # extended columns.
    tpx = Timepix3(run, pulses=XrayPulses(run, sase=1))
    data = getattr(tpx, method)(extended_columns=True)

    np.testing.assert_equal(
        data.index.get_level_values('trainId').value_counts().sort_index(),
        np.arange(10, 100))
    np.testing.assert_equal(
        data.index.get_level_values('pulseId').value_counts().sort_index(),
        np.array([2302, 303, 2048, 252]))

    for _, pulse_data in data.groupby(['trainId', 'pulseId']):
        # Ensure the difference ToA and ToF is always constant.
        diff = pulse_data['t'] - pulse_data['toa']
        assert (diff == diff.min()).all()

    # Obtain SA1 data with leading/trailing virtual pulses, offset ToA
    # to get a leading pulse.
    data = getattr(tpx, method)(toa_offset=0.5, out_of_pulse_events=True)

    np.testing.assert_equal(
        data.index.get_level_values('trainId').value_counts().sort_index(),
        np.arange(1, 100))  # Now includes empty trains!

    pulse_dist = data.index.get_level_values('pulseId') \
        .value_counts().sort_index()
    np.testing.assert_allclose(
        pulse_dist.index, [-np.inf, 1000.0, 1006.0, 1012.0, 1018.0, np.inf])
    np.testing.assert_equal(
        pulse_dist, np.array([531, 2262, 304, 1681, 136, 36]))

    # Try with PumpProbePulses.
    tpx = Timepix3(run, pulses=PumpProbePulses(
        run, instrument=(1, 1024), bunch_table_position=1000))
    data = getattr(tpx, method)(out_of_pulse_events=True)
    assert data.index.names == ['trainId', 'pulseId', 'fel', 'ppl']


def test_timepix3_pixel_events(mock_sqs_timepix_run):
    tpx = Timepix3(mock_sqs_timepix_run.deselect('SQS_EXTRA*'))

    # Extended columns.
    events = tpx.pixel_events(extended_columns=True)

    np.testing.assert_array_equal(
        events.columns, ['x', 'y', 't', 'tot', 'toa', 'pos', 'label'])
    np.testing.assert_equal(events.loc[10010]['label'], np.arange(1, 11))
    for _, d in events.groupby(['trainId', 'pulseId']):
        N = len(d)
        np.testing.assert_allclose(d['toa'], np.linspace(0.0, 5.0, N))
        np.testing.assert_array_equal(d['pos'], np.arange(N))
        assert d['label'].iloc[0] == 1 and d['label'].iloc[-1] == 10

    orig_toa = events['toa'].copy()  # For test below.

    # Timewalk LUT, apply negative-only to not shift hits before the
    # first pulse for easier comparison.
    walk = -2e-9 * np.arange(1000)

    np.save(lut_buf := BytesIO(), walk)
    events = tpx.pixel_events(timewalk_lut=BytesIO(lut_buf.getbuffer()),
                              extended_columns=True)
    np.testing.assert_allclose(orig_toa.to_numpy() - events['toa'].to_numpy(),
                               walk[(events['tot'] // 25) - 1] * 1e6)


def test_timepix3_centroid_events(mock_sqs_timepix_run):
    tpx = Timepix3(mock_sqs_timepix_run.deselect('SQS_EXTRA*'))

    # Extended columns.
    centroids = tpx.centroid_events(extended_columns=True)

    np.testing.assert_array_equal(
        centroids.columns, ['x', 'y', 't', 'tot', 'tot_avg', 'tot_max', 'toa',
                            'centroid_size', 'label'])

    for _, d in centroids.groupby(['trainId', 'pulseId']):
        N = len(d)
        np.testing.assert_allclose(d['tot_avg'], np.linspace(100, 400, N))
        np.testing.assert_array_equal(
            d['tot_max'], np.linspace(150, 600, N, dtype=np.uint16))
        np.testing.assert_array_equal(
            d['centroid_size'], np.linspace(1, 10, N, dtype=np.int16))
        np.testing.assert_array_equal(d['label'], np.arange(N))
