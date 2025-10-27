import os
import re
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from IPython.core.formatters import MarkdownFormatter

import extra.calibration
from extra.calibration import (
    CalCatAPIClient,
    AGIPDConditions,
    CalibrationData,
    DSSCConditions,
    JUNGFRAUConditions,
    LPDConditions,
    SingleConstant,
)

# These tests all use saved HTTP responses by default (with pytest-recording).
# To ignore these & use exflcalproxy, run pytest with the --disable-recording flag.
# To update the saved cassettes from exflcalproxy, pass --record-mode=rewrite.

TEST_DIR = Path(__file__).parent


def drop_cookie_header(response):
    response['headers'].pop('Set-Cookie', None)
    return response


@pytest.fixture(scope="module")
def vcr_config():
    # These shouldn't be necessary, but are here as an extra layer to prevent
    # recording potential credentials.
    return {
        "filter_headers": ["authorization"],
        "before_record_response": drop_cookie_header,
    }

@pytest.fixture(autouse=True)
def reset_calibration_client():
    """Reset the global client object after each test, to avoid leaking state"""
    yield
    extra.calibration.global_client = None


@pytest.mark.vcr
def test_AGIPD_CalibrationData_metadata():
    """Test CalibrationData with AGIPD condition"""
    cond = AGIPDConditions(
        # From: https://in.xfel.eu/calibration/calibration_constants/5754#condition
        sensor_bias_voltage=300,  # V
        memory_cells=352,
        acquisition_rate=2.2,  # MHz
        gain_mode=0,
        gain_setting=1,
        integration_time=12,
        source_energy=9.2,
    )
    agipd_cd = CalibrationData.from_condition(
        cond,
        "MID_DET_AGIPD1M-1",
        event_at="2022-09-01 13:26:48.00",
        calibrations=["Offset", "SlopesFF"],
    )
    assert agipd_cd.detector_name == "MID_DET_AGIPD1M-1"
    assert "Offset" in agipd_cd
    assert set(agipd_cd["Offset"].constants) == {f"AGIPD{m:02}" for m in range(16)}
    assert isinstance(agipd_cd["Offset", "AGIPD00"], SingleConstant)
    assert agipd_cd["Offset", "Q1M2"] == agipd_cd["Offset", "AGIPD01"]

    bva = agipd_cd["Offset", "AGIPD00"].metadata("begin_validity_at")
    assert bva == "2022-09-02T07:42:33.000+02:00"
    metadata = agipd_cd["Offset", "AGIPD00"].metadata_dict()
    assert metadata["begin_validity_at"] == bva

    assert re.search(
        r"\[2022-\d{2}-\d{2} \d{2}:\d{2}\]\(https://in.xfel.eu/", agipd_cd.markdown_table()
    )

@pytest.mark.vcr
def test_AGIPD_merge():
    cond = AGIPDConditions(
        # From: https://in.xfel.eu/calibration/calibration_constants/5754#condition
        sensor_bias_voltage=300,  # V
        memory_cells=352,
        acquisition_rate=2.2,  # MHz
        gain_mode=0,
        gain_setting=1,
        integration_time=12,
        source_energy=9.2,
    )
    agipd_cd = CalibrationData.from_condition(
        cond,
        "MID_DET_AGIPD1M-1",
        event_at="2022-09-01 13:26:48.00",
        calibrations=["Offset", "SlopesFF"],
    )

    modnos_q1 = list(range(0, 4))
    modnos_q4 = list(range(12, 16))
    merged = agipd_cd.select_modules(modnos_q1).merge(
        agipd_cd.select_modules(modnos_q4)
    )
    assert merged.module_nums == modnos_q1 + modnos_q4

    offset_only = agipd_cd.select_calibrations(["Offset"])
    slopes_only = agipd_cd.select_calibrations(["SlopesFF"])
    assert set(offset_only) == {"Offset"}
    assert set(slopes_only) == {"SlopesFF"}
    merged_cals = offset_only.merge(slopes_only)
    assert set(merged_cals) == {"Offset", "SlopesFF"}
    assert merged_cals.module_nums == list(range(16))


@pytest.mark.vcr
def test_AGIPD_CalibrationData_metadata_SPB():
    """Test CalibrationData with AGIPD condition"""
    cond = AGIPDConditions(
        sensor_bias_voltage=300,
        memory_cells=352,
        acquisition_rate=1.1,
        integration_time=12,
        source_energy=9.2,
        gain_mode=0,
        gain_setting=0,
    )
    agipd_cd = CalibrationData.from_condition(
        cond,
        "SPB_DET_AGIPD1M-1",
        event_at="2020-01-07 13:26:48.00",
    )
    assert "Offset" in agipd_cd
    assert set(agipd_cd["Offset"].constants) == {f"AGIPD{m:02}" for m in range(16)}
    assert agipd_cd["Offset"].module_nums == list(range(16))
    assert agipd_cd["Offset"].qm_names == [
        f"Q{(m // 4) + 1}M{(m % 4) + 1}" for m in range(16)
    ]
    assert isinstance(agipd_cd["Offset", 0], SingleConstant)


@pytest.mark.skipif(not os.path.isdir("/gpfs/exfel/d"), reason="GPFS not available")
@pytest.mark.vcr
def test_AGIPD_load_data():
    cond = AGIPDConditions(
        sensor_bias_voltage=300,
        memory_cells=352,
        acquisition_rate=1.1,
        integration_time=12,
        source_energy=9.2,
        gain_mode=0,
        gain_setting=0,
    )
    agipd_cd = CalibrationData.from_condition(
        cond,
        "SPB_DET_AGIPD1M-1",
        event_at="2020-01-07 13:26:48.00",
    )
    arr = agipd_cd["Offset"].select_modules(list(range(4))).xarray()
    assert arr.shape == (4, 128, 512, 352, 3)
    assert arr.dims[0] == "module"
    np.testing.assert_array_equal(arr.coords["module"], np.arange(0, 4))
    assert arr.dtype == np.float64

    # Load parallel
    arr_p = agipd_cd["Offset"].select_modules(list(range(4))).xarray(parallel=4)
    xr.testing.assert_identical(arr_p, arr)


@pytest.mark.vcr
def test_DSSC_modules_missing():
    dssc_cd = CalibrationData.from_condition(
        DSSCConditions(sensor_bias_voltage=100, memory_cells=600),
        "SQS_DET_DSSC1M-1",
        event_at="2023-11-29 00:00:00",
    )
    # DSSC was used with only 3 quadrants at this point
    modnos = list(range(4)) + list(range(8, 16))
    assert dssc_cd.aggregator_names == [f"DSSC{m:02}" for m in modnos]
    assert dssc_cd.module_nums == modnos
    assert dssc_cd.qm_names == [f"Q{(m // 4) + 1}M{(m % 4) + 1}" for m in modnos]

    offset = dssc_cd["Offset"]
    assert offset.module_nums == modnos

    # test ModulesConstantVersions.select_modules()
    modnos_q3 = list(range(8, 12))
    aggs_q3 = [f"DSSC{m:02}" for m in modnos_q3]
    qm_q3 = [f"Q3M{i}" for i in range(1, 5)]
    assert offset.select_modules(modnos_q3).module_nums == modnos_q3
    assert offset.select_modules(aggregator_names=aggs_q3).module_nums == modnos_q3
    assert offset.select_modules(qm_names=qm_q3).module_nums == modnos_q3

    # test CalibrationData.select_modules()
    assert dssc_cd.select_modules(modnos_q3).module_nums == modnos_q3
    assert dssc_cd.select_modules(aggregator_names=aggs_q3).module_nums == modnos_q3
    assert dssc_cd.select_modules(qm_names=qm_q3).module_nums == modnos_q3


@pytest.mark.vcr
def test_LPD_constant_missing():
    lpd_cd = CalibrationData.from_condition(
        LPDConditions(memory_cells=200, sensor_bias_voltage=250),
        "FXE_DET_LPD1M-1",
        event_at="2022-05-22T02:00:00",
    )
    # Constants are missing for 1 module (LPD05), but it was still included in
    # the PDUs for the detector, so it should still appear in the lists.
    assert lpd_cd.aggregator_names == [f"LPD{m:02}" for m in range(16)]
    assert lpd_cd.module_nums == list(range(16))
    assert lpd_cd.qm_names == [f"Q{(m // 4) + 1}M{(m % 4) + 1}" for m in range(16)]

    # When we look at a specific constant, module LPD05 is missing
    modnos_w_constant = list(range(0, 5)) + list(range(6, 16))
    assert lpd_cd["Offset"].module_nums == modnos_w_constant

    # Test CalibrationData.require_constant()
    assert lpd_cd.require_calibrations(["Offset"]).module_nums == modnos_w_constant


@pytest.mark.vcr
def test_JUNGFRAU_constant():
    cond = JUNGFRAUConditions(
        sensor_bias_voltage=90.,
        memory_cells=1,
        integration_time=400.,
        gain_setting=0,
        sensor_temperature=291.,
    )
    jf_cd = CalibrationData.from_condition(
        cond,
        "FXE_XAD_JF1M",
        event_at="2024-03-04 17:56:05.172132+00:00",
    )
    assert jf_cd.aggregator_names == ["JNGFR01", "JNGFR02"]
    assert set(jf_cd) >= {"Offset10Hz", "BadPixelsDark10Hz", "RelativeGain10Hz"}


@pytest.mark.vcr
def test_AGIPD_CalibrationData_report():
    """Test CalibrationData with data from report"""
    # Report ID: https://in.xfel.eu/calibration/reports/3757
    agipd_cd = CalibrationData.from_report(3757)
    assert agipd_cd.detector_name == "SPB_DET_AGIPD1M-1"
    assert set(agipd_cd) == {"Offset", "Noise", "ThresholdsDark", "BadPixelsDark"}
    assert agipd_cd.aggregator_names == [f"AGIPD{n:02}" for n in range(16)]
    assert isinstance(agipd_cd["Offset", "AGIPD00"], SingleConstant)


def test_AGIPD_from_correction_minimal():
    agipd_cd = CalibrationData.from_correction_minimal(
        TEST_DIR / "files" / "cal-metadata-p900508-r22.yml",
    )

    assert agipd_cd.detector_name == "SPB_DET_AGIPD1M-1"
    assert set(agipd_cd) == {
        "Offset", "Noise", "ThresholdsDark", "BadPixelsDark", "SlopesPC", "BadPixelsPC",
    }
    assert agipd_cd.aggregator_names == [f"AGIPD{n:02}" for n in range(16)]
    assert isinstance(agipd_cd["Offset", "AGIPD00"], SingleConstant)
    assert agipd_cd["Offset", "AGIPD00"].ccv_id == 229094


@pytest.mark.vcr
def test_AGIPD_from_correction():
    agipd_cd = CalibrationData.from_correction(
        TEST_DIR / "files" / "cal-metadata-p900508-r22.yml",
    )

    assert agipd_cd.detector_name == "SPB_DET_AGIPD1M-1"
    assert set(agipd_cd) == {
        "Offset", "Noise", "ThresholdsDark", "BadPixelsDark", "SlopesPC", "BadPixelsPC",
    }
    assert agipd_cd.aggregator_names == [f"AGIPD{n:02}" for n in range(16)]
    assert agipd_cd.module_nums == list(range(16))
    assert agipd_cd.qm_names == [f"Q{(m // 4) + 1}M{(m % 4) + 1}" for m in range(16)]
    assert isinstance(agipd_cd["Offset", "AGIPD00"], SingleConstant)
    assert agipd_cd["Offset", "AGIPD00"].ccv_id == 229094
    # Using the private attribute to check metadata is already loaded, not just
    # available for lazy loading.
    assert agipd_cd["Offset", "AGIPD00"]._metadata['report_id'] == 6512


@pytest.mark.vcr
def test_JUNGFRAU_from_correction():
    jf_cd = CalibrationData.from_correction(
        TEST_DIR / "files" / "cal-metadata-p900491-r487.yml"
    )

    assert jf_cd.detector_name == "FXE_XAD_JF1M"
    assert set(jf_cd) == {
        "Offset10Hz", "BadPixelsDark10Hz", "RelativeGain10Hz", "BadPixelsFF10Hz",
    }
    assert jf_cd.aggregator_names == ["JNGFR01", "JNGFR02"]
    assert jf_cd.pdu_names == ["Jungfrau_M530", "Jungfrau_M512"]
    assert jf_cd["Offset10Hz", "JNGFR01"].ccv_id == 242621


def test_format_time(mock_spb_aux_run):
    by_run = datetime.fromisoformat(
        CalCatAPIClient.format_time(mock_spb_aux_run))

    assert by_run.tzinfo is not None
    assert (by_run - datetime.now(tz=by_run.tzinfo)) < timedelta(minutes=10)


def test_conditions_markdown():
    cond = LPDConditions()
    md = MarkdownFormatter()(cond)  # Smoketest
    assert isinstance(md, str)
