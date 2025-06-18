from unittest.mock import patch

import pint
import pytest
import numpy as np
import pandas as pd
import xarray as xr

from extra.components import XGM
from extra.components.xgm import PropertyGroup
from extra_data.exceptions import MultiRunError


def test_create_xgm(multi_xgm_run):
    run = multi_xgm_run

    # Test finding the only XGM in a run
    XGM(run.select(["SA2*"]))

    # With multiple XGMs it should throw an error
    with pytest.raises(RuntimeError):
        XGM(run)

    # By selecting the SPB and SQS XGM we can trick the class into thinking that
    # the run is from SASE 3. If we then select a SASE 3 XGM it should
    # automatically set `default_sase` to 3
    sase3_run = run.select(["SPB*", "SQS*"])
    xgm = XGM(sase3_run, "sqs")
    assert xgm._default_pg == PropertyGroup.SASE3

    # But if we select an XGM from a different SASE it should require us to
    # explicitly set `default_sase`.
    with pytest.raises(RuntimeError):
        XGM(sase3_run, "spb")
    xgm = XGM(sase3_run, "spb", default_sase=1)
    assert xgm._default_pg == PropertyGroup.SASE1

    source = "SA2_XTD1_XGM/XGM/DOOCS"
    control_source = run[source]
    instrument_source = run[f"{source}:output"]

    # We should be able to pass a device name, alias, lower-case substring,
    # upper-case substring, SourceData, and KeyData.
    for device in [source, "sa2-xgm", "sa2", "SA2",
                   run[source], run[source, "pulseEnergy.numberOfBunchesActual"]]:
        xgm = XGM(run, device)
        assert xgm.control_source == control_source
        assert xgm.instrument_source == instrument_source

# Test all the functions that read data from files
@pytest.mark.parametrize("source_sase", [("sa2", 2), ("sqs", 3),
                                         ("non_actualized", None), ("hobbit", None)])
def test_xgm_getters(multi_xgm_run, source_sase):
    source, sase = source_sase
    xgm = XGM(multi_xgm_run, source, default_sase=sase)
    assert isinstance(xgm.wavelength(), pint.Quantity)
    xgm.wavelength_by_train()
    assert isinstance(xgm.photon_energy(with_units=False), (float, np.floating))
    xgm.photon_energy_by_train()
    xgm.doocs_server()
    xgm.pulse_energy()
    xgm.slow_train_energy()
    xgm.npulses()
    xgm.pulse_counts()
    xgm.max_npulses()
    # xgm.photon_flux()

def test_multisase_xgm(multi_xgm_run):
    run = multi_xgm_run.select(["SQS*"])
    control = "SQS_DIAG1_XGMD/XGM/DOOCS"
    instrument = f"{control}:output"

    # Test results without setting `default_sase`
    xgm = XGM(run, control)
    # It should automatically pick SASE 3 based on the source name
    assert xgm.pulse_energy().name == "Energy"
    assert xgm.pulse_counts().name == f"{control}.pulseEnergy.numberOfSa3BunchesActual"
    assert xgm.slow_train_energy().name == f"{control}.controlData.slowTrainSa3"

    # And with a `default_sase`
    xgm = XGM(run, control, default_sase=1)
    assert xgm.pulse_energy().name == "Energy"
    assert xgm.pulse_counts().name == f"{control}.pulseEnergy.numberOfSa1BunchesActual"
    assert xgm.slow_train_energy().name == f"{control}.controlData.slowTrainSa1"

    # Test specifying an explicit non-default SASE
    assert xgm.pulse_energy(0).name == "Energy"
    assert xgm.pulse_counts(0).name == f"{control}.pulseEnergy.numberOfBunchesActual"
    assert xgm.slow_train_energy(0).name == f"{control}.controlData.slowTrain"

    assert xgm.pulse_energy(3).name == "Energy"
    assert xgm.pulse_counts(3).name == f"{control}.pulseEnergy.numberOfSa3BunchesActual"
    assert xgm.slow_train_energy(3).name == f"{control}.controlData.slowTrainSa3"

    # Smoke tests
    mock_pulse_energy = xr.DataArray(np.random.rand(1000, 100),
                                  dims=("trainId", "pulseIndex"))
    with patch.object(xgm, "pulse_energy", return_value=mock_pulse_energy):
        # Test both a specific SASE (the current `default_sase`) and the main
        # properties (with `sase=0`).
        xgm.plot_pulse_energy()
        xgm.plot_pulse_energy(0)
        xgm.plot_energy_per_pulse()
        xgm.plot_energy_per_pulse(0)
        xgm.plot_energy_per_train()
        xgm.plot_energy_per_train(0)
        xgm.plot()

    assert xgm.is_constant_pulse_count()
    repr(xgm)
    xgm.info()

def test_run_union(multi_xgm_run, mock_spb_aux_run):
    run = multi_xgm_run.select("*SQS*").union(mock_spb_aux_run)
    assert not run.is_single_run

    # We should be able to create the XGM object at least
    xgm = XGM(run, "spb", default_sase=0)

    # But getting data from the RUN section will fail
    with pytest.raises(MultiRunError):
        xgm.doocs_server()

    # Smoke test to check that non-RUN values work
    xgm.pulse_energy()

def test_xgm_pulse_energy_series(mock_spb_aux_run):
    run = mock_spb_aux_run
    xgm = XGM(run)

    energy_series = xgm.pulse_energy(series=True)
    assert isinstance(energy_series, pd.Series)
    assert energy_series.attrs["units"] == "µJ"

    # By default the tests write zeros, so the Series should be empty
    assert len(energy_series) == 0

    # Fill the first 10 pulses of the first 10 trains with fake data
    xgm._pulse_energy[xgm._default_pg][:10, :10] = 800
    energy_series = xgm.pulse_energy(series=True)

    # We should only have 100 pulses with data
    assert len(energy_series) == 100

    # Check their index values
    assert np.array_equal(energy_series.index.get_level_values(0).unique(),
                          run.train_ids[:10])
    assert np.array_equal(energy_series.index.get_level_values(1).unique(),
                          np.arange(10))

def test_wrong_pulse_counts(mock_spb_aux_run):
    run = mock_spb_aux_run
    xgm = XGM(mock_spb_aux_run)

    # Create a mock pulse_counts array
    n_trains = len(run.train_ids)
    mock_pulse_counts = xr.DataArray(np.ones(n_trains),
                                     dims=("trainId",),
                                     coords={"trainId": run.train_ids},
                                     name="slow_counts")

    # And a mock pulse_energy array with a different number of pulses
    mock_pulse_energy = xr.DataArray(np.ones((n_trains, 100)),
                                     dims=("trainId", "pulseIndex"),
                                     coords={"trainId": run.train_ids})

    with patch("extra.components.xgm.KeyData.xarray", return_value=mock_pulse_counts), \
         patch.object(xgm, "pulse_energy", return_value=mock_pulse_energy):
        with pytest.warns():
            # npulses() will call pulse_counts() internally which should emit a
            # warning when the number of pulses differ.
            assert xgm.npulses() == 100

            # pulse_counts() should use the fast data array by default, which
            # will not have a name.
            assert xgm.pulse_counts().name == None

            # Otherwise it should return the slow data counts which should have
            # a name.
            assert xgm.pulse_counts(force_slow_data=True).name == mock_pulse_counts.name
