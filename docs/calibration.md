# Calibration constants

The `extra.calibration` module helps you to find & load detector calibration
constants stored in the [Calibration Catalog](https://in.xfel.eu/calibration/).

To find constants by conditions for a particular point in time, create a
condition object for the relevant detector type and use
`CalibrationData.from_condition`:

```python
from extra.calibration import CalibrationData, LPDConditions

lpd_cd = CalibrationData.from_condition(
    LPDConditions(memory_cells=200, sensor_bias_voltage=250),
    "FXE_DET_LPD1M-1",
    event_at="2022-05-22T02:00:00",
)

# Load one constant for all found modules
offset = lpd_cd["Offset"].ndarray()
```

You can also find a group of constants produced by one characterisation process
by specifying a CalCat report, e.g. [Report 3757](https://in.xfel.eu/calibration/reports/3757):

```python
from extra.calibration import CalibrationData

agipd_cd = CalibrationData.from_report(3757)
```

## Found constants

::: extra.calibration.CalibrationData

::: extra.calibration.MultiModuleConstant

::: extra.calibration.SingleConstant

## Conditions objects

::: extra.calibration.AGIPDConditions

::: extra.calibration.LPDConditions

::: extra.calibration.DSSCConditions
