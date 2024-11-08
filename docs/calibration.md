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

In a Jupyter notebook, you can show the selected constants in a table:

```python
agipd_cd.display_markdown_table()
```

| Modules | BadPixelsDark                                                                           | Noise                                                                                   | Offset                                                                                  | ThresholdsDark                                                                          |
|--------:|:----------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------|
|       0 | [2023-07-29 19:34](https://in.xfel.eu/calibration/calibration_constant_versions/172433) | [2023-07-29 19:34](https://in.xfel.eu/calibration/calibration_constant_versions/172424) | [2023-07-29 19:34](https://in.xfel.eu/calibration/calibration_constant_versions/172409) | [2023-07-29 19:34](https://in.xfel.eu/calibration/calibration_constant_versions/172442) |
|       1 | [2023-07-29 19:34](https://in.xfel.eu/calibration/calibration_constant_versions/172402) | [2023-07-29 19:34](https://in.xfel.eu/calibration/calibration_constant_versions/172395) | [2023-07-29 19:34](https://in.xfel.eu/calibration/calibration_constant_versions/172390) | [2023-07-29 19:34](https://in.xfel.eu/calibration/calibration_constant_versions/172416) |
|       2 | [2023-07-29 19:34](https://in.xfel.eu/calibration/calibration_constant_versions/172425) | [2023-07-29 19:34](https://in.xfel.eu/calibration/calibration_constant_versions/172410) | [2023-07-29 19:34](https://in.xfel.eu/calibration/calibration_constant_versions/172388) | [2023-07-29 19:34](https://in.xfel.eu/calibration/calibration_constant_versions/172435) |
|     ... |                                                                                      |
|      14 | [2023-07-29 19:34](https://in.xfel.eu/calibration/calibration_constant_versions/172407) | [2023-07-29 19:34](https://in.xfel.eu/calibration/calibration_constant_versions/172399) | [2023-07-29 19:34](https://in.xfel.eu/calibration/calibration_constant_versions/172392) | [2023-07-29 19:34](https://in.xfel.eu/calibration/calibration_constant_versions/172418) |
|      15 | [2023-07-29 19:34](https://in.xfel.eu/calibration/calibration_constant_versions/172421) | [2023-07-29 19:34](https://in.xfel.eu/calibration/calibration_constant_versions/172415) | [2023-07-29 19:34](https://in.xfel.eu/calibration/calibration_constant_versions/172406) | [2023-07-29 19:34](https://in.xfel.eu/calibration/calibration_constant_versions/172429) |


## Found constants

::: extra.calibration.CalibrationData
    options:
      merge_init_into_class: false
      members_order: source
      group_by_category: false

::: extra.calibration.MultiModuleConstant

::: extra.calibration.SingleConstant

## Conditions objects

::: extra.calibration.AGIPDConditions

::: extra.calibration.DSSCConditions

::: extra.calibration.JUNGFRAUConditions

::: extra.calibration.LPDConditions

::: extra.calibration.ShimadzuHPVX2Conditions

## Bad pixel values

The calibration pipeline produces masks along with corrected data, in keys
called `image.mask` or `data.mask` depending on the detector. Zeros in the mask
represent normal, good pixels, while any other value indicates one or more
reasons why the data may be atypical.

Most of these values indicate different kinds of 'bad' data. But it also
includes values like `NON_STANDARD_SIZE` for pixels, usually at sensor edges,
which are intentionally larger than most, and thus capture more photons.

::: extra.calibration.BadPixels
    options:
      show_if_no_docstring: true
