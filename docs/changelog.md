<!-- Let's try to follow the naming guidelines here: https://keepachangelog.com/en/1.0.0/#how. -->

<!-- Note: before making a new release make sure to update the 'Unreleased' link at -->
<!-- the bottom to compare to the new version. -->

# Changelog

## [2025.1]

<!-- !!! note -->
<!--     All of the changes here are deployed to our current environment, even though -->
<!--     a release hasn't been made for them yet. If you want to have these updates -->
<!--     in a personal environment you'll need to install the package from git. -->

<!--     ```bash title="Installation command" -->
<!--     pip install git+https://github.com/European-XFEL/EXtra.git -->
<!--     ``` -->


Added:

- [TOFResponse][extra.recipes.TOFResponse] to estimate, deconvolve and denoise the instrumental response in eTOFs (!304).
- [CookieboxCalibration][extra.recipes.CookieboxCalibration] to calibrate data from eTOFs after taking a calibration run (!284).
- [Grating2DCalibration][extra.recipes.Grating2DCalibration] to calibrate data from a 2D grating detector (!284).
- [Grating1DCalibration][extra.recipes.Grating1DCalibration] to calibrates 1D grating information (e.g.: collected from the Gotthard detector in SQS) (!318).
- Exposed detector data components from `extra_data` in `extra.components`
  (AGIPD1M, AGIPD500K, DSSC1M, JUNGFRAU, LPD1M) (!177).
- [imshow2][extra.utils.imshow2] now supports plotting 2D
  [DataArray][xarray.DataArray]s properly (!333).
- Added [hyperslicer2()][extra.utils.hyperslicer2] to make plotting image arrays
  easier (!348).
- [reorder_axes_to_shape][extra.utils.reorder_axes_to_shape] utility function (!349).
- [ShimadzuHPVX2Conditions][extra.calibration.ShimadzuHPVX2Conditions] to
  retrieve constants for dynamic flat-field correction (!254).

Changed:

- [Timepix3.spatial_bins()][extra.components.Timepix3.spatial_bins] is now a static method.
- The [XGM][extra.components.XGM] component will now emit warnings when it
  detects the wrong number of pulses have been saved in the slow data
  property. There is also a new `force_slow_data` argument to
  [XGM.pulse_counts()][extra.components.XGM.pulse_counts] to always return whatever was
  saved in the slow data (!161).
- [LPDConditions][extra.calibration.LPDConditions] now accepts
  `parallel_gain` (!254).
- [JUNGFRAUConditions][extra.calibration.JUNGFRAUConditions] now accepts
  `exposure_timeout` (!254).
- [CalibrationData.from_condition][extra.calibration.CalibrationData.from_condition] has a new `begin_at_strategy` parameter (!254).

Fixed:

- Fixed [PumpProbePulses.is_constant_pattern()][extra.components.PumpProbePulses.is_constant_pattern] to properly take pump probe flags into account when determining whether a pattern is constant (!313).
- [AdqRawChannel.pulse_edges()][extra.components.AdqRawChannel.pulse_edges] now also supports data where the trace is too short for the actual number of pulses present (!312).
- Fixed issues with pulse separation in [AdqRawChannel][extra.components.AdqRawChannel] with variable pulse patterns and those with trains missing ADQ data (!310).
- [AdqRawChannel][extra.components.AdqRawChannel] now properly enumerates channels starting with 1 rather than 0 as in the Karabo device.
- Fixed reading of the
  [Scantool.acquisition_time][extra.components.Scantool.acquisition_time]
  property for newer Scantool versions (!303).
- [CookieboxCalibration][extra.recipes.CookieboxCalibration]: allow flagging of bad calibration points, to allow using others in the `mask_calibration_point(tof_id, energy)` function (!318).
- [CookieboxCalibration][extra.recipes.CookieboxCalibration]: Order `plot_calibration_data` y axis by energy to avoid unordered axis in case the energy scan was done out of monotonically increasing order (!318).
- [CookieboxCalibration][extra.recipes.CookieboxCalibration]: Use Auger-Meitner mean to align all plots in `plot_calibration_data` by the Auger-Meitner peak (!318).
- [CookieboxCalibration][extra.recipes.CookieboxCalibration]: New function `plot_calibrations`, which plots only the calibration factors in the same plot for diagnostics. Often this is all that one is interested in (!318).
- [Grating2DCalibration][extra.recipes.Grating2DCalibration]: Crop image based on rotation angle to avoid artifacts caused by out-of-bound effects. Add background root-mean-squared error as uncertainty band in the output (!318).
- [Grating2DCalibration][extra.recipes.Grating2DCalibration]: If provided, use extra motor information as independent variable in the fit (useful when data includes motor movements additionally) (!318).
- Fixed [Scan.plot_bin_by_steps()][extra.components.Scan.plot_bin_by_steps] to show 2D
  data (!320).
- Restrict the version of Cython used to build while we figure out an issue with
  Cython 3.1 (!328).
- Fixed behaviour of
  [Scan.plot_bin_by_steps()][extra.components.Scan.plot_bin_by_steps] when
  passed a custom axis (!334).

## [2024.2]
Added:

- [Timepix3][extra.components.Timepix3] to access raw hits and centroids from the Timepix3 detector (!231).
- [Scan.plot()][extra.components.Scan.plot] now allows passing a `figsize` (!262).
- [JF4MHalfMotors][extra.components.JF4MHalfMotors] to access positions of motors moving halfs of Jungfrau detector (!224).

Fixed:

- Karabacon 3.0.10 is now supported by the [Scantool][extra.components.Scantool]
  (!212).
- [`Scan.plot_bin_by_steps()`][extra.components.Scan.plot_bin_by_steps] would
  previously ignore the `title`/`xlabel`/`ylabel` arguments, now it actually
  uses them (!237).
- [`Scan.bin_by_steps()][extra.components.Scan.plot_bin_by_steps] now preserves
  any additional dimensions in the data to be binned, so it can produce e.g.
  an average spectrum per scan step (!269).
- [`AdqRawChannel.pulse_data()`][extra.components.AdqRawChannel.pulse_data] no longer erroneously reads in the same train data for every pulse if there is only a single pulse per train (!259).

Changed:

- [`gaussian()`][extra.utils.gaussian] has a new `norm` parameter to allow
  disabling normalization, which [`fit_gaussian()`][extra.utils.fit_gaussian] will
  use by default (!221).
- [`fit_gaussian()`][extra.utils.fit_gaussian] has a new `A_sign` parameter to
  specify the expected orientation of the peak (!222).
- [`AdqRawChannel.pulse_data()`][extra.components.AdqRawChannel.pulse_data] no longer throws an exception if the pulse pattern refers to data beyond the acquired traces, but instead fills this with NaN or -1 depending on data type (!245).

## [2024.1.2]
Added:

- [AGIPD1MQuadrantMotors][extra.components.AGIPD1MQuadrantMotors] to access positions of motors moving quadrants of AGIPD1M detector (!148).
- A module to expose the [DAMNIT API](damnit.md).
- [AdqRawChannel][extra.components.AdqRawChannel] to access and process raw traces saved by fast ADQ digitizers (!35).
- Hits obtained via [DelayLineDetector.hits()][extra.components.DelayLineDetector.hits] can now optionally be restricted to hits reconstructed with a certain method. (!170).
- [DldPulses][extra.components.DldPulses] can optionally enumerate PPL-only pulses using negative indices to stay compatible with PPL-unaware data. (!167).
- The `slowTrains` XGM property is available through
  [XGM.slow_train_energy()][extra.components.XGM.slow_train_energy] (!162).
- [XGM.pulse_energy()][extra.components.XGM.pulse_energy] can now take a
  `series=True` argument to return a 1D [Series][pandas.Series] instead of a 2D
  [DataArray][xarray.DataArray] (!162).
- Data indexed by pulse or train can now be aligned with and added as additional columns to sparse DLD data (!166, !173).
- [OpticalLaserDelay][extra.components.OpticalLaserDelay] to obtain the time delays between FEL and optical laser pulses (!165).
- Empty trains can now optionally be included when determining constant pulse patterns via [XrayPulses.is_constant_pattern()][extra.components.XrayPulses.is_constant_pattern] (!156).
- Check whether any pulse patterns are interleaved with `is_interleaved_with()` or directly SA1/SA3 with `is_sa1_interleaved_with_sa3()` (!155).
- Obtain [MachinePulses][extra.components.MachinePulses] from any other timeserver-based pulse components directly via `machine_pulses()` or get machine repetition rate directly from `machine_repetition_rate()` (!155).
- A helper function named [fit_gaussian()][extra.utils.fit_gaussian] (!131).
- A new method [Scan.split_by_steps()][extra.components.Scan.split_by_steps] (!169).
- [PumpProbePulses.pulse_mask][extra.components.PumpProbePulses.pulse_mask] now
  has an option to give a mask for only FEL or only pump laser pulses (!174).
- [BadPixels][extra.calibration.BadPixels] flag values for interpreting masks
  in corrected detector data (!172).
- Deprecate pulse dimension `time` in favor of `pulseTime` (!178).
- Constant fraction and dynamic leading edge discriminators to determine arrival time of fast pulses on an analog signal (!25).
- Sinc interpolation method (!25).
- [extra.calibration.CalibrationData][extra.calibration.CalibrationData] now has methods
  to create & display a markdown table of the constants found. This is primarily for
  use in Jupyter notebooks (!202).
- More informative plots from [Scan.plot()][extra.components.Scan.plot] (!201).

Fixed:

- Sometimes an XGM will record the wrong number of pulses in its slow data
  property, which would cause the pulse energy to not be retrieved properly. Now
  the XGM will rely only on the fast data to find the number of pulses in
  `.pulse_energy()` (!153).
- Attempt to convert calibration condition values to floats before defaulting to strings. This prevents misinterpretation of boolean values as strings (e.g., "True") (!193).
- Add tabulate to the package dependencies.

## [2024.1.1]

Added:

- [MachinePulses][extra.components.MachinePulses] to investigate pulse pattern
  data beyond only X-rays and optical lasers (!138).
- Implemented [Scan.bin_by_steps()][extra.components.Scan.bin_by_steps] and
  [Scan.plot_bin_by_steps()][extra.components.Scan.plot_bin_by_steps] to help
  with averaging data over scan steps (!124).
- `pulse_periods()`, `pulse_repetition_rates()` and `train_durations()` methods
  to obtain statistics about the pulses in all [pulse
  pattern](components/pulse-patterns.md) components (!114).
- `extra.calibration` module to find and load calibration constants.

## [2024.1]
Added:

- [DelayLineDetector][extra.components.DelayLineDetector] to access raw and reconstructed results from the offline delay line detector processing pipeline (!103).
- An [XGM][extra.components.XGM] component to access XGM devices (!53).
- [PumpProbePulses][extra.components.PumpProbePulses] to combine X-ray and
  optical laser pulses in a single pattern (!24).
- The [Scan][extra.components.Scan] component to automatically detect steps
  within a motor scan (!4).
- [DldPulses][extra.components.DldPulses] to access pulse information saved
  during delay line detector event reconstruction (!42).
- The helper function [imshow2][extra.utils.imshow2] to provide good defaults
  when plotting images (!38).

Changed:

- The `get_` prefix was deprecated for some method names in the [pulse pattern
  components](components/pulse-patterns.md) (!106).
- All methods in [XrayPulses][extra.components.XrayPulses] and
  [OpticalLaserPulses][extra.components.OpticalLaserPulses] now support labelled
  results and default to it (!40).
- [Scantool][extra.components.Scantool]'s `__repr__()` functionality to print
  information was moved to [Scantool.info()][extra.components.Scantool.info]
  (!29).

## [2023.2]
The initial release! :tada: :sparkles:

Added:

- A [Scantool][extra.components.Scantool] component for the EuXFEL scantool, in !2.
- Components for extracting the X-ray and optical laser pulse patterns:
  [XrayPulses][extra.components.XrayPulses] and
  [OpticalLaserPulses][extra.components.OpticalLaserPulses] in !5.
- A collection of [utility](utilities.md) functions.
- Sub-modules to forward [EXtra-data](reading-data.md),
  [EXtra-geom](detector-geometry.md), and [karabo-bridge-py](karabo-bridge.md).


[Unreleased]: https://github.com/European-XFEL/EXtra/compare/2025.1...master
[2025.1]: https://github.com/European-XFEL/EXtra/releases/tag/2025.1
[2024.2]: https://github.com/European-XFEL/EXtra/releases/tag/2024.2
[2024.1.2]: https://github.com/European-XFEL/EXtra/releases/tag/2024.1.2
[2024.1.1]: https://github.com/European-XFEL/EXtra/releases/tag/2024.1.1
[2024.1]: https://github.com/European-XFEL/EXtra/releases/tag/2024.1
[2023.2]: https://github.com/European-XFEL/EXtra/releases/tag/2023.2
