<!-- Let's try to follow the naming guidelines here: https://keepachangelog.com/en/1.0.0/#how. -->

<!-- Note: before making a new release make sure to update the 'Unreleased' link at -->
<!-- the bottom to compare to the new version. -->

# Changelog

## [Unreleased]

!!! note
    All of the changes here are deployed to our current environment, even though
    a release hasn't been made for them yet. If you want to have these updates
    in a personal environment you'll need to install the package from git.

    ```bash title="Installation command"
    pip install git+https://github.com/European-XFEL/EXtra.git
    ```
Added:

- The `slowTrains` XGM property is available through
  [XGM.slow_train_energy()][extra.components.XGM.slow_train_energy] (!162).
- [XGM.pulse_energy()][extra.components.XGM.pulse_energy] can now take a
  `series=True` argument to return a 1D [Series][pandas.Series] instead of a 2D
  [DataArray][xarray.DataArray] (!162).
- Data indexed by pulse can now be aligned with and added as additional columns to sparse DLD data (!166).
- Empty trains can now optionally be included when determining constant pulse patterns via [XrayPulses.is_constant_pattern()][extra.components.XrayPulses.is_constant_pattern] (!156).
- Check whether any pulse patterns are interleaved with `is_interleaved_with()` or directly SA1/SA3 with `is_sa1_interleaved_with_sa3()` (!155).
- Obtain [MachinePulses][extra.components.MachinePulses] from any other timeserver-based pulse components directly via `machine_pulses()` or get machine repetition rate directly from `machine_repetition_rate()` (!155).
- A helper function named [fit_gaussian()][extra.utils.fit_gaussian] (!131).

Fixed:

- Sometimes an XGM will record the wrong number of pulses in its slow data
  property, which would cause the pulse energy to not be retrieved properly. Now
  the XGM will rely only on the fast data to find the number of pulses in
  `.pulse_energy()` (!153).

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


[Unreleased]: https://github.com/European-XFEL/EXtra/compare/2024.1.1...master
[2024.1.1]: https://github.com/European-XFEL/EXtra/releases/tag/2024.1.1
[2024.1]: https://github.com/European-XFEL/EXtra/releases/tag/2024.1
[2023.2]: https://github.com/European-XFEL/EXtra/releases/tag/2023.2
