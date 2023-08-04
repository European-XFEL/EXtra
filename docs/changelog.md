<!-- Let's try to follow the naming guidelines here: https://keepachangelog.com/en/1.0.0/#how -->

# Changelog

## Unreleased

!!! note
    All of the changes here are deployed to our current environment, even though
    a release hasn't been made for them yet. If you want to have these updates
    in a personal environment you'll need to install the package from git.

    ```bash title="Installation command"
    pip install git+https://github.com/European-XFEL/EXtra.git
    ```

Changed:

- [Scantool][extra.components.Scantool]'s `__repr__()` functionality to print
  information was moved to [Scantool.info()][extra.components.Scantool.info]
  (!29).

## 2023.2
The initial release! :tada: :sparkles:

Added:

- A [Scantool][extra.components.Scantool] component for the EuXFEL scantool, in !2.
- Components for extracting the X-ray and optical laser pulse patterns:
  [XrayPulses][extra.components.XrayPulses] and
  [OpticalLaserPulses][extra.components.OpticalLaserPulses] in !5.
- A collection of [utility](utilities.md) functions.
- Sub-modules to forward [EXtra-data](reading-data.md),
  [EXtra-geom](detector-geometry.md), and [karabo-bridge-py](karabo-bridge.md).
