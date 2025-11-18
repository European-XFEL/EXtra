
Analog to European XFEL's timing structure, its data is also organized in trains
identifed by a unique and global train ID. Some data however, in particular many
of the primary detectors in an experiment like AGIPD, DSSC, LPD or fast digitizers,
also resolves every or at least some of the underlying pulses a train is composed of.

![European XFEL timing structure](../images/trains_pulses.png){ width="75%" align=center }

The `PulsePattern` family of components allows access to pulse patterns recorded
by various sources. For example, in order to obtain the FEL pulses for a particular
SASE:

```python
p = XrayPulses(run)
p.pulse_ids()
```

There are different types available depending on the source that should be used for
the pulse pattern information, but they all offer the same interface as
[`PulsePattern`][extra.components.pulses.PulsePattern] for access. Furthermore,
many other components like [`AdqRawChannel`][extra.components.AdqRawChannel] can be
given a [`PulsePattern`][extra.components.pulses.PulsePattern] component to customize
how pulse-resolved data is interpreted.

The primary source of pulse pattern information is the bunch pattern table from
the machine side, which is recorded either through a so called timeserver or a
pulse pattern decoder source. There are three different components available to access
different parts of this data:

- [`XrayPulses`][extra.components.XrayPulses] for FEL pulses.
- [`OpticalLaserPulses`][extra.components.OpticalLaserPulses] for optical laser
  or PPL pulses.
- [`MachinePulses`][extra.components.MachinePulses] for machine-related pulses
  or any other data in the bunch pattern table.

There is a special component available for pump-probe experiments, which can combine
FEL and PPL pulses into a single pattern with corresponding labels which of these two
is present in a given pulse:

- [`PumpProbePulses`][extra.components.PumpProbePulses] for combined FEL/PPL
  pulse patterns.

All of the components above offer the interface of
[`TimeserverPulses`][extra.components.pulses.TimeserverPulses] in addition to the
general interface of [`PulsePattern`][extra.components.pulses.PulsePattern].

Apart from that, there are components to access other sources of pulse pattern
information via the same [`PulsePattern`][extra.components.pulses.PulsePattern]
interface:

- [`ManualPulses`][extra.components.ManualPulses] for manually created pulse
  pattern or modifications of existing patterns via
  [`select_pulses()`][extra.components.pulses.PulsePattern.select_pulses], 
  [`deselect_pulses()`][extra.components.pulses.PulsePattern.select_pulses] or
  [`union()`][extra.components.pulses.PulsePattern.union].

- [`DldPulses`][extra.components.DldPulses] for pulse informations generated as
  part of processed DLD (delay line detector) data.

As opposed to trains, there is no single global mechanism how to record pulses and
some sources use their own identification scheme. Therefore, these components aim to
refer to the location of pulses in the machine bunch pattern table wherever possible
as a shared and universal identification called *pulse ID*. An enumeration of pulses
for a single SASE, instrument or device is called *pulse index*. If desired, the
*pulse time* uses the relative time to the beginning of a subtrain.

::: extra.components.pulses.PulsePattern

::: extra.components.pulses.TimeserverPulses
	options:
		inherited_members: no

::: extra.components.XrayPulses
	options:
		inherited_members: no

::: extra.components.OpticalLaserPulses
	options:
		inherited_members: no

::: extra.components.PumpProbePulses
	options:
		inherited_members: no

::: extra.components.MachinePulses
	options:
		inherited_members: no

::: extra.components.ManualPulses
    options:
		inherited_members: false

::: extra.components.DldPulses
    options:
		inherited_members: false
