from enum import Enum
from warnings import warn
from textwrap import dedent

import numpy as np

from extra_data import SourceData, KeyData, MultiRunError

from .. import ureg
from .utils import SASE_TOPICS, identify_sase


class PropertyGroup(Enum):
    """Enum representing the different 'groups' of XGM properties.

    XGMs in SASE 1 and SASE 3 hold data for both SASEs, as well as their
    combined data in the default properties (e.g. `data.intensityTD`). To
    simplify access to these, this enum defines groups for 'SASE-resolved'
    properties. However, this is purely an internal abstraction. To users we
    only expose `sase` arguments in the XGM methods (which may be `0` to refer
    to the main group).
    """
    MAIN = 0
    SASE1 = 1
    SASE3 = 3

def _find_xgm(run, device):
    """Helper function to find an XGM device."""
    # First look for XGMs in the run
    available_xgms = [source for source in run.control_sources
                      if run[source].device_class in XGM.DEVICE_CLASSES]
    if len(available_xgms) == 0:
        available_xgms = [source for source in run.control_sources
                          if "XGM/DOOCS" in source or "XGMD/DOOCS" in source]


    # If the user doesn't choose a device, attempt to pick one automatically
    if device is None:
        if len(available_xgms) == 0:
            raise RuntimeError("Couldn't find an XGM in the run, please pass the device name with the `device` argument")
        elif len(available_xgms) > 1:
            raise RuntimeError(f"Multiple XGMs found, please select one with the `device` argument: {available_xgms}")
        elif len(available_xgms) == 1:
            device = available_xgms[0]

    elif isinstance(device, KeyData):
        # If a KeyData is passed, assume it belongs to an XGM source
        device = _find_xgm(run, device.source)

    elif isinstance(device, SourceData):
        # If it's an instrument source, assume it's the output channel
        if device.is_instrument and device.source.endswith(":output"):
            device = device.source.removesuffix(":output")
        else:
            device = device.source

    elif isinstance(device, str):
        if device in run.all_sources:
            # We allow complete device names
            pass
        elif device in run.alias:
            # And aliases
            device = _find_xgm(run, run.alias[device])
        else:
            # And unique substrings of available XGMs
            matches = [xgm_name for xgm_name in available_xgms
                       if device.upper() in xgm_name]

            if len(matches) == 0:
                raise RuntimeError(f"Couldn't identify an XGM from '{device}'; please pass a valid device name, alias, or unique substring")
            elif len(matches) == 1:
                device = matches[0]
            elif len(matches) > 1:
                raise RuntimeError(f"Multiple XGMs found matching '{device}', please be more specific: {matches}")
    else:
        raise TypeError(f"Unrecognized `device` type '{type(device)}', must be a SourceData, KeyData, or string")

    return device

class XGM:
    """Interface to an XGM (X-ray Gas Monitor).

    Example usage in a Jupyter notebook:
    ```python
            -----------------------------------------------------------
    In [1]: |xgm = XGM(run) # This will work with a single XGM        |
            |xgm                                                      |
            -----------------------------------------------------------
    Out[1]: <XGM for SA1_XTD2_XGM/XGM/DOOCS at 9.3 keV>

            -----------------------------------------------------------
    In [2]: |# With multiple XGMs, one needs to be chosen explicitly. |
            |# Either by explicit device name:                        |
            |xgm = XGM(run, "SPB_XTD9_XGM/XGM/DOOCS")                 |
            |                                                         |
            |# Or by passing a SourceData object:                     |
            |xgm = XGM(run, run.alias["spb-xgm"])                     |
            |                                                         |
            |# Or a KeyData object:                                   |
            |xgm = XGM(run, run.alias["spb-xgm"]["data.intensityTD"]) |
            |                                                         |
            |# Or alias name                                          |
            |xgm = XGM(run, "spb-xgm")                                |
            |                                                         |
            |# Or a unique (case-insensitive) substring among XGMs    |
            |xgm = XGM(run, "spb")                                    |
            -----------------------------------------------------------
    Out[2]: <XGM for SPB_XTD9_XGM/XGM/DOOCS at 9.3 keV>
    ```
    """

    # There are three different XGM device classes, though all are very
    # similar. DoocsXGM is the main one, and DoocsXGMD and DoocsXGMReduced are
    # used by FXE and SQS respectively. For more information see the original MR
    # to create them: https://git.xfel.eu/karaboDevices/doocsDevices/-/merge_requests/117
    DEVICE_CLASSES = ["DoocsXGM", "DoocsXGMD", "DoocsXGMReduced"]

    def __init__(self, run, device=None, default_sase=None):
        """
        Args:
            run (extra_data.DataCollection): A run containing at least one XGM.
            device (str, SourceData, KeyData): Specify an XGM to use, necessary
                if a run contains more than one XGM. This can be any of:

                  - The device name of the control source.
                  - A `SourceData` or [KeyData][extra_data.KeyData] of either
                    the control source (e.g. `SA2_XTD1_XGM/XGM/DOOCS`) or
                    instrument source (e.g. `SA2_XTD1_XGM/XGM/DOOCS:output`) of
                    an XGM.
                  - The alias name of either a `SourceData` or
                    [KeyData][extra_data.KeyData] belonging to an XGM.
                  - A unique (case-insensitive) substring of an XGM source
                    name. For example if a run contains `HED_XTD6_XGM/XGM/DOOCS`
                    and `SA2_XTD1_XGM/XGM/DOOCS`, then passing `device="hed"` or
                    `device="sa2"` will select those XGMs, respectively (passing
                    `xtd1` or `xtd6` would also work).

            default_sase (int): Only for XGMs with data for multiple SASE's, to
                specify a default SASE to retrieve data for. This is useful when
                working with XGMs on SASE 1 or SASE 3 and you're only interested
                in data for a specific SASE. This setting applies to all the
                SASE-specific methods of the class, such as
                [XGM.pulse_energy()][extra.components.XGM.pulse_energy] and
                [XGM.npulses()][extra.components.XGM.npulses].

                If it is not set the class will try to guess the SASE of the
                instrument the data was recorded at, and it will throw an error
                if:

                - It couldn't guess the SASE of the instrument at all.
                - It did guess the SASE of the instrument, but it's a different
                  SASE from the XGM selected. For example, if a run recorded at
                  SQS has both a SASE 1 and SQS XGM and the SQS XGM is selected,
                  `default_sase` will automatically be set to `3`. But if the
                  SASE 1 XGM is selected it will throw an error and ask the user
                  to set `default_sase` explicitly.

                Valid values are: 0, 1, and 3.
                Passing 0 always selects the main properties of a device,
                e.g. `data.intensityTD` instead of the SASE-specific ones.
        """
        device = _find_xgm(run, device)

        self._control_source = run[device]
        self._instrument_source = run[f"{device}:output"]

        self._extra_sases = [sase for sase in [1, 3]
                             if f"pulseEnergy.numberOfSa{sase}BunchesActual" in self._control_source]

        if len(self._extra_sases) > 0 and default_sase is None:
            try:
                default_sase = identify_sase(run)
            except ValueError:
                raise RuntimeError("Could not identify a SASE, please pass one explicitly with `default_sase`")

            xgm_topic = device[:device.find("_")]
            if xgm_topic not in SASE_TOPICS[default_sase]:
                raise RuntimeError("XGM doesn't belong to the same SASE as the instrument, please pass an explicit `default_sase`")

        elif default_sase is None:
            default_sase = 0

        # Note the special case for SASE 2 XGMs, which don't have SASE-specific
        # properties.
        self._default_pg = PropertyGroup(default_sase) if default_sase != 2 else PropertyGroup.MAIN
        self._main_nbunches_key = None

        self._wavelength = None
        self._wavelength_by_train = None
        self._photon_energy_by_train = None
        self._photon_flux = None
        self._doocs_server = None
        self._pulse_energy = { }
        self._slow_train_energy = { }
        self._npulses = { }
        self._pulse_counts = { }
        self._slow_pulse_counts = { }
        self._max_pulses = { }

        self._proposal = None
        self._run_no = None
        if run.is_single_run:
            metadata = run.run_metadata()
            self._proposal = metadata.get("proposalNumber")
            self._run_no = metadata.get("runNumber")

    @property
    def control_source(self) -> SourceData:
        """The `SourceData` object for the control source of the XGM
        (e.g. `SA2_XTD1_XGM/XGM/DOOCS`)."""
        return self._control_source

    @property
    def instrument_source(self) -> SourceData:
        """The `SourceData` object for the instrument source of the XGM
        (e.g. `SA2_XTD1_XGM/XGM/DOOCS:output`)."""
        return self._instrument_source

    def wavelength(self, with_units=True):
        """The nominal wavelength of the X-rays in nanometers.

        This calls
        [KeyData.as_single_value()][extra_data.KeyData.as_single_value]
        internally, which means it will throw an exception if the wavelength is
        not constant.

        Args:
            with_units (bool): Whether to return a [pint.Quantity][] or a [float][].

        Raises:
            ValueError: Will be thrown if the number of pulses is not constant.
        """
        if self._wavelength is None:
            self._wavelength = self._control_source["pulseEnergy.wavelengthUsed"].as_single_value() * ureg.nm

        return self._wavelength if with_units else self._wavelength.magnitude

    def wavelength_by_train(self):
        """Return a 1D [DataArray][xarray.DataArray] of the nominal wavelength
        of each train.
        """
        if self._wavelength_by_train is None:
            self._wavelength_by_train = self._control_source["pulseEnergy.wavelengthUsed"].xarray()
            self._wavelength_by_train.attrs["units"] = "nm"

        return self._wavelength_by_train

    def photon_energy(self, with_units=True):
        """The nominal photon energy in keV.

        This calls
        [KeyData.as_single_value()][extra_data.KeyData.as_single_value]
        internally, which means it will throw an exception if the wavelength is
        not constant.

        Args:
            with_units (bool): Whether to return a [pint.Quantity][] or a [float][].
        """
        energy = self.wavelength().to("keV", "xfel")
        return energy if with_units else energy.magnitude

    def photon_energy_by_train(self):
        """The nominal photon energy in keV of each train."""
        if self._photon_energy_by_train is None:
            import xarray as xr

            wavelengths = self.wavelength_by_train()
            energies = (wavelengths * ureg.nm).data.to("keV", "xfel").magnitude
            energies = xr.DataArray(energies,
                                    dims=("trainId",),
                                    coords=dict(trainId=wavelengths.trainId),
                                    attrs=dict(units="keV"))
            self._photon_energy_by_train = energies

        return self._photon_energy_by_train

    def doocs_server(self) -> str:
        """The DOOCS server that the Karabo XGM device is connected to.

        This information was not stored in older versions of the XGM, so the
        method may return `None` if the DOOCS server was not found.

        Raises:
            extra_data.MultiRunError: A `MultiRunError` will be thrown whenever
                the [DataCollection][extra_data.DataCollection] contains data
                from more than one run. This XGM property is stored in the `RUN`
                section of EXDF files, and when a run has been
                [unioned][extra_data.DataCollection.union] with another it is
                not possible to retrieve `RUN` values.
        """
        if self._doocs_server is None:
            self._doocs_server = self._control_source.run_values().get("location.value")

        return self._doocs_server

    def _check_sase_arg(self, sase):
        """Helper function to return the correct PropertyGroup."""
        if len(self._extra_sases) == 0 and sase is not None:
            raise RuntimeError("This XGM doesn't have data from any other SASEs")
        if sase is not None and sase not in [0, 1, 3]:
            raise RuntimeError(f"Invalid SASE '{sase}', it must be either 0, 1, or 3")

        return PropertyGroup(sase) if sase is not None else self._default_pg

    def pulse_energy(self, sase=None, series=False):
        """Returns the energy per-pulse and per-train in microjoules.

        If `series=False` (the default) this will return a 2D
        [DataArray][xarray.DataArray] with dimensions of `(trainId,
        pulseIndex)`. For runs with a varying number of pulses, the data will be
        sliced to the *maximum number* of pulses. e.g. if a run has 100 trains
        with only one train containing 10 pulses and all the others 0, the
        returned array will have a shape of `(100, 10)`.

        If `series=True` this will return a 1D [Series][pandas.Series] where all
        entries with 0 pulses have been dropped, indexed by `trainId` and
        `pulseIndex`.

        Note:
            This uses the `data.intensityTD` property of the XGM.

        Args:
            sase (int): Specify a SASE to retrieve data for. For XGMs from SASE
                1 and SASE 3 this can be either `1` or `3` or `0`. Passing `0`
                is a special case that refers to the main properties of an XGM,
                i.e. it will look at `data.intensityTD` instead of
                `data.intensitySa1TD` with `sase=1`. This setting overrides the
                `default_sase` argument to the
                [constructor][extra.components.XGM].
            series (bool): Whether to return a 2D [DataArray][xarray.DataArray]
                or 1D [Series][pandas.Series].
        """
        pg = self._check_sase_arg(sase)
        if pg not in self._pulse_energy:
            if pg == PropertyGroup.MAIN:
                key = "data.intensityTD"
            else:
                key = f"data.intensitySa{pg.value}TD"

            pulse_energy = self.instrument_source[key].xarray()

            # Find the maximum number of pulses as recorded in the fast data. We
            # do this instead of trusting .max_npulses() because that can save
            # the wrong number sometimes.
            empty_indices = np.where(pulse_energy.mean("trainId") == 1)[0]
            if len(empty_indices) == 0:
                # If there are no empty indices then apparently there were 1000
                # pulses (only needed for the tests).
                max_pulses = pulse_energy.shape[1]
            else:
                # Otherwise we select the index of the first empty element, and
                # that's our number of pulses.
                max_pulses = empty_indices[0]

            pulse_energy = pulse_energy[:, :max_pulses]

            # Assign a pulseIndex dimension and coordinate
            pulse_energy = pulse_energy.rename(dim_0="pulseIndex").assign_coords(pulseIndex=np.arange(max_pulses))
            # Replace the default fill value of 1 with 0
            pulse_energy.data[pulse_energy.data == 1] = 0
            # Add units
            pulse_energy.attrs["units"] = self.instrument_source[key].units or "µJ"
            # Set a meaningful name
            pulse_energy.name = "Energy"

            self._pulse_energy[pg] = pulse_energy

        pulse_energy = self._pulse_energy[pg]
        if series:
            # Use .where() to convert 0's to nans, then convert to a Series,
            # then drop the nan entries.
            pulse_energy = pulse_energy.where(pulse_energy != 0).to_series().dropna()
            pulse_energy.attrs["units"] = self._pulse_energy[pg].attrs["units"]

        return pulse_energy

    def slow_train_energy(self, sase=None):
        """Return the slow train energy from the XGM in microjoules.

        This is an average pulse energy, averaged over all pulses for 10-20s.

        Args:
            sase (int): Same meaning as in
                [XGM.pulse_energy()][extra.components.XGM.pulse_energy].
        """
        pg = self._check_sase_arg(sase)
        if pg not in self._slow_train_energy:
            if pg == PropertyGroup.MAIN:
                key = "controlData.slowTrain"
            else:
                key = f"controlData.slowTrainSa{pg.value}"

            energy = self.control_source[key].xarray()
            energy.attrs["units"] = self.control_source[key].units or "µJ"

            self._slow_train_energy[pg] = energy

        return self._slow_train_energy[pg]

    def _get_main_nbunches_key(self):
        """Helper function to find the main key for the number of bunches.

        For historical reasons, this has differed in wording and preferred meal.
        """
        if self._main_nbunches_key is None:
            possible_keys = ["numberOfBunchesActual", "nummberOfBrunches", "numberOfBunches"]
            for key in possible_keys:
                if f"pulseEnergy.{key}" in self.control_source:
                    self._main_nbunches_key = f"pulseEnergy.{key}"
                    break

        return self._main_nbunches_key

    def npulses(self, sase=None) -> int:
        """The nominal number of pulses.

        This will throw an exception if the number of pulses is not constant.

        Args:
            sase (int): Same meaning as in
                [XGM.pulse_energy()][extra.components.XGM.pulse_energy].

        Raises:
            ValueError: Will be thrown if the number of pulses is not constant.
        """
        pg = self._check_sase_arg(sase)
        if pg not in self._npulses:
            pulse_counts = self.pulse_counts(sase=sase)
            if not np.allclose(pulse_counts[0], pulse_counts):
                raise ValueError("Number of pulses is changing, there is no nominal number.")

            self._npulses[pg] = int(pulse_counts[0])

        return self._npulses[pg]

    def pulse_counts(self, sase=None, force_slow_data=False):
        """Return a 1D [DataArray][xarray.DataArray] of the number of pulses in each train.

        Because the slow data `pulseEnergy.numberOf[SAx]BunchesActual` property
        can be unreliable this will always check the slow data counts against
        the counts in the fast data from
        [XGM.pulse_energy()][extra.components.XGM.pulse_energy] and return the
        fast data counts if there is a difference. This can be overridden by
        passing `force_slow_data=True`.

        Warning:
            Using `force_slow_data=True` can give unreliable results, only use
            it if you specifically want to find what numbers the XGM
            recorded. In general, prefer using something like the
            [XrayPulses][extra.components.XrayPulses] component to find the real
            number of pulses from the bunch pattern table.

        Args:
            sase (int): Same meaning as in
                [XGM.pulse_energy()][extra.components.XGM.pulse_energy].
        """
        import xarray as xr

        pg = self._check_sase_arg(sase)
        if pg not in self._pulse_counts:
            if pg == PropertyGroup.MAIN:
                key = self._get_main_nbunches_key()
            else:
                key = f"pulseEnergy.numberOfSa{pg.value}BunchesActual"
            slow_counts = self._control_source[key].xarray()
            self._slow_pulse_counts[pg] = slow_counts

            pulse_energy = self.pulse_energy(sase=sase)
            common_tids = np.intersect1d(pulse_energy.trainId, slow_counts.trainId)
            fast_counts = np.count_nonzero(pulse_energy.sel(trainId=common_tids), axis=1)
            fast_counts = xr.DataArray(fast_counts, dims=("trainId",),
                                       coords={"trainId": pulse_energy.trainId})

            counts_match = np.allclose(fast_counts, slow_counts.sel(trainId=common_tids))
            if not counts_match:
                warn(f"Slow data pulse counts ({key}) don't match the counts from fast data (data.intensityTD), data may be invalid!")
                self._pulse_counts[pg] = fast_counts
            else:
                self._pulse_counts[pg] = slow_counts

        return self._pulse_counts[pg] if not force_slow_data else self._slow_pulse_counts[pg]

    def max_npulses(self, sase=None) -> int:
        """The maximum number of pulses.

        Args:
            sase (int): Same meaning as in
                [XGM.pulse_energy()][extra.components.XGM.pulse_energy].
        """
        pg = self._check_sase_arg(sase)
        if pg not in self._max_pulses:
            self._max_pulses[pg] = int(self.pulse_counts(sase).max().item())

        return self._max_pulses[pg]

    def is_constant_pulse_count(self, sase=None) -> bool:
        """Return whether or not the number of pulses is constant.

        Args:
            sase (int): Same meaning as in
                [XGM.pulse_energy()][extra.components.XGM.pulse_energy].
        """
        try:
            self.npulses(sase)
            return True
        except ValueError:
            return False

    # TODO: ask XPD what this really is and what it should be named.
    # def photon_flux(self):
    #     """A 1D array of the photon flux in uJ for each train."""
    #     if self._photon_flux is None:
    #         self._photon_flux = self.control_source["pulseEnergy.photonFlux"].xarray()

    #     return self._photon_flux

    def _get_run_prefix(self):
        run_metadata_available = self._proposal != None
        return f"p{self._proposal}, r{self._run_no}: " if run_metadata_available else ""

    def _get_device_label(self, sase):
        if sase is None:
            sase = self._default_pg
        else:
            sase = PropertyGroup(sase)

        sase_label = f", SASE {sase.value}" if sase != PropertyGroup.MAIN else ""
        return f"{self.control_source.source}{sase_label}"

    def _set_plot_title(self, title, ax, sase, minimal):
        """Set a plot title with appropriate metadata."""
        if minimal:
            ax.set_title(title)
            return

        run_prefix = self._get_run_prefix()
        padding = 20 if run_prefix != "" else 0
        ax.set_title(f"{run_prefix}{title}", pad=padding)

        if run_prefix != "":
            device_label = self._get_device_label(sase)
            ax.text(0.5, 1.02, device_label,
                    horizontalalignment="center",
                    transform=ax.transAxes)

    def plot(self, sase=None, figsize=(9, 7)):
        """Plot an overview of data from the XGM.

        This combines [XGM.plot_pulse_energy()][extra.components.XGM.plot_pulse_energy],
        [XGM.plot_energy_per_pulse()][extra.components.XGM.plot_energy_per_pulse],
        and
        [XGM.plot_energy_per_train()][extra.components.XGM.plot_energy_per_train]
        in a single figure.

        Example plot:
        ![](../images/xgm-plot.png)

        Args:
            sase (int): Same meaning as in
                [XGM.pulse_energy()][extra.components.XGM.pulse_energy].
            figsize (tuple): The size of the [Figure][matplotlib.figure.Figure].
        """
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        fig = plt.figure(figsize=figsize)
        gs = GridSpec(2, 2, figure=fig)
        ax1 = fig.add_subplot(gs[0, :])
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[1, 1])

        self.plot_energy_per_train(sase=sase, ax=ax1, minimal_title=True)
        self.plot_pulse_energy(sase=sase, ax=ax3, minimal_title=True)
        self.plot_energy_per_pulse(sase=sase, ax=ax2, minimal_title=True)

        run_prefix = self._get_run_prefix()
        device_label = self._get_device_label(sase)
        fig.suptitle(f"{run_prefix}{device_label}")

        fig.tight_layout()

        return fig.get_axes()[0]

    def plot_pulse_energy(self, sase=None, ax=None, minimal_title=False):
        """Plot a heatmap of the pulse energies.

        Example plot:
        ![](../images/xgm-plot-pulse-energy.png)

        Args:
            sase (int): Same meaning as in
                [XGM.pulse_energy()][extra.components.XGM.pulse_energy].
            ax (matplotlib.axes.Axes): The axis to plot in. This will default to
                [plt][matplotlib.pyplot] if not provided.
            minimal_title (bool): Whether to include the proposal/run
                information and XGM device name in the title.
        """
        if ax is None:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(9, 5))
        else:
            fig = ax.get_figure()

        pulse_energy = self.pulse_energy(sase)

        from extra.utils import imshow2
        imshow2(pulse_energy, ax=ax)

        self._set_plot_title("XGM pulse energy heatmap", ax, sase, minimal_title)

        fig.tight_layout()

        return ax

    def plot_energy_per_pulse(self, sase=None, ax=None, minimal_title=False):
        """Plot the average pulse energy.

        Example plot:
        ![](../images/xgm-plot-energy-per-pulse.png)

        Args:
            sase (int): Same meaning as in
                [XGM.pulse_energy()][extra.components.XGM.pulse_energy].
            ax (matplotlib.axes.Axes): The axis to plot in. This will default to
                [plt][matplotlib.pyplot] if not provided.
            minimal_title (bool): Whether to include the proposal/run
                information and XGM device name in the title.
        """
        return self._plot_axis_energy("pulse", "train", sase, ax, fmt="o-",
                                         markersize=3, minimal_title=minimal_title)

    def plot_energy_per_train(self, sase=None, window_trains=None, ax=None, minimal_title=False):
        """Plot the average train energy.

        Example plot:
        ![](../images/xgm-plot-energy-per-train.png)

        Args:
            sase (int): Same meaning as in
                [XGM.pulse_energy()][extra.components.XGM.pulse_energy].
            window_trains (int): The number of trains to use when plotting the
                rolling average. By default this is chosen automatically based
                on the number of trains in the run.
            ax (matplotlib.axes.Axes): The axis to plot in. This will default to
                [plt][matplotlib.pyplot] if not provided.
            minimal_title (bool): Whether to include the proposal/run
                information and XGM device name in the title.
        """
        pulse_energy = self.pulse_energy(sase)

        if window_trains is None:
            # Select a number of trains such that the number of points
            # will be about 500. The goal is to smooth the data just enough
            # to see something visually useful no matter how long the run is.
            window_trains = max(1, int(pulse_energy.shape[0] / 500 * 10))

        ax = self._plot_axis_energy("train", "pulse", sase, ax, minimal_title=minimal_title)

        if window_trains > 1 and len(self._control_source.train_ids) > 300:
            rolling_mean = pulse_energy.mean(dim="pulseIndex").rolling(trainId=window_trains, center=True).mean()
            ax.plot(rolling_mean, label=f"{window_trains / 10:.1f}s rolling average", linewidth=2)

        ax.legend()

        return ax

    def _plot_axis_energy(self, axis_name, other_axis_name, sase, ax, fmt="-", minimal_title=False, **kwargs):
        if ax is None:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(9, 5))

        pulse_energy = self.pulse_energy(sase)

        get_dim = lambda x: "trainId" if x == "train" else "pulseIndex"
        axis_energy = pulse_energy.mean(dim=get_dim(other_axis_name))
        axis_std = pulse_energy.std(dim=get_dim(other_axis_name))
        xs = np.arange(len(axis_energy))

        ax.plot(axis_energy, fmt if len(xs) > 1 else "o", label="Raw data", **kwargs)

        if len(xs) > 1:
            ax.fill_between(xs,
                            axis_energy - axis_std, axis_energy + axis_std,
                            alpha=0.5)
        else:
            ax.errorbar(xs, axis_energy, yerr=axis_std, capsize=10, fmt="none")

        self._set_plot_title(f"Mean {axis_name} energy (averaged over {other_axis_name}s)",
                             ax, sase, minimal_title)
        ax.set_xlabel(axis_name.capitalize())
        ax.set_ylabel("Energy [μJ]")
        ax.grid()

        return ax

    def info(self):
        run_str = ""
        if self._proposal is not None:
            run_str = f" for p{self._proposal}, r{self._run_no}"

        try:
            doocs_server = self.doocs_server()
        except MultiRunError:
            doocs_server = "N/A"
        if doocs_server is None:
            doocs_server = "N/A"

        info_str = f"""
        {self.control_source.source} properties{run_str}:
          Avg. nominal wavelength:    {self.wavelength_by_train().mean().item():.3f} nm
          Avg. nominal photon energy: {self.photon_energy_by_train().mean().item():.2f} keV
          Avg. pulse energy:          {self.pulse_energy().mean().item():.2f} µJ

          Max pulses:                 {self.max_npulses()}
          Mean pulses:                {self.pulse_counts().mean().item():.3f}

          DOOCS server:               {doocs_server}
        """

        print(dedent(info_str))

    def __repr__(self):
        kev = self.photon_energy_by_train().mean().item()
        return f"<XGM for {self.control_source.source} with nominal mean photon energy {kev:.1f} keV>"
