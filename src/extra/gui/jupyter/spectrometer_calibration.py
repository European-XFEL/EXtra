import datetime
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path

import ipympl
import ipywidgets as ipw
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from extra_data import open_run
from extra_data.read_machinery import find_proposal
from IPython.display import clear_output, display

from extra.gui.widgets.peak_selection import PeakSelectorWidget
from extra.gui.widgets.roi_selection import ROISelectorWidget

__all__ = ['SpectrometerCalibration', 'plot_from_calibration_file']

log = logging.getLogger(__name__)


class CalibratedPlotter:
    """
    Plots one or more 1D projections (ROI lineouts) on a single graph,
    applying individual linear energy calibrations to each projection's x-axis.
    """

    @staticmethod
    def pixel_to_energy(pixel_indices, slope, intercept):
        """
        Converts an array of pixel indices to energies using specific calibration parameters.

        Args:
            pixel_indices (np.ndarray): A 1D array of pixel indices.
            slope (float): The slope of the linear calibration (e.g., eV/pixel).
            intercept (float): The intercept of the linear calibration (e.g., eV at pixel 0).

        Returns:
            np.ndarray: A 1D array of corresponding energies in eV.
        """
        return slope * pixel_indices + intercept

    def plot_calibrated_projections(
        self,
        projections,
        calibrations,
        labels=None,
        title="Calibrated ROI Projections",
        xlabel="Energy (eV)",
        ylabel="Integrated Intensity (Arb. Units)",
    ):
        """
        Plots multiple 1D projections with individual energy calibrations.

        Args:
            projections (list[np.ndarray]):
                A list/tuple of 1D numpy arrays (the ROI lineouts/projections).
            calibrations (list[dict]):
                A list/tuple of dictionaries, corresponding to the 'projections'.
                Each dict must contain 'slope' and 'intercept' keys for that projection.
                Example: [{'slope': -0.1, 'intercept': 7100}, {'slope': -0.11, 'intercept': 7105}, ...]
            labels (list[str], optional):
                A list of labels for the legend, corresponding to the projections.
                If None, default labels ('Projection 0', 'Projection 1', ...) are used. Defaults to None.
            title (str, optional): Title for the plot. Defaults to "Calibrated ROI Projections".
            xlabel (str, optional): Label for the x-axis. Defaults to "Energy (eV)".
            ylabel (str, optional): Label for the y-axis. Defaults to "Integrated Intensity (Arb. Units)".

        Returns:
            tuple: (matplotlib.figure.Figure, matplotlib.axes._axes.Axes) The figure and axes objects.
                   Returns (None, None) if inputs are invalid or no data is plotted.
        """
        if not isinstance(projections, (list, tuple)) or not isinstance(
            calibrations, (list, tuple)
        ):
            raise TypeError("'projections' and 'calibrations' must be lists or tuples.")
        if len(projections) != len(calibrations):
            raise ValueError("Length of 'projections' and 'calibrations' must match.")
        if labels is not None and len(projections) != len(labels):
            raise ValueError(
                "If 'labels' are provided, length must match 'projections'."
            )
        if not all(isinstance(p, np.ndarray) and p.ndim == 1 for p in projections):
            log.debug(projections)
            raise TypeError("All items in 'projections' must be 1D numpy arrays.")
        if not all(
            isinstance(c, dict) and "slope" in c and "intercept" in c
            for c in calibrations
        ):
            raise TypeError(
                "All items in 'calibrations' must be dicts with 'slope' and 'intercept' keys."
            )

        if not projections:
            log.warning("No projections provided to plot.")
            return None, None

        # Create Plot
        fig, ax = plt.subplots(figsize=(8, 5))
        plotted_something = False

        for i, projection in enumerate(projections):
            if projection.size == 0:
                log.warning(f"Projection {i} is empty, skipping.")
                continue

            calib = calibrations[i]
            slope = calib["slope"]
            intercept = calib["intercept"]

            pixel_axis = np.arange(projection.shape[0])
            energy_axis = self.pixel_to_energy(pixel_axis, slope, intercept)
            plot_label = labels[i] if labels is not None else f"Projection {i}"

            ax.plot(energy_axis, projection, label=plot_label)
            plotted_something = True

        if not plotted_something:
            log.warning("No valid projections were plotted.")
            plt.close(fig)
            return None, None

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, linestyle=":")
        ax.legend()

        plt.tight_layout()
        return fig, ax


class SpectrometerCalibration:
    """
    An interactive Jupyter widget for performing energy calibration on 2D X-ray
    spectrometer detector data.

    This tool provides an interface within Jupyter to guide the user through:

    - Selecting regions of interest (ROIs)
    - Identifying peaks within the 1D projections of those ROIs
    - Providing reference energies
    - Performing linear calibration fits for each ROI
    - Plotting the calibrated spectra
    - Saving the results

    # Usage

    Provide either:

    - A `proposal`, `run_number` and  data `source`: the data will be a
        average image of the run.
    - a 2D numpy array as `image_data`

    `image_data` will be used first if provided. If data is loaded from a run,
    the generated average will be cached at
    `${Proposal}/scratch/.EXtra-gui-jupyter-SpectrometerCalibration-cache`

    ```python
    # Ensure the ipympl backend is active for interactive plots
    %matplotlib widget

    import numpy as np
    from extra.gui.jupyter import SpectrometerCalibration

    # Create an instance of the widget.
    # You can either provide data from an EuXFEL run
    widget = SpectrometerCalibration(
        proposal=1234, # Your proposal number
        run=10,        # Your run number
        source=('SQS_NQS_PNCCD1MP/CAL/PNCCD_FOC_SUM', 'image.data')  # Example source
    )
    # or a pre-loaded NumPy array
    my_xes_image = np.load("my_data.npy")
    widget = SpectrometerCalibration(image_data=my_xes_image)

    # Display the widget in your Jupyter cell
    widget.display()

    # Use the tabs in the displayed widget to perform calibration

    # Save results to a file
    widget.save_results()

    # When finished, close the widget's figures
    widget.close_all()
    ```

    # Widget interaction
    ## Tab 1: ROI selection

    - Click and drag vertically on the image to define rectangular ROIs.
    - To remove an ROI, click an existing ROI to select it (turns red), then use
      the "Delete Selected ROI" button.

    ![](images/spectrometer-calibration-widget-roi-selection.png)

    ## Tab 2: Peak selection

    - View the 1D projections for each ROI.
    - Left-click on a trace to add a peak marker.
    - Left-click and drag an existing marker line to move it.
    - Right-click near a marker line to delete it.

    ![](images/spectrometer-calibration-widget-peak-selection.png)

    ## Tab 3: Calibration

    - Review selected peak pixel positions.
    - Enter known reference energies (in eV) for each *peak index* ("Peak
      1", "Peak 2", etc.).
    - Click "Calibrate per ROI". Results (slope, intercept per ROI) will appear.
    - If calibration succeeds, "Plot Calibrated ROIs" and "Save Results" buttons
      become active.
        - Click "Plot Calibrated ROIs" to view the energy-calibrated spectra.
        - Click "Save Results" to save all data (ROIs, peaks, fits, spectra) to
          a `.txt` file (e.g.,
          `spectrometer_calibration_results_YYYY-MM-DD_HH-MM-SS.txt`).

    ![](images/spectrometer-calibration-widget-results.png)

    """

    def __init__(
        self,
        proposal: int = None,
        run: int = None,
        source: tuple[str, str] = None,
        image_data: np.ndarray = None,
        *,
        use_cache: bool = True,
    ):
        """

        Args:
            proposal (int): Proposal number
            run (int): run number
            source (tuple[str, str]): data source
            image_data (np.ndarray): 2D array
            use_cache (bool): try loading data from cache if True (default).
              If set to False, data will neither be loaded from nor written to cache.
        """
        self.image_data = image_data
        if image_data is None:
            self.image_data = self._load_data(proposal, run, source, use_cache)

        self.processed_image_data = self.image_data.copy()
        self.roi_widget_instance = None
        self.peak_widget_instance = None
        self.peak_index_energy_inputs = {}
        self.calibration_results = {}  # {roi_idx: {'slope': float, 'intercept': float}}
        self.plotter = CalibratedPlotter()

        # Output widgets
        self.output_tab1 = ipw.Output(
            layout={"border": "1px solid black", "min_height": "650px"}
        )
        self.output_tab2 = ipw.Output(
            layout={"border": "1px solid black", "min_height": "650px"}
        )
        self.output_tab3 = ipw.Output(
            layout={
                "border": "1px solid black",
                "min_height": "650px",
                "padding": "10px",
            }
        )

        # Widgets for Calibration Tab (Tab 3)
        self.peak_display_area = ipw.HTML(
            value="<i>Select ROIs (Tab 1) and Peaks (Tab 2)</i>"
        )
        self.energy_input_container = ipw.VBox([])
        self.energy_input_header = ipw.HTML(
            "<hr><b>Reference Energies (per Peak Index):</b><br><i>(Applies to the Nth peak in every ROI)</i>"
        )
        self.calibrate_button = ipw.Button(
            description="Calibrate per ROI",
            button_style="info",
            icon="cogs",
            layout={"margin": "10px 0 0 0"},
        )
        self.calibration_output = ipw.Output(
            layout={
                "border": "1px solid lightgray",
                "padding": "5px",
                "margin_top": "10px",
                "min_height": "200px",
            }
        )  # Min height for text/plot
        self.plot_calibrated_button = ipw.Button(
            description="Plot Calibrated ROIs",
            button_style="success",
            icon="line-chart",
            layout={"margin": "10px 0 0 0"},
            disabled=True,
        )
        self.save_results_button = ipw.Button(
            description="Save Results",
            button_style="primary",
            icon="save",
            layout={"margin": "10px 0 0 0"},
            disabled=True,
        )
        # Container for the lower buttons
        self.button_row = ipw.HBox(
            [self.plot_calibrated_button, self.save_results_button]
        )

        # Arrange Tab 3 content
        self.tab3_content = ipw.VBox(
            [
                ipw.HTML("<b>Selected Peak Positions (by ROI):</b>"),
                self.peak_display_area,
                self.energy_input_header,
                self.energy_input_container,
                self.calibrate_button,
                ipw.HTML("<hr><b>Calibration Result (per ROI):</b>"),
                self.calibration_output,
                self.button_row,
            ]
        )

        # Tab Widget
        self.tab_widget = ipw.Tab(
            children=[self.output_tab1, self.output_tab2, self.output_tab3]
        )
        self.tab_widget.set_title(0, "1. ROI Selection")
        self.tab_widget.set_title(1, "2. Peak Selection")
        self.tab_widget.set_title(2, "3. Calibration")

        # Connect Buttons
        self.calibrate_button.on_click(self._on_calibrate_button_clicked)
        self.plot_calibrated_button.on_click(self._on_plot_calibrated_clicked)
        self.save_results_button.on_click(self._on_save_results_clicked)

        # Initialize
        self._init_roi_widget()
        self._init_peak_widget([])
        self._init_calibration_tab()

    @staticmethod
    def _load_data(proposal: int, run: int, source: tuple[str, str], use_cache: bool):
        # define cache location
        cache_path = (
            Path(find_proposal(f"p{proposal:06}"))
            / "scratch/.EXtra-gui-jupyter-SpectrometerCalibration-cache"
        )
        str_src = re.sub(r"[^a-zA-Z0-9_-]", "-", "-".join(source))
        fname = f"spec-p{proposal:06}-r{run:04}-{str_src}-mean.npy"

        # load, if cache file exist
        if use_cache and (cache_path / fname).is_file():
            return np.load(cache_path / fname)

        kd = open_run(proposal, run, data="all")[source]
        data = kd.xarray().squeeze()
        mean_data = data.mean(dim="trainId")

        # cache data for faster loading unless use_cache=False
        if use_cache:
            cache_path.mkdir(mode=666, exist_ok=True, parents=True)
            np.save(cache_path / fname, mean_data)

        return mean_data

    # Widget Initialization Methods
    def _init_roi_widget(self):
        with self.output_tab1:
            clear_output(wait=True)
            try:
                if (
                    self.roi_widget_instance
                    and hasattr(self.roi_widget_instance, "fig")
                    and plt.fignum_exists(self.roi_widget_instance.fig.number)
                ):
                    self.roi_widget_instance.close()

                with plt.ioff():
                    self.roi_widget_instance = ROISelectorWidget(self.image_data)

                self.roi_widget_instance.register_roi_update_callback(self._roi_updated)
                display(self.roi_widget_instance.fig.canvas)
                log.debug("ROI Selector Initialized in Tab 1.")
            except Exception:
                log.exception(f"Error initializing ROI Selector")

    def _init_peak_widget(self, rois):
        with self.output_tab2:
            clear_output(wait=True)
            try:
                if (
                    self.peak_widget_instance
                    and hasattr(self.peak_widget_instance, "fig")
                    and plt.fignum_exists(self.peak_widget_instance.fig.number)
                ):
                    self.peak_widget_instance.close()

                if not rois:
                    log.warning("Peak Selector: No ROIs defined yet.")
                    self.peak_widget_instance = None
                    self._update_calibration_tab_display()
                    return

                log.debug(f"Initializing Peak Selector with {len(rois)} ROIs.")
                with plt.ioff():
                    self.peak_widget_instance = PeakSelectorWidget(
                        self.processed_image_data, rois
                    )

                self.peak_widget_instance.register_peak_update_callback(
                    self._update_calibration_tab_display
                )
                display(self.peak_widget_instance.fig.canvas)
                log.debug("Peak Selector Initialized in Tab 2.")
                self._update_calibration_tab_display()

            except Exception:
                log.exception(f"Error initializing Peak Selector")
                self.peak_widget_instance = None
                self._update_calibration_tab_display()

    def _init_calibration_tab(self):
        with self.output_tab3:
            clear_output(wait=True)
            display(self.tab3_content)
            self._update_calibration_tab_display()

    # Update Logic
    def _roi_updated(self):
        log.debug("ROI update detected. Re-initializing Peak Selector...")
        if self.roi_widget_instance:
            current_rois = self.roi_widget_instance.get_rois()
            # Update the processed image to reflect any flips in the ROI widget
            self.processed_image_data = (
                self.roi_widget_instance.get_current_image_data()
            )
            self._init_peak_widget(current_rois)
        else:
            log.warning("ROI widget instance not found.")

    def _update_calibration_tab_display(self):
        """Updates peak display and generates energy inputs based on MAX peak index."""
        log.debug("Updating calibration tab display...")

        # Clear previous dynamic widgets and data
        self.energy_input_container.children = []
        self.peak_index_energy_inputs.clear()
        self.calibration_results.clear()  # Clear old calibration results too
        # Disable plot button until calibration is done
        self.plot_calibrated_button.disabled = True

        with self.calibration_output:  # Clear output display
            clear_output()
            print(
                "Enter reference energies for peak indices and click 'Calibrate per ROI'."
            )

        new_energy_widgets = []
        max_peak_count = 0  # Find the max number of peaks selected in any ROI

        # Update Peak Display Area & Find Max Peak Count
        if self.peak_widget_instance:
            peak_data = self.peak_widget_instance.get_selected_peaks()
            if not peak_data:
                html_content = "<i>No ROIs available or peaks selected yet.</i>"
            else:
                html_content = "<ul>"
                for roi_info in peak_data:
                    roi_idx = roi_info["roi_index"]
                    num_peaks_in_roi = len(roi_info["peaks"])
                    if num_peaks_in_roi > max_peak_count:
                        max_peak_count = num_peaks_in_roi  # Update max

                    peaks_str = ", ".join([f"{p:.2f}" for p in roi_info["peaks"]])
                    if not peaks_str:
                        peaks_str = "<i>None</i>"
                    html_content += f"<li><b>ROI {roi_idx}:</b> Peaks at Pixels = [{peaks_str}]</li>"
                html_content += "</ul>"
            self.peak_display_area.value = html_content
        else:
            self.peak_display_area.value = "<i>Peak selector not initialized.</i>"
            max_peak_count = 0  # Ensure it's 0 if no peak widget

        # Create Dynamic Energy Input Widgets based on Max Peak Count
        if max_peak_count > 0:
            for i in range(1, max_peak_count + 1):
                peak_index_label = ipw.Label(
                    f"Peak {i} Energy (eV):",
                    layout={"width": "auto", "margin": "0 5px 0 0"},
                )
                energy_input = ipw.FloatText(
                    value=None,
                    description="",
                    disabled=False,
                    layout={"width": "150px"},
                )
                self.peak_index_energy_inputs[i] = energy_input
                hbox = ipw.HBox(
                    [peak_index_label, energy_input], layout={"margin": "2px 0"}
                )
                new_energy_widgets.append(hbox)

        # Set the children of the container VBox
        self.energy_input_container.children = tuple(new_energy_widgets)

        # Ensure plot and save buttons are disabled initially or on reset
        self.plot_calibrated_button.disabled = True
        self.save_results_button.disabled = True

    # Calibration Logic
    def _on_calibrate_button_clicked(self, b):
        """Performs linear calibration for EACH ROI based on shared peak index energies."""
        with self.calibration_output:
            clear_output(wait=True)
            print("Performing calibration for each ROI...")
            self.calibration_results.clear()
            self.plot_calibrated_button.disabled = True

            # 1. Get data from Peak Selector
            if not self.peak_widget_instance:
                print("Error: Peak Selector widget is not available.")
                return
            selected_peaks_data = self.peak_widget_instance.get_selected_peaks()
            if not selected_peaks_data:
                print("Error: No ROIs/Peaks found in Peak Selector.")
                return

            # 2. Get entered reference energies (keyed by peak index)
            reference_energies = {
                idx: widget.value
                for idx, widget in self.peak_index_energy_inputs.items()
                if widget.value is not None
            }
            if not reference_energies:
                print("Error: No reference energies entered for any peak index.")
                return
            print(f"\nUsing reference energies: {reference_energies}")

            # Disable buttons during calibration
            self.plot_calibrated_button.disabled = True
            self.save_results_button.disabled = True

            # 3. Iterate through each ROI and perform individual fits
            calibration_successful_count = 0
            roi_indices_with_results = []  # Keep track of successful ROIs
            for roi_info in selected_peaks_data:
                roi_idx = roi_info["roi_index"]
                print(f"\n--- Processing ROI {roi_idx} ---")
                roi_pixels, roi_energies, points_used_roi = [], [], 0

                # Collect valid (pixel, energy) pairs for *this* ROI
                for peak_list_idx, pixel_value in enumerate(roi_info["peaks"]):
                    peak_index = peak_list_idx + 1  # 1-based index
                    if peak_index in reference_energies:
                        ref_energy = reference_energies[peak_index]
                        roi_pixels.append(pixel_value)
                        roi_energies.append(ref_energy)
                        points_used_roi += 1

                if points_used_roi < 2:
                    print(
                        f"Calibration Failed for ROI {roi_idx}: Need >= 2 points (found {points_used_roi})."
                    )
                    continue

                # Perform linear fit for *this* ROI
                try:
                    pixels_np, energies_np = (
                        np.array(roi_pixels),
                        np.array(roi_energies),
                    )
                    slope, intercept = np.polyfit(pixels_np, energies_np, 1)

                    # Store results
                    self.calibration_results[roi_idx] = {
                        "slope": slope,
                        "intercept": intercept,
                    }
                    calibration_successful_count += 1
                    roi_indices_with_results.append(roi_idx)

                    # Display results
                    print(f"  Calibration Success for ROI {roi_idx}:")
                    print(f"    Fit: Energy = ({slope:.4f} * Pixel) + {intercept:.2f}")

                except Exception as e:
                    print(f"  Error during calibration fitting for ROI {roi_idx}: {e}")

            print("\n--- Calibration Summary ---")
            print(
                f"Successfully calibrated {calibration_successful_count} out of {len(selected_peaks_data)} ROIs."
            )
            if self.calibration_results:
                print(
                    f"Successfully calibrated ROIs: {sorted(roi_indices_with_results)}"
                )
                self.plot_calibrated_button.disabled = False
                self.save_results_button.disabled = False
            else:
                print("No ROIs were successfully calibrated.")

    def _prepare_spectra_data(self):
        """Helper function to prepare calibrated spectra data.

        Returns:
            tuple: (list_of_energy_axes, list_of_projections, list_of_labels, list_of_roi_indices)
                   Returns empty lists if data cannot be prepared.
        """
        if not self.calibration_results:
            log.debug("Save/Plot: No successful calibration results available.")
            return [], [], [], []
        if not self.roi_widget_instance:
            log.debug(
                "Save/Plot: ROI definitions not available (ROI widget missing)."
            )
            return [], [], [], []
        if self.processed_image_data is None:
            log.debug("Save/Plot: Image data is missing.")
            return [], [], [], []

        all_energy_axes = []
        all_projections = []
        all_labels = []
        all_roi_indices = []  # Keep track of which ROI index corresponds to which spectrum

        all_roi_defs = self.roi_widget_instance.get_rois()
        img_height, img_width = self.processed_image_data.shape
        pixel_axis = np.arange(img_width)
        roi_defs_dict = {
            roi.get("roi_index"): roi
            for roi in all_roi_defs
            if roi.get("roi_index") is not None
        }

        for roi_idx, calib_params in sorted(
            self.calibration_results.items()
        ):  # Sort for consistent order
            roi_def = roi_defs_dict.get(roi_idx)
            if not roi_def:
                log.debug(
                    f"Save/Plot: Could not find ROI definition for calibrated ROI {roi_idx}."
                )
                continue

            y_start = int(max(0, np.floor(roi_def["y_start"])))
            y_end = int(min(img_height, np.ceil(roi_def["y_end"])))

            if y_end <= y_start:
                log.debug(
                    f"Save/Plot: ROI {roi_idx} has invalid height ({y_start}-{y_end})."
                )
                continue

            projection = np.sum(self.processed_image_data[y_start:y_end, :], axis=0)
            energy_axis = CalibratedPlotter.pixel_to_energy(
                pixel_axis, calib_params["slope"], calib_params["intercept"]
            )

            all_energy_axes.append(energy_axis)
            all_projections.append(projection)
            all_labels.append(f"ROI {roi_idx}")
            all_roi_indices.append(roi_idx)

        return all_energy_axes, all_projections, all_labels, all_roi_indices

    # Plotting Logic
    def _on_plot_calibrated_clicked(self, b):
        """Gathers data and calls the plotter to display calibrated ROI projections."""
        with self.calibration_output:
            clear_output(wait=True)
            print("Generating plot of calibrated ROI projections...")

            energy_axes, projections, labels, _ = self._prepare_spectra_data()

            if not projections:
                print(
                    "Error: No valid projections could be generated for calibrated ROIs."
                )
                return

            # Prepare calibration dicts for the plotter
            calibrations_for_plotter = []
            # Need to get the roi_indices again to map to calibration_results
            for label_str in labels:
                try:
                    roi_idx_from_label = int(label_str.split("ROI ")[1])
                    if roi_idx_from_label in self.calibration_results:
                        calibrations_for_plotter.append(
                            self.calibration_results[roi_idx_from_label]
                        )
                    else:
                        print(
                            f"Warning: Could not find calibration for {label_str} during plotting prep."
                        )
                        calibrations_for_plotter.append(
                            {"slope": 1, "intercept": 0}
                        )  # Dummy
                except Exception:
                    print(
                        f"Warning: Could not parse ROI index from label '{label_str}' for plotting."
                    )
                    calibrations_for_plotter.append(
                        {"slope": 1, "intercept": 0}
                    )  # Dummy

            try:
                fig, ax = self.plotter.plot_calibrated_projections(
                    projections=projections,
                    calibrations=calibrations_for_plotter,
                    labels=labels,
                )
                if fig:
                    display(fig.canvas)
                else:
                    print("Plot generation failed (plotter returned None).")
            except Exception as e:
                print(f"Error during plotting: {e}")
                import traceback

                traceback.print_exc()

    def _on_save_results_clicked(self, b):
        """Saves the current ROI, peak, energy, calibration data, and calibrated spectra to a text file."""
        now = datetime.datetime.now().isoformat(sep="_", timespec="seconds")
        default_filename = f"spectrometer_calibration_results_{now}.txt"

        # Gather Data
        # 1. ROI Definitions
        rois = []
        if self.roi_widget_instance:
            try:
                rois = self.roi_widget_instance.get_rois()
            except Exception:
                log.exception(f"Could not get ROI definitions")
        else:
            log.error("ROI widget instance not available.")

        # 2. Peak Selections
        peaks_data_raw = []
        if self.peak_widget_instance:
            try:
                peaks_data_raw = self.peak_widget_instance.get_selected_peaks()
            except Exception:
                log.exception(f"Could not get peak selections")
        else:
            log.error("Peak widget instance not available.")

        # 3. Entered Reference Energies
        ref_energies_input = self.get_entered_peak_index_energies()

        # 4. Calibration Results
        calib_fits = self.get_calibration_results()

        # Format Basic Info
        lines = []
        lines.append("# === XES Calibration Widget Results ===")
        lines.append(f"# Saved on: {now}")
        lines.append("-" * 35)

        # ROI Definitions
        lines.append("[ROI Definitions]")
        if rois:
            for roi in sorted(rois, key=lambda x: x.get("roi_index", -1)):
                idx = roi.get("roi_index", "N/A")
                ystart = roi.get("y_start", "N/A")
                yend = roi.get("y_end", "N/A")
                try:
                    ystart_str = f"{ystart:.2f}"
                except TypeError:
                    ystart_str = str(ystart)
                try:
                    yend_str = f"{yend:.2f}"
                except TypeError:
                    yend_str = str(yend)
                lines.append(f"ROI {idx}: y_start = {ystart_str}, y_end = {yend_str}")
        else:
            lines.append("No ROI definitions available.")
        lines.append("-" * 35)

        # Peak Selections (per ROI)
        lines.append("[Peak Selections (Pixels)]")
        if peaks_data_raw:
            for roi_peaks in sorted(
                peaks_data_raw, key=lambda x: x.get("roi_index", -1)
            ):
                idx = roi_peaks.get("roi_index", "N/A")
                pixel_values = roi_peaks.get("peaks", [])
                pixels_str = (
                    ", ".join([f"{p:.3f}" for p in pixel_values])
                    if pixel_values
                    else "None"
                )
                lines.append(f"ROI {idx}: [{pixels_str}]")
        else:
            lines.append("No peak selections available.")
        lines.append("-" * 35)

        # Reference Energies (per Peak Index)
        lines.append("[Reference Energies Entered (eV)]")
        if ref_energies_input:
            for peak_idx in sorted(ref_energies_input.keys()):
                energy = ref_energies_input[peak_idx]
                lines.append(f"Peak Index {peak_idx}: {energy:.4f}")
        else:
            lines.append("No reference energies were entered or available.")
        lines.append("-" * 35)

        # Calibration Results (per ROI)
        lines.append("[Calibration Fit Results (per ROI)]")
        lines.append("# Format: Energy = slope * Pixel + intercept")
        if calib_fits:
            points_per_roi = defaultdict(list)
            if peaks_data_raw and ref_energies_input:
                for roi_peaks_data in peaks_data_raw:
                    roi_idx_calib = roi_peaks_data.get("roi_index")
                    if roi_idx_calib is None:
                        continue
                    for peak_list_idx, pixel_val in enumerate(
                        roi_peaks_data.get("peaks", [])
                    ):
                        peak_idx_calib = peak_list_idx + 1
                        if peak_idx_calib in ref_energies_input:
                            points_per_roi[roi_idx_calib].append(
                                f"(Pix={pixel_val:.2f}, E={ref_energies_input[peak_idx_calib]:.2f})"
                            )
            for roi_idx_calib in sorted(calib_fits.keys()):
                results = calib_fits[roi_idx_calib]
                slope, intercept = results["slope"], results["intercept"]
                points_str = ", ".join(points_per_roi.get(roi_idx_calib, ["N/A"]))
                lines.append(
                    f"ROI {roi_idx_calib}: slope = {slope:.6f}, intercept = {intercept:.4f}"
                )
                lines.append(f"  - Points Used: {points_str}")
        else:
            lines.append("No successful calibration results available.")
        lines.append("-" * 35)

        # Calibrated Spectra Data (CSV Format) using helper
        lines.append("[Calibrated Spectra Data (CSV Format)]")

        all_energy_axes, all_projections, _, all_roi_indices_spectra = (
            self._prepare_spectra_data()
        )

        if all_projections:
            # Determine a common energy grid for all spectra to align them in the CSV
            min_e_global, max_e_global = np.inf, -np.inf
            max_len = 0
            for e_axis in all_energy_axes:
                if len(e_axis) > 0:
                    min_e_global = min(min_e_global, np.min(e_axis))
                    max_e_global = max(max_e_global, np.max(e_axis))
                    max_len = max(max_len, len(e_axis))

            if np.isfinite(min_e_global) and np.isfinite(max_e_global) and max_len > 0:
                # Create a common, high-resolution energy grid
                num_points_common_grid = max(max_len, 1000)
                common_energy_grid = np.linspace(
                    min_e_global, max_e_global, num_points_common_grid
                )

                spectra_df_data = {"Energy_eV": common_energy_grid}

                for i, original_roi_idx in enumerate(all_roi_indices_spectra):
                    current_energy_axis = all_energy_axes[i]
                    current_projection = all_projections[i]

                    # Interpolate projection onto the common energy grid
                    # Ensure energy axes are monotonic for interpolation
                    sort_indices = np.argsort(current_energy_axis)
                    sorted_energy = current_energy_axis[sort_indices]
                    sorted_projection = current_projection[sort_indices]

                    unique_energies, unique_indices_interp = np.unique(
                        sorted_energy, return_index=True
                    )
                    if len(unique_energies) < 2:
                        log.warning(
                            f"Not enough unique energy points for ROI {original_roi_idx} to interpolate for saving. Skipping spectrum."
                        )
                        # Fill with NaNs if skipping or add a placeholder
                        spectra_df_data[f"Intensity_ROI_{original_roi_idx}"] = (
                            np.full_like(common_energy_grid, np.nan)
                        )
                        continue

                    interp_projection = np.interp(
                        common_energy_grid,
                        unique_energies,
                        sorted_projection[unique_indices_interp],
                        left=np.nan,
                        right=np.nan,
                    )
                    spectra_df_data[f"Intensity_ROI_{original_roi_idx}"] = (
                        interp_projection
                    )

                try:
                    df_spectra = pd.DataFrame(spectra_df_data)
                    float_format_str = "%.4f"
                    csv_lines = df_spectra.to_csv(
                        index=False, na_rep="NaN", float_format=float_format_str
                    ).splitlines()
                    lines.extend(csv_lines)
                except Exception as e_df:
                    lines.append(f"# Error creating/saving spectra DataFrame: {e_df}")
            else:
                lines.append(
                    "# Could not determine a common energy grid or no valid spectra to save."
                )
        else:
            lines.append("No calibrated spectra data could be prepared for saving.")
        lines.append("# === End of Results ===")

        # Write to File
        try:
            # Using ipywidgets does not give direct access to a file dialog easily.
            # We will save directly to the specified filename in the kernel's current working directory.
            # Inform the user where the file is saved.
            output_path = default_filename
            with open(output_path, "w") as f:
                for line in lines:
                    f.write(line + "\n")

            with self.calibration_output:
                print("\n--- Results Saved ---")
                print(
                    f"Calibration data and spectra saved successfully to:\n{output_path}"
                )
        except IOError as e:
            with self.calibration_output:
                print("\n--- Error Saving Results ---")
                print(f"Could not write to file '{output_path}': {e}")
        except Exception as e:
            with self.calibration_output:
                print("\n--- Error Saving Results ---")
                print(f"An unexpected error occurred during saving: {e}")
                import traceback

                traceback.print_exc(file=sys.stdout)

    # Getter Methods
    def display(self):
        """Display the widget."""
        display(self.tab_widget)

    def get_selected_peak_data(self) -> list[dict[str, any]] | None:
        """Get the selected peak data.

        Returns:
            list: Selected peak data.
        """
        return (
            self.peak_widget_instance.get_selected_peaks()
            if self.peak_widget_instance
            else None
        )

    def get_entered_peak_index_energies(self) -> dict[int, float]:
        """Get the entered peak index energies.

        Returns:
            dict: Entered peak index energies.
        """
        return {
            idx: w.value
            for idx, w in self.peak_index_energy_inputs.items()
            if w.value is not None
        }

    def get_calibration_results(self) -> dict[int, dict[str, float]]:
        """Get the calibration results.

        Returns:
            dict: Calibration results.
        """
        return self.calibration_results

    def get_calibrated_plotter_for_roi(self, roi_index) -> CalibratedPlotter | None:
        """Get the calibrated plotter for a specific ROI.

        Args:
            roi_index (int): The index of the ROI.

        Returns:
            CalibratedPlotter: The calibrated plotter for the ROI.
        """
        result = self.calibration_results.get(roi_index)
        return (
            CalibratedPlotter(slope=result["slope"], intercept=result["intercept"])
            if result
            else None
        )

    # Cleanup
    def close_all(self):
        """Close all widget figures."""
        log.debug("Closing all widget figures...")
        if (
            self.roi_widget_instance
            and hasattr(self.roi_widget_instance, "fig")
            and plt.fignum_exists(self.roi_widget_instance.fig.number)
        ):
            self.roi_widget_instance.close()
        if (
            self.peak_widget_instance
            and hasattr(self.peak_widget_instance, "fig")
            and plt.fignum_exists(self.peak_widget_instance.fig.number)
        ):
            self.peak_widget_instance.close()
        self.roi_widget_instance = None
        self.peak_widget_instance = None
        self.calibration_results.clear()
        self.output_tab1.clear_output()
        self.output_tab2.clear_output()
        self.output_tab3.clear_output(wait=True)
        log.debug("Widgets closed and outputs cleared.")


def plot_from_calibration_file(filepath):
    """
    Reads the output file from the SpectrometerCalibration widget and plots the
    calibrated spectra.

    This function looks for the "[Calibrated Spectra Data (CSV Format)]" section
    and uses pandas to parse the subsequent data. It also reads the
    calibration fit results to display them in the plot legend.

    Args:
        filepath (str): The path to the calibration results text file.

    Returns:
        tuple: (matplotlib.figure.Figure, matplotlib.axes._axes.Axes)
               The figure and axes objects of the generated plot.
               Returns (None, None) if the file cannot be read or parsed.
    """
    try:
        with open(filepath, "r") as f:
            lines = f.readlines()
    except FileNotFoundError:
        log.error(f"File not found at '{filepath}'")
        return None, None
    except Exception:
        log.exception()
        return None, None

    # Find the start of the CSV data section
    try:
        # Find the line number where the CSV data starts
        # The actual data begins on the line AFTER the header.
        csv_start_line = (
            next(
                i
                for i, line in enumerate(lines)
                if "[Calibrated Spectra Data (CSV Format)]" in line
            )
            + 1
        )
    except StopIteration:
        log.error(
            "Could not find the '[Calibrated Spectra Data (CSV Format)]' section in the file."
        )
        return None, None

    # Use pandas to read the CSV data from that specific point
    from io import StringIO

    csv_string_block = "".join(lines[csv_start_line:])
    end_marker = "# === End of Results ==="
    if end_marker in csv_string_block:
        csv_string_block = csv_string_block.split(end_marker)[0].strip()

    # Read the block into a pandas DataFrame
    try:
        df = pd.read_csv(StringIO(csv_string_block))
    except pd.errors.EmptyDataError:
        log.error("The CSV data section in the file is empty.")
        return None, None
    except Exception:
        log.exception(f"Error parsing CSV data with pandas")
        return None, None

    # Extract Calibration Fit Info for Legend
    fit_info = {}
    try:
        fit_start_line = next(
            i
            for i, line in enumerate(lines)
            if "[Calibration Fit Results (per ROI)]" in line
        )
        for line in lines[fit_start_line:]:
            if line.strip().startswith("ROI"):
                match = re.search(
                    r"ROI\s*(\d+):\s*slope\s*=\s*(-?\d+\.\d+),\s*intercept\s*=\s*(-?\d+\.\d+)",
                    line,
                )
                if match:
                    roi_idx, slope, intercept = match.groups()
                    fit_info[int(roi_idx)] = (float(slope), float(intercept))
    except (StopIteration, ValueError, IndexError):
        log.warning("Could not parse calibration fit results for the legend.")

    # Create Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # The first column is assumed to be the energy axis
    x_col_name = df.columns[0]
    if "energy" not in x_col_name.lower():
        log.warning(
            f"The first column is named '{x_col_name}', assuming it is the energy axis."
        )

    energy_axis = df[x_col_name]

    # Iterate through the other columns (intensities)
    plotted_something = False
    for col_name in df.columns[1:]:
        if "intensity" in col_name.lower():
            intensity_data = df[col_name]

            # Construct the label
            label = col_name
            try:
                # extract ROI index from column name like 'Intensity_ROI_5'
                roi_idx = int(re.search(r"\d+", col_name).group())
                if roi_idx in fit_info:
                    slope, intercept = fit_info[roi_idx]
                    label = f"ROI {roi_idx} (slope={slope:.4f})"
            except (AttributeError, ValueError):
                pass  # Use the full column name if parsing fails

            ax.plot(energy_axis, intensity_data, label=label)
            plotted_something = True

    if not plotted_something:
        log.error("No intensity columns found to plot.")
        plt.close(fig)
        return None, None

    ax.set_title(f"Calibrated Spectra from '{filepath}'")
    ax.set_xlabel(x_col_name.replace("_", " "))
    ax.set_ylabel("Integrated Intensity (Arb. Units)")
    ax.legend()
    ax.grid(True, linestyle=":")
    plt.tight_layout()
    plt.show()

    return fig, ax


if __name__ == "__main__":
    # Create dummy image data
    image_height, image_width = 100, 300
    yy, xx = np.mgrid[0:image_height, 0:image_width]
    dummy_image = np.random.rand(image_height, image_width) * 20
    # ROI 1 area (Peaks ~150, ~210)
    mask1 = (yy > 10) & (yy < 20)
    dummy_image[mask1] += 50 * np.exp(-((xx[mask1] - 150) ** 2 / (2 * 10**2)))
    dummy_image[mask1] += 30 * np.exp(-((xx[mask1] - 210) ** 2 / (2 * 8**2)))
    # ROI 2 area (Peak ~185)
    mask2 = (yy > 50) & (yy < 65)
    dummy_image[mask2] += 60 * np.exp(-((xx[mask2] - 185) ** 2 / (2 * 12**2)))

    combined_widget = SpectrometerCalibration(image_data=dummy_image)
    combined_widget.display()
