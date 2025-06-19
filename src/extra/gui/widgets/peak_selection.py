import logging
import warnings

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backend_bases import MouseButton
from matplotlib.lines import Line2D

__all__ = ['PeakSelectorWidget']

log = logging.getLogger(__name__)

warnings.filterwarnings("ignore", message="Unable to determine Axes to steal")


class PeakSelectorWidget:
    """
    An interactive Matplotlib widget to select an arbitrary number of peak
    positions on 1D projection traces derived from ROIs of Spectrometer detector
    data. Adds sequential labels to markers ("Peak 1", "Peak 2", ...) and allows
    dragging.

    Features:

    - Takes 2D image data and a list of ROI definitions as input.
    - Calculates and plots 1D projections (integrations along y) for each ROI.
    - Left-clicking on a trace places a new peak marker (vertical line) with a
      label.
    - Right-clicking near an existing marker deletes it.
    - Labels show peak number ("Peak N") and pixel position.
    - Allows interactively dragging existing markers (and their labels) to
      refine position.
    - Provides standard zoom/pan functionality via Matplotlib toolbar.
    - Returns the selected pixel positions for each ROI.
    """

    # Display constants
    DEFAULT_LINEWIDTH = 1.5
    HIGHLIGHT_LINEWIDTH = 2.5
    MARKER_PICKER_TOLERANCE = 5.0  # Pixel distance threshold for dragging/deleting
    LINE_PICKER_TOLERANCE = 5
    CLICK_TOLERANCE = 5  # Pixel distance threshold to distinguish click from drag
    MARKER_ALPHA = 0.9
    LABEL_ALPHA = 0.7
    MARKER_LINESTYLE = "-"  # Use solid lines for all peaks

    # Position constants
    LABEL_Y_POS = 0.95  # Y position in axes coordinates for all labels
    LABEL_AREA_HEIGHT = 0.15  # Fraction of y-axis reserved for labels (approx)

    # Color constants
    HIGHLIGHT_COLOR = "red"
    LABEL_BG_COLOR = "white"
    LABEL_BORDER_COLOR = "none"

    def __init__(self, image_data, roi_definitions):
        """
        Initializes the widget.

        Args:
            image_data (np.ndarray): The 2D detector image data.
            roi_definitions (list): A list of dictionaries, where each dict represents
                                    an ROI and must contain 'y_start' and 'y_end' keys.
                                    Example: [{'y_start': 40, 'y_end': 60}, ...]
        """
        if image_data.ndim != 2:
            raise ValueError("Input image_data must be a 2D numpy array.")
        if not isinstance(roi_definitions, list) or not all(
            "y_start" in r and "y_end" in r for r in roi_definitions
        ):
            raise ValueError(
                "roi_definitions must be a list of dicts with 'y_start' and 'y_end'."
            )

        self.image_data = image_data
        self.img_height, self.img_width = self.image_data.shape

        # ROI data storage:
        # {'roi_def': {'y_start', 'y_end'},
        #  'projection': array,
        #  'line': Line2D,
        #  'peaks': list of dicts [{'pixel': float, 'vline': Line2D, 'label': Text, 'id': int}],
        #  'color': color,
        #  'roi_index': int} # Original index from input list
        self.rois_data = []

        # Stores info about the marker being dragged:
        # {'roi_idx': int, 'peak_list_idx': int, 'vline': Line2D}
        self.drag_info = None
        # Stores ({'button', 'x', 'y', 'xdata', 'ydata'}) from button_press to check on release
        self._click_info = None
        self._peak_update_callback = None

        # Calculate Projections and Populate self.rois_data
        prop_cycle = plt.rcParams["axes.prop_cycle"]
        colors = prop_cycle.by_key()["color"]
        valid_roi_count = 0
        for i, roi_def in enumerate(roi_definitions):
            y_start, y_end = roi_def["y_start"], roi_def["y_end"]
            y_start_idx = int(max(0, np.floor(y_start)))
            y_end_idx = int(min(self.img_height, np.ceil(y_end)))

            if y_end_idx <= y_start_idx:
                print(
                    f"ROI {i} has zero or negative height in pixels ({y_start_idx}-{y_end_idx}). Skipping."
                )
                continue

            projection = np.sum(self.image_data[y_start_idx:y_end_idx, :], axis=0)

            self.rois_data.append(
                {
                    "roi_def": roi_def,
                    "projection": projection,
                    "line": None,
                    "peaks": [],
                    "color": colors[valid_roi_count % len(colors)],
                    "roi_index": i,
                }
            )
            valid_roi_count += 1

        if not self.rois_data:
            raise ValueError("No valid ROIs found or calculated.")

        # Create Figure and Axes
        self.fig, self.ax_proj = plt.subplots(figsize=(10, 6))
        plt.subplots_adjust(bottom=0.1, top=0.9)  # Adjusted bottom slightly

        # Plot Projections
        self._plot_projections()

        # Setup Axes
        self.ax_proj.set_title(
            "ROI Projections - Left-Click Trace to Add Peak, Right-Click Peak to Delete, Drag to Move"
        )
        self.ax_proj.set_xlabel("Dispersion Axis (Pixels)")
        self.ax_proj.set_ylabel("Integrated Intensity (Arb. Units)")
        self.ax_proj.grid(True, linestyle=":")
        self.ax_proj.autoscale(enable=True, axis="x", tight=True)

        # Connect Events
        self.cid_press = self.fig.canvas.mpl_connect(
            "button_press_event", self._on_press
        )
        self.cid_motion = self.fig.canvas.mpl_connect(
            "motion_notify_event", self._on_motion
        )
        self.cid_release = self.fig.canvas.mpl_connect(
            "button_release_event", self._on_release
        )

        print("- Left-Click on a trace to place a new peak marker.")
        print("- Right-Click near an existing marker to delete it.")
        print("- Click and drag an existing marker to move it.")
        print("- Use the toolbar (top left) for Zoom/Pan.")

    # Methods for callback registration and notification
    def register_peak_update_callback(self, callback):
        """Register a function to be called when peak selections are updated."""
        self._peak_update_callback = callback

    def _notify_peak_update(self):
        """Calls the registered peak update callback function, if any."""
        if callable(self._peak_update_callback):
            log.debug("Notifying peak update...")
            try:
                self._peak_update_callback()
            except Exception:
                log.exception(f"Error in peak update callback")

    # Plotting Methods
    def _plot_projections(self):
        """Plots all projection lines and their current markers and labels."""
        self.ax_proj.clear()
        x_pixels = np.arange(self.img_width)
        legend_handles = []

        for i, roi_data in enumerate(self.rois_data):
            # Plot the trace
            (line,) = self.ax_proj.plot(
                x_pixels,
                roi_data["projection"],
                label=f"ROI {roi_data['roi_index']}",
                color=roi_data["color"],
                linewidth=self.DEFAULT_LINEWIDTH,
                picker=self.LINE_PICKER_TOLERANCE,
            )
            line._roi_idx = i  # Internal index used by the widget
            roi_data["line"] = line

            # Re-draw all existing peaks for this ROI
            for peak_list_idx, peak_info in enumerate(roi_data["peaks"]):
                # Ensure vline and label are recreated or updated correctly
                new_vline, new_label = self._update_peak_visuals(
                    i, peak_list_idx, peak_info["pixel"]
                )
                # Store the potentially new matplotlib objects back
                peak_info["vline"] = new_vline
                peak_info["label"] = new_label

            # Create legend handle
            handle = Line2D(
                [0],
                [0],
                marker="s",
                color="w",
                label=f"ROI {roi_data['roi_index']}",  # Label uses original index
                markerfacecolor=roi_data["color"],
                markersize=10,
            )
            legend_handles.append(handle)

        self.ax_proj.set_title(
            "ROI Projections - Left-Click Trace to Add Peak, Right-Click Peak to Delete, Drag to Move"
        )
        self.ax_proj.set_xlabel("Dispersion Axis (Pixels)")
        self.ax_proj.set_ylabel("Integrated Intensity (Arb. Units)")
        self.ax_proj.grid(True, linestyle=":")

        if legend_handles:
            self.ax_proj.legend(handles=legend_handles, fontsize="small", loc="best")

        self.ax_proj.relim()
        self.ax_proj.autoscale_view(True, True, True)

    def _update_peak_visuals(self, roi_idx, peak_list_idx, x_pixel, highlight=False):
        """
        Creates, updates, or removes visuals (vline and label) for a specific peak.

        Args:
            roi_idx: Internal index of the ROI in self.rois_data.
            peak_list_idx: Index of the peak within the ROI's 'peaks' list.
            x_pixel: X-coordinate position (in data units) or None to remove.
            highlight: Whether to highlight the visuals (e.g., during dragging).

        Returns:
            tuple: (vline_object or None, label_object or None)
        """
        roi_data = self.rois_data[roi_idx]
        try:
            peak_info = roi_data["peaks"][peak_list_idx]
        except IndexError:
            # This might happen if called during deletion before list is updated, handle gracefully
            log.warning(
                f"Peak index {peak_list_idx} out of bounds for ROI {roi_idx}. Skipping visual update."
            )
            return None, None

        vline_obj = peak_info.get("vline")
        label_obj = peak_info.get("label")

        # Removal Case
        if x_pixel is None:
            if vline_obj and vline_obj in self.ax_proj.lines:
                try:
                    vline_obj.remove()
                except ValueError:
                    pass
            if label_obj and label_obj in self.ax_proj.texts:
                try:
                    label_obj.remove()
                except ValueError:
                    pass
            return None, None  # Visuals removed

        # Common Properties
        color = self.HIGHLIGHT_COLOR if highlight else roi_data["color"]
        linewidth = self.HIGHLIGHT_LINEWIDTH if highlight else self.DEFAULT_LINEWIDTH
        peak_number = peak_list_idx + 1  # 1-based index for display
        label_text = f"Peak {peak_number}: {x_pixel:.1f}"
        y_pos_axes = self.LABEL_Y_POS  # Use the single defined Y position

        # Update or Create Vline
        if vline_obj and vline_obj in self.ax_proj.lines:
            vline_obj.set_xdata([x_pixel, x_pixel])
            vline_obj.set_color(color)
            vline_obj.set_linestyle(self.MARKER_LINESTYLE)
            vline_obj.set_linewidth(linewidth)
            vline_obj.set_visible(True)
            new_vline = vline_obj
        else:
            new_vline = self.ax_proj.axvline(
                x_pixel,
                color=color,
                linestyle=self.MARKER_LINESTYLE,
                linewidth=linewidth,
                ymin=0.0,
                ymax=1.0,
                alpha=self.MARKER_ALPHA,
                picker=False,
            )
            peak_info["vline"] = new_vline

        # Update or Create Label
        if label_obj and label_obj in self.ax_proj.texts:
            label_obj.set_position((x_pixel, y_pos_axes))
            label_obj.set_text(label_text)
            label_obj.set_visible(True)
            label_obj.set_color(color)
            label_obj.set_bbox(
                dict(
                    boxstyle="round,pad=0.2",
                    fc=self.LABEL_BG_COLOR,
                    ec=self.LABEL_BORDER_COLOR,
                    alpha=self.LABEL_ALPHA,
                )
            )
            new_label = label_obj
        else:
            new_label = self.ax_proj.text(
                x_pixel,
                y_pos_axes,
                label_text,
                transform=self.ax_proj.get_xaxis_transform(),  # X data, Y axes coords
                ha="center",
                va="bottom",
                color=color,
                fontsize="small",
                clip_on=True,
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    fc=self.LABEL_BG_COLOR,
                    ec=self.LABEL_BORDER_COLOR,
                    alpha=self.LABEL_ALPHA,
                ),
            )
            peak_info["label"] = new_label

        return new_vline, new_label

    def _renumber_and_update_labels(self, roi_idx, start_list_idx=0):
        """Renumbers peak labels from a given index onwards for an ROI."""
        roi_data = self.rois_data[roi_idx]
        for i, peak_info in enumerate(
            roi_data["peaks"][start_list_idx:], start=start_list_idx
        ):
            peak_number = i + 1
            label_obj = peak_info.get("label")
            if label_obj and label_obj in self.ax_proj.texts:
                new_text = f"Peak {peak_number}: {peak_info['pixel']:.1f}"
                label_obj.set_text(new_text)

    # Event Handlers
    def _on_press(self, event):
        """Handle mouse button press for marker placement, deletion, drag initiation."""
        if not self._is_valid_axes_interaction(event):
            return

        self._click_info = {
            "button": event.button,
            "x": event.x,
            "y": event.y,
            "xdata": event.xdata,
            "ydata": event.ydata,
        }

        # Check for drag initiation (LEFT CLICK ONLY)
        if event.button == MouseButton.LEFT:
            nearby_marker_info = self._find_nearby_marker(event)
            if nearby_marker_info:
                self.drag_info = (
                    nearby_marker_info  # Includes {'roi_idx', 'peak_list_idx', 'vline'}
                )
                self._start_marker_drag()
                return  # Don't process as a click if drag started

        # Deletion (RIGHT CLICK) and placement (LEFT CLICK) handled on release

    def _is_valid_axes_interaction(self, event):
        """Check if the event is a potentially valid interaction within the axes."""
        if event.inaxes != self.ax_proj:
            return False

        return True

    def _start_marker_drag(self):
        """Set up marker dragging state and visuals."""
        roi_idx = self.drag_info["roi_idx"]
        peak_list_idx = self.drag_info["peak_list_idx"]
        roi_orig_idx = self.rois_data[roi_idx]["roi_index"]
        peak_number = peak_list_idx + 1

        log.debug(f"Initiating drag for Peak {peak_number} of ROI {roi_orig_idx}")

        # Highlight dragged line and label
        x_pixel = self.rois_data[roi_idx]["peaks"][peak_list_idx]["pixel"]
        self._update_peak_visuals(roi_idx, peak_list_idx, x_pixel, highlight=True)
        self.fig.canvas.draw_idle()

    def _on_motion(self, event):
        """Handle mouse motion for dragging markers and their labels."""
        if not self._is_active_drag(event):
            return

        x_pixel = event.xdata
        if x_pixel is None:  # Moved outside axes
            return

        self._update_dragged_marker(x_pixel)
        self.fig.canvas.draw_idle()

    def _is_active_drag(self, event):
        """Check if there's an active drag operation."""
        return (
            self.drag_info is not None
            and event.inaxes == self.ax_proj
            # Drag only happens with left mouse button pressed
            and event.button == MouseButton.LEFT
        )

    def _update_dragged_marker(self, x_pixel):
        """Update the position and label of the marker being dragged."""
        if not self.drag_info:
            return

        roi_idx = self.drag_info["roi_idx"]
        peak_list_idx = self.drag_info["peak_list_idx"]

        # Update the stored pixel value temporarily during drag
        self.rois_data[roi_idx]["peaks"][peak_list_idx]["pixel"] = x_pixel

        # Update the visuals (vline and label) with highlight
        self._update_peak_visuals(roi_idx, peak_list_idx, x_pixel, highlight=True)

    def _on_release(self, event):
        """Handle mouse button release for placing/deleting markers or ending drag."""
        if not self._click_info or event.inaxes != self.ax_proj:
            self._reset_interaction_state()
            return

        peak_changed = False  # Flag to check if we need to notify

        # Finishing a drag (was initiated with LEFT)
        if self.drag_info and self._click_info["button"] == MouseButton.LEFT:
            self._finish_marker_drag(event)
            peak_changed = True

        # Not dragging, check for simple click
        elif self._is_click(event):
            # Left Click -> Place Marker
            if event.button == MouseButton.LEFT:
                marker_placed = self._handle_add_peak_click(event)
                if marker_placed:
                    peak_changed = True
            # Right Click -> Delete Marker
            elif event.button == MouseButton.RIGHT:
                marker_deleted = self._handle_delete_peak_click(event)
                if marker_deleted:
                    peak_changed = True

        # Reset interaction state regardless
        self._reset_interaction_state()

        # Notify if a peak was added, deleted, or moved
        if peak_changed:
            self.fig.canvas.draw_idle()  # Update the view
            self._notify_peak_update()  # Notify parent widget

    def _reset_interaction_state(self):
        """Reset interaction state variables."""
        self.drag_info = None
        self._click_info = None

    def _finish_marker_drag(self, event):
        """Complete a marker drag operation."""
        if not self.drag_info:
            return

        roi_idx = self.drag_info["roi_idx"]
        peak_list_idx = self.drag_info["peak_list_idx"]
        roi_orig_idx = self.rois_data[roi_idx]["roi_index"]
        peak_number = peak_list_idx + 1

        # Final update of position
        # If released outside axes, keep the last valid position during drag
        final_x_pixel = (
            event.xdata
            if event.xdata is not None
            else self.rois_data[roi_idx]["peaks"][peak_list_idx]["pixel"]
        )

        log.debug(
            f"Finished drag for Peak {peak_number} of ROI {roi_orig_idx} at pixel {final_x_pixel:.1f}"
        )

        # Update the stored pixel position permanently
        self.rois_data[roi_idx]["peaks"][peak_list_idx]["pixel"] = final_x_pixel

        # Update visuals back to normal state (no highlight)
        self._update_peak_visuals(
            roi_idx, peak_list_idx, final_x_pixel, highlight=False
        )

        # No need to reset self.drag_info here, done in on_release

    def _handle_add_peak_click(self, event):
        """Handle a left click on a trace to add a new peak."""
        if event.xdata is not None:
            clicked_roi_idx = self._find_clicked_trace(event)
            if clicked_roi_idx is not None:
                self._add_peak(clicked_roi_idx, event.xdata)
                return True  # Peak was added
        return False  # No peak added

    def _add_peak(self, roi_idx, x_pixel):
        """Adds a new peak marker and label to the specified ROI."""
        roi_data = self.rois_data[roi_idx]
        roi_orig_idx = roi_data["roi_index"]
        new_peak_list_idx = len(roi_data["peaks"])
        peak_number = new_peak_list_idx + 1

        log.debug(
            f"Adding Peak {peak_number} for ROI {roi_orig_idx} at pixel {x_pixel:.1f}"
        )

        # Create the peak data structure (visuals added by _update_peak_visuals)
        new_peak_info = {"pixel": x_pixel, "vline": None, "label": None}
        roi_data["peaks"].append(new_peak_info)

        # Create/update the visuals for the new peak
        vline, label = self._update_peak_visuals(roi_idx, new_peak_list_idx, x_pixel)

        # Store the created matplotlib objects
        new_peak_info["vline"] = vline
        new_peak_info["label"] = label

    def _handle_delete_peak_click(self, event):
        """Handle a right click near a marker to delete it."""
        nearby_marker_info = self._find_nearby_marker(event)
        if nearby_marker_info:
            roi_idx = nearby_marker_info["roi_idx"]
            peak_list_idx = nearby_marker_info["peak_list_idx"]
            self._delete_peak(roi_idx, peak_list_idx)
            return True  # Peak deleted
        return False  # No peak deleted

    def _delete_peak(self, roi_idx, peak_list_idx):
        """Deletes a specific peak and updates subsequent peak labels."""
        roi_data = self.rois_data[roi_idx]
        roi_orig_idx = roi_data["roi_index"]

        if 0 <= peak_list_idx < len(roi_data["peaks"]):
            log.debug(
                f"Deleting Peak {peak_list_idx + 1} (List Index {peak_list_idx}) from ROI {roi_orig_idx}"
            )

            # Remove visuals first
            self._update_peak_visuals(roi_idx, peak_list_idx, None)

            # Remove from data list
            del roi_data["peaks"][peak_list_idx]

            # Renumber and update labels for subsequent peaks in the same ROI
            self._renumber_and_update_labels(roi_idx, start_list_idx=peak_list_idx)
        else:
            log.error(
                f"Attempted to delete non-existent peak index {peak_list_idx} in ROI {roi_idx}"
            )

    def _find_nearby_marker(self, event):
        """Check if the event is close to any existing vertical marker."""
        if event.xdata is None or event.ydata is None:
            return None

        # Calculate the y-range available for clicking markers (excluding label area)
        ymin, ymax = self.ax_proj.get_ylim()
        clickable_ymax = ymax - (ymax - ymin) * self.LABEL_AREA_HEIGHT

        # Check if the click is within the clickable y-range of the plot
        if not (ymin <= event.ydata <= clickable_ymax):
            return None

        # Tolerance in display coords squared
        min_dist_sq = (self.MARKER_PICKER_TOLERANCE / self.fig.dpi) ** 2

        for roi_idx, roi_data in enumerate(self.rois_data):
            for peak_list_idx, peak_info in enumerate(roi_data["peaks"]):
                vline = peak_info.get("vline")
                pixel_pos = peak_info.get("pixel")
                if vline and pixel_pos is not None and vline.get_visible():
                    # Check horizontal distance in data coordinates first
                    if abs(event.xdata - pixel_pos) < self.MARKER_PICKER_TOLERANCE:
                        # More robust check: distance in display coordinates
                        x_display, _ = vline.get_transform().transform((pixel_pos, 0))
                        # Only check horizontal distance for vline picking
                        dist_sq = (event.x - x_display) ** 2

                        # If we find a candidate within tolerance, return it
                        if dist_sq < min_dist_sq * 100:
                            return {
                                "roi_idx": roi_idx,
                                "peak_list_idx": peak_list_idx,
                                "vline": vline,
                            }

        return None  # No marker found nearby

    def _find_clicked_trace(self, event):
        """Find which projection trace was clicked, prioritizing closer lines."""
        candidate_lines = []
        for line in self.ax_proj.get_lines():
            # Only consider the main ROI projection lines which have _roi_idx
            if hasattr(line, "_roi_idx"):
                contains, props = line.contains(event)
                if contains:
                    # Calculate vertical distance if multiple lines contain the point
                    x_data = line.get_xdata()
                    y_data = line.get_ydata()
                    if (
                        len(x_data) > 0
                        and event.xdata is not None
                        and event.ydata is not None
                    ):
                        # Find index of x_data closest to the click
                        idx = np.argmin(np.abs(x_data - event.xdata))
                        # Ensure index is valid
                        if 0 <= idx < len(y_data):
                            # Use distance squared in display coordinates for better comparison
                            _, y_display_line = self.ax_proj.transData.transform(
                                (event.xdata, y_data[idx])
                            )
                            dist_sq = (event.y - y_display_line) ** 2
                            candidate_lines.append(
                                {"dist_sq": dist_sq, "roi_idx": line._roi_idx}
                            )

        if not candidate_lines:
            return None
        # Return the internal index of the closest line vertically
        candidate_lines.sort(key=lambda x: x["dist_sq"])

        return candidate_lines[0]["roi_idx"]

    def _is_click(self, event, tolerance=None):
        """Check if the button_release event corresponds to a click (minimal drag)."""
        if not self._click_info:
            return False
        if tolerance is None:
            tolerance = self.CLICK_TOLERANCE
        # Check distance in display coordinates
        dx = abs(event.x - self._click_info["x"])
        dy = abs(event.y - self._click_info["y"])
        return dx <= tolerance and dy <= tolerance

    # Data Retrieval
    def get_selected_peaks(self):
        """
        Returns the selected peak positions for all ROIs.

        Returns:
            list: A list of dictionaries, one for each ROI processed. Each dict contains:
                  - 'roi_def': The original ROI definition dictionary.
                  - 'roi_index': The original index of the ROI in the input list.
                  - 'peaks': A list of selected peak pixel positions for this ROI,
                             sorted by pixel value. Returns an empty list if no peaks
                             were selected for that ROI.
        """
        results = []
        for roi_data in self.rois_data:
            # Extract pixel values and sort them
            peak_pixels = sorted([peak["pixel"] for peak in roi_data["peaks"]])
            results.append(
                {
                    "roi_def": roi_data["roi_def"],
                    "roi_index": roi_data["roi_index"],
                    "peaks": peak_pixels,
                }
            )
        return results

    def show(self):
        """Display the plot window."""
        plt.show()

    def close(self):
        """Close the plot window and disconnect events."""
        log.debug("Closing figure and disconnecting events.")
        if hasattr(self, "cid_press") and self.cid_press:
            self.fig.canvas.mpl_disconnect(self.cid_press)
            self.cid_press = None
        if hasattr(self, "cid_motion") and self.cid_motion:
            self.fig.canvas.mpl_disconnect(self.cid_motion)
            self.cid_motion = None
        if hasattr(self, "cid_release") and self.cid_release:
            self.fig.canvas.mpl_disconnect(self.cid_release)
            self.cid_release = None
        plt.close(self.fig)


if __name__ == "__main__":
    # Dummy image data
    yy, xx = np.mgrid[0:200, 0:300]
    image = np.random.rand(200, 300) * 50
    peak_centers_y = [50, 100, 150]
    peak_width_y = 15
    peak_intensity = 2000

    # Add multiple peaks per ROI trace
    base_peak_x = 120
    peak_sep_x = 80
    peak_sigma_x = 15

    for i, center_y in enumerate(peak_centers_y):
        y_mask = (yy > center_y - peak_width_y / 2) & (yy < center_y + peak_width_y / 2)
        gauss_y = np.exp(-((yy - center_y) ** 2) / (2 * (peak_width_y / 4) ** 2))

        # Add Peak 1
        peak1_x = base_peak_x + i * 10  # Slightly shift base peak per ROI
        intensity_profile1 = peak_intensity * (1 - 0.1 * np.abs(xx - peak1_x) / 150)
        gauss_x1 = np.exp(-((xx - peak1_x) ** 2) / (2 * peak_sigma_x**2))
        image[y_mask] += (intensity_profile1 * gauss_y * gauss_x1)[y_mask]

        # Add Peak 2 (if not the first ROI)
        if i > 0:
            peak2_x = (
                peak1_x + peak_sep_x + np.random.uniform(-5, 5)
            )  # Add second peak further out
            intensity_profile2 = (
                peak_intensity * 0.6 * (1 - 0.1 * np.abs(xx - peak2_x) / 150)
            )
            gauss_x2 = np.exp(
                -((xx - peak2_x) ** 2) / (2 * (peak_sigma_x * 1.2) ** 2)
            )  # Wider peak
            image[y_mask] += (intensity_profile2 * gauss_y * gauss_x2)[y_mask]

        # Add a third, smaller peak for the last ROI
        if i == len(peak_centers_y) - 1:
            peak3_x = peak1_x + peak_sep_x / 2 + np.random.uniform(-3, 3)
            intensity_profile3 = peak_intensity * 0.3
            gauss_x3 = np.exp(
                -((xx - peak3_x) ** 2) / (2 * (peak_sigma_x * 0.8) ** 2)
            )  # Narrower peak
            image[y_mask] += (intensity_profile3 * gauss_y * gauss_x3)[y_mask]

    image += yy * 0.5

    # Define ROIs
    input_rois = [
        {"y_start": 42.5, "y_end": 57.5},
        {"y_start": 95.0, "y_end": 110.0},
        {"y_start": 140.0, "y_end": 165.0},
        {"y_start": 10, "y_end": 12},  # Test very narrow/potentially invalid ROI
    ]

    # Run the Peak Selector Widget
    print("Creating PeakSelectorWidget (Arbitrary Peaks)...")
    peak_selector = None
    try:
        peak_selector = PeakSelectorWidget(image, input_rois)

        def my_callback():
            print("--- Callback: Peaks updated! ---")
            updated_peaks = peak_selector.get_selected_peaks()
            print(f"Current Peaks: {updated_peaks}")
            print("-------------------------------")

        peak_selector.register_peak_update_callback(my_callback)

        peak_selector.show()

        print("\nWidget closed.")
        selected_peaks = peak_selector.get_selected_peaks()

        print("Selected Peak Positions:")
        if selected_peaks:
            for roi_result in selected_peaks:
                roi_idx = roi_result["roi_index"]
                peaks_str = ", ".join([f"{p:.1f}" for p in roi_result["peaks"]])
                if not peaks_str:
                    peaks_str = "None"
                print(
                    f"  ROI {roi_idx} (y=[{roi_result['roi_def']['y_start']:.1f}, {roi_result['roi_def']['y_end']:.1f}]): "
                    f"Peaks at Pixels = [{peaks_str}]"
                )
        else:
            print("  No ROIs were processed or widget not fully initialized.")

    except ValueError as e:
        print(f"Error initializing or running widget: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        if peak_selector is not None:
            peak_selector.close()
