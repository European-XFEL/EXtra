from typing import overload

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
import numpy as np
from extra.utils import imshow2
from matplotlib.backend_bases import MouseButton

PATCH_COLOR = "red"
UNSELECTED_COLOR = "lime"
SELECTING_COLOR = "yellow"
SELECTED_COLOR = "red"


class ROISelectorWidget:
    """
    An interactive Matplotlib widget for selecting vertical ROIs on a 2D image,
    typically used for analyzing detector data in X-ray Spectroscopy.

    Allows adding ROIs by clicking and dragging vertically, selecting existing
    ROIs by clicking on them, and deleting the selected ROI using a button.
    """

    def __init__(self, image_data):
        """

        Args:
            image_data (np.ndarray): A 2D numpy array representing the detector image.
        """
        if image_data.ndim != 2:
            raise ValueError("Input image_data must be a 2D numpy array.")

        self.image_data = image_data
        self.rois = []  # ROI definitions: {'patch': patch, 'y_start': y1, 'y_end': y2}
        self.press_y = None
        self.current_rect = None  # Temporary rectangle during drag
        self.selected_roi_index = None  # Index of the selected ROI in self.rois

        # Create Figure and Axes
        self.fig, self.ax = plt.subplots(figsize=(8, 7))
        plt.subplots_adjust(bottom=0.2)

        # Display Image
        self.im_display = imshow2(
            self.image_data, ax=self.ax, cmap="viridis", aspect="auto"
        )
        self.ax.set_title("Detector Image - Click & Drag Vertically to Define ROIs")
        self.ax.set_xlabel("Energy Axis (Pixels)")
        self.ax.set_ylabel("Spatial Axis (Pixels)")
        self.img_height, self.img_width = self.image_data.shape

        # Connect Events
        self.cid_press = self.fig.canvas.mpl_connect(
            "button_press_event", self.on_press
        )
        self.cid_release = self.fig.canvas.mpl_connect(
            "button_release_event", self.on_release
        )
        self.cid_motion = self.fig.canvas.mpl_connect(
            "motion_notify_event", self.on_motion
        )
        self.cid_pick = self.fig.canvas.mpl_connect("pick_event", self.on_pick)

        # Callbacks
        self._roi_update_callback = None

        # Add Delete Button
        ax_delete = plt.axes([0.7, 0.05, 0.2, 0.075])  # [left, bottom, width, height]
        self.btn_delete = widgets.Button(ax_delete, "Delete Selected ROI")
        self.btn_delete.on_clicked(self.delete_selected_roi)

        print("Widget Initialized. Instructions:")
        print("- Click and drag vertically on the image to define an ROI.")
        print("- Click on an existing ROI rectangle to select it (it will turn red).")
        print("- Click the 'Delete Selected ROI' button to remove the selected ROI.")

    def register_roi_update_callback(self, callback):
        """Register a function to be called when ROIs are updated."""
        self._roi_update_callback = callback

    def _notify_roi_update(self):
        """Calls the registered callback function, if any."""
        if callable(self._roi_update_callback):
            print("ROISelectorWidget: Notifying ROI update...")
            try:
                self._roi_update_callback()
            except Exception as e:
                print(f"ROISelectorWidget: Error in ROI update callback: {e}")

    @overload
    def on_press(self, event):
        """Callback for mouse button press events."""
        # Ignore clicks outside the main axes or with the wrong button
        if event.inaxes != self.ax or event.button != MouseButton.LEFT:
            return
        if self.fig.canvas.toolbar.mode != "":
            return

        # Check if the click was specifically on any existing patch
        for roi_data in self.rois:
            contains, _ = roi_data["patch"].contains(event)
            if contains:
                return  # Let on_pick handle selection

        # Start defining a new ROI
        self.press_y = event.ydata
        # Create a temporary rectangle for visual feedback during drag
        self.current_rect = patches.Rectangle(
            (0, self.press_y),
            self.img_width,
            1,  # Initial small height
            edgecolor=SELECTING_COLOR,
            facecolor=SELECTING_COLOR,
            alpha=0.3,
            linewidth=1,
            animated=True,
        )
        self.ax.add_patch(self.current_rect)
        self.fig.canvas.draw_idle()

    @overload
    def on_motion(self, event):
        """Callback for mouse motion events."""
        # Only act if dragging within the axes
        if self.press_y is None or event.inaxes != self.ax or self.current_rect is None:
            return

        y0 = self.press_y
        y1 = event.ydata
        if y1 is None:
            return  # Moved out of axes

        # Update the temporary rectangle's height and position
        self.current_rect.set_y(min(y0, y1))
        self.current_rect.set_height(abs(y0 - y1))

        self.fig.canvas.draw_idle()

    @overload
    def on_release(self, event):
        """Callback for mouse button release events."""
        # Check if we were dragging to create a new ROI
        if (
            self.press_y is None
            or event.button != MouseButton.LEFT
            or self.current_rect is None
        ):
            # If not dragging, but released over an axis, maybe deselect?
            if event.inaxes == self.ax and self.selected_roi_index is not None:
                # Check if the click was outside *any* ROI patch
                is_on_patch = False
                for i, roi_data in enumerate(self.rois):
                    contains, _ = roi_data["patch"].contains(event)
                    if contains:
                        is_on_patch = True
                        break
                if not is_on_patch:
                    # Clicked on background, deselect
                    print(f"Deselecting ROI {self.selected_roi_index}")
                    self._update_selection(None)  # Deselect visually

            # Cleanup in case something went wrong with motion/press state
            if self.current_rect:
                self.current_rect.remove()
                self.current_rect = None
            self.press_y = None
            self.fig.canvas.draw_idle()
            return

        # Finalize ROI
        y0 = self.press_y
        y1 = event.ydata
        self.press_y = None  # Reset starting position

        # Remove the temporary rectangle
        self.current_rect.remove()
        self.current_rect = None

        if y1 is None:  # Released outside axes
            self.fig.canvas.draw_idle()
            return

        # Ensure start < end
        y_start, y_end = min(y0, y1), max(y0, y1)

        # Optional: Ignore tiny ROIs
        min_height = 1.0
        if abs(y_start - y_end) < min_height:
            print("ROI too small, ignoring.")
            self.fig.canvas.draw_idle()
            return

        # Create the permanent ROI patch
        roi_patch = patches.Rectangle(
            (0, y_start),
            self.img_width,
            y_end - y_start,
            edgecolor=UNSELECTED_COLOR,
            facecolor=UNSELECTED_COLOR,
            alpha=0.4,
            linewidth=1.5,
            picker=5,  # Allow picking this patch (tolerance in points)
        )

        # Store ROI data and add patch to axes
        new_roi_index = len(self.rois)
        roi_data = {
            "patch": roi_patch,
            "y_start": y_start,
            "y_end": y_end,
        }
        # Add custom attribute to link patch back to index
        roi_patch._roi_index = new_roi_index

        self.rois.append(roi_data)
        self.ax.add_patch(roi_patch)

        print(f"Added ROI {new_roi_index}: y=[{y_start:.2f}, {y_end:.2f}]")
        # Deselect any previously selected ROI when adding a new one
        self._update_selection(None)

        self.fig.canvas.draw_idle()
        self._notify_roi_update()

    @overload
    def on_pick(self, event):
        """Callback for pick events (clicking on a patch)."""
        # Ensure the picked artist is one of our ROI patches
        if isinstance(event.artist, patches.Rectangle) and hasattr(
            event.artist, "_roi_index"
        ):
            picked_index = event.artist._roi_index

            # Prevent starting a drag if we just picked
            self.press_y = None
            if self.current_rect:
                self.current_rect.remove()
                self.current_rect = None

            # Update selection state
            if self.selected_roi_index == picked_index:
                # Clicked on already selected ROI - deselect it
                print(f"Deselecting ROI {picked_index}")
                self._update_selection(None)
            else:
                print(f"Selected ROI {picked_index}")
                self._update_selection(picked_index)

            self.fig.canvas.draw_idle()

    def _update_selection(self, new_selected_index):
        """Internal helper to visually update selected/deselected ROIs."""
        # Deselect the old one (if any)
        if (
                self.selected_roi_index is not None and
                self.selected_roi_index < len(self.rois)
        ):
            try:
                old_patch = self.rois[self.selected_roi_index]["patch"]
                old_patch.set_edgecolor(UNSELECTED_COLOR)
                old_patch.set_facecolor(UNSELECTED_COLOR)
                old_patch.set_linewidth(1.5)
            except IndexError:
                print(
                    f"Warning: Could not find patch for previously selected index {self.selected_roi_index}"
                )

        self.selected_roi_index = new_selected_index

        # Select the new one (if any)
        if self.selected_roi_index is not None:
            try:
                new_patch = self.rois[self.selected_roi_index]["patch"]
                new_patch.set_edgecolor(SELECTED_COLOR)
                new_patch.set_facecolor(SELECTED_COLOR)
                new_patch.set_linewidth(2.0)
            except IndexError:
                print(
                    f"Warning: Could not find patch for newly selected index {self.selected_roi_index}"
                )
                self.selected_roi_index = None  # Reset if index is bad

    def delete_selected_roi(self, event):
        """Callback for the delete button."""
        if self.selected_roi_index is None:
            print("No ROI selected to delete.")
            return

        if self.selected_roi_index < 0 or self.selected_roi_index >= len(self.rois):
            print(
                f"Error: Invalid selected index {self.selected_roi_index}. Cannot delete."
            )
            self.selected_roi_index = None  # Reset selection
            return

        print(f"Deleting ROI {self.selected_roi_index}...")

        # Get the ROI data to remove
        roi_to_remove = self.rois[self.selected_roi_index]

        # Remove the patch from the axes
        roi_to_remove["patch"].remove()

        # Remove the ROI data from our list
        del self.rois[self.selected_roi_index]

        # Update indices stored in subsequent patches
        for i in range(self.selected_roi_index, len(self.rois)):
            self.rois[i]["patch"]._roi_index = i

        # Reset selection
        self.selected_roi_index = None

        print("Deletion complete.")
        self.fig.canvas.draw_idle()
        self._notify_roi_update()

    def get_rois(self):
        """Returns the list of defined ROIs."""
        return [
            {"roi_index": index, "y_start": roi["y_start"], "y_end": roi["y_end"]}
            for index, roi in enumerate(self.rois)
        ]

    def show(self):
        """Display the plot."""
        plt.show()

    def close(self):
        print("ROISelectorWidget: Closing figure.")
        plt.close(self.fig)


if __name__ == "__main__":
    # Create sample 2D data
    yy, xx = np.mgrid[0:200, 0:300]
    data = np.random.rand(200, 300) * 0.2

    # Add some simulated K-beta peaks
    peak_centers = [50, 100, 150]
    peak_width = 15
    peak_intensity = 1.5

    for center in peak_centers:
        mask = (yy > center - peak_width / 2) & (yy < center + peak_width / 2)
        # Make peak intensity vary slightly across x
        intensity_profile = peak_intensity * (1 - 0.3 * np.abs(xx - 150) / 150)
        data[mask] += intensity_profile[mask] * np.exp(
            -((yy[mask] - center) ** 2) / (2 * (peak_width / 3) ** 2)
        )
    data += yy * 0.001

    print("Creating ROISelectorWidget...")
    calibrator_widget = ROISelectorWidget(data)
    calibrator_widget.show()
    print("\nWidget closed.")
    final_rois = calibrator_widget.get_rois()
    print("Final defined ROIs:")
    if final_rois:
        for i, roi in enumerate(final_rois):
            print(f"  ROI {i}: y_start={roi['y_start']:.2f}, y_end={roi['y_end']:.2f}")
    else:
        print("  No ROIs were defined.")
