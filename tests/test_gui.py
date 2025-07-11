import os
import tempfile
from unittest.mock import Mock, patch

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.backend_bases import MouseButton

from extra.gui.jupyter import SpectrometerCalibration
from extra.gui.jupyter.spectrometer_calibration import (
    CalibratedPlotter, plot_from_calibration_file)
from extra.gui.widgets.peak_selection import PeakSelectorWidget
from extra.gui.widgets.roi_selection import ROISelectorWidget


class TestPeakSelectorWidget:

    @pytest.fixture
    def sample_image_data(self):
        np.random.seed(42)
        return np.random.rand(100, 150) * 100

    @pytest.fixture
    def sample_roi_definitions(self):
        return [
            {"y_start": 10, "y_end": 20},
            {"y_start": 30, "y_end": 40},
            {"y_start": 60, "y_end": 80}
        ]

    @pytest.fixture
    def widget(self, sample_image_data, sample_roi_definitions):
        with patch('matplotlib.pyplot.show'):  # Prevent actual display during tests
            widget = PeakSelectorWidget(sample_image_data, sample_roi_definitions)
            yield widget
            widget.close()

    def test_initialization_valid_input(self, sample_image_data, sample_roi_definitions):
        with patch('matplotlib.pyplot.show'):
            widget = PeakSelectorWidget(sample_image_data, sample_roi_definitions)

            assert widget.image_data.shape == (100, 150)
            assert len(widget.rois_data) == 3
            assert widget.img_height == 100
            assert widget.img_width == 150

            # Check that projections were calculated
            for roi_data in widget.rois_data:
                assert roi_data["projection"] is not None
                assert len(roi_data["projection"]) == 150  # Width of image
                assert roi_data["peaks"] == []  # Should start with no peaks

            widget.close()

    def test_initialization_invalid_image_data(self, sample_roi_definitions):
        # Test 1D array
        with pytest.raises(ValueError, match="Input image_data must be a 2D numpy array"):
            PeakSelectorWidget(np.array([1, 2, 3]), sample_roi_definitions)

        # Test 3D array
        with pytest.raises(ValueError, match="Input image_data must be a 2D numpy array"):
            PeakSelectorWidget(np.random.rand(10, 10, 10), sample_roi_definitions)

    def test_initialization_invalid_roi_definitions(self, sample_image_data):
        # Test empty list
        with pytest.raises(ValueError, match="No valid ROIs found"):
            PeakSelectorWidget(sample_image_data, [])

        # Test invalid ROI format
        invalid_rois = [{"start": 10, "end": 20}]  # Missing y_start, y_end
        with pytest.raises(ValueError):
            PeakSelectorWidget(sample_image_data, invalid_rois)

    def test_initialization_zero_height_rois(self, sample_image_data):
        invalid_rois = [{"y_start": 10, "y_end": 10}]  # Zero height
        with pytest.raises(ValueError, match="No valid ROIs found"):
            PeakSelectorWidget(sample_image_data, invalid_rois)

    def test_projection_calculation(self, widget):
        roi_data = widget.rois_data[0]
        roi_def = roi_data["roi_def"]

        y_start = int(max(0, np.floor(roi_def["y_start"])))
        y_end = int(min(widget.img_height, np.ceil(roi_def["y_end"])))
        expected_projection = np.sum(widget.image_data[y_start:y_end, :], axis=0)

        np.testing.assert_array_equal(roi_data["projection"], expected_projection)

    def test_add_peak(self, widget):
        roi_idx = 0
        x_pixel = 75.5

        # Add a peak
        widget._add_peak(roi_idx, x_pixel)

        # Check that peak was added
        assert len(widget.rois_data[roi_idx]["peaks"]) == 1
        assert widget.rois_data[roi_idx]["peaks"][0]["pixel"] == x_pixel
        assert widget.rois_data[roi_idx]["peaks"][0]["vline"] is not None
        assert widget.rois_data[roi_idx]["peaks"][0]["label"] is not None

    def test_delete_peak(self, widget):
        roi_idx = 0
        x_pixel = 75.5

        # Add a peak first
        widget._add_peak(roi_idx, x_pixel)
        assert len(widget.rois_data[roi_idx]["peaks"]) == 1

        # Delete the peak
        widget._delete_peak(roi_idx, 0)
        assert len(widget.rois_data[roi_idx]["peaks"]) == 0

    def test_delete_peak_invalid_index(self, widget):
        roi_idx = 0

        # Try to delete from empty peaks list
        with patch('extra.gui.widgets.peak_selection.log') as mock_log:
            widget._delete_peak(roi_idx, 0)
            mock_log.error.assert_called_once()

    def test_multiple_peaks_same_roi(self, widget):
        """Test adding multiple peaks to the same ROI"""
        roi_idx = 0
        x_pixels = [25.0, 50.0, 100.0]

        # Add multiple peaks
        for x_pixel in x_pixels:
            widget._add_peak(roi_idx, x_pixel)

        # Check all peaks were added
        assert len(widget.rois_data[roi_idx]["peaks"]) == 3
        stored_pixels = [peak["pixel"] for peak in widget.rois_data[roi_idx]["peaks"]]
        assert stored_pixels == x_pixels

    def test_peaks_across_multiple_rois(self, widget):
        # Add peaks to different ROIs
        widget._add_peak(0, 25.0)
        widget._add_peak(1, 50.0)
        widget._add_peak(2, 75.0)

        # Check each ROI has one peak
        for i in range(3):
            assert len(widget.rois_data[i]["peaks"]) == 1

    def test_get_selected_peaks_empty(self, widget):
        """Test getting selected peaks when no peaks are selected"""
        result = widget.get_selected_peaks()

        assert len(result) == 3  # Three ROIs
        for roi_result in result:
            assert roi_result["peaks"] == []
            assert "roi_def" in roi_result
            assert "roi_index" in roi_result

    def test_get_selected_peaks_with_data(self, widget):
        # Add peaks to different ROIs
        widget._add_peak(0, 25.0)
        widget._add_peak(0, 75.0)
        widget._add_peak(1, 50.0)

        result = widget.get_selected_peaks()
        print(result)

        # Check ROI 0 has 2 peaks (sorted)
        assert len(result[0]["peaks"]) == 2
        assert result[0]["peaks"] == [25.0, 75.0]

        # Check ROI 1 has 1 peak
        assert len(result[1]["peaks"]) == 1
        assert result[1]["peaks"] == [50.0]

        # Check ROI 2 has no peaks
        assert len(result[2]["peaks"]) == 0

    def test_find_nearby_marker(self, widget):
        # Add a peak
        widget._add_peak(0, 50.0)

        # Create mock event near the marker
        mock_event = Mock()
        mock_event.xdata = 52.0  # Close to the marker at 50.0
        mock_event.ydata = 50.0  # Within valid y range

        # Mock the y-axis limits
        widget.ax_proj.get_ylim = Mock(return_value=(0, 100))

        result = widget._find_nearby_marker(mock_event)

        assert result is not None
        assert result["roi_idx"] == 0
        assert result["peak_list_idx"] == 0

    def test_find_nearby_marker_none_found(self, widget):
        mock_event = Mock()
        mock_event.xdata = 50.0
        mock_event.ydata = 50.0

        # Mock the y-axis limits
        widget.ax_proj.get_ylim = Mock(return_value=(0, 100))

        result = widget._find_nearby_marker(mock_event)
        assert result is None

        # Add a peak
        widget._add_peak(0, 50.0)

        # marker too far
        mock_event = Mock()
        mock_event.xdata = 56.0  # far from the marker at 50.0
        mock_event.ydata = 50.0

        widget.ax_proj.get_ylim = Mock(return_value=(0, 100))

        result = widget._find_nearby_marker(mock_event)

        assert result is None

    def test_is_click_true(self, widget):
        """Test click detection"""
        # Set up click info
        widget._click_info = {
            "x": 100, "y": 100, "xdata": 50.0, "ydata": 50.0
        }

        # Create mock event with minimal movement
        mock_event = Mock()
        mock_event.x = 102  # 2 pixels
        mock_event.y = 101  # 1 pixel

        assert widget._is_click(mock_event) is True

    def test_is_click_false(self, widget):
        """Test drag detection"""
        # Set up click info
        widget._click_info = {
            "x": 100, "y": 100, "xdata": 50.0, "ydata": 50.0
        }

        # Create mock event with movement
        mock_event = Mock()
        mock_event.x = 120  # 20 pixels
        mock_event.y = 110  # 10 pixels

        assert widget._is_click(mock_event) is False

    def test_is_valid_axes_interaction(self, widget):
        # Valid interaction
        mock_event = Mock()
        mock_event.inaxes = widget.ax_proj
        assert widget._is_valid_axes_interaction(mock_event) is True

        # Invalid interaction (different axes)
        mock_event.inaxes = Mock()
        assert widget._is_valid_axes_interaction(mock_event) is False

    def test_peak_update_callback(self, widget):
        callback_called = False

        def test_callback():
            nonlocal callback_called
            callback_called = True

        widget.register_peak_update_callback(test_callback)
        widget._notify_peak_update()

        assert callback_called is True

    def test_peak_update_callback_exception_handling(self, widget):
        def failing_callback():
            raise Exception("Test exception")

        widget.register_peak_update_callback(failing_callback)

        # Should not raise exception
        with patch('extra.gui.widgets.peak_selection.log') as mock_log:
            widget._notify_peak_update()
            mock_log.exception.assert_called_once()

    def test_renumber_and_update_labels(self, widget):
        """Test renumbering of peak labels after deletion"""
        roi_idx = 0

        # Add three peaks
        widget._add_peak(roi_idx, 25.0)
        widget._add_peak(roi_idx, 50.0)
        widget._add_peak(roi_idx, 75.0)

        # Delete the middle peak
        widget._delete_peak(roi_idx, 1)

        # Check that remaining peaks are renumbered correctly
        assert len(widget.rois_data[roi_idx]["peaks"]) == 2

        # Labels should be updated to "Peak 1" and "Peak 2"
        labels = [peak["label"] for peak in widget.rois_data[roi_idx]["peaks"]]
        assert all(label.get_text().startswith(f'Peak {idx}')
                   for idx, label in enumerate(labels, start=1))

    def test_widget_close(self, widget):
        # Check that event connections exist before closing
        assert widget.cid_press is not None
        assert widget.cid_motion is not None
        assert widget.cid_release is not None

        widget.close()

        # Check that event connections are cleaned up
        assert widget.cid_press is None
        assert widget.cid_motion is None
        assert widget.cid_release is None

    def test_narrow_roi_skipped(self, sample_image_data, capsys):
        """Test that ROIs with zero or negative height are skipped"""
        roi_definitions = [
            {"y_start": 10, "y_end": 20},  # Valid ROI
            {"y_start": 50, "y_end": 50},  # Zero height - should be skipped
            {"y_start": 80, "y_end": 70},  # Negative height - should be skipped
        ]

        with patch('matplotlib.pyplot.show'):
            widget = PeakSelectorWidget(sample_image_data, roi_definitions)

            # Should only have one valid ROI
            assert len(widget.rois_data) == 1
            assert widget.rois_data[0]["roi_index"] == 0

            captured = capsys.readouterr()
            assert "ROI 1 has zero or negative height in pixels" in captured.out
            assert "ROI 2 has zero or negative height in pixels" in captured.out

            widget.close()


class TestROISelectorWidget:

    @pytest.fixture
    def sample_image(self):
        np.random.seed(42)
        return np.random.rand(100, 150)

    @pytest.fixture
    def widget(self, sample_image):
        plt.switch_backend('Agg')
        widget = ROISelectorWidget(sample_image)
        yield widget
        widget.close()

    def test_init_with_valid_image(self, sample_image):
        plt.switch_backend('Agg')
        widget = ROISelectorWidget(sample_image)

        assert widget.original_image_data.shape == sample_image.shape
        assert np.array_equal(widget.original_image_data, sample_image)
        assert np.array_equal(widget.image_data, sample_image)
        assert widget.rois == []
        assert widget.selected_roi_index is None
        assert widget._is_flipped_v is False
        assert widget._is_flipped_h is False

        widget.close()

    def test_init_with_invalid_image(self):
        plt.switch_backend('Agg')

        # Test with 1D array
        with pytest.raises(ValueError, match="Input image_data must be a 2D numpy array"):
            ROISelectorWidget(np.array([1, 2, 3]))

        # Test with 3D array
        with pytest.raises(ValueError, match="Input image_data must be a 2D numpy array"):
            ROISelectorWidget(np.random.rand(10, 10, 3))

    def test_roi_callback_registration(self, widget):
        callback_mock = Mock()
        widget.register_roi_update_callback(callback_mock)

        # Trigger a callback
        widget._notify_roi_update()
        callback_mock.assert_called_once()

    def test_roi_callback_exception_handling(self, widget):
        def failing_callback():
            raise Exception("Test exception")

        widget.register_roi_update_callback(failing_callback)

        # Should not raise exception
        widget._notify_roi_update()

    def test_get_current_image_data(self, widget):
        current_data = widget.get_current_image_data()
        assert np.array_equal(current_data, widget.image_data)

    def test_get_rois_empty(self, widget):
        rois = widget.get_rois()
        assert rois == []

    def test_flip_vertical_toggle(self, widget):
        original_data = widget.image_data.copy()

        widget.check_flip.get_status = Mock(return_value=(True, False))

        # Trigger flip
        widget._on_flip_toggled("Flip Vertical")

        # Check that image was flipped
        expected_flipped = np.flipud(original_data)
        assert np.array_equal(widget.image_data, expected_flipped)
        assert widget._is_flipped_v is True
        assert widget._is_flipped_h is False

    def test_flip_horizontal_toggle(self, widget):
        original_data = widget.image_data.copy()

        widget.check_flip.get_status = Mock(return_value=(False, True))

        # Trigger flip
        widget._on_flip_toggled("Flip Horizontal")

        # Check that image was flipped
        expected_flipped = np.fliplr(original_data)
        assert np.array_equal(widget.image_data, expected_flipped)
        assert widget._is_flipped_v is False
        assert widget._is_flipped_h is True

    def test_flip_both_directions(self, widget):
        original_data = widget.image_data.copy()

        widget.check_flip.get_status = Mock(return_value=(True, True))

        # Trigger flip
        widget._on_flip_toggled("Flip Vertical")

        # Check that image was flipped in both directions
        expected_flipped = np.fliplr(np.flipud(original_data))
        assert np.array_equal(widget.image_data, expected_flipped)
        assert widget._is_flipped_v is True
        assert widget._is_flipped_h is True

    def test_roi_creation_simulation(self, widget):
        press_event = Mock()
        press_event.inaxes = widget.ax
        press_event.button = MouseButton.LEFT
        press_event.ydata = 20.0

        # Simulate press
        widget._on_press(press_event)

        assert widget.press_y == 20.0
        assert widget.current_rect is not None

        # Create mock event for release
        release_event = Mock()
        release_event.inaxes = widget.ax
        release_event.button = MouseButton.LEFT
        release_event.ydata = 40.0

        # Simulate release
        widget._on_release(release_event)

        # Check that ROI was created
        assert len(widget.rois) == 1
        assert widget.rois[0]['y_start'] == 20.0
        assert widget.rois[0]['y_end'] == 40.0
        assert widget.press_y is None
        assert widget.current_rect is None

    def test_roi_creation_with_inverted_coordinates(self, widget):
        press_event = Mock()
        press_event.inaxes = widget.ax
        press_event.button = MouseButton.LEFT
        press_event.ydata = 40.0

        release_event = Mock()
        release_event.inaxes = widget.ax
        release_event.button = MouseButton.LEFT
        release_event.ydata = 20.0

        # Simulate press and release
        widget._on_press(press_event)
        widget._on_release(release_event)

        # Check that ROI coordinates are normalized (start < end)
        assert len(widget.rois) == 1
        assert widget.rois[0]['y_start'] == 20.0
        assert widget.rois[0]['y_end'] == 40.0

    def test_roi_selection_update(self, widget):
        roi_patch = patches.Rectangle((0, 10), 150, 20, picker=5)
        roi_patch._roi_index = 0
        roi_data = {
            'patch': roi_patch,
            'y_start': 10.0,
            'y_end': 30.0
        }
        widget.rois.append(roi_data)
        widget.ax.add_patch(roi_patch)

        # Test selection
        widget._update_selection(0)
        assert widget.selected_roi_index == 0
        assert roi_patch.get_edgecolor() == (1.0, 0.0, 0.0, 1.0)  # Red color

        # Test deselection
        widget._update_selection(None)
        assert widget.selected_roi_index is None
        assert roi_patch.get_edgecolor() == (0.0, 1.0, 0.0, 1.0)  # Lime color

    def test_delete_selected_roi(self, widget):
        roi_patch = patches.Rectangle((0, 10), 150, 20, picker=5)
        roi_patch._roi_index = 0
        roi_data = {
            'patch': roi_patch,
            'y_start': 10.0,
            'y_end': 30.0
        }
        widget.rois.append(roi_data)
        widget.ax.add_patch(roi_patch)
        widget.selected_roi_index = 0

        # Delete the ROI
        widget.delete_selected_roi(None)

        # Check that ROI was removed
        assert len(widget.rois) == 0
        assert widget.selected_roi_index is None

    def test_delete_no_selection(self, widget):
        initial_roi_count = len(widget.rois)

        # Try to delete when nothing is selected
        widget.delete_selected_roi(None)

        # Should not change anything
        assert len(widget.rois) == initial_roi_count
        assert widget.selected_roi_index is None

    def test_roi_flip_coordinate_transformation(self, widget):
        roi_patch = patches.Rectangle((0, 20), 150, 20, picker=5)
        roi_patch._roi_index = 0
        roi_data = {
            'patch': roi_patch,
            'y_start': 20.0,
            'y_end': 40.0
        }
        widget.rois.append(roi_data)
        widget.ax.add_patch(roi_patch)

        # Mock checkbox for vertical flip
        widget.check_flip.get_status = Mock(return_value=(True, False))

        # Trigger flip
        widget._on_flip_toggled("Flip Vertical")

        # Check that ROI coordinates were transformed
        img_height = widget.img_height
        expected_y_start = img_height - 40.0  # Original y_end
        expected_y_end = img_height - 20.0    # Original y_start

        assert widget.rois[0]['y_start'] == expected_y_start
        assert widget.rois[0]['y_end'] == expected_y_end

    def test_ignore_small_rois(self, widget):
        # Create events for a very small ROI
        press_event = Mock()
        press_event.inaxes = widget.ax
        press_event.button = MouseButton.LEFT
        press_event.ydata = 20.0

        release_event = Mock()
        release_event.inaxes = widget.ax
        release_event.button = MouseButton.LEFT
        release_event.ydata = 20.5  # Very small height

        # Simulate press and release
        widget._on_press(press_event)
        widget._on_release(release_event)

        # Check that no ROI was created
        assert len(widget.rois) == 0

    def test_ignore_clicks_outside_axes(self, widget):
        # Create event outside axes
        press_event = Mock()
        press_event.inaxes = None  # Outside axes
        press_event.button = MouseButton.LEFT
        press_event.ydata = 20.0

        widget._on_press(press_event)

        # Should not start ROI creation
        assert widget.press_y is None
        assert widget.current_rect is None

    def test_ignore_wrong_mouse_button(self, widget):
        # Create event with right mouse button
        press_event = Mock()
        press_event.inaxes = widget.ax
        press_event.button = MouseButton.RIGHT
        press_event.ydata = 20.0

        widget._on_press(press_event)

        # Should not start ROI creation
        assert widget.press_y is None
        assert widget.current_rect is None

    def test_toolbar_mode_blocks_interaction(self, widget):
        # Create valid press event
        press_event = Mock()
        press_event.inaxes = widget.ax
        press_event.button = MouseButton.LEFT
        press_event.ydata = 20.0

        class MockToolbar:
            mode = ""

        # Set toolbar to zoom mode
        widget.fig.canvas.toolbar = MockToolbar()
        widget.fig.canvas.toolbar.mode = "zoom rect"

        widget._on_press(press_event)

        # Should not start ROI creation
        assert widget.press_y is None
        assert widget.current_rect is None


class TestCalibratedPlotter:

    def test_pixel_to_energy_conversion(self):
        pixels = np.array([0, 100, 200])
        slope = -0.1
        intercept = 7000

        expected_energies = np.array([7000, 6990, 6980])
        result = CalibratedPlotter.pixel_to_energy(pixels, slope, intercept)

        np.testing.assert_array_almost_equal(result, expected_energies)

    def test_plot_calibrated_projections_valid_input(self):
        plotter = CalibratedPlotter()

        # Create test data
        projections = [np.array([1, 2, 3, 4]), np.array([2, 3, 4, 5])]
        calibrations = [
            {'slope': -0.1, 'intercept': 7000},
            {'slope': -0.11, 'intercept': 7100}
        ]
        labels = ['ROI 1', 'ROI 2']

        # Mock matplotlib to avoid actual plotting
        with patch('matplotlib.pyplot.subplots') as mock_subplots:
            mock_fig = Mock()
            mock_ax = Mock()
            mock_subplots.return_value = (mock_fig, mock_ax)

            fig, ax = plotter.plot_calibrated_projections(
                projections, calibrations, labels
            )

            # Verify the plot was called correctly
            assert mock_subplots.called
            assert mock_ax.plot.call_count == 2
            assert mock_ax.set_xlabel.called
            assert mock_ax.set_ylabel.called
            assert mock_ax.set_title.called
            assert mock_ax.legend.called
            assert mock_ax.grid.called

    def test_plot_calibrated_projections_invalid_input(self):
        plotter = CalibratedPlotter()

        # Test mismatched lengths
        with pytest.raises(ValueError, match="Length of 'projections' and 'calibrations' must match"):
            plotter.plot_calibrated_projections(
                [np.array([1, 2, 3])], 
                [{'slope': 1, 'intercept': 0}, {'slope': 2, 'intercept': 1}]
            )

        # Test wrong types
        with pytest.raises(TypeError, match="'projections' and 'calibrations' must be lists or tuples"):
            plotter.plot_calibrated_projections(
                np.array([1, 2, 3]), 
                [{'slope': 1, 'intercept': 0}]
            )

        # Test invalid projection type
        with pytest.raises(TypeError, match="All items in 'projections' must be 1D numpy arrays"):
            plotter.plot_calibrated_projections(
                [[1, 2, 3]], 
                [{'slope': 1, 'intercept': 0}]
            )

        # Test invalid calibration format
        with pytest.raises(TypeError, match="All items in 'calibrations' must be dicts"):
            plotter.plot_calibrated_projections(
                [np.array([1, 2, 3])], 
                [{'slope': 1}]  # missing 'intercept'
            )

    def test_plot_calibrated_projections_empty_projections(self):
        plotter = CalibratedPlotter()

        fig, ax = plotter.plot_calibrated_projections([], [])

        assert fig is None
        assert ax is None


class TestSpectrometerCalibration:

    @pytest.fixture
    def dummy_image(self):
        np.random.seed(42)
        return np.random.rand(100, 300) * 20

    @pytest.fixture
    def widget_with_image(self, dummy_image):
        with patch('matplotlib.pyplot.ioff'), \
             patch('extra.gui.widgets.roi_selection.ROISelectorWidget'), \
             patch('extra.gui.widgets.peak_selection.PeakSelectorWidget'):
            return SpectrometerCalibration(image_data=dummy_image)

    def test_initialization_with_image_data(self, dummy_image):
        with patch('matplotlib.pyplot.ioff'), \
             patch('extra.gui.widgets.roi_selection.ROISelectorWidget'), \
             patch('extra.gui.widgets.peak_selection.PeakSelectorWidget'):

            widget = SpectrometerCalibration(image_data=dummy_image)

            np.testing.assert_array_equal(widget.image_data, dummy_image)
            assert widget.calibration_results == {}
            assert widget.peak_index_energy_inputs == {}

    def test_get_entered_peak_index_energies(self, widget_with_image):
        mock_widget1 = Mock()
        mock_widget1.value = 7000.0
        mock_widget2 = Mock()
        mock_widget2.value = None
        mock_widget3 = Mock()
        mock_widget3.value = 6950.0

        widget_with_image.peak_index_energy_inputs = {
            1: mock_widget1,
            2: mock_widget2,
            3: mock_widget3
        }

        result = widget_with_image.get_entered_peak_index_energies()

        expected = {1: 7000.0, 3: 6950.0}
        assert result == expected

    def test_get_calibration_results(self, widget_with_image):
        test_results = {
            0: {'slope': -0.1, 'intercept': 7000},
            1: {'slope': -0.11, 'intercept': 7100}
        }
        widget_with_image.calibration_results = test_results

        result = widget_with_image.get_calibration_results()
        assert result == test_results

    def test_prepare_spectra_data_no_calibration(self, widget_with_image):
        result = widget_with_image._prepare_spectra_data()

        # Should return empty lists
        assert result == ([], [], [], [])

    def test_prepare_spectra_data_with_calibration(self, widget_with_image):
        mock_roi = Mock()
        mock_roi.get_rois.return_value = [
            {'roi_index': 0, 'y_start': 10, 'y_end': 20}
        ]
        widget_with_image.roi_widget_instance = mock_roi

        # Setup calibration results
        widget_with_image.calibration_results = {
            0: {'slope': -0.1, 'intercept': 7000}
        }

        # Setup image data
        widget_with_image.processed_image_data = np.ones((100, 300)) * 5

        energy_axes, projections, labels, roi_indices = widget_with_image._prepare_spectra_data()

        assert len(energy_axes) == 1
        assert len(projections) == 1
        assert len(labels) == 1
        assert labels[0] == 'ROI 0'
        assert roi_indices[0] == 0
        assert len(projections[0]) == 300  # Width of image

    def test_update_calibration_tab_display(self, widget_with_image):
        mock_peak_widget = Mock()
        mock_peak_widget.get_selected_peaks.return_value = [
            {'roi_index': 0, 'peaks': [100.0, 200.0]},
            {'roi_index': 1, 'peaks': [150.0]}
        ]
        widget_with_image.peak_widget_instance = mock_peak_widget

        # Call the method
        widget_with_image._update_calibration_tab_display()

        # Check that energy input widgets were created (max 2 peaks)
        assert len(widget_with_image.peak_index_energy_inputs) == 2
        assert 1 in widget_with_image.peak_index_energy_inputs
        assert 2 in widget_with_image.peak_index_energy_inputs

    @patch('builtins.open', create=True)
    def test_save_results_basic(self, mock_open, widget_with_image):
        # Setup mock file handle
        mock_file = Mock()
        mock_open.return_value.__enter__.return_value = mock_file

        # Setup widget state
        widget_with_image.calibration_results = {
            0: {'slope': -0.1, 'intercept': 7000}
        }

        # Mock the ROI widget
        mock_roi = Mock()
        mock_roi.get_rois.return_value = [
            {'roi_index': 0, 'y_start': 10, 'y_end': 20}
        ]
        widget_with_image.roi_widget_instance = mock_roi

        # Mock the peak widget
        mock_peak = Mock()
        mock_peak.get_selected_peaks.return_value = [
            {'roi_index': 0, 'peaks': [100.0, 200.0]}
        ]
        widget_with_image.peak_widget_instance = mock_peak

        # Call save results
        widget_with_image._on_save_results_clicked(None)

        # Verify file was opened for writing
        mock_open.assert_called_once()
        args, kwargs = mock_open.call_args
        assert args[1] == 'w'  # Write mode

        # Verify file.write was called multiple times (for different sections)
        assert mock_file.write.call_count > 10


class TestPlotFromCalibrationFile:

    def test_plot_from_calibration_file_valid(self):
        # Create temporary test file
        test_content = """# === XES Calibration Widget Results ===
# Saved on: 2025-01-01_12-00-00
-------------------------------------
[ROI Definitions]
ROI 0: y_start = 10.00, y_end = 20.00
-------------------------------------
[Peak Selections (Pixels)]
ROI 0: [100.000, 200.000]
-------------------------------------
[Reference Energies Entered (eV)]
Peak Index 1: 7000.0000
Peak Index 2: 6950.0000
-------------------------------------
[Calibration Fit Results (per ROI)]
# Format: Energy = slope * Pixel + intercept
ROI 0: slope = -0.100000, intercept = 7100.0000
  - Points Used: (Pix=100.00, E=7000.00), (Pix=200.00, E=6950.00)
-------------------------------------
[Calibrated Spectra Data (CSV Format)]
Energy_eV,Intensity_ROI_0
7100.0000,10.5
7099.9000,11.2
7099.8000,12.1
7099.7000,13.0
# === End of Results ===
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(test_content)
            temp_path = f.name

        try:
            with patch('matplotlib.pyplot.subplots') as mock_subplots, \
                 patch('matplotlib.pyplot.show'):
                mock_fig = Mock()
                mock_ax = Mock()
                mock_subplots.return_value = (mock_fig, mock_ax)

                fig, ax = plot_from_calibration_file(temp_path)

                # Verify plot was created
                assert mock_subplots.called
                assert mock_ax.plot.called
                assert mock_ax.set_title.called
                assert mock_ax.set_xlabel.called
                assert mock_ax.set_ylabel.called
                assert mock_ax.legend.called
                assert mock_ax.grid.called
        finally:
            os.unlink(temp_path)

    def test_plot_from_calibration_file_not_found(self):
        fig, ax = plot_from_calibration_file('/non/existent/file.txt')

        assert fig is None
        assert ax is None

    def test_plot_from_calibration_file_no_csv_section(self):
        test_content = """# === XES Calibration Widget Results ===
Some other content without CSV section
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(test_content)
            temp_path = f.name

        try:
            fig, ax = plot_from_calibration_file(temp_path)

            assert fig is None
            assert ax is None
        finally:
            os.unlink(temp_path)
