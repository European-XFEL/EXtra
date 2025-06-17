from datetime import timedelta

import numpy as np

from extra_data import SourceData, KeyData, by_id
from .utils import _isinstance_no_import

class Scan:
    """Detect steps in a motor scan.

    Example usage in a Jupyter notebook:

    ```python
            -----------------------------------------------------------
    In [1]: |s = Scan(run["MOTOR/MCMOTORYFACE"])                      |
            |s.info()                                                 |
            -----------------------------------------------------------
    Out[1]: Scan over MOTOR/MCMOTORYFACE.actualPosition from -0.5800 to -0.5600:
             Steps: 21
             Scan time: 0:17:54
             Average step length: 511.33 trains (51.13s)
             Average step size: 1.00e-03 [arb. u.]

            Detection parameters:
             resolution: 5.00e-04
             min_trains: 235

            ----------------------------------------------------------
    In [2]: |s.plot() # Check the detection results                  |
            ----------------------------------------------------------
    ```
    """

    def __init__(self, motor, name=None, resolution=None, min_trains=None, intra_step_filtering=1):
        """The class tries to detect the best parameters automatically, but it will
        still fail in some situations. If that happens either the `resolution`
        or `min_trains` parameters (unlikely to be both) will need to be set
        manually, typically on a per-motor rather than per-run basis.

        Note:
            You should always check the results of the detection with
            [Scan.plot()][extra.components.Scan.plot] to verify that they're
            correct.

        Args:
            motor (SourceData, KeyData, DataArray): An
                object that contains the motor positions and train IDs. If it's a
                `SourceData` object the `actualPosition` key will be used. Note that
                if a [xarray.DataArray][] is passed it *must* have a `trainId`
                coordinate with the train IDs (i.e. obtained with
                [KeyData.xarray()][extra_data.KeyData.xarray]). Raw Numpy arrays are not
                allowed.
            resolution (float): The usable resolution of the motor. Ideally this
                should be slightly above the noise level of the motor (and below the
                step size of the scan). It will be guessed if not passed explicitly.

                This is the trickiest parameter to guess automatically. For
                example, if it's set too high then multiple real steps will be
                detected as single steps. And if it's set too low and there's a
                lot of drift in the motor steps, there can be multiple detected
                steps within a single real step.
            min_trains (int): The minimum number of trains per-step in the
                scan. It will be guessed if not passed explicitly.

                The automatic guessing rarely fails badly. Usually you will only
                need to tweak this if the length of the steps varies a lot and
                some of the smaller steps are being cut out (this often happens
                with energy scans).
            intra_step_filtering (float): A factor that influences how
                aggressively noisy trains within a step will be filtered
                out. Higher values mean less aggressive and lower values mean
                more aggressive.

                This factor is used to remove trains in a step that are outside
                a certain range of the mean value in the step, the main purpose
                being to remove trains on the boundaries of a step. Try tweaking
                this value if there are too many/too few trains being filtered
                around a step.
        """
        self._steps = []
        self._resolution = None
        self._min_trains = None
        self._intra_step_filtering = intra_step_filtering

        # Debugging variables
        self._diff = None

        from xarray import DataArray

        if isinstance(motor, SourceData):
            self._input_pos = motor["actualPosition"].xarray()
            default_name = f"{motor.source}.actualPosition"
        elif isinstance(motor, KeyData):
            if not motor.ndim == 1:
                raise ValueError(f"KeyData of motor positions must be 1-dimensional, is actually {motor.ndim}-dimensional")

            self._input_pos = motor.xarray()
            default_name = f"{motor.source}.{motor.key.removesuffix('.value')}"
        elif isinstance(motor, DataArray):
            if not "trainId" in motor.coords:
                raise ValueError("DataArray of motor positions must have a trainId coordinate")

            self._input_pos = motor
            default_name = motor.name if motor.name is not None else "motor"
        else:
            raise TypeError(f"Unrecognized input type: {type(motor)}")

        self._name = name if name is not None else default_name

        steps = self._get_motor_steps(self._input_pos, resolution,
                                      min_trains, intra_step_filtering)

        # Sometimes motors that have jitter are erroneously detected as a
        # single-step scan, so we ignore those and only take scans with more
        # than 1 step detected.
        if len(steps) > 1:
            self._steps = steps

        self._positions = np.array([pos for pos, _ in self.steps])
        self._positions_train_ids = [tids for _, tids in self.steps]

    @property
    def name(self) -> str:
        """Name of the device being scanned."""
        return self._name

    @property
    def steps(self) -> list:
        """List of `(position, train_ids)` tuples for each step in the scan."""
        return self._steps

    @property
    def positions(self) -> np.ndarray:
        """Array of positions for each step."""
        return self._positions

    @property
    def positions_train_ids(self) -> list:
        """List of train IDs for each position."""
        return self._positions_train_ids

    def plot(self, figsize=(10, 6), ax=None):
        """Visualize the scan steps.

        Each step is plotted in a different color on top of the motor
        positions. On a long scan with lots of steps you'll probably want to
        zoom in to check the individual steps.

        Example plot:
        ![](../images/scan-plot.png)
        """
        from matplotlib.collections import PatchCollection
        from matplotlib.patches import Rectangle
        if ax is None:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=figsize)

        # Show all the motor values
        ax.plot(self._input_pos.trainId, self._input_pos)

        # Label up to 8 of the steps
        n_steps = len(self._steps)
        label_steps = range(0, n_steps, n_steps // min(n_steps, 8))

        max_tid, min_tid = self._input_pos.trainId.max(), self._input_pos.trainId.min()
        mid_train_id = (max_tid + min_tid) / 2

        step_centres_x, step_centres_y = [], []
        rects = []

        for i, (pos, train_ids) in enumerate(self.steps):
            left, right = train_ids.min(), train_ids.max()
            width = right - left
            centre_x = (left + right) / 2
            bottom, top = (pos - self._resolution), (pos + self._resolution)
            rects.append(Rectangle((left, bottom), width, top - bottom))
            step_centres_x.append(centre_x)
            step_centres_y.append(pos)

            if i in label_steps:
                if centre_x < mid_train_id:
                    # Left half of plot - label to the right
                    lbl_x = min(right + 2 * width, max_tid)
                    line_x_min = min(right + 0.5 * width, max_tid)
                    line_x_max = min(right + 1.75 * width, max_tid)
                    ha = 'left'
                else:
                    # Right half of plot - label to the left
                    lbl_x = max(left - 2 * width, min_tid)
                    line_x_max = max(left - 0.5 * width, min_tid)
                    line_x_min = max(left - 1.75 * width, min_tid)
                    ha = 'right'
                ax.text(lbl_x, pos, str(i),
                        verticalalignment='center_baseline', horizontalalignment=ha)
                ax.plot([line_x_min, line_x_max], [pos, pos], color='k')

        # Highlight the step position +/- the resolution
        ax.add_collection(
            PatchCollection(rects, facecolor=(1., 0.75, 1.), edgecolor=None)
        )
        ax.scatter(step_centres_x, step_centres_y, marker='x', color='k')

        ax.set_xlabel("Train ID")
        ax.set_title(f"Scan over {self.name} with {len(self.steps)} steps")

        return ax

    def bin_by_steps(self, data, uncertainty_method="std"):
        """Average train-resolved data within each scan step.

        This will return a [DataArray][xarray.DataArray] containing the averaged
        value of `data` over each step in the scan, along with `position`,
        `counts`, and `uncertainty` coordinates containing the
        position/counts/uncertainty for each step. The `.name` property will be
        set to `data.name`, and the following attributes will be set:

        - `motor`: `scan.name`
        - `uncertainty_method`: the `uncertainty_method` that was passed,
          indicating either the standard deviation or standard error.

        Args:
            data (xarray.DataArray): A train-resolved (i.e. with a `trainId`
                coordinate) array to bin.
            uncertainty_method (str): Can be set to `std` to use the standard
                deviation (i.e. the 1-sigma region) or `stderr` to use the
                [standard error](https://en.wikipedia.org/wiki/Standard_error).
        """
        import xarray as xr
        if not isinstance(data, xr.DataArray) or "trainId" not in data.coords:
            raise TypeError("Input must be a DataArray with a `trainId` coordinate")
        elif uncertainty_method != "std" and uncertainty_method != "stderr":
            raise ValueError(f"`uncertainty_method` must be either 'std' or 'stderr', not: {uncertainty_method}")

        chunks = self.split_by_steps(data)

        res = xr.concat(
            [c.mean(dim='trainId') for c in chunks],
            dim=xr.Variable("position", self.positions)
        )

        counts = np.array([len(c.coords['trainId']) for c in chunks])
        uncertainty = np.array([c.std().item() for c in chunks])
        if uncertainty_method == "stderr":
            uncertainty /= np.sqrt(counts)

        return res.assign_coords({
            "uncertainty": ("position", uncertainty),
            "counts": ("position", counts),
        }).assign_attrs({
            "motor": self.name,
            "uncertainty_method": uncertainty_method,
        }).rename(data.name)

    def plot_bin_by_steps(self, data, uncertainty_method="std",
                          title=None, xlabel=None, ylabel=None,
                          ax=None, figsize=(9, 5)):
        """Plot step-averaged data.

        This calls [Scan.bin_by_steps()][extra.components.Scan.bin_by_steps] and
        plots the result. Note that while it's possible to explicitly specify
        the title/xlabel/ylabel, it's recommended to set `data.name` and
        `scan.name` and let the plot settings be inferred automatically.

        Example plot with `data.name == "ROI intensity"` and `scan.name ==
        "Theta"`:
        ![](../images/scan-plot-bin-by-steps.png)

        Args:
            data (xarray.DataArray): A train-resolved array to pass to
                [Scan.bin_by_steps()][extra.components.Scan.bin_by_steps].
            uncertainty_method (str): Same as in
                [Scan.bin_by_steps()][extra.components.Scan.bin_by_steps].
            title (str): The title of the plot.
            xlabel (str): The xlabel of the plot.
            ylabel (str): The ylabel of the plot.
            ax (matplotlib.axes.Axes): The axes to plot on. A figure will be
                created if this is not set.
            figsize (tuple): The figure size. Only used if `ax=None`.

        """
        if ax is None:
            import matplotlib.pyplot as plt
            _, ax = plt.subplots(figsize=figsize)

        binned_data = self.bin_by_steps(data)

        if binned_data.ndim == 1:
            uncertainty_label = "standard deviation" if uncertainty_method == "std" else "standard error"
            binned_data.plot.line("-o", markersize=4, label=f"Uncertainty: {uncertainty_label}", ax=ax)
            ax.fill_between(binned_data.position,
                            binned_data - binned_data.uncertainty,
                            binned_data + binned_data.uncertainty,
                            alpha=0.5)
            ax.grid()
            ax.legend()

            if binned_data.name is not None:
                yaxis = binned_data.name
            else:
                ylabel = "Signal [arb. u.]" if ylabel is None else ylabel
                yaxis = "Signal"

            if xlabel is None:
                xlabel = self.name

            if title is None:
                title = f"{yaxis} vs {self.name}"
        else:  # 2D
            binned_data.plot(ax=ax)

            if ylabel is None:
                ylabel = self.name

        if title:
            ax.set_title(title)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)

        return ax

    def split_by_steps(self, data):
        """Split an EXtra or EXtra-data object, or a DataArray, into scan steps.

        This will yield one shorter object of the same type for each step.
        It should work for any object with a ``.select_trains()`` method, and
        for xarray DataArray objects with a 'trainId' coordinate.
        """
        if _isinstance_no_import(data, 'xarray', 'DataArray') and "trainId" in data.coords:
            return [
                data.sel(trainId=np.intersect1d(tids, data.coords['trainId']))
                for tids in self.positions_train_ids
            ]
        elif hasattr(data, 'select_trains'):
            return [
                data.select_trains(by_id[tids]) for tids in self.positions_train_ids
            ]
        else:
            raise TypeError("Input must either have a select_trains method, or "
                            "be a DataArray with a `trainId` coordinate")

    def _plot_resolution_data(self):
        """Plot the data points that used to guess the resolution.

        This is an internal function meant to help with debugging.
        """
        if len(self.steps) == 0:
            print("No data, automatic resolution detection was not used.")
            return

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 5))

        diff_filt_masked = self._diff.copy()
        diff_filt_masked[~self._filter_positions_mask(self._diff)] = 0

        plt.plot(self._diff, "*", label="Full diff points")
        plt.plot(diff_filt_masked, "*", label="Filtered points")
        plt.axhline(self._diff_lb, c="k")
        plt.axhline(self._diff_ub, c="k")
        plt.legend()

        return ax

    @classmethod
    def _mkscan(cls, n_steps, step_size=10, step_length=10, step_length_rnd=0):
        """Create a mock scan with specific parameters.

        This is an internal function to help with debugging and testing.
        """
        # Helper lambda to calculate an offset to add to the step length to
        # randomize it, such that the step length is between 0x-2x step_length.
        length_offset = lambda: int(step_length_rnd * np.random.uniform(-step_length, step_length))
        steps = [np.full(max(1, step_length + length_offset()),
                         i * step_size)
                 for i in range(n_steps)]

        motor = np.concatenate(steps, dtype=np.float64)

        from xarray import DataArray
        motor = DataArray(motor,
                          name="fake-motor",
                          dims=("trainId",),
                          coords={
                              "trainId": np.arange(len(motor))
                          })

        return cls(motor), steps

    def _filter_positions_mask(self, positions):
        return (self._diff_lb <= positions) & (positions <= self._diff_ub)

    def _guess_resolution(self, position):
        # When guessing the resolution we only take the middle 50% of trains to
        # try to cut out any backlash around the ends.
        cutoff_trains = int(len(position) * 0.25)
        self._position_subset = position[cutoff_trains:len(position) - cutoff_trains]

        self._diff = np.diff(self._position_subset)
        self._diff = self._diff[self._diff != 0]

        # If there is no non-zero diffs the motor definitely isn't moving at all
        if len(self._diff) == 0:
            return None

        # Filter the deltas by two standard deviations. This is
        # partly to remove any backlash from the motor moving into
        # the start position, partly to try to remove any movements
        # smaller than the step size.
        diff_std = np.nanstd(self._diff)
        diff_mean = np.nanmean(self._diff)
        self._diff_lb = diff_mean - diff_std * 2
        self._diff_ub = diff_mean + diff_std * 2
        diff_filt = self._diff[self._filter_positions_mask(self._diff)]

        # If everything has been filtered out then the motor probably isn't
        # moving at all.
        if len(diff_filt) == 0:
            return None

        # If all the diff's sum to 0 then taking the weighted average will fail
        # later with a ZeroDivisionError. This only happens in very specific
        # situations where the motor value is jittering between two values such
        # that the diffs have a constant magnitude `C`and jump between +C and
        # -C. And if there are equal counts of +C and -C then the total sum will
        # be 0.
        if np.sum(diff_filt) == 0:
            return None

        # Take the average of the diffs, weighted by their value
        # to try to ignore jitter or drift within the steps, which
        # will show up as lots of little diffs.
        est_step_size = np.average(diff_filt, weights=diff_filt)

        # Rule of thumb that seems to work: use a resolution of half the
        # estimated step size.
        return np.abs(est_step_size) / 2

    def _guess_min_trains(self, steps):
        step_lengths = np.array([len(tids) for _, tids in steps
                                if len(tids) > 1])
        mean_step_length = np.nanmean(step_lengths)

        # Rule of thumb that seems to work: set min_trains to roughly half the
        # mean step size (absolute minimum is 2 trains).
        return int(max(2, mean_step_length // 2))

    def _get_motor_steps(self, actual_pos, resolution, min_trains, intra_step_filtering):
        if resolution is None:
            resolution = self._guess_resolution(actual_pos)

            # If the motor isn't moving at all then _guess_resolution() will
            # fail and return None.
            if resolution is None:
                return []

        self._resolution = resolution

        # Initialize the positions
        steps = [ (actual_pos[0].item(), [actual_pos.trainId[0].item()]) ]

        for x in actual_pos[1:]:
            # Assume that the current step is the last position added
            current_step = steps[-1][0]

            if current_step - resolution < x < current_step + resolution:
                # If we're within bounds of the current step, add the train ID
                steps[-1][1].append(x.trainId.item())
            else:
                # Otherwise create a new step.
                new_step = x.item()
                steps.append((new_step, [x.trainId.item()]))

        if min_trains is None:
            min_trains = self._guess_min_trains(steps)
        self._min_trains = min_trains

        # Final filtering
        step_idxs_to_delete = []
        for i in range(len(steps)):
            _, tids = steps[i]

            # Remove trains that are outside `intra_step_filtering` standard
            # deviations of the mean position of the step.
            step_positions = actual_pos.sel(trainId=tids)
            std = np.std(step_positions).item() * intra_step_filtering
            mean = np.nanmean(step_positions).item()

            for tid in tids.copy():
                step_pos = step_positions.sel(trainId=tid).item()
                if not (mean - std <= step_pos <= mean + std):
                    tids.remove(tid)

            # Remove all positions with too few trains
            if len(tids) < min_trains:
                step_idxs_to_delete.append(i)

        # Delete from list in reverse order so that we don't invalidate any
        # indices in the process.
        for i in sorted(step_idxs_to_delete, reverse=True):
            del steps[i]

        # Ensure that the range of trains selected for each step is
        # contiguous. We assume that there's no need to remove trains from the
        # middle of a step.
        all_tids = actual_pos.trainId.data
        for i in range(len(steps)):
            step_tids = steps[i][1]
            start_tid_idx = np.argwhere(all_tids == step_tids[0])[0][0]
            end_tid_idx = np.argwhere(all_tids == step_tids[-1])[0][0]

            # Note that we add 1 to end_tid_idx so that the end train ID will be
            # included too, otherwise it'd be skipped as the stop value of a
            # range.
            final_tids = all_tids[start_tid_idx:end_tid_idx + 1]

            # Go over the positions and set them to the mean value of the motor for
            # their train IDs. Right now they're set to the value of the first train
            # in the step, which can be subtly off from what a user would expect.
            final_position = actual_pos.sel(trainId=final_tids).mean().item()

            steps[i] = (final_position, final_tids)

        # Detect backlash at the beginning by comparing the step direction of
        # the first and second steps. If there's backlash they're typically not
        # the same, so we remove the first step if so.
        if len(steps) >= 3:
            first_step = steps[1][0] - steps[0][0]
            second_step = steps[2][0] - steps[1][0]
            if np.sign(first_step) != np.sign(second_step):
                del steps[0]

        return steps

    def __repr__(self):
        return f"<{self.format(compact=True)}>"

    def info(self, compact=False):
        """Print information about the scan from [Scan.format()][extra.components.Scan.format]"""
        print(self.format(compact=compact))

    def format(self, compact=False) -> str:
        """Format information about the scan as a string.

        Args:
            compact (bool): Whether to format the information in a one-line or
                multi-line string.
        """
        if compact:
            return f"Scan over {self.name} with {len(self.steps)} steps"

        if len(self.steps) == 0:
            return f"No steps detected for {self.name}"

        start_pos = self.positions[0]
        end_pos = self.positions[-1]
        n_steps = len(self.positions)
        avg_step_length = np.nanmean([len(tids) for tids in self.positions_train_ids])
        avg_step_time = avg_step_length / 10
        avg_step_size = np.nanmean(np.diff(self.positions))

        start_train = self.positions_train_ids[0][0]
        end_train = self.positions_train_ids[-1][-1]
        scan_time = timedelta(seconds=round((end_train - start_train) / 10))

        return f"Scan over {self.name} from {start_pos:.4f} to {end_pos:.4f}:\n" \
               f" Steps: {n_steps}\n" \
               f" Scan time: {scan_time}\n" \
               f" Average step length: {avg_step_length:.2f} trains ({avg_step_time:.2f}s)\n" \
               f" Average step size: {avg_step_size:.2e} [arb. u.]\n" \
               "\n" \
               f"Detection parameters:\n" \
               f" resolution: {self._resolution:.2e}\n" \
               f" min_trains: {self._min_trains}\n" \
               f" intra_step_filtering: {self._intra_step_filtering}"
