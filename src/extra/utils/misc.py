import sys

import numpy as np

from typing import Any


def find_nearest_index(array, value: Any) -> np.int64:
    """Find array index for the nearest value.

    Args:
        array (array_like): Array to search.
        value (Any): Value to search.

    Returns:
        (np.int64): Index of the nearest array value.
    """

    return np.argmin(np.abs(array - value))


def find_nearest_value(array, value: Any) -> Any:
    """Find the nearest array value.

    Args:
        array (array_like): Array to search.
        value (Any): Value to search.

    Returns:
        (Any): Nearest array value.
    """

    return array[find_nearest_index(array, value)]


def reorder_axes_to_shape(a, target_shape):
    """Transpose an array to match the axis order specified by a shape tuple.

    All dimensions must have different sizes. One axis in target_shape may be
    None, a wildcard for the remainining axis in the array shape.
    """
    t = target_shape
    if len(set(t)) != len(t):
        raise ValueError(f"Target shape {t} has non-unique axes")
    if len(t) != len(a.shape):
        raise ValueError(f"Number of dimensions differs: {a.shape} -> {t}")
    if None in t:
        unmatched = set(a.shape) - set(t)
        if len(unmatched) != 1:
            raise ValueError(f"Cannot rearrange array shape {a.shape} to {t}")
        t = list(t)
        t[t.index(None)] = unmatched.pop()

    if set(t) != set(a.shape):
        raise ValueError(f"Cannot rearrange array shape {a.shape} to {t}")

    order = tuple([a.shape.index(l) for l in t])
    return a.transpose(order)


def _isinstance_no_import(obj, mod: str, cls: str):
    """Check if isinstance(obj, mod.cls) without loading mod"""
    m = sys.modules.get(mod)
    if m is None:
        return False

    return isinstance(obj, getattr(m, cls))


def imshow2(image, *args, colorbar=True, lognorm=False, ax=None, **kwargs):
    """Display an image with reasonable defaults.

    This function wraps [plt.imshow()][matplotlib.axes.Axes.imshow] to
    automatically set some defaults:

    - Try to set `vmin`/`vmax` to reasonable values. Note that setting
      `vmin`/`vmax` is incompatible with the `norm` argument, so they will only
      be set if `norm` is not passed.
    - Use an `auto` aspect ratio if the images aspect ratio is too skewed
      (useful for displaying heatmaps).
    - Set `interpolation="none"`.
    - Draw a colorbar.

    All arguments other than the ones listed below are passed to
    [plt.imshow()][matplotlib.axes.Axes.imshow], and explicitly passing any of
    `vmin`/`vmax`/`aspect`/`interpolation` will override the defaults.

    Args:
        image (array_like): The image to display.
        colorbar (bool): Whether to draw a colorbar.
        lognorm (bool): Whether to display the image in a log color scale.
        ax (matplotlib.axes.Axes): The axis to plot the image in.
    """
    import matplotlib.pyplot as plt
    is_dataarray = _isinstance_no_import(image, "xarray", "DataArray")

    # Disable interpolation by default
    if "interpolation" not in kwargs:
        kwargs["interpolation"] = "none"

    # Enable log color scale if requested and `norm` is not already set
    if lognorm and "norm" not in kwargs:
        from matplotlib.colors import LogNorm
        kwargs["norm"] = LogNorm()

    # Set the vmin/vmax if we're not using `norm`
    if "norm" not in kwargs and np.issubdtype(image.dtype, np.number):
        if "vmin" not in kwargs:
            vmin = np.nanquantile(image, 0.01)
            kwargs["vmin"] = vmin
        if "vmax" not in kwargs:
            vmax = np.nanquantile(image, 0.99)
            kwargs["vmax"] = vmax

    # Set a default aspect ratio (except for DataArray's because it doesn't
    # support string `aspect` values).
    if "aspect" not in kwargs and not is_dataarray:
        aspect_ratio = max(image.shape) / min(image.shape)
        if aspect_ratio > 4:
            kwargs["aspect"] = "auto"

    if ax is None:
        ax = plt.gca()
    fig = ax.get_figure()

    if is_dataarray:
        im = image.plot.imshow(*args, ax=ax, **kwargs)
    else:
        im = ax.imshow(image, *args, **kwargs)

    # Xarray will automatically add a colorbar so we only need to add one
    # explicitly for regular arrays.
    if colorbar and not is_dataarray:
        fig.colorbar(im, ax=ax)

    return im

def hyperslicer2(arr, *args, ax=None, lognorm=False, colorbar=True, **kwargs):
    """Interactively visualize arrays of images.

    This is a lightweight wrapper around
    [hyperslicer()][mpl_interactions.generic.hyperslicer] with some useful defaults:

    - Try to set `vmin`/`vmax` to reasonable values. Note that setting
      `vmin`/`vmax` is incompatible with the `norm` argument, so they will only
      be set if `norm` is not passed.
    - Set `interpolation="none"`.
    - Enable the play buttons.
    - Draw a colorbar.

    Example usage:
    ```python
    plt.figure()
    # Note the trailing semi-colon to swallow the return value. hyperslicer2()
    # returns a `controls` object by default that displays the play buttons, so
    # returning it from a notebook cell will end up displaying the play buttons
    # twice.
    hyperslicer2(images);
    ```
    ![](images/hyperslicer2.gif)

    All arguments other than the ones listed below are passed to
    [hyperslicer()][mpl_interactions.generic.hyperslicer], and explicitly
    passing any of `vmin`/`vmax`/`interpolation`/`play_buttons` will override
    the defaults.

    Args:
        arr (array_like): The array of images to display. Should have at least
            three dimensions.
        ax (matplotlib.axes.Axes): The axis to plot the image in.
        lognorm (bool): Whether to display the images in a log color scale.
        colorbar (bool): Whether to display a colorbar.
    """
    import matplotlib.pyplot as plt
    from mpl_interactions import hyperslicer

    # Enable the controls by default
    if "play_buttons" not in kwargs:
        kwargs["play_buttons"] = True

    # Disable interpolation by default
    if "interpolation" not in kwargs:
        kwargs["interpolation"] = "none"

    # Enable log color scale if requested and `norm` is not already set
    if lognorm and "norm" not in kwargs:
        from matplotlib.colors import LogNorm
        kwargs["norm"] = LogNorm()

    # Set the vmin/vmax if we're not using `norm`
    if "norm" not in kwargs and np.issubdtype(arr.dtype, np.number):
        if "vmin" not in kwargs:
            kwargs["vmin"] = np.nanquantile(arr, 0.01)
        if "vmax" not in kwargs:
            kwargs["vmax"] = np.nanquantile(arr, 0.99)

    if ax is None:
        ax = plt.gca()
    fig = ax.get_figure()

    controls = hyperslicer(arr, *args, ax=ax, **kwargs)

    if colorbar:
        fig.colorbar(ax.get_images()[-1], ax=ax)

    return controls


def ridgeplot(
        data, *, fig=None, overlap=0.5, xlabel=None, ylabel="Per-line values",
        ylim=None, yline=None, stack_label=None, stack_ticklabels=None
):
    """Make a ridgeline plot showing a sequence of similar lines

    A ridgeline plot spreads out the different lines vertically to make their
    order clear, but allowing them to overlap. It's an alternative to a heatmap,
    especially if there are relatively few rows (around 5-20).

    Args:
        data (array_like): A 2D array, each row of which will be plotted as one
            line, starting at the top of the plot. Pass an xarray DataArray to
            use its labels by default.
        fig (matplotlib.figure.Figure): Plot into an existing matplotlib figure.
        overlap (float): Number from 0 (no overlap) to 1, the fraction of each
            plot's area covered by the next plot.
        xlabel (str): Label for the shared x axis.
        ylabel (str): Label for the y axis (drawn on the bottom plot).
        ylim (tuple): Lower & upper limits for the y axis of each line.
        yline (float): Y value at which to draw a horizontal marker for each line.
        stack_label (str): Label for the stacking axis (shown on the right)
        stack_ticklabels (array_like): Labels for each line (shown on the right
            next to the zero line of each plot).
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as grid_spec
    from matplotlib.transforms import blended_transform_factory

    if isinstance(data, (list, tuple)):
        data = np.asarray(data)
    if data.ndim != 2:
        raise TypeError(f"Expected a 2D array (got {data.ndim}D)")

    if fig is None:
        fig = plt.figure(figsize=(8, 6), layout="constrained")

    gs = grid_spec.GridSpec(len(data), 1, hspace=-overlap)

    if _isinstance_no_import(data, "xarray", "DataArray"):
        x_data = data.coords[data.dims[1]]
        xlabel = xlabel or data.dims[1]
        stack_label = stack_label or data.dims[0]
        if stack_ticklabels is None and data.dims[0] in data.coords:
            stack_ticklabels = data.coords[data.dims[0]].values
    else:  # Numpy array
        x_data = np.arange(data.shape[1])

    x_range = np.nanmin(x_data), np.nanmax(x_data)
    if ylim is not None:
        y_min, y_max = ylim
    else:
        y_min, y_max = np.nanmin(data), np.nanmax(data)
        if y_min > 0 and (y_max / y_min) > 20:
            y_min = 0  # Data from just above 0
        elif y_max < 0 and (y_min / y_max) > 20:
            y_max = 0  # Data from just below 0

    if yline is None:
        if y_min <= 0 <= y_max:
            yline = 0
        else:
            yline = np.median(data)

    for i, trace in enumerate(data):
        ax = fig.add_subplot(gs[i:i + 1, 0:])
        ax.patch.set_alpha(0)  # Transparent background
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.plot(x_data, trace)

        # Draw a light line to mark each separate dataset
        ax.axhline(yline, color='0.7', linewidth=1., zorder=0)

        # Use the same scale on each axes
        ax.set_ylim(y_min, y_max)
        ax.set_xlim(*x_range)

        if i < len(data) - 1:
            # Clear stuff from all but the bottom axes
            ax.set_xticklabels([])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_yticklabels([])
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
        else:
            # Bottom axes
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

        # the x coords of this transformation are axes, and the y coords are data
        if stack_ticklabels is not None:
            trans = blended_transform_factory(ax.transAxes, ax.transData)
            ax.text(1.02, yline, str(stack_ticklabels[i]), ha="left", va="center", transform=trans)

    if stack_label:
        fig.supylabel(stack_label, x=1., ha="right")
    return fig
