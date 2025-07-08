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


def imshow2(image, *args, lognorm=False, ax=None, **kwargs):
    """Display an image with reasonable defaults.

    This function wraps [plt.imshow()][matplotlib.axes.Axes.imshow] to
    automatically set some defaults:

    - Try to set `vmin`/`vmax` to reasonable values. Note that setting
      `vmin`/`vmax` is incompatible with the `norm` argument, so they will only
      be set if `norm` is not passed.
    - Use an `auto` aspect ratio if the images aspect ratio is too skewed
      (useful for displaying heatmaps).
    - Set `interpolation="none"`.

    All arguments other than the ones listed below are passed to
    [plt.imshow()][matplotlib.axes.Axes.imshow], and explicitly passing any of
    `vmin`/`vmax`/`aspect`/`interpolation` will override the defaults.

    Args:
        image (array_like): The image to display.
        lognorm (bool): Whether to display the image in a log color scale.
        ax (matplotlib.axes.Axes): The axis to plot the image in. This will
            default to [plt][matplotlib.pyplot] if none is explicitly passed.
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

    if is_dataarray:
        return image.plot.imshow(*args, ax=ax, **kwargs)
    else:
        if ax is None:
            ax = plt
        return ax.imshow(image, *args, **kwargs)

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
