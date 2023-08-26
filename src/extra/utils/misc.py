
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
    if ax is None:
        import matplotlib.pyplot as plt
        ax = plt

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

    # Set a default aspect ratio
    if "aspect" not in kwargs:
        aspect_ratio = max(image.shape) / min(image.shape)
        if aspect_ratio > 4:
            kwargs["aspect"] = "auto"

    return ax.imshow(image, *args, **kwargs)
