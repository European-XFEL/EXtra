
import numpy as np

from typing import Any
from numpy.typing import ArrayLike


def find_nearest_index(array: ArrayLike, value: Any) -> np.int64:
    """Find array index for the nearest value.

    Args:
        array (ArrayLike): Array to search.
        value (Any): Value to search.

    Returns:
        (np.int64) Index of the nearest array value.
    """

    return np.argmin(np.abs(array - value))


def find_nearest_value(array: ArrayLike, value: Any) -> Any:
    """Find the nearest array value.

    Args:
        array (ArrayLike): Array to search.
        value (Any): Value to search.

    Returns:
        (Any) Nearest array value.
    """

    return array[find_nearest_index(array, value)]
