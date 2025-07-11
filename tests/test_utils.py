import numpy as np
import pytest
import xarray as xr

from extra.utils import (
    imshow2, hyperslicer2, fit_gaussian, gaussian, reorder_axes_to_shape,
)


def test_imshow2():
    # Smoke test
    image = np.random.rand(100, 100)
    imshow2(image)

    image = xr.DataArray(image, dims=("foo", "bar"))
    imshow2(image)


def test_hyperslicer2():
    # Smoke test
    images = np.random.rand(10, 100, 100)
    hyperslicer2(images)

def test_fit_gaussian():
    # Test with auto-generated xdata and nans/infs
    params = [0, 1, 20, 5]
    data = gaussian(np.arange(100), *params, norm=False)
    data[50] = np.nan
    data[51] = np.inf
    popt = fit_gaussian(data)
    assert np.allclose(popt, params)

    # Test with A constrained to be >0
    popt_A_pos = fit_gaussian(data, A_sign=1)
    assert np.allclose(popt_A_pos, params)

    # Test with provided xdata
    params = [0, 1, 0.5, 0.2]
    xdata = np.arange(0, 1, 0.01)
    data = gaussian(xdata, *params, norm=False)
    popt = fit_gaussian(data, xdata=xdata)
    assert np.allclose(popt, params)

    # Test with a downwards-pointing peak
    params = [0, -3, 20, 5]
    data = gaussian(np.arange(100), *params, norm=False)
    popt = fit_gaussian(data, A_sign=-1)
    assert np.allclose(popt, params)


def test_reorder_axes_to_shape():
    arr = np.zeros((512, 1024, 16), dtype=np.float32)  # E.g. burst mode JUNGFRAU data
    res = reorder_axes_to_shape(arr, (16, 512, 1024))
    assert res.shape == (16, 512, 1024)
    assert res.base is arr

    res = reorder_axes_to_shape(arr, (None, 512, 1024))
    assert res.shape == (16, 512, 1024)
    assert res.base is arr

    with pytest.raises(ValueError):
        reorder_axes_to_shape(arr, (12, 512, 1024))  # Wrong dimension sizes

    with pytest.raises(ValueError):
        reorder_axes_to_shape(arr, (16, 1, 512, 1024))  # Wrong number of dimensions

    with pytest.raises(ValueError):
        reorder_axes_to_shape(arr[:, :512], (16, 512, 512))  # Ambiguous order

    with pytest.raises(ValueError):
        reorder_axes_to_shape(arr, (None, None, 1024))  # Only 1 None allowed

    with pytest.raises(ValueError):
        reorder_axes_to_shape(arr, (None, 256, 1024))  # Wildcard & wrong number

    # Check we've transposed, not reshaped
    arr = np.arange(15).reshape(3, 5)
    res = reorder_axes_to_shape(arr, (5, 3))
    assert res.shape == (5, 3)
    np.testing.assert_array_equal(res[0], [0, 5, 10])
