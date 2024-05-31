
import numpy as np

from extra.utils import imshow2, fit_gaussian, gaussian


def test_scaled_imshow():
    # Smoke test
    image = np.random.rand(100, 100)
    imshow2(image)


def test_fit_gaussian():
    # Test with auto-generated xdata and nans/infs
    params = [0, 1, 20, 5]
    data = gaussian(np.arange(100), *params)
    data[50] = np.nan
    data[51] = np.inf
    popt = fit_gaussian(data)
    assert np.allclose(popt, params)

    # Test with provided xdata
    params = [0, 1, 0.5, 0.2]
    xdata = np.arange(0, 1, 0.01)
    data = gaussian(xdata, *params)
    popt = fit_gaussian(data, xdata=xdata)
    assert np.allclose(popt, params)
