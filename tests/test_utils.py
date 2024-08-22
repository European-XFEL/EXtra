
import numpy as np

from extra.utils import imshow2, fit_gaussian, gaussian


def test_scaled_imshow():
    # Smoke test
    image = np.random.rand(100, 100)
    imshow2(image)


def test_fit_gaussian():
    # Test with auto-generated xdata and nans/infs
    params = [0, 1, 20, 5]
    data = gaussian(np.arange(100), *params, norm=False)
    data[50] = np.nan
    data[51] = np.inf
    popt = fit_gaussian(data)
    assert np.allclose(popt, params)

    # Test with A constrained to be >0
    popt_A_pos = fit_gaussian(data, Asign=1)
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
    popt = fit_gaussian(data, Asign=-1)
    assert np.allclose(popt, params)
