import numpy as np

from extra.utils import imshow2, fit_gaussian, gaussian
from extra.utils.ftd import cfd, dled, config_sinc_interpolation


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


def test_ftd_sinc_interpolation():
    assert config_sinc_interpolation() == {
        'search_iterations': 10, 'window': 200}

    assert config_sinc_interpolation(window=100) == {
        'search_iterations': 10, 'window': 100}

    assert config_sinc_interpolation(search_iterations=15) == {
        'search_iterations': 15, 'window': 100}

    assert config_sinc_interpolation(search_iterations=5, window=150) == {
        'search_iterations': 5, 'window': 150}

    assert config_sinc_interpolation() == {
        'search_iterations': 5, 'window': 150}
