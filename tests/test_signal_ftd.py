
from extra.signal import cfd, dled, config_ftd_interpolation


def test_config_ftd_interpolation():
    assert config_ftd_interpolation() == {
        'sinc_search_iterations': 10, 'sinc_window': 200}

    assert config_ftd_interpolation(sinc_window=100) == {
        'sinc_search_iterations': 10, 'sinc_window': 100}

    assert config_ftd_interpolation(sinc_search_iterations=15) == {
        'sinc_search_iterations': 15, 'sinc_window': 100}

    assert config_ftd_interpolation(sinc_search_iterations=5,
                                    sinc_window=150) == {
        'sinc_search_iterations': 5, 'sinc_window': 150}

    assert config_ftd_interpolation() == {
        'sinc_search_iterations': 5, 'sinc_window': 150}
