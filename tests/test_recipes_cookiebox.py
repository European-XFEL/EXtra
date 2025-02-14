from unittest.mock import patch

import pint
import pytest
import numpy as np
import pandas as pd
import xarray as xr

from extra.recipes import CookieboxCalibration

def test_create_cookiebox_calibration():
    # instantiates it without doing any calibration, only to check for syntax errors
    cal = CookieboxCalibration()

