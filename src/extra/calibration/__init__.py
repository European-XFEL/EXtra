
from enum import IntFlag

from .calcat import CalCatAPIError, CalCatAPIClient, get_client
from .detector import DetectorData, DetectorModule
from .conditions import AutoConditionsError, AGIPDConditions, LPDConditions, \
    DSSCConditions, JUNGFRAUConditions, ShimadzuHPVX2Conditions
from .constants import SingleConstant, MultiModuleConstant, CalibrationData, \
    lpd_dark_consts_with_fallback


class BadPixels(IntFlag):
    """Bad pixel reasons, as used in masks in corrected detector data"""
    OFFSET_OUT_OF_THRESHOLD  = 1 << 0
    NOISE_OUT_OF_THRESHOLD   = 1 << 1
    OFFSET_NOISE_EVAL_ERROR  = 1 << 2
    NO_DARK_DATA             = 1 << 3
    CI_GAIN_OF_OF_THRESHOLD  = 1 << 4
    CI_LINEAR_DEVIATION      = 1 << 5
    CI_EVAL_ERROR            = 1 << 6
    FF_GAIN_EVAL_ERROR       = 1 << 7
    FF_GAIN_DEVIATION        = 1 << 8
    FF_NO_ENTRIES            = 1 << 9
    CI2_EVAL_ERROR           = 1 << 10
    VALUE_IS_NAN             = 1 << 11
    VALUE_OUT_OF_RANGE       = 1 << 12
    GAIN_THRESHOLDING_ERROR  = 1 << 13
    DATA_STD_IS_ZERO         = 1 << 14
    ASIC_STD_BELOW_NOISE     = 1 << 15
    INTERPOLATED             = 1 << 16
    NOISY_ADC                = 1 << 17
    OVERSCAN                 = 1 << 18
    NON_SENSITIVE            = 1 << 19
    NON_LIN_RESPONSE_REGION  = 1 << 20
    WRONG_GAIN_VALUE         = 1 << 21
    NON_STANDARD_SIZE        = 1 << 22


__all__ = [
    "BadPixels",
    "CalCatAPIError",
    "CalCatAPIClient",
    "SingleConstant",
    "MultiModuleConstant",
    "CalibrationData",
    "AGIPDConditions",
    "LPDConditions",
    "DSSCConditions",
    "JUNGFRAUConditions",
    "ShimadzuHPVX2Conditions",
    "DetectorData",
    "DetectorModule"
    "AutoConditionsError"
]


