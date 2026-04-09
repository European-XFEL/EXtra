from unittest.mock import patch

import pint
import pytest

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from extra.data import open_run, by_id
from extra.components import XGM, Scan, AdqRawChannel
from extra.applications import CookieboxCalibration
from extra.applications.cookiebox import TofFitResult
from extra.applications.cookiebox_deconvolve import TOFAnalogResponse


# this produces mock data
def produce_mock_fit_result():
    tof_fit_result = dict()
    calibration_mean_xgm = dict()
    correct = dict()
    #tof_fit_result[0] = TofFitResult(energy=np.array([969.966, 979.971, 989.969, 999.978, 1009.957, 1019.964, 1029.956, 1039.950, 1049.967, 1059.949]),
    #                                        mu=np.array([82.000, 214.000, 175.000, 164.000, 155.000, 148.000, 142.000, 138.000, 134.000, 130.000]),
    #                                        sigma=np.array([1.500, 4.250, 2.074, 2.047, 2.087, 1.555, 1.884, 2.264, 2.083, 2.219]),
    #                                        A=np.array([52.871, 155.973, 101.682, 164.055, 161.534, 84.793, 105.596, 154.320, 160.713, 121.223]),
    #                                        Aa=np.array([108.801, 81.371, 68.864, 124.867, 132.353, 94.637, 91.424, 109.867, 131.269, 186.073]),
    #                                        offset=np.array([-4.153, -6.107, -6.197, -9.619, -8.846, -7.339, -7.101, -8.543, -8.839, -6.793]),
    #                                        mu_auger=np.array([51.000, 51.000, 51.000, 51.000, 51.000, 51.000, 51.000, 51.000, 51.000, 51.000]),
    #                                        )
    #calibration_mean_xgm[0] = np.array([3.231, 3.239, 3.220, 3.198, 3.240, 3.229, 3.186, 3.175, 3.147, 3.127])
    tof_fit_result[0] = TofFitResult(energy=np.array([969.966, 979.971, 989.969, 999.978, 1009.957, 1019.964, 1029.956, 1039.950, 1049.967, 1059.949]),
                                            mu=np.array([199.000, 172.000, 155.000, 143.000, 135.000, 128.000, 122.000, 118.000, 114.000, 110.000]),
                                            sigma=np.array([3.042, 2.524, 2.371, 2.237, 2.043, 2.107, 1.894, 2.003, 1.916, 2.076]),
                                            A=np.array([112.561, 162.165, 203.319, 222.322, 246.422, 194.105, 151.967, 193.293, 185.952, 191.278]),
                                            Aa=np.array([102.905, 122.542, 124.238, 159.981, 156.384, 128.887, 118.458, 130.968, 138.698, 116.351]),
                                            offset=np.array([-3.239, -5.287, -5.633, -9.093, -8.863, -6.499, -5.745, -6.976, -7.848, -6.009]),
                                            mu_auger=np.array([62.000, 62.000, 62.000, 62.000, 62.000, 62.000, 62.000, 62.000, 62.000, 62.000]),
                                            )
    calibration_mean_xgm[0] = np.array([3.231, 3.239, 3.220, 3.198, 3.240, 3.229, 3.186, 3.175, 3.147, 3.127])
    tof_fit_result[1] = TofFitResult(energy=np.array([969.966, 979.971, 989.969, 999.978, 1009.957, 1019.964, 1029.956, 1039.950, 1049.967, 1059.949]),
                                            mu=np.array([199.000, 172.000, 155.000, 143.000, 135.000, 128.000, 122.000, 118.000, 114.000, 110.000]),
                                            sigma=np.array([3.042, 2.524, 2.371, 2.237, 2.043, 2.107, 1.894, 2.003, 1.916, 2.076]),
                                            A=np.array([112.561, 162.165, 203.319, 222.322, 246.422, 194.105, 151.967, 193.293, 185.952, 191.278]),
                                            Aa=np.array([102.905, 122.542, 124.238, 159.981, 156.384, 128.887, 118.458, 130.968, 138.698, 116.351]),
                                            offset=np.array([-3.239, -5.287, -5.633, -9.093, -8.863, -6.499, -5.745, -6.976, -7.848, -6.009]),
                                            mu_auger=np.array([62.000, 62.000, 62.000, 62.000, 62.000, 62.000, 62.000, 62.000, 62.000, 62.000]),
                                            )
    calibration_mean_xgm[1] = np.array([3.231, 3.239, 3.220, 3.198, 3.240, 3.229, 3.186, 3.175, 3.147, 3.127])
    tof_fit_result[2] = TofFitResult(energy=np.array([969.966, 979.971, 989.969, 999.978, 1009.957, 1019.964, 1029.956, 1039.950, 1049.967, 1059.949]),
                                            mu=np.array([184.000, 156.000, 140.000, 128.000, 119.000, 112.000, 107.000, 102.000, 98.000, 94.000]),
                                            sigma=np.array([3.233, 2.362, 2.054, 1.858, 1.828, 1.820, 1.731, 1.747, 1.753, 1.782]),
                                            A=np.array([402.555, 467.256, 521.181, 502.549, 535.726, 445.624, 451.416, 454.178, 436.428, 411.789]),
                                            Aa=np.array([306.405, 305.538, 297.467, 309.662, 322.777, 263.064, 266.821, 270.217, 260.312, 247.270]),
                                            offset=np.array([-2.559, -3.079, -3.090, -4.691, -6.172, -1.773, -2.801, -3.747, -2.996, -2.422]),
                                            mu_auger=np.array([46.000, 46.000, 46.000, 46.000, 46.000, 46.000, 46.000, 46.000, 46.000, 46.000]),
                                            )
    calibration_mean_xgm[2] = np.array([3.231, 3.239, 3.220, 3.198, 3.240, 3.229, 3.186, 3.175, 3.147, 3.127])
    tof_fit_result[3] = TofFitResult(energy=np.array([969.966, 979.971, 989.969, 999.978, 1009.957, 1019.964, 1029.956, 1039.950, 1049.967, 1059.949]),
                                            mu=np.array([184.000, 156.000, 140.000, 128.000, 119.000, 112.000, 107.000, 102.000, 98.000, 94.000]),
                                            sigma=np.array([3.233, 2.362, 2.054, 1.858, 1.828, 1.820, 1.731, 1.747, 1.753, 1.782]),
                                            A=np.array([402.555, 467.256, 521.181, 502.549, 535.726, 445.624, 451.416, 454.178, 436.428, 411.789]),
                                            Aa=np.array([306.405, 305.538, 297.467, 309.662, 322.777, 263.064, 266.821, 270.217, 260.312, 247.270]),
                                            offset=np.array([-2.559, -3.079, -3.090, -4.691, -6.172, -1.773, -2.801, -3.747, -2.996, -2.422]),
                                            mu_auger=np.array([46.000, 46.000, 46.000, 46.000, 46.000, 46.000, 46.000, 46.000, 46.000, 46.000]),
                                            )
    calibration_mean_xgm[3] = np.array([3.231, 3.239, 3.220, 3.198, 3.240, 3.229, 3.186, 3.175, 3.147, 3.127])
    #tof_fit_result[3] = TofFitResult(energy=np.array([969.966, 979.971, 989.969, 999.978, 1009.957, 1019.964, 1029.956, 1039.950, 1049.967, 1059.949]),
    #                                        mu=np.array([117.000, 174.000, 158.000, 146.000, 138.000, 131.000, 125.000, 120.000, 116.000, 113.000]),
    #                                        sigma=np.array([3.845, 2.325, 2.073, 2.142, 2.204, 1.894, 1.889, 2.203, 2.331, 2.353]),
    #                                        A=np.array([153.024, 103.103, 117.284, 156.301, 160.879, 101.516, 91.213, 139.554, 147.449, 135.230]),
    #                                        Aa=np.array([69.084, 70.826, 78.883, 92.380, 97.501, 76.276, 65.611, 78.368, 82.910, 68.494]),
    #                                        offset=np.array([-6.190, -6.595, -7.560, -9.107, -9.658, -7.377, -6.196, -7.639, -8.134, -6.547]),
    #                                        mu_auger=np.array([65.000, 65.000, 65.000, 65.000, 65.000, 65.000, 65.000, 65.000, 65.000, 65.000]),
    #                                        )
    #calibration_mean_xgm[3] = np.array([3.231, 3.239, 3.220, 3.198, 3.240, 3.229, 3.186, 3.175, 3.147, 3.127])
    tof_fit_result[4] = TofFitResult(energy=np.array([969.966, 979.971, 989.969, 999.978, 1009.957, 1019.964, 1029.956, 1039.950, 1049.967, 1059.949]),
                                            mu=np.array([192.000, 165.000, 149.000, 137.000, 129.000, 122.000, 116.000, 112.000, 108.000, 104.000]),
                                            sigma=np.array([3.338, 2.584, 2.351, 2.158, 2.078, 1.929, 1.952, 1.916, 1.913, 1.863]),
                                            A=np.array([345.986, 346.322, 385.901, 403.805, 416.736, 409.737, 390.229, 415.471, 424.858, 403.130]),
                                            Aa=np.array([349.084, 343.774, 341.385, 348.326, 329.244, 319.818, 298.250, 296.483, 292.756, 281.999]),
                                            offset=np.array([-1.380, -1.326, -1.548, -2.662, -2.298, -2.071, -1.184, -1.419, -1.484, -1.363]),
                                            mu_auger=np.array([56.000, 56.000, 56.000, 56.000, 56.000, 56.000, 56.000, 56.000, 56.000, 56.000]),
                                            )
    calibration_mean_xgm[4] = np.array([3.231, 3.239, 3.220, 3.198, 3.240, 3.229, 3.186, 3.175, 3.147, 3.127])
    tof_fit_result[5] = TofFitResult(energy=np.array([969.966, 979.971, 989.969, 999.978, 1009.957, 1019.964, 1029.956, 1039.950, 1049.967, 1059.949]),
                                            mu=np.array([194.000, 166.000, 149.000, 138.000, 129.000, 122.000, 117.000, 112.000, 108.000, 104.000]),
                                            sigma=np.array([3.375, 2.446, 2.283, 2.173, 2.077, 2.068, 2.116, 2.040, 2.058, 2.034]),
                                            A=np.array([551.446, 584.136, 594.861, 625.268, 611.601, 606.633, 597.904, 582.895, 574.338, 531.194]),
                                            Aa=np.array([418.499, 407.655, 396.545, 403.101, 388.302, 372.153, 361.537, 353.582, 353.536, 330.539]),
                                            offset=np.array([-0.807, -0.884, -0.746, -1.187, -1.571, -0.966, -1.075, -0.881, -1.472, -0.737]),
                                            mu_auger=np.array([56.000, 56.000, 56.000, 56.000, 56.000, 56.000, 56.000, 56.000, 56.000, 56.000]),
                                            )
    calibration_mean_xgm[5] = np.array([3.231, 3.239, 3.220, 3.198, 3.240, 3.229, 3.186, 3.175, 3.147, 3.127])
    tof_fit_result[6] = TofFitResult(energy=np.array([969.966, 979.971, 989.969, 999.978, 1009.957, 1019.964, 1029.956, 1039.950, 1049.967, 1059.949]),
                                            mu=np.array([175.000, 148.000, 132.000, 120.000, 112.000, 105.000, 99.000, 94.000, 90.000, 87.000]),
                                            sigma=np.array([3.095, 2.420, 2.271, 2.134, 2.118, 2.071, 2.021, 2.040, 2.045, 2.024]),
                                            A=np.array([2863.327, 3097.026, 3219.167, 3417.134, 3426.451, 3278.033, 3143.847, 2974.612, 2812.425, 2647.466]),
                                            Aa=np.array([1881.455, 1861.154, 1796.645, 1756.476, 1694.830, 1651.598, 1628.246, 1597.002, 1543.864, 1492.822]),
                                            offset=np.array([-0.827, -2.280, -1.544, -2.219, -2.266, -2.456, -2.205, -1.978, -1.270, -1.422]),
                                            mu_auger=np.array([39.000, 39.000, 39.000, 39.000, 39.000, 39.000, 39.000, 39.000, 39.000, 39.000]),
                                            )
    calibration_mean_xgm[6] = np.array([3.231, 3.239, 3.220, 3.198, 3.240, 3.229, 3.186, 3.175, 3.147, 3.127])
    tof_fit_result[7] = TofFitResult(energy=np.array([969.966, 979.971, 989.969, 999.978, 1009.957, 1019.964, 1029.956, 1039.950, 1049.967, 1059.949]),
                                            mu=np.array([173.000, 146.000, 130.000, 118.000, 110.000, 103.000, 97.000, 93.000, 89.000, 85.000]),
                                            sigma=np.array([2.861, 2.222, 2.002, 1.956, 1.847, 1.805, 1.828, 1.845, 1.863, 1.775]),
                                            A=np.array([2164.230, 2383.270, 2607.639, 2596.795, 2520.969, 2356.247, 2161.105, 2065.747, 1925.785, 1689.271]),
                                            Aa=np.array([1139.639, 1118.926, 1101.224, 1107.817, 1051.972, 1024.533, 998.036, 989.061, 952.272, 921.526]),
                                            offset=np.array([-2.676, -3.717, -4.069, -5.038, -4.383, -4.203, -3.753, -3.941, -3.402, -2.328]),
                                            mu_auger=np.array([37.000, 37.000, 37.000, 37.000, 37.000, 37.000, 37.000, 37.000, 37.000, 37.000]),
                                            )
    calibration_mean_xgm[7] = np.array([3.231, 3.239, 3.220, 3.198, 3.240, 3.229, 3.186, 3.175, 3.147, 3.127])
    tof_fit_result[8] = TofFitResult(energy=np.array([969.966, 979.971, 989.969, 999.978, 1009.957, 1019.964, 1029.956, 1039.950, 1049.967, 1059.949]),
                                            mu=np.array([170.000, 143.000, 126.000, 115.000, 106.000, 99.000, 93.000, 88.000, 84.000, 81.000]),
                                            sigma=np.array([3.492, 2.607, 2.335, 2.302, 2.160, 2.137, 2.086, 2.122, 2.140, 1.990]),
                                            A=np.array([705.567, 717.764, 733.814, 758.276, 704.809, 696.471, 703.093, 694.286, 681.787, 606.602]),
                                            Aa=np.array([466.190, 466.398, 453.045, 463.824, 443.796, 424.693, 405.628, 413.204, 414.021, 390.915]),
                                            offset=np.array([-2.037, -3.292, -3.005, -4.280, -4.048, -3.029, -2.674, -3.330, -4.116, -3.469]),
                                            mu_auger=np.array([33.000, 33.000, 33.000, 33.000, 33.000, 33.000, 33.000, 33.000, 33.000, 33.000]),
                                            )
    calibration_mean_xgm[8] = np.array([3.231, 3.239, 3.220, 3.198, 3.240, 3.229, 3.186, 3.175, 3.147, 3.127])
    tof_fit_result[9] = TofFitResult(energy=np.array([969.966, 979.971, 989.969, 999.978, 1009.957, 1019.964, 1029.956, 1039.950, 1049.967, 1059.949]),
                                            mu=np.array([172.000, 145.000, 129.000, 117.000, 108.000, 102.000, 96.000, 91.000, 87.000, 84.000]),
                                            sigma=np.array([3.337, 2.358, 2.118, 1.948, 1.908, 1.888, 1.818, 1.815, 1.819, 1.812]),
                                            A=np.array([3103.212, 3344.723, 3539.962, 3500.884, 3306.640, 3295.435, 3135.436, 2960.729, 2763.412, 2595.917]),
                                            Aa=np.array([1682.130, 1641.305, 1616.644, 1610.479, 1549.766, 1510.755, 1467.439, 1442.409, 1415.082, 1382.602]),
                                            offset=np.array([-1.516, -2.758, -3.791, -5.020, -5.421, -5.931, -4.968, -5.164, -3.944, -4.354]),
                                            mu_auger=np.array([36.000, 36.000, 36.000, 36.000, 36.000, 36.000, 36.000, 36.000, 36.000, 36.000]),
                                            )
    calibration_mean_xgm[9] = np.array([3.231, 3.239, 3.220, 3.198, 3.240, 3.229, 3.186, 3.175, 3.147, 3.127])
    tof_fit_result[10] = TofFitResult(energy=np.array([969.966, 979.971, 989.969, 999.978, 1009.957, 1019.964, 1029.956, 1039.950, 1049.967, 1059.949]),
                                            mu=np.array([174.000, 147.000, 131.000, 120.000, 111.000, 104.000, 99.000, 94.000, 90.000, 86.000]),
                                            sigma=np.array([3.273, 2.399, 2.062, 2.044, 1.939, 1.882, 1.953, 1.903, 1.901, 1.862]),
                                            A=np.array([635.244, 730.672, 763.755, 846.638, 754.066, 714.770, 706.029, 678.916, 633.954, 580.495]),
                                            Aa=np.array([411.538, 405.854, 395.222, 410.290, 407.516, 370.108, 361.856, 365.161, 359.966, 356.076]),
                                            offset=np.array([-1.605, -1.753, -1.519, -2.759, -3.778, -1.629, -1.547, -2.280, -2.389, -3.036]),
                                            mu_auger=np.array([39.000, 39.000, 39.000, 39.000, 39.000, 39.000, 39.000, 39.000, 39.000, 39.000]),
                                            )
    calibration_mean_xgm[10] = np.array([3.231, 3.239, 3.220, 3.198, 3.240, 3.229, 3.186, 3.175, 3.147, 3.127])
    tof_fit_result[11] = TofFitResult(energy=np.array([969.966, 979.971, 989.969, 999.978, 1009.957, 1019.964, 1029.956, 1039.950, 1049.967, 1059.949]),
                                            mu=np.array([168.000, 141.000, 125.000, 113.000, 104.000, 97.000, 92.000, 87.000, 83.000, 79.000]),
                                            sigma=np.array([3.854, 3.032, 2.800, 3.027, 3.111, 2.918, 2.451, 2.984, 2.504, 2.547]),
                                            A=np.array([425.056, 512.046, 581.996, 728.882, 694.960, 636.096, 563.828, 595.908, 540.055, 506.923]),
                                            Aa=np.array([373.504, 399.580, 437.434, 504.472, 476.920, 433.931, 379.274, 384.104, 381.842, 368.152]),
                                            offset=np.array([-3.069, -5.018, -7.993, -12.255, -11.663, -9.322, -6.354, -7.049, -7.199, -7.129]),
                                            mu_auger=np.array([32.000, 32.000, 32.000, 32.000, 32.000, 32.000, 32.000, 32.000, 32.000, 32.000]),
                                            )
    calibration_mean_xgm[11] = np.array([3.231, 3.239, 3.220, 3.198, 3.240, 3.229, 3.186, 3.175, 3.147, 3.127])
    tof_fit_result[12] = TofFitResult(energy=np.array([969.966, 979.971, 989.969, 999.978, 1009.957, 1019.964, 1029.956, 1039.950, 1049.967, 1059.949]),
                                            mu=np.array([188.000, 162.000, 145.000, 134.000, 125.000, 119.000, 113.000, 108.000, 104.000, 101.000]),
                                            sigma=np.array([3.033, 2.461, 2.235, 2.229, 2.228, 2.207, 2.050, 2.106, 2.193, 2.278]),
                                            A=np.array([594.491, 722.971, 790.569, 852.026, 848.373, 791.047, 732.021, 675.843, 664.856, 608.463]),
                                            Aa=np.array([458.061, 440.204, 464.142, 460.831, 489.716, 418.807, 397.643, 391.720, 396.985, 376.972]),
                                            offset=np.array([-2.459, -2.093, -4.303, -4.544, -7.883, -3.402, -2.692, -2.857, -3.539, -3.006]),
                                            mu_auger=np.array([53.000, 53.000, 53.000, 53.000, 53.000, 53.000, 53.000, 53.000, 53.000, 53.000]),
                                            )
    calibration_mean_xgm[12] = np.array([3.231, 3.239, 3.220, 3.198, 3.240, 3.229, 3.186, 3.175, 3.147, 3.127])
    tof_fit_result[13] = TofFitResult(energy=np.array([969.966, 979.971, 989.969, 999.978, 1009.957, 1019.964, 1029.956, 1039.950, 1049.967, 1059.949]),
                                            mu=np.array([184.000, 158.000, 141.000, 130.000, 122.000, 115.000, 109.000, 105.000, 101.000, 97.000]),
                                            sigma=np.array([3.499, 2.767, 2.557, 2.504, 2.480, 2.398, 2.236, 2.361, 2.282, 2.536]),
                                            A=np.array([522.164, 651.719, 727.776, 861.150, 877.040, 808.610, 685.787, 725.143, 668.640, 679.214]),
                                            Aa=np.array([499.620, 515.655, 536.836, 603.032, 594.111, 534.009, 475.448, 462.708, 440.345, 437.715]),
                                            offset=np.array([-3.518, -5.471, -8.216, -13.742, -14.241, -10.455, -6.841, -6.371, -5.452, -6.174]),
                                            mu_auger=np.array([50.000, 50.000, 50.000, 50.000, 50.000, 50.000, 50.000, 50.000, 50.000, 50.000]),
                                            )
    calibration_mean_xgm[13] = np.array([3.231, 3.239, 3.220, 3.198, 3.240, 3.229, 3.186, 3.175, 3.147, 3.127])
    tof_fit_result[14] = TofFitResult(energy=np.array([969.966, 979.971, 989.969, 999.978, 1009.957, 1019.964, 1029.956, 1039.950, 1049.967, 1059.949]),
                                            mu=np.array([176.000, 150.000, 134.000, 122.000, 113.000, 107.000, 101.000, 96.000, 92.000, 89.000]),
                                            sigma=np.array([3.459, 2.426, 2.357, 2.229, 2.181, 2.176, 2.041, 2.118, 2.067, 2.156]),
                                            A=np.array([690.366, 733.165, 830.742, 919.780, 939.585, 897.465, 845.078, 871.793, 861.005, 837.457]),
                                            Aa=np.array([614.873, 619.851, 619.859, 683.328, 668.022, 585.407, 529.526, 552.633, 535.706, 495.319]),
                                            offset=np.array([-2.954, -4.795, -5.380, -10.876, -11.257, -6.321, -3.146, -5.203, -4.984, -3.617]),
                                            mu_auger=np.array([41.000, 41.000, 41.000, 41.000, 41.000, 41.000, 41.000, 41.000, 41.000, 41.000]),
                                            )
    calibration_mean_xgm[14] = np.array([3.231, 3.239, 3.220, 3.198, 3.240, 3.229, 3.186, 3.175, 3.147, 3.127])
    tof_fit_result[15] = TofFitResult(energy=np.array([969.966, 979.971, 989.969, 999.978, 1009.957, 1019.964, 1029.956, 1039.950, 1049.967, 1059.949]),
                                            mu=np.array([178.000, 151.000, 135.000, 123.000, 114.000, 107.000, 101.000, 97.000, 92.000, 89.000]),
                                            sigma=np.array([3.389, 2.492, 2.226, 2.069, 2.013, 1.984, 2.042, 2.018, 2.029, 1.996]),
                                            A=np.array([397.125, 425.260, 483.265, 509.147, 518.803, 464.732, 465.127, 477.613, 462.203, 418.617]),
                                            Aa=np.array([312.599, 316.463, 332.680, 342.935, 321.642, 290.574, 278.439, 284.293, 279.913, 262.764]),
                                            offset=np.array([-1.486, -2.407, -3.982, -5.013, -4.256, -2.570, -2.206, -2.977, -3.038, -2.446]),
                                            mu_auger=np.array([40.000, 40.000, 40.000, 40.000, 40.000, 40.000, 40.000, 40.000, 40.000, 40.000]),
                                            )
    calibration_mean_xgm[15] = np.array([3.231, 3.239, 3.220, 3.198, 3.240, 3.229, 3.186, 3.175, 3.147, 3.127])

    #correct[0] = np.array([-374804.121, 1035.472, 296.177])
    correct[0] = np.array([570800.136, 947.666, 39.012])
    correct[1] = np.array([570800.136, 947.666, 39.012])
    correct[2] = np.array([593583.691, 947.426, 21.721])
    correct[3] = np.array([593583.691, 947.426, 21.721])
    #correct[3] = np.array([6627137.571, 892.084, -100.600])
    correct[4] = np.array([629045.617, 945.559, 30.053])
    correct[5] = np.array([575369.595, 947.833, 32.769])
    correct[6] = np.array([612478.903, 946.483, 13.321])
    correct[7] = np.array([629045.660, 945.559, 11.053])
    correct[8] = np.array([619636.068, 946.709, 6.773])
    correct[9] = np.array([586431.727, 947.138, 11.724])
    correct[10] = np.array([604862.698, 946.527, 13.359])
    correct[11] = np.array([616862.214, 946.593, 5.546])
    correct[12] = np.array([590432.506, 946.720, 28.630])
    correct[13] = np.array([583796.646, 946.697, 25.542])
    correct[14] = np.array([610640.951, 946.263, 15.493])
    correct[15] = np.array([634612.517, 946.329, 14.148])

    return tof_fit_result, calibration_mean_xgm, correct

def test_create_cookiebox_calibration():
    # instantiates it without doing any calibration, only to check for syntax errors
    cal = CookieboxCalibration()

def test_fit(tmp_path):
    calibration_energies = []
    data = []

    # fake calibration
    tof_fit_result, calibration_mean_xgm, correct = produce_mock_fit_result()

    cal = CookieboxCalibration()
    # create some fake data to avoid the first steps, which would depend on much more data
    cal._energy_axis = np.linspace(965, 1070, 160)
    cal._auger_start_roi = {tof_id: 1 for tof_id in range(16)}
    cal._start_roi = {tof_id: 75 for tof_id in range(16)}
    cal._stop_roi = {tof_id: 320 for tof_id in range(16)}
    cal.mask = {tof_id: True for tof_id in range(16)}
    cal.mask[0] = False
    cal.mask[1] = False
    cal.mask[3] = False
    cal.mask[8] = False
    cal.mask[11] = False
    cal.kwargs_adq = {tof_id: dict() for tof_id in range(16)}
    cal.tof_fit_result = tof_fit_result
    cal.calibration_mean_xgm = calibration_mean_xgm

    cal.calibration_mask = {tof_id: [True]*10 for tof_id in range(16)}

    # fit!
    cal.update_calibration()

    # check if it worked
    for tof_id, v in correct.items():
        assert np.allclose(cal.model_params[tof_id], v, rtol=1e-2, atol=1e-2)

    d = tmp_path / "data"
    d.mkdir()
    fpath = str(d / "cookiebox_test.h5")
    cal.to_file(fpath)
    cal_read = CookieboxCalibration.from_file(fpath)

    # check if serialization worked
    for tof_id, v in correct.items():
        assert np.allclose(cal_read.model_params[tof_id], v, rtol=1e-2, atol=1e-2)

def test_avg_and_fit_single_channel(mock_sqs_etof_calibration_run, tmp_path, mock_etof_mono_energies, mock_etof_calibration_constants):
    # use mock data
    pulse_timing = 'SQS_RR_UTC/TSYS/TIMESERVER'
    monochromator_energy = 'SA3_XTD10_MONO/MDL/PHOTON_ENERGY'
    digitizer = 'SQS_DIGITIZER_UTC4/ADC/1:network'
    digitizer_control = 'SQS_DIGITIZER_UTC4/ADC/1'
    pulse_energy = 'SQS_DIAG1_XGMD/XGM/DOOCS'
    mock_sqs_etof_calibration_run = mock_sqs_etof_calibration_run.select([pulse_timing,
                                  digitizer, digitizer_control,
                                  pulse_energy, f"{pulse_energy}:output",
                                  monochromator_energy], require_all=True).select_trains(np.s_[10:])
    channel_name = "1_A"
    tof_ids = [0]
    tof_channel = {}
    tof_channel[0] = AdqRawChannel(mock_sqs_etof_calibration_run,
                                   channel_name,
                                   digitizer=digitizer,
                                   first_pulse_offset=1000)
    scan = Scan(mock_sqs_etof_calibration_run[monochromator_energy, "actualEnergy"], resolution=2)
    energy_axis = np.linspace(965, 1070, 160)
    xgm = XGM(mock_sqs_etof_calibration_run, pulse_energy)
    cal = CookieboxCalibration(
                    auger_start_roi=1,
                    start_roi=75,
                    stop_roi=320,
    )
    cal.setup(run=mock_sqs_etof_calibration_run, energy_axis=energy_axis, tof_settings=tof_channel,
              xgm=xgm,
              scan=scan)
    correct_energies = np.unique(mock_etof_mono_energies)
    correct_constants = np.array(mock_etof_calibration_constants)
    for tof_id in tof_ids:
        assert np.allclose(cal.tof_fit_result[tof_id].energy, correct_energies, rtol=1e-2, atol=1e-2)

        energy = correct_energies

        # get calibration curve
        c, e0, t0 = cal.model_params[tof_id]
        ts = t0 + np.sqrt(c/(energy - e0))

        c_true, e0_true, t0_true = correct_constants
        ts_true = t0_true + np.sqrt(c_true/(energy - e0_true))

        # check how well it matches
        assert np.allclose(ts, ts_true, rtol=1e-2, atol=1e-2)

    d = tmp_path / "data"
    d.mkdir()
    fpath = str(d / "cookiebox_test.h5")
    cal.to_file(fpath)

    # make some plots
    plt.figure(figsize=(10, 8))
    cal.plot_calibration_data()
    plt.savefig(str(d / "data.pdf"))

    plt.figure(figsize=(10, 8))
    cal.plot_fit(0)
    plt.savefig(str(d / "fit.pdf"))

    plt.figure(figsize=(10, 8))
    cal.plot_calibrations()
    plt.savefig(str(d / "calibrations.pdf"))

    plt.figure(figsize=(10, 8))
    cal.plot_jacobians()
    plt.savefig(str(d / "jacobians.pdf"))

    plt.figure(figsize=(10, 8))
    cal.plot_transmissions()
    plt.savefig(str(d / "transmissions.pdf"))

    cal_read = CookieboxCalibration.from_file(fpath)

    # test if serialization worked
    for tof_id in tof_ids:
        assert np.allclose(cal_read.model_params[tof_id], cal.model_params[tof_id], rtol=1e-2, atol=1e-2)

    data = cal_read.load_data(mock_sqs_etof_calibration_run.select_trains(np.s_[10:20]))
    spectrum_two_steps = cal_read.calibrate(data)

    # now do it subtracting the offset to check if that does not crash either
    spectrum_subtracted = cal_read.calibrate(data, subtract_offset=True)

    # now call simpler function to do both at once
    spectrum_one_go = cal_read.apply(mock_sqs_etof_calibration_run.select_trains(np.s_[10:20]))

    assert np.allclose(spectrum_one_go, spectrum_two_steps, rtol=1e-2, atol=1e-2)


# tests data reading without parallelization
def test_no_parallel(mock_sqs_etof_calibration_run, tmp_path, mock_etof_mono_energies, mock_etof_calibration_constants):
    # same as above, but tests only if a crash happens in `calc_mean`
    # somehow parallelization means that `calc_mean` is not shown in the coverage
    pulse_timing = 'SQS_RR_UTC/TSYS/TIMESERVER'
    monochromator_energy = 'SA3_XTD10_MONO/MDL/PHOTON_ENERGY'
    digitizer = 'SQS_DIGITIZER_UTC4/ADC/1:network'
    digitizer_control = 'SQS_DIGITIZER_UTC4/ADC/1'
    pulse_energy = 'SQS_DIAG1_XGMD/XGM/DOOCS'
    mock_sqs_etof_calibration_run = mock_sqs_etof_calibration_run.select([pulse_timing,
                                  digitizer, digitizer_control,
                                  pulse_energy, f"{pulse_energy}:output",
                                  monochromator_energy], require_all=True).select_trains(np.s_[10:])
    channel_name = "1_A"
    tof_ids = [0]
    tof_channel = {}
    tof_channel[0] = AdqRawChannel(mock_sqs_etof_calibration_run,
                                   channel_name,
                                   digitizer=digitizer,
                                   first_pulse_offset=1000)
    scan = Scan(mock_sqs_etof_calibration_run[monochromator_energy, "actualEnergy"], resolution=2)
    energy_axis = np.linspace(965, 1070, 160)
    xgm = XGM(mock_sqs_etof_calibration_run, pulse_energy)
    cal = CookieboxCalibration(
                    auger_start_roi=1,
                    start_roi=75,
                    stop_roi=320,
                    parallel=False,
    )
    cal.setup(run=mock_sqs_etof_calibration_run, energy_axis=energy_axis, tof_settings=tof_channel,
              xgm=xgm,
              scan=scan)

    correct_energies = np.unique(mock_etof_mono_energies)
    correct_constants = np.array(mock_etof_calibration_constants)
    for tof_id in tof_ids:
        assert np.allclose(cal.tof_fit_result[tof_id].energy, correct_energies, rtol=1e-2, atol=1e-2)

        energy = correct_energies

        # get calibration curve
        c, e0, t0 = cal.model_params[tof_id]
        ts = t0 + np.sqrt(c/(energy - e0))

        c_true, e0_true, t0_true = correct_constants
        ts_true = t0_true + np.sqrt(c_true/(energy - e0_true))

        # check how well it matches
        assert np.allclose(ts, ts_true, rtol=1e-2, atol=1e-2)

def test_deconvolve(mock_sqs_etof_calibration_run, tmp_path):
    # use mock data and do the same as before, but with deconvolution
    # it should improve resolution, but lead to the same calibration constants
    pulse_timing = 'SQS_RR_UTC/TSYS/TIMESERVER'
    monochromator_energy = 'SA3_XTD10_MONO/MDL/PHOTON_ENERGY'
    digitizer = 'SQS_DIGITIZER_UTC4/ADC/1:network'
    digitizer_control = 'SQS_DIGITIZER_UTC4/ADC/1'
    pulse_energy = 'SQS_DIAG1_XGMD/XGM/DOOCS'
    mock_sqs_etof_calibration_run = mock_sqs_etof_calibration_run.select([pulse_timing,
                                  digitizer, digitizer_control,
                                  pulse_energy, f"{pulse_energy}:output",
                                  monochromator_energy], require_all=True).select_trains(np.s_[10:])
    channel_name = "1_A"
    tof_ids = [0]
    tof_channel = {}
    tof_channel[0] = AdqRawChannel(mock_sqs_etof_calibration_run,
                                   channel_name,
                                   digitizer=digitizer,
                                   first_pulse_offset=1000)
    scan = Scan(mock_sqs_etof_calibration_run[monochromator_energy, "actualEnergy"], resolution=2)

    # setup tof response
    tof_response = TOFAnalogResponse(roi=slice(75, None), n_samples=150)
    tof_response.setup(tof_channel[0], scan)

    d = tmp_path / "data"
    d.mkdir()

    # make some plots
    plt.figure(figsize=(10, 8))
    tof_response.plot()
    plt.savefig(str(d / "response.pdf"))

    # test serialization
    fpath = str(d / "response.h5")
    tof_response.to_file(fpath)
    tof_response_read = TOFAnalogResponse.from_file(fpath)


    # create calibration object to read data in the appropriate format
    energy_axis = np.linspace(965, 1070, 160)
    xgm = XGM(mock_sqs_etof_calibration_run, pulse_energy)
    cal = CookieboxCalibration(
                    auger_start_roi=1,
                    start_roi=75,
                    stop_roi=320,
    )
    cal.setup(run=mock_sqs_etof_calibration_run, energy_axis=energy_axis, tof_settings=tof_channel,
              xgm=xgm,
              scan=scan)

    # read data
    data = cal.load_data(mock_sqs_etof_calibration_run.select_trains(np.s_[10:12]))
    data = data.stack(pulse=("trainId", "pulseIndex"))
    # apply NN method
    tof_response.apply(data.sel(tof=0), nonneg=True, method="nn_matrix", n_iter=10)
    # apply TV method
    tof_response.apply(data.sel(tof=0), nonneg=True, method="tv_matrix", n_iter=10, Lambda=1e-5)
    # apply standard method
    tof_response.apply(data.sel(tof=0), method="standard")

    # check if we can get the response
    h = tof_response.get_response()


