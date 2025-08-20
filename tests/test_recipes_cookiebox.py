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
from extra.recipes import CookieboxCalibration
from extra.recipes.cookiebox import TofFitResult
from extra.recipes.cookiebox_deconvolve import TOFAnalogResponse

from .mockdata.utils import (mock_etof_calibration_constants,
                             mock_etof_mono_energies)

# this produces mock data
def produce_mock_fit_result():
    tof_fit_result = dict()
    calibration_mean_xgm = dict()
    tof_fit_result[0] = TofFitResult(energy=np.array([969.966, 979.971, 989.969, 999.977, 1009.957, 1019.964, 1029.956, 1039.950, 1049.967, 1059.948]),
                                            mu=np.array([82.295, 244.115, 181.515, 163.538, 170.006, 150.682, 142.277, 137.487, 151.709, 136.544]),
                                            sigma=np.array([1.235, 2.156, 1.005, 1.252, 14.557, 3.495, 1.128, 1.385, 16.149, 1.037]),
                                            A=np.array([15.130, 22.051, 19.500, 33.493, 43.768, 44.432, 25.401, 37.861, 59.123, 20.661]),
                                            Aa=np.array([924075.198, 991195.439, 1058832.648, 1015218.896, 1002736.344, 925559.489, 914417.844, 886635.529, 852280.619, 834658.563]),
                                            offset=np.array([0.159, -0.004, -0.118, -0.084, -0.139, -0.149, -0.034, -0.034, -0.124, 0.027]),
                                            mu_auger=np.array([27.562, 25.722, 24.076, 23.066, 22.714, 22.385, 22.525, 22.550, 22.439, 22.787]),
                                            )
    calibration_mean_xgm[0] = np.array([3.233, 3.241, 3.221, 3.200, 3.242, 3.232, 3.188, 3.176, 3.150, 3.129])
    tof_fit_result[1] = TofFitResult(energy=np.array([969.966, 979.971, 989.969, 999.977, 1009.957, 1019.964, 1029.956, 1039.950, 1049.967, 1059.948]),
                                            mu=np.array([200.113, 172.262, 155.726, 143.951, 135.353, 132.821, 122.713, 117.998, 113.823, 122.164]),
                                            sigma=np.array([2.128, 1.926, 2.009, 1.869, 1.930, 28.195, 1.702, 1.838, 1.778, 29.549]),
                                            A=np.array([46.930, 74.734, 111.740, 84.199, 114.757, 290.372, 81.203, 82.407, 85.506, 302.155]),
                                            Aa=np.array([67.781, 64.108, 62.141, 60.107, 59.227, 57.663, 55.628, 54.628, 52.638, 50.273]),
                                            offset=np.array([1.210, 1.059, 0.783, 1.016, 0.864, 0.156, 1.037, 1.074, 1.015, 0.181]),
                                            mu_auger=np.array([62.507, 62.501, 62.510, 62.501, 62.541, 62.523, 62.522, 62.536, 62.515, 62.532]),
                                            )
    calibration_mean_xgm[1] = np.array([3.233, 3.241, 3.221, 3.200, 3.242, 3.232, 3.188, 3.176, 3.150, 3.129])
    tof_fit_result[2] = TofFitResult(energy=np.array([969.966, 979.971, 989.969, 999.977, 1009.957, 1019.964, 1029.956, 1039.950, 1049.967, 1059.948]),
                                            mu=np.array([184.509, 156.886, 140.156, 128.517, 119.752, 112.897, 107.169, 102.380, 98.325, 94.787]),
                                            sigma=np.array([3.136, 2.318, 2.085, 1.956, 1.880, 1.857, 1.800, 1.807, 1.803, 1.820]),
                                            A=np.array([301.648, 382.453, 439.376, 430.411, 446.154, 412.673, 384.698, 376.797, 367.397, 352.865]),
                                            Aa=np.array([260.121, 254.284, 246.285, 241.142, 238.340, 225.992, 218.870, 210.685, 209.030, 202.004]),
                                            offset=np.array([3.266, 3.210, 3.252, 3.458, 3.491, 3.646, 3.649, 3.752, 3.800, 3.770]),
                                            mu_auger=np.array([46.587, 46.603, 46.610, 46.623, 46.623, 46.638, 46.637, 46.644, 46.652, 46.642]),
                                            )
    calibration_mean_xgm[2] = np.array([3.233, 3.241, 3.221, 3.200, 3.242, 3.232, 3.188, 3.176, 3.150, 3.129])
    tof_fit_result[3] = TofFitResult(energy=np.array([969.966, 979.971, 989.969, 999.977, 1009.957, 1019.964, 1029.956, 1039.950, 1049.967, 1059.948]),
                                            mu=np.array([101.243, 307.194, 265.747, 243.454, 231.156, 225.498, 221.744, 139.642, 116.327, 133.883]),
                                            sigma=np.array([17.120, 24.670, 23.112, 25.064, 27.356, 26.613, 26.227, 16.075, 1.406, 16.175]),
                                            A=np.array([49.086, -75.445, -54.532, -84.043, -117.301, -125.127, -134.132, 62.317, 33.742, 64.616]),
                                            Aa=np.array([15.350, 13.780, 13.448, 13.517, 13.533, 13.007, 12.965, 12.641, 12.710, 12.574]),
                                            offset=np.array([-0.187, 0.084, 0.161, 0.273, 0.504, 0.525, 0.556, -0.204, -0.067, -0.179]),
                                            mu_auger=np.array([65.128, 65.126, 65.170, 65.191, 65.197, 65.236, 65.242, 65.252, 65.271, 65.261]),
                                            )
    calibration_mean_xgm[3] = np.array([3.233, 3.241, 3.221, 3.200, 3.242, 3.232, 3.188, 3.176, 3.150, 3.129])
    tof_fit_result[4] = TofFitResult(energy=np.array([969.966, 979.971, 989.969, 999.977, 1009.957, 1019.964, 1029.956, 1039.950, 1049.967, 1059.948]),
                                            mu=np.array([193.410, 166.020, 149.486, 137.918, 129.279, 122.384, 116.850, 112.017, 107.993, 104.535]),
                                            sigma=np.array([3.152, 2.438, 2.262, 2.161, 2.128, 2.043, 2.056, 1.982, 1.943, 1.978]),
                                            A=np.array([257.699, 286.902, 317.391, 332.679, 340.068, 362.492, 360.537, 362.251, 372.464, 361.179]),
                                            Aa=np.array([331.262, 325.934, 321.456, 315.408, 300.194, 293.309, 281.339, 276.056, 270.758, 261.394]),
                                            offset=np.array([3.524, 3.414, 3.284, 3.341, 3.397, 3.502, 3.557, 3.665, 3.718, 3.711]),
                                            mu_auger=np.array([56.561, 56.559, 56.583, 56.573, 56.582, 56.594, 56.600, 56.599, 56.593, 56.596]),
                                            )
    calibration_mean_xgm[4] = np.array([3.233, 3.241, 3.221, 3.200, 3.242, 3.232, 3.188, 3.176, 3.150, 3.129])
    tof_fit_result[5] = TofFitResult(energy=np.array([969.966, 979.971, 989.969, 999.977, 1009.957, 1019.964, 1029.956, 1039.950, 1049.967, 1059.948]),
                                            mu=np.array([194.502, 166.680, 150.001, 138.361, 129.686, 122.703, 117.137, 112.334, 108.281, 104.743]),
                                            sigma=np.array([3.322, 2.436, 2.222, 2.144, 2.099, 2.056, 2.084, 2.043, 2.035, 2.040]),
                                            A=np.array([449.110, 518.401, 550.821, 558.562, 539.187, 540.703, 522.684, 509.578, 490.889, 466.577]),
                                            Aa=np.array([391.849, 380.294, 371.217, 372.244, 352.581, 344.431, 332.645, 326.286, 318.599, 304.862]),
                                            offset=np.array([4.564, 4.527, 4.535, 4.570, 4.734, 4.840, 4.885, 5.020, 5.087, 5.021]),
                                            mu_auger=np.array([56.687, 56.693, 56.716, 56.714, 56.724, 56.727, 56.735, 56.736, 56.743, 56.738]),
                                            )
    calibration_mean_xgm[5] = np.array([3.233, 3.241, 3.221, 3.200, 3.242, 3.232, 3.188, 3.176, 3.150, 3.129])
    tof_fit_result[6] = TofFitResult(energy=np.array([969.966, 979.971, 989.969, 999.977, 1009.957, 1019.964, 1029.956, 1039.950, 1049.967, 1059.948]),
                                            mu=np.array([175.502, 148.557, 132.185, 120.644, 111.988, 105.130, 99.474, 94.809, 90.785, 87.206]),
                                            sigma=np.array([3.278, 2.556, 2.383, 2.244, 2.167, 2.133, 2.129, 2.106, 2.152, 2.083]),
                                            A=np.array([2463.428, 2786.150, 2984.042, 3187.814, 3101.974, 2934.906, 2824.153, 2689.922, 2527.713, 2304.650]),
                                            Aa=np.array([1687.940, 1649.953, 1596.887, 1549.525, 1488.429, 1443.984, 1423.230, 1397.883, 1350.555, 1300.408]),
                                            offset=np.array([22.297, 22.406, 22.568, 23.040, 23.631, 24.477, 24.733, 25.348, 25.851, 26.127]),
                                            mu_auger=np.array([39.406, 39.411, 39.412, 39.421, 39.422, 39.423, 39.432, 39.437, 39.417, 39.444]),
                                            )
    calibration_mean_xgm[6] = np.array([3.233, 3.241, 3.221, 3.200, 3.242, 3.232, 3.188, 3.176, 3.150, 3.129])
    tof_fit_result[7] = TofFitResult(energy=np.array([969.966, 979.971, 989.969, 999.977, 1009.957, 1019.964, 1029.956, 1039.950, 1049.967, 1059.948]),
                                            mu=np.array([173.715, 146.811, 130.403, 118.905, 110.276, 103.371, 97.777, 93.013, 89.009, 85.462]),
                                            sigma=np.array([2.996, 2.315, 2.096, 1.989, 1.927, 1.899, 1.901, 1.887, 1.884, 1.891]),
                                            A=np.array([1945.062, 2258.984, 2402.333, 2548.302, 2383.288, 2203.990, 2032.280, 1875.111, 1740.255, 1539.499]),
                                            Aa=np.array([1042.545, 1009.708, 988.826, 983.378, 933.850, 907.489, 886.316, 870.647, 839.116, 821.156]),
                                            offset=np.array([13.332, 13.779, 14.221, 14.671, 15.167, 15.476, 15.681, 16.021, 16.186, 15.899]),
                                            mu_auger=np.array([37.838, 37.839, 37.853, 37.865, 37.871, 37.880, 37.892, 37.885, 37.873, 37.902]),
                                            )
    calibration_mean_xgm[7] = np.array([3.233, 3.241, 3.221, 3.200, 3.242, 3.232, 3.188, 3.176, 3.150, 3.129])
    tof_fit_result[8] = TofFitResult(energy=np.array([969.966, 979.971, 989.969, 999.977, 1009.957, 1019.964, 1029.956, 1039.950, 1049.967, 1059.948]),
                                            mu=np.array([170.742, 143.245, 126.612, 115.062, 106.264, 99.397, 93.696, 88.997, 85.055, 81.435]),
                                            sigma=np.array([3.512, 2.647, 2.342, 2.284, 2.209, 2.126, 2.153, 2.094, 2.113, 2.065]),
                                            A=np.array([563.595, 607.604, 604.373, 646.835, 583.596, 585.357, 593.332, 595.863, 573.028, 527.815]),
                                            Aa=np.array([379.518, 365.874, 356.750, 352.303, 335.915, 330.603, 316.966, 315.062, 305.197, 292.081]),
                                            offset=np.array([5.110, 5.065, 5.307, 5.242, 5.642, 5.701, 5.826, 6.016, 6.225, 6.276]),
                                            mu_auger=np.array([33.628, 33.629, 33.634, 33.646, 33.666, 33.662, 33.677, 33.674, 33.680, 33.696]),
                                            )
    calibration_mean_xgm[8] = np.array([3.233, 3.241, 3.221, 3.200, 3.242, 3.232, 3.188, 3.176, 3.150, 3.129])
    tof_fit_result[9] = TofFitResult(energy=np.array([969.966, 979.971, 989.969, 999.977, 1009.957, 1019.964, 1029.956, 1039.950, 1049.967, 1059.948]),
                                            mu=np.array([172.908, 145.636, 129.083, 117.536, 108.880, 101.998, 96.365, 91.643, 87.642, 84.073]),
                                            sigma=np.array([3.372, 2.487, 2.192, 2.064, 1.967, 1.909, 1.913, 1.883, 1.899, 1.854]),
                                            A=np.array([2667.318, 3041.809, 3232.132, 3384.897, 3194.018, 3026.880, 2891.898, 2753.113, 2558.753, 2313.681]),
                                            Aa=np.array([1536.216, 1487.233, 1445.328, 1428.098, 1362.680, 1316.722, 1285.184, 1253.661, 1238.718, 1199.309]),
                                            offset=np.array([20.568, 21.027, 21.593, 22.252, 22.889, 23.700, 24.162, 24.857, 25.409, 25.429]),
                                            mu_auger=np.array([36.491, 36.503, 36.501, 36.508, 36.518, 36.521, 36.532, 36.535, 36.545, 36.531]),
                                            )
    calibration_mean_xgm[9] = np.array([3.233, 3.241, 3.221, 3.200, 3.242, 3.232, 3.188, 3.176, 3.150, 3.129])
    tof_fit_result[10] = TofFitResult(energy=np.array([969.966, 979.971, 989.969, 999.977, 1009.957, 1019.964, 1029.956, 1039.950, 1049.967, 1059.948]),
                                            mu=np.array([175.275, 148.080, 131.546, 120.116, 111.511, 104.618, 99.022, 94.351, 90.400, 86.804]),
                                            sigma=np.array([3.264, 2.435, 2.121, 2.074, 2.001, 1.926, 1.933, 1.964, 1.945, 1.912]),
                                            A=np.array([533.004, 650.617, 684.356, 740.786, 685.372, 654.859, 634.674, 617.524, 565.420, 512.129]),
                                            Aa=np.array([355.346, 349.117, 341.909, 344.091, 330.531, 317.279, 309.206, 304.328, 297.394, 288.041]),
                                            offset=np.array([4.966, 5.054, 5.296, 5.232, 5.485, 5.631, 5.646, 5.672, 5.798, 5.851]),
                                            mu_auger=np.array([39.380, 39.400, 39.401, 39.403, 39.415, 39.418, 39.421, 39.428, 39.427, 39.444]),
                                            )
    calibration_mean_xgm[10] = np.array([3.233, 3.241, 3.221, 3.200, 3.242, 3.232, 3.188, 3.176, 3.150, 3.129])
    tof_fit_result[11] = TofFitResult(energy=np.array([969.966, 979.971, 989.969, 999.977, 1009.957, 1019.964, 1029.956, 1039.950, 1049.967, 1059.948]),
                                            mu=np.array([168.659, 141.624, 125.134, 113.788, 105.129, 98.253, 92.425, 88.010, 84.117, -2561.378]),
                                            sigma=np.array([3.289, 2.709, 2.557, 2.806, 2.782, 2.764, 2.283, 2.727, 2.487, 280.223]),
                                            A=np.array([264.806, 346.403, 361.989, 427.323, 402.093, 398.863, 388.097, 410.041, 363.388, 367130718344518840614912.000]),
                                            Aa=np.array([271.626, 271.404, 266.567, 270.849, 254.731, 246.832, 237.537, 232.785, 229.069, 218.191]),
                                            offset=np.array([4.104, 3.983, 4.211, 4.169, 4.328, 4.441, 4.511, 4.438, 4.699, 2.215]),
                                            mu_auger=np.array([32.907, 32.911, 32.915, 32.911, 32.916, 32.924, 32.940, 32.928, 32.937, 32.934]),
                                            )
    calibration_mean_xgm[11] = np.array([3.233, 3.241, 3.221, 3.200, 3.242, 3.232, 3.188, 3.176, 3.150, 3.129])
    tof_fit_result[12] = TofFitResult(energy=np.array([969.966, 979.971, 989.969, 999.977, 1009.957, 1019.964, 1029.956, 1039.950, 1049.967, 1059.948]),
                                            mu=np.array([189.153, 162.306, 145.857, 134.449, 125.924, 119.104, 113.498, 108.751, 104.927, 101.344]),
                                            sigma=np.array([2.880, 2.444, 2.203, 2.274, 2.305, 2.212, 2.100, 2.220, 2.117, 2.394]),
                                            A=np.array([443.346, 608.009, 682.265, 732.581, 698.339, 665.382, 610.812, 586.735, 540.248, 504.934]),
                                            Aa=np.array([421.642, 408.665, 404.509, 398.675, 384.307, 371.980, 359.365, 351.322, 347.140, 334.008]),
                                            offset=np.array([5.879, 5.813, 5.841, 5.865, 5.972, 6.057, 6.188, 6.279, 6.483, 6.368]),
                                            mu_auger=np.array([54.005, 54.020, 54.025, 54.028, 54.034, 54.040, 54.035, 54.039, 54.039, 54.054]),
                                            )
    calibration_mean_xgm[12] = np.array([3.233, 3.241, 3.221, 3.200, 3.242, 3.232, 3.188, 3.176, 3.150, 3.129])
    tof_fit_result[13] = TofFitResult(energy=np.array([969.966, 979.971, 989.969, 999.977, 1009.957, 1019.964, 1029.956, 1039.950, 1049.967, 1059.948]),
                                            mu=np.array([184.504, 158.241, 142.142, 130.788, 122.269, 115.682, 109.914, 105.285, 101.500, 97.994]),
                                            sigma=np.array([3.218, 2.668, 2.423, 2.436, 2.499, 2.316, 2.223, 2.331, 2.189, 2.654]),
                                            A=np.array([361.280, 484.299, 551.889, 604.312, 598.749, 579.293, 537.174, 565.293, 537.914, 538.378]),
                                            Aa=np.array([433.716, 424.599, 410.892, 406.560, 391.931, 381.132, 369.500, 363.199, 352.688, 340.257]),
                                            offset=np.array([4.563, 4.350, 4.245, 4.283, 4.389, 4.560, 4.724, 4.728, 4.799, 4.694]),
                                            mu_auger=np.array([50.475, 50.485, 50.486, 50.503, 50.512, 50.515, 50.502, 50.528, 50.522, 50.522]),
                                            )
    calibration_mean_xgm[13] = np.array([3.233, 3.241, 3.221, 3.200, 3.242, 3.232, 3.188, 3.176, 3.150, 3.129])
    tof_fit_result[14] = TofFitResult(energy=np.array([969.966, 979.971, 989.969, 999.977, 1009.957, 1019.964, 1029.956, 1039.950, 1049.967, 1059.948]),
                                            mu=np.array([177.362, 150.365, 134.074, 122.505, 113.862, 107.137, 101.406, 96.757, 92.779, 89.287]),
                                            sigma=np.array([3.254, 2.453, 2.381, 2.276, 2.260, 2.177, 2.095, 2.135, 2.056, 2.236]),
                                            A=np.array([532.968, 587.325, 669.487, 727.741, 747.416, 747.688, 728.644, 748.139, 723.819, 716.258]),
                                            Aa=np.array([526.339, 507.572, 501.319, 495.038, 475.726, 457.691, 444.197, 439.884, 426.085, 404.517]),
                                            offset=np.array([5.260, 5.132, 5.074, 5.224, 5.399, 5.574, 5.827, 5.840, 6.035, 5.995]),
                                            mu_auger=np.array([41.652, 41.648, 41.655, 41.660, 41.684, 41.679, 41.692, 41.686, 41.688, 41.698]),
                                            )
    calibration_mean_xgm[14] = np.array([3.233, 3.241, 3.221, 3.200, 3.242, 3.232, 3.188, 3.176, 3.150, 3.129])
    tof_fit_result[15] = TofFitResult(energy=np.array([969.966, 979.971, 989.969, 999.977, 1009.957, 1019.964, 1029.956, 1039.950, 1049.967, 1059.948]),
                                            mu=np.array([178.974, 151.695, 135.028, 123.336, 114.594, 107.666, 101.973, 97.081, 92.934, 89.343]),
                                            sigma=np.array([3.141, 2.486, 2.228, 2.099, 2.076, 2.073, 2.033, 2.010, 2.018, 2.078]),
                                            A=np.array([308.571, 349.418, 395.520, 399.319, 417.045, 420.839, 395.461, 381.850, 369.179, 358.170]),
                                            Aa=np.array([264.795, 256.977, 253.242, 250.294, 240.044, 230.531, 223.772, 219.166, 214.223, 205.269]),
                                            offset=np.array([3.428, 3.423, 3.367, 3.622, 3.704, 3.787, 3.909, 4.051, 4.043, 3.986]),
                                            mu_auger=np.array([40.868, 40.878, 40.892, 40.901, 40.903, 40.906, 40.909, 40.920, 40.919, 40.925]),
                                            )
    calibration_mean_xgm[15] = np.array([3.233, 3.241, 3.221, 3.200, 3.242, 3.232, 3.188, 3.176, 3.150, 3.129])

    correct = {2:  [612272.284, 946.822, 21.275],
               4:  [616545.768, 946.333, 30.869],
               5:  [615576.869, 946.638, 31.079],
               6:  [613029.508, 946.339, 13.824],
               7:  [623419.734, 946.026, 11.527],
               9:  [611060.957, 946.512, 10.740],
               10: [609107.546, 946.376, 13.634],
               12: [607427.441, 946.218, 28.303],
               13: [574359.906, 946.874, 26.794],
               14: [594706.838, 946.768, 16.850],
               15: [654273.605, 945.733, 13.706]
               }

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

def test_avg_and_fit_single_channel(mock_sqs_etof_calibration_run, tmp_path):
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
    correct_energies = np.unique(mock_etof_mono_energies())
    correct_constants = np.array(mock_etof_calibration_constants())
    for tof_id in tof_ids:
        assert np.allclose(cal.tof_fit_result[tof_id].energy, correct_energies, rtol=1e-2, atol=1e-2)
        assert np.allclose(cal.model_params[tof_id], correct_constants, rtol=1e-2, atol=1e-2)

    d = tmp_path / "data"
    d.mkdir()
    fpath = str(d / "cookiebox_test.h5")
    cal.to_file(fpath)

    # make some plots
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
    cal.plot_diagnostics(0)
    plt.savefig(str(d / "diagnostics.pdf"))

    cal_read = CookieboxCalibration.from_file(fpath)

    # test if serialization worked
    for tof_id in tof_ids:
        assert np.allclose(cal_read.model_params[tof_id], correct_constants, rtol=1e-2, atol=1e-2)

    data = cal_read.load_data(mock_sqs_etof_calibration_run.select_trains(np.s_[10:20]))
    spectrum = cal_read.calibrate(data)

    # now do it subtracting the offset to check if that does not crash either
    spectrum = cal_read.calibrate(data, subtract_offset=True)


# tests data reading without parallelization
def test_no_parallel(mock_sqs_etof_calibration_run, tmp_path):
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
    energy_axis = np.linspace(965, 1070, 160)
    xgm = XGM(mock_sqs_etof_calibration_run, pulse_energy)
    cal = CookieboxCalibration(
                    auger_start_roi=1,
                    start_roi=75,
                    stop_roi=320,
    )
    # setup tof response
    tof_response = {0: TOFAnalogResponse(roi=slice(75, None))}
    tof_response[0].setup(tof_channel[0], scan)
    cal.setup(run=mock_sqs_etof_calibration_run, energy_axis=energy_axis, tof_settings=tof_channel,
              xgm=xgm,
              scan=scan,
              tof_response=tof_response,
              )
    correct_energies = np.unique(mock_etof_mono_energies())
    correct_constants = np.array(mock_etof_calibration_constants())
    for tof_id in tof_ids:
        assert np.allclose(cal.tof_fit_result[tof_id].energy, correct_energies, rtol=1e-2, atol=1e-2)
        assert np.allclose(cal.model_params[tof_id], correct_constants, rtol=1e-2, atol=1e-2)

    d = tmp_path / "data"
    d.mkdir()
    fpath = str(d / "cookiebox_test.h5")
    cal.to_file(fpath)

    # make some plots
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
    cal.plot_diagnostics()
    plt.savefig(str(d / "diagnostics.pdf"))

    plt.figure(figsize=(10, 8))
    cal.plot_transmissions()
    plt.savefig(str(d / "transmissions.pdf"))

    plt.figure(figsize=(10, 8))
    cal.plot_offsets()
    plt.savefig(str(d / "offsets.pdf"))

    cal_read = CookieboxCalibration.from_file(fpath)

    # test if serialization worked
    for tof_id in tof_ids:
        assert np.allclose(cal_read.model_params[tof_id], correct_constants, rtol=1e-2, atol=1e-2)

    data = cal_read.load_data(mock_sqs_etof_calibration_run.select_trains(np.s_[10:20]))
    spectrum = cal_read.calibrate(data)

