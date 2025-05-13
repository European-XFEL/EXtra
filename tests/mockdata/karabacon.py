import h5py
import numpy as np
from extra_data.tests.mockdata.base import DeviceBase

str_dtype = h5py.special_dtype(vlen=str)

class Karabacon(DeviceBase):
    control_keys = [
        ("isMoving", "uint8", ()),
        ("deviceEnv/acquisitionMode", str_dtype, (1,)),
        ("deviceEnv/acquisitionTimes", "f8", (10_000,)),
        ("deviceEnv/activeMotors", str_dtype, (1000,)),
        ("scanEnv/scanType", str_dtype, (1,)),
        ("scanEnv/steps", "int32", (6,)),
        ("scanEnv/startPoints", "f8", (6,)),
        ("scanEnv/stopPoints", "f8", (6,)),
        ("actualConfiguration", str_dtype, (1,))
    ]

    def write_control(self, f):
        super().write_control(f)

        # Values taken from p7948, run 154
        mock_values = {
            "deviceEnv/acquisitionMode": np.array([b'Continuous Averaged'], dtype=object),
            "deviceEnv/acquisitionTimes": np.insert(np.ones(9999), 0, 30),
            "deviceEnv/activeMotors": np.array([b"ppl_odl"] + [b""] * 999, dtype=object),
            "scanEnv/scanType": np.array([b'dscan'], dtype=object),
            "scanEnv/steps": np.array([20,  1,  1,  1,  1,  1], dtype=np.int32),
            "scanEnv/startPoints": np.array([-15., 1., 1., 1., 1., 1.]),
            "scanEnv/stopPoints": np.array([25., 1., 1., 1., 1., 1.]),
            "actualConfiguration": np.array([b"--- Motors: ['FXE_AUXT_LIC/DOOCS/PPODL:default']--- Data Sources: ['FXE_EXP_ONC/METRO/USER_XAS:output0.schema.data.value', 'FXE_EXP_ONC/METRO/USER_XAS:output1.schema.data.value', 'FXE_EXP_ONC/METRO/USER_XAS:output2.schema.data.value', 'FXE_EXP_ONC/METRO/USER_XAS:output6.schema.data.value', 'FXE_EXP_ONC/METRO/USER_XAS:output7.schema.data.value', 'FXE_EXP_ONC/METRO/USER_XAS:output8.schema.data.value']--- Triggers: []"],
                                                  dtype=object)
        }


        for key, value in mock_values.items():
            ds_key = f"CONTROL/{self.device_id}/{key}/value"
            ds = f[ds_key]
            ntrains = ds.shape[0]
            for i in range(ntrains):
                ds[i, :] = value
