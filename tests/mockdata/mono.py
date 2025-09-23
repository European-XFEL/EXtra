
import numpy as np

from extra_data.tests.mockdata.base import DeviceBase

class MonoMdl(DeviceBase):
    control_keys = [
        ("actualEnergy", "f4", ()),
        ]

    def __init__(self, *args, energy_data, **kwargs):
        self.energy_data = energy_data
        super().__init__(*args, **kwargs)

    def write_instrument(self, f):
        super().write_instrument(f)

        ds = f[f'CONTROL/{self.device_id}/actualEnergy/value']
        ds[:] = self.energy_data
