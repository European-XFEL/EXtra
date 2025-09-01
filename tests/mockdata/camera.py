import numpy as np

from extra_data.tests.mockdata.base import DeviceBase

class GotthardIIWithData(DeviceBase):
    output_channels = ('daqOutput/data',)

    instrument_keys = [
        ("adc", "f4", (12,1000)),
    ]

    def __init__(self, *args, data, **kwargs):
        self.data = data
        super().__init__(*args, **kwargs)

    def write_instrument(self, f):
        super().write_instrument(f)

        ds = f[f'INSTRUMENT/{self.device_id}:daqOutput/data/adc']
        ds[:] = self.data

class CameraWithData(DeviceBase):
    output_channels = ('daqOutput/data',)

    instrument_keys = [
        ("image/pixels", "f4", (10, 1000)),
    ]

    def __init__(self, *args, data, **kwargs):
        self.data = data
        super().__init__(*args, **kwargs)

    def write_instrument(self, f):
        super().write_instrument(f)

        ds = f[f'INSTRUMENT/{self.device_id}:daqOutput/data/image/pixels']
        ds[:] = self.data

