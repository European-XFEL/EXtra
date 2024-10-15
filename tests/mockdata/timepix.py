
from extra_data.tests.mockdata.base import DeviceBase

import numpy as np


pixel_shape = (1000,)
centroid_shape = (1000,)

centroid_dt = np.dtype([('x', np.float64),
                        ('y', np.float64),
                        ('toa', np.float64),
                        ('tot', np.float64),
                        ('tot_avg', np.float64),
                        ('tot_max', np.uint16),
                        ('size', np.int16)])

stat_dt = np.dtype([('N_centroids', np.int64),
                    ('missing_centroids', np.bool_),
                    ('fraction_px_in_centroids', np.float64)])


class TimepixCommon(DeviceBase):
    def _fill_hit_data(self, dset, min_val, max_val):
        val = np.nan if np.issubdtype(dset.dtype, np.floating) else 1
        dset[:self.ntrains] = np.stack([
            np.concatenate([
                np.linspace(min_val, max_val, min(N, pixel_shape[0]),
                            dtype=dset.dtype),
                np.full(max(pixel_shape[0] - N, 0), val, dtype=dset.dtype)
            ])
            for N in range(self.ntrains)])


class Timepix3Receiver(TimepixCommon):
    control_keys = [
        ('biasVoltage', 'f8', ()),
        ('chip0.ibiasIkrum', 'u1', ())
    ]

    output_channels = ('daqOutput.chip0/data',)

    instrument_keys = {
        ('x', 'u1', pixel_shape),
        ('y', 'u1', pixel_shape),
        ('toa', 'f8', pixel_shape),
        ('tot', 'u2', pixel_shape),
        ('size', 'u4', ())
    }

    extra_run_values = [
        ('classId', None, 'Timepix3SingleReceiver')
    ]

    def write_instrument(self, f):
        super().write_instrument(f)

        root = f[f'INSTRUMENT/{self.device_id}:daqOutput.chip0/data']
        self._fill_hit_data(root['x'], 0, 256)
        self._fill_hit_data(root['y'], 256, 0)
        self._fill_hit_data(root['toa'], 0.0, 5.0e-6)
        self._fill_hit_data(root['tot'], 200, 800)

        if self.ntrains > pixel_shape[0]:
            size = np.clip(np.arange(self.ntrains), 0, pixel_shape[0])
        else:
            size = np.arange(self.ntrains)

        root['size'][:self.ntrains] = size


class Timepix3Centroids(TimepixCommon):
    control_keys = [
        ('fakeScalar', 'u4', ()),
    ]

    output_channels = ('daqOutput.chip0/data',)

    instrument_keys = {
        ('centroids', centroid_dt, centroid_shape),
        ('labels', 'i4', pixel_shape),
        ('stats', stat_dt, centroid_shape)
    }

    def write_instrument(self, f):
        super().write_instrument(f)

        root = f[f'INSTRUMENT/{self.device_id}:daqOutput.chip0/data']

        centroids = np.zeros((self.ntrains,) + centroid_shape,
                              dtype=centroid_dt)
        self._fill_hit_data(centroids['x'], 0, 256)
        self._fill_hit_data(centroids['y'], 256, 0)
        self._fill_hit_data(centroids['toa'], 0.0, 5.0e-6)
        self._fill_hit_data(centroids['tot'], 200, 800)
        self._fill_hit_data(centroids['tot_avg'], 100, 400)
        self._fill_hit_data(centroids['tot_max'], 150, 600)
        self._fill_hit_data(centroids['size'], 1, 10)
        root['centroids'][:self.ntrains] = centroids

        self._fill_hit_data(root['labels'], 1, 10)
