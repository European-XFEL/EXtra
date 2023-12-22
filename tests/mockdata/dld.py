
from itertools import product

import numpy as np

from extra_data.tests.mockdata.base import DeviceBase


trigger_dt = np.dtype([('start', np.int32), ('stop', np.int32),
                       ('offset', np.float64), ('pulse', np.int16),
                       ('fel', bool), ('ppl', bool)])

edge_dt = np.dtype('f8')

hit_dt = np.dtype([('x', np.float64), ('y', np.float64),
                   ('t', np.float64), ('m', np.int32)], align=True)

signal_dt = np.dtype([('u1', np.float64), ('u2', np.float64),
                      ('v1', np.float64), ('v2', np.float64),
                      ('w1', np.float64), ('w2', np.float64),
                      ('mcp', np.float64)], align=True)


class ReconstructedDld(DeviceBase):
    max_rows = 10

    control_keys = [
        # Needed to work around a bug in DeviceBase.
        ('fakeScalar', 'u4', ()),
    ]

    output_channels = ('output/raw', 'output/rec')

    instrument_keys = {
        'raw': [('triggers', trigger_dt, ()),
                ('edges', edge_dt, (7, max_rows))],
        'rec': [('signals', signal_dt, (max_rows,)),
                ('hits', hit_dt, (max_rows,))],
    }

    extra_run_values = [
        ('digitizer/baseline_region', None, ':1000')
    ]

    def __init__(self, device_id):
        super().__init__(device_id)

    @property
    def pulses_per_train(self):
        # Increase number of pulses monotonically with trains.
        return np.arange(self.ntrains).astype(np.uint64)

    def write_instrument(self, f):
        source = f'{self.device_id}:output'

        # INDEX
        for idx_group, keys in self.instrument_keys.items():
            idx_root = f.create_group(f'INDEX/{source}/{idx_group}')

            i_first = idx_root.create_dataset(
                'first', (self.ntrains,), 'u8', maxshape=(None,))
            i_first[1:] = np.cumsum(self.pulses_per_train)[:-1]

            i_count = idx_root.create_dataset(
                'count', (self.ntrains,), 'u8', maxshape=(None,))
            i_count[:] = self.pulses_per_train

        num_entries = int(self.pulses_per_train.sum())  # from np.uint64

        for index_group, keys in self.instrument_keys.items():
            for key, dtype, entry_shape in keys:
                f.create_dataset(f'INSTRUMENT/{source}/{index_group}/{key}',
                                 maxshape=(None,) + entry_shape, dtype=dtype,
                                 shape=(num_entries,) + entry_shape)

        data_root = f[f'INSTRUMENT/{source}']

        triggers = np.zeros(num_entries, dtype=trigger_dt)
        pulse_ids = np.concatenate([np.arange(num_pulses).astype(np.int16)
                                    for num_pulses in self.pulses_per_train])
        triggers['pulse'][:num_entries] = pulse_ids
        triggers['start'][:num_entries] = 100 + pulse_ids * 1000
        triggers['stop'][:num_entries] = 150 + pulse_ids * 1000
        triggers['offset'][:num_entries] = 0.0
        triggers['fel'][:num_entries] = True
        triggers['ppl'][:num_entries] = (pulse_ids % 2) == 0

        data_root['raw/triggers'][:num_entries] = triggers
        data_root['raw/edges'][:num_entries] = np.nan
        data_root['rec/signals'][:num_entries] = np.nan
        data_root['rec/hits'][:num_entries] = (np.nan, np.nan, np.nan, -1)

        for i, ch in product(range(5), range(7)):
            # Alternate between 1-channel edges per pulse with random
            # values in [0, 1] + (7-channel).
            data_root['raw/edges'][i:num_entries:7, ch, :ch+1] = (7 - ch)  \
                + 0.5 + np.random.rand(1 + (num_entries - i - 1) // 7, ch+1)

        for i in range(5):
            # Alternate between 0-4 hits per pulse with random values.
            data_root['rec/signals'][i:num_entries:5, :i] = \
                tuple(np.random.rand(7))
            data_root['rec/hits'][i:num_entries:5, :i] = (
                *np.random.rand(3), np.random.randint(0, 18))
