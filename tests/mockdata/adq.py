
import numpy as np

from extra_data.tests.mockdata.base import DeviceBase

from extra.utils import gaussian


class AdqDigitizer(DeviceBase):
    # still needed?
    control_keys = [
        ('fakeScalar', 'u4', ()),
    ]

    output_channels = ('network/digitizers',)

    instrument_keys = [
        ('channel_{board}_{ch_letter}/raw/length', 'u4', ()),
        ('channel_{board}_{ch_letter}/raw/position', 'u4', ()),
        ('channel_{board}_{ch_letter}/raw/triggerId', 'u8', ()),
        ('channel_{board}_{ch_letter}/raw/samples', 'i2', (150000,))
    ]

    extra_run_values = [
        ('classId', None, 'AdqDigitizer'),
        ('board{board}/enable', None, True),
        ('board{board}/interleavedMode', None, False),
        ('board{board}/enable_raw', None, True),
        ('board{board}/channel_{ch_number}/offset', None, 0),
        ('board{board}/channel_{ch_number}/enable', None, True),
    ]

    def __init__(self, *args, channels, **kwargs):
        self.channel_labels = []

        # These are dicts for now to have no duplicate keys, their
        # values are turned into lists afterwards.
        instrument_keys = {}
        extra_run_values = {}

        cls = self.__class__
        format_fields = {}
        for board_idx, num_channels in enumerate(channels):
            format_fields['board'] = board_idx + 1

            for ch_index in range(num_channels):
                format_fields['ch_number'] = ch_index + 1
                format_fields['ch_letter'] = chr(ord('A') + ch_index)
                self.channel_labels.append(
                    '{board}_{ch_letter}'.format(**format_fields))

                for key, dtype, shape in cls.instrument_keys:
                    full_key = key.format(**format_fields)
                    instrument_keys[full_key] = (full_key, dtype, shape)

                for key, dtype, value in cls.extra_run_values:
                    full_key = key.format(**format_fields)
                    extra_run_values[full_key] = (full_key, dtype, value)

        self.instrument_keys = list(instrument_keys.values())
        self.extra_run_values = list(extra_run_values.values())

        super().__init__(*args, **kwargs)

    def write_instrument(self, f):
        super().write_instrument(f)

        root_grp = f[f'INSTRUMENT/{self.device_id}:network/digitizers']
        x = np.arange(150000)

        for i, ch_label in enumerate(self.channel_labels):
            # Add a channel-dependent baseline shift and gaussian to
            # each channel.
            root_grp[f'channel_{ch_label}/raw/samples'][:] += -10 * (i+1) \
                + gaussian(x, 0, i*80, i*1000, 50).astype(np.int16)
