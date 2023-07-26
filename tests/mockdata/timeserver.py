
import numpy as np

from euxfel_bunch_pattern import DESTINATION_T4D, DESTINATION_T5D, \
    PHOTON_LINE_DEFLECTION, PPL_BITS, is_sase, is_laser
from extra_data.tests.mockdata.base import DeviceBase


def _fill_bunch_pattern_table(table, num_rows):
    # SASE 1
    table[:num_rows//2, 1000:1300:6] |= DESTINATION_T4D
    table[num_rows//2:, 1000:1300:12] |= DESTINATION_T4D

    # SASE 2
    table[:, 1500:2000:8] |= DESTINATION_T5D

    # SASE 3
    table[:, 200] |= (DESTINATION_T4D | PHOTON_LINE_DEFLECTION)

    # LP_SPB
    table[:, 0:300:6] |= PPL_BITS.LP_SPB


class Timeserver(DeviceBase):
    control_keys = [
        # TODO: Value increases soon.
        ('bunchPatternTable', 'u4', (2700,)),
        ('periodActual', 'f8', ()),
        ('periodSet', 'f8', ()),
        ('readBunchPatternTable', 'u1', ()),
        ('tickFactor', 'u4', ()),
    ]

    output_channels = ('outputBunchPattern/data',)

    instrument_keys = [
        ('bunchPatternTable', 'u4', (2700,))
    ]

    extra_run_values = [
        ('classId', None, 'TimeServer')
    ]

    def write_control(self, f):
        super().write_control(f)

        if self.ntrains > 0 and not self.no_ctrl_data:
            _fill_bunch_pattern_table(f[
                f'CONTROL/{self.device_id}/bunchPatternTable/value'],
                self.ntrains)

    def write_instrument(self, f):
        super().write_instrument(f)

        if self.nsamples > 0:
            _fill_bunch_pattern_table(f[
                f'INSTRUMENT/{self.device_id}:outputBunchPattern/'
                f'data/bunchPatternTable'], self.nsamples)


class PulsePatternDecoder(DeviceBase):
    control_keys = sum(
        [[(f'{loc}/pulseIds', 'i4', (2700,)), (f'{loc}/nPulses', 'i4', ())]
         for loc in ['laser', 'maindump', 'sase1', 'sase2', 'sase3']],
        [])

    extra_run_values = [
        ('classId', None, 'TimeServer2'),
        ('laserSource', None, 'LP_SPB')
    ]

    def write_control(self, f):
        super().write_control(f)

        if self.ntrains > 0 and not self.no_ctrl_data:
            table = np.zeros((self.ntrains, 2700), dtype=np.uint32)
            _fill_bunch_pattern_table(table, self.ntrains)

            grp = f[f'CONTROL/{self.device_id}']

            for loc, pulse_mask in [
                ('sase1', is_sase(table, sase=1)),
                ('sase2', is_sase(table, sase=2)),
                ('sase3', is_sase(table, sase=3)),
                ('laser', is_laser(table, laser=PPL_BITS.LP_SPB))
            ]:
                grp[f'{loc}/nPulses/value'][:] = pulse_mask.sum(axis=1)
                id_dset = grp[f'{loc}/pulseIds/value']

                for i, mask in enumerate(pulse_mask):
                    pulse_ids = mask.nonzero()[0]
                    id_dset[i, :len(pulse_ids)] = pulse_ids
