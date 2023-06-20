
from euxfel_bunch_pattern import DESTINATION_T4D, DESTINATION_T5D, \
    PHOTON_LINE_DEFLECTION, PPL_BITS
from extra_data.tests.mockdata.base import DeviceBase


def _fill_bunch_pattern_table(table):
    # SASE 1
    table[:table.shape[0]//4, 1000:1300:6] |= DESTINATION_T4D
    table[table.shape[0]//4:, 1000:1300:12] |= DESTINATION_T4D

    # SASE 2
    table[:, 1500:2000:8] |= DESTINATION_T5D

    # SASE 3
    table[:, 200] |= (DESTINATION_T4D | PHOTON_LINE_DEFLECTION)

    # LP_SPB
    table[:, 0:300:6] |= PPL_BITS.LP_SPB


class Timeserver(DeviceBase):
    control_keys = [
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
                f'CONTROL/{self.device_id}/bunchPatternTable/value'])

    def write_instrument(self, f):
        super().write_instrument(f)

        if self.nsamples > 0:
            _fill_bunch_pattern_table(f[
                f'INSTRUMENT/{self.device_id}:outputBunchPattern/'
                f'data/bunchPatternTable'])
