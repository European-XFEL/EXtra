
from extra_data.tests.mockdata.base import DeviceBase


# This is a more accurate representation of an XGM than the XGM class from the
# extra_data tests (it has some incorrect property names).
class XGM(DeviceBase):
    extra_run_values = [
        ('classId', None, 'DoocsXGM'),
        ("location", None, "XGM.2595.T6")
    ]

    output_channels = ('output/data',)

    instrument_keys = [
        ('intensityTD', 'f4', (1000,)),
    ]

    def __init__(self, device_id, main_nbunches_property="numberOfBunchesActual"):
        self.control_keys = [
            ("controlData/slowTrain", "f4", ()),
            (f"pulseEnergy/{main_nbunches_property}", "f4", ()),
            ('pulseEnergy/photonFlux', 'f4', ()),
            ('pulseEnergy/wavelengthUsed', 'f4', ())
        ]
        super().__init__(device_id)


class XGMD(DeviceBase):
    extra_run_values = [
        ("classId", None, "DoocsXGMD"),
        ("location", None, "XGM.3331.FXE")
    ]

    control_keys = [
        ("controlData/slowTrain", "f4", ()),
        ("controlData/slowTrainSa1", "f4", ()),
        ("controlData/slowTrainSa3", "f4", ()),
        ("pulseEnergy/numberOfBunchesActual", "f4", ()),
        ('pulseEnergy/photonFlux', 'f4', ()),
        ('pulseEnergy/wavelengthUsed', 'f4', ()),
        ("pulseEnergy/numberOfSa1BunchesActual", "f4", ()),
        ("pulseEnergy/numberOfSa3BunchesActual", "f4", ())
    ]

    output_channels = ('output/data',)

    instrument_keys = XGM.instrument_keys + [
        ("intensitySa1TD", "f4", (1000,)),
        ("intensitySa3TD", "f4", (1000,))
    ]


class XGMReduced(XGMD):
    extra_run_values = [
        ("classId", None, "DoocsXGMReduced"),
        ("location", None, "XGM.3356.SQS")
    ]
