
from .scantool import Scantool  # noqa
from .pulses import XrayPulses, OpticalLaserPulses, PumpProbePulses, DldPulses  # noqa
from .scan import Scan  # noqa
from .xgm import XGM  # noqa


def identify_instrument(data):
    """Try to identify instrument data was taken at.

    Args:
        data (DataCollection): Data

    Returns
        (str or None) Instrument if it can be identified.
    """

    if data.files and (instrument := data.files[0].instrument) is not None:
        return instrument

    from collections import Counter
    topic, num = Counter([
        s[:s.index('_')] for s in data.all_sources]).most_common()[0]

    if num > 0.5 * len(data.all_sources):
        # Only accept if it's in the majority of sources
        return topic


def instrument_to_sase(instrument):
    if instrument in {'SA1', 'LA1', 'SPB', 'FXE'}:
        return 1
    elif instrument in {'SA2', 'LA2', 'MID', 'HED'}:
        return 2
    elif instrument in {'SA3', 'LA3', 'SCS', 'SQS', 'SXP'}:
        return 3


def identify_sase(data):
    return instrument_to_sase(identify_instrument(data))


def select_subcomponent_trains(src, dst=None, keys=None):
    if dst is None:
        from copy import copy
        dst = copy(src)

    if keys is None:
        keys = [k for k, v in dir(src).items() if hasattr(v, 'select_trains')]

    for key in keys:
        setattr(res, key, getattr(self, key).select_trains(trains))

    return res
