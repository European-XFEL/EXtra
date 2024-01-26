
def _identify_instrument(data, threshold=0.5):
    """Try to identify instrument some data was taken at.

    Args:
        data (DataCollection): Data to identify instrument of.
        threshold (float, optional): Minimum ratio among sources for the
            most common instrument prefix found, 0.5 by default.

    Returns
        (str or None) Instrument if it can be identified.
    """

    if data.files and (instrument := data.files[0].instrument) is not None:
        # Skip this method if the result is XMPL.
        if instrument != 'XMPL':
            return instrument

    from collections import Counter
    topic, num = Counter([
        s[:s.index('_')] for s in data.all_sources]).most_common()[0]

    if num > threshold * len(data.all_sources):
        # Only accept if it's in the majority of sources
        return topic


def _instrument_to_sase(instrument):
    if instrument in {'SA1', 'LA1', 'SPB', 'FXE'}:
        return 1
    elif instrument in {'SA2', 'LA2', 'MID', 'HED'}:
        return 2
    elif instrument in {'SA3', 'LA3', 'SCS', 'SQS', 'SXP'}:
        return 3


def _identify_sase(data):
    return _instrument_to_sase(_identify_instrument(data))


def _select_subcomponent_trains(src, keys, dst=None):
    if dst is None:
        from copy import copy
        dst = copy(src)

    for key in keys:
        prop = getattr(self, key)

        if prop is not None:
            setattr(dst, key, prop.select_trains(trains))

    return dst
