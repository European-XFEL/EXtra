
from ..utils.misc import _isinstance_no_import

# Source prefixes in use at each SASE.
SASE_TOPICS = {
    1: {'SA1', 'LA1', 'SPB', 'FXE'},
    2: {'SA2', 'LA2', 'MID', 'HED'},
    3: {'SA3', 'LA3', 'SCS', 'SQS', 'SXP'}
}


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
        s.partition('_')[0] for s in data.all_sources]).most_common()[0]

    if num > threshold * len(data.all_sources):
        # Only accept if it's in the majority of sources
        return topic


def identify_sase(run):
    """Try to identify which SASE a run belongs to."""

    sases = {sase
             for src in run.all_sources
             for sase, topics in SASE_TOPICS.items()
             if src[:src.find('_')] in topics}

    if len(sases) == 1:
        return sases.pop()
    elif sases == {1, 3}:
        # SA3 data often contains one or more SA1 sources
        # from shared upstream components.
        return 3
    else:
        raise ValueError('sources from multiple SASE branches {} found, '
                         'please pass the SASE beamline explicitly'.format(
                             ', '.join(map(str, sases))))


def _instrument_to_sase(instrument):
    if instrument in {'SA1', 'LA1', 'SPB', 'FXE'}:
        return 1
    elif instrument in {'SA2', 'LA2', 'MID', 'HED'}:
        return 2
    elif instrument in {'SA3', 'LA3', 'SCS', 'SQS', 'SXP'}:
        return 3


def _select_subcomponent_trains(src, keys, dst=None):
    if dst is None:
        from copy import copy
        dst = copy(src)

    for key in keys:
        prop = getattr(self, key)

        if prop is not None:
            setattr(dst, key, prop.select_trains(trains))

    return dst


from collections.abc import Sequence
from typing import Protocol, Self, runtime_checkable
import pandas as pd
from extra_data import by_id


@runtime_checkable
class TrainData(Protocol):
    train_ids: Sequence[int]

    def select_trains(self, train_sel) -> Self:
        raise NotImplementedError('select_trains')

    def data_counts(self, labelled=True) -> pd.Series:
        raise NotImplementedError('data_counts')


def align(*unaligned):
    counts = pd.concat([
        obj.data_counts() for obj in unaligned
        if isinstance(obj, TrainData)
    ], axis=1, join='inner')

    if len(counts.columns) != len(unaligned):
        raise TypeError('one or more passed objects do not implement '
                        'the TrainData protocol')

    train_ids = counts.index[(counts > 0).all(axis=1)].to_numpy()

    return [obj.select_trains(by_id[train_ids]) for obj in unaligned]
