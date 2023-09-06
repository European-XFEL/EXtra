# Source prefixes in use at each SASE.
SASE_TOPICS = {
    1: {'SA1', 'LA1', 'SPB', 'FXE'},
    2: {'SA2', 'LA2', 'MID', 'HED'},
    3: {'SA3', 'LA3', 'SCS', 'SQS', 'SXP'}
}

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
