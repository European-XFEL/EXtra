
warning = 'The extra.staging package is NOT considered a stable API and may ' \
          'be removed or change in ANY future release.\n' \
          'Do not expect this import to continue working and APIs to be ' \
          'stable!'

from logging import getLogger, WARNING
log = getLogger(__name__)

if log.level <= WARNING:
    # If our logger's warnings are visible, emit the warning here.
    log.warn(warning)
else:
    # If our logger's warnings are silenced, print to stderr.
    import sys
    print(warning, file=sys.stderr)
