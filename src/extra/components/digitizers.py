import numpy as np

def _pull_to_baselevel(
        signal: np.ndarray, 
        out: np.ndarray,
        baseline: slice | np.ndarray,
        baselevel=None,
    ):
    """Pull baseline to a certain level."""

    if isinstance(baseline, slice):
        baseline = signal[..., baseline]

    correction = baseline.mean(axis=signal.ndim - 1)
    if baselevel is not None:
        correction -= baselevel

    np.subtract(signal, correction[..., None], out=out, casting='unsafe')