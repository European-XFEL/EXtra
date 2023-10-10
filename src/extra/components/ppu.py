import logging
from typing import List, Union

import numpy as np
import pandas as pd
from extra_data import by_id
from extra_data.keydata import KeyData
from extra_data.reader import DataCollection
from extra_data.sourcedata import SourceData

log = logging.getLogger(__name__)


def _find_ppu(run: DataCollection, device: str = None):
    """Helper function to find a PPU device."""

    # fast path, we don't validate if the type or name match
    if isinstance(device, SourceData):
        return device
    elif isinstance(device, KeyData):
        return run[device.source]
    elif isinstance(device, str):
        if device in run.control_sources:
            return run[device]
        elif device in run.alias:
            return _find_ppu(run, run.alias[device])
        # else search substring for match
    elif device is not None:
        raise KeyError(f"ppu must be a SourceData or str, not {type(device).__name__}")

    # Then we list all PPU device in the run
    available_ppus = [
        source
        for source in run.control_sources
        if run[source].device_class in PPU._DEVICE_CLASSES
    ]
    if len(available_ppus) == 0:
        available_ppus = [s for s in run.control_sources if "MDL/PPU" in s]

    if len(available_ppus) == 0:
        raise KeyError("Could not find a PPU device in this data")
    elif len(available_ppus) == 1:
        return run[available_ppus[0]]
    else:  # len(available_ppus) > 1
        if device:
            # And unique substrings of available PPU
            matches = [name for name in available_ppus if device.upper() in name]
            if len(matches) == 1:
                return run[matches[0]]
            elif len(matches) == 0:
                raise KeyError(
                    f"Couldn't identify a PPU from '{device}'; please pass a valid device name, alias, or unique substring"
                )
            else:
                raise KeyError(
                    f"Multiple PPUs found matching '{device}', please be more specific: {matches}"
                )
        raise KeyError(f"Multiple PPU devices found in that data: {available_ppus}")


class PPU:
    """Interface to a Pulse Picker Unit (PPU).

    Despite its name, the PPU selects a bunch train from within the 10Hz
    structure and block the remainder of the beam.

    Technical description:
        A motor-driven absorber rotor is rotated into the beam axis in order to
        block the XFEL beam when triggered. The rotor is contained within a UHV
        chamber. In terms of temporal structure, the beam pipe is blocked by an
        absorbing rotor for up to 9/10ths of a second or vice versa,
        synchronized to the facility clock/trigger.
    """

    _DEVICE_CLASSES = ["PulsePickerTrainTrigger", "PulsePickerTrainTriggerCopy"]

    def __init__(
        self, data: DataCollection, ppu: Union[KeyData, SourceData, str] = None
    ):
        """

        Args:
            data (DataCollection):
            ppu (Union[KeyData, SourceData, str], optional):
                Specify a Pulse Picker Unit device to use, necessary if a run
                contains more than one PPU. This can be any of:
                  - The device name of the source.
                  - A `SourceData` or [KeyData][extra_data.KeyData] of the
                    control source (e.g. `HED_XTD6_PPU/MDL/PPU_TRIGGER`) of a
                    PPU.
                  - The alias name of either a `SourceData` or
                    [KeyData][extra_data.KeyData] belonging to a PPU.
                  - A unique (case-insensitive) substring of a PPU source name.

        Raises:
            KeyError: If we can't identify a unique PPU device from the
            arguments.
        """
        self.data = data
        self.device = _find_ppu(data, ppu)

    def train_ids(
        self, offset: int = 0, labelled: bool = False
    ) -> Union[List[int], pd.Series]:
        """All train IDs picked by the PPU.

        Args:
            offset (int, optional):
                offset to add to the selected trains. Defaults to 0.
            labelled (bool, optional):
                Returns a Pandas Series if set to True, where this index represents the
                trigger sequence a train ID is part of. Defaults to False.

        Returns:
            Union[List[int], pd.Series]: Train IDs picked by the PPU.
        """
        seq_start = self.device["trainTrigger.sequenceStart"].ndarray()
        # The trains picked are the unique values of trainTrigger.sequenceStart
        # minus the first (previous trigger before this run).
        start_train_ids = np.unique(seq_start)[1:] + offset

        train_ids = []
        sequences = []
        for seq, train_id in enumerate(start_train_ids):
            n_trains = self.device["trainTrigger.numberOfTrains"]
            n_trains = n_trains.select_trains(by_id[[train_id]]).ndarray()[0]
            train_ids.extend(np.arange(train_id, train_id + n_trains).tolist())
            sequences.extend([seq] * n_trains)

        log.info(
            f"PPU device {self.device.source} triggered for {len(train_ids)} train(s) across {len(sequences)} sequence(s)."
        )

        if labelled:
            train_ids = pd.Series(train_ids, index=sequences)
        return train_ids

    def trains(
        self, split_sequence: bool = False, offset: int = 0
    ) -> Union[DataCollection, List[DataCollection]]:
        """Returns a subset of the data only with Trains selected by the PPU.

        Args:
            split_sequence (bool, optional): Split data per PPU trigger sequence. Defaults to False.
            offset (int, optional): offset to apply to train IDs to be selected. Defaults to 0.

        Returns:
            Union[DataCollection, List[DataCollection]]:
                DataCollection(s) containing only trains triggered by the PPU
        """
        train_ids = self.train_ids(labelled=True, offset=offset)
        if split_sequence:
            return [
                self.data.select_trains(by_id[seq.values])
                for _, seq in train_ids.groupby(train_ids.index)
            ]
        return self.data.select_trains(by_id[train_ids.values])
