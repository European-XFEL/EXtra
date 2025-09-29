from typing import Optional, Union, Dict, List, Tuple, Any

import h5py
from dataclasses import dataclass, asdict, is_dataclass

def save_dict(h5grp, obj: Dict[str, Any]):
    """Save dictionary object into HDF5 group recursively."""
    for k, v in obj.items():
        if isinstance(v, dict):
            newgrp = h5grp.create_group(f"{k}")
            save_dict(newgrp, v)
        elif isinstance(v, str) or isinstance(v, int) or isinstance(v, float):
            h5grp.attrs[f"{k}"] = v
        elif isinstance(v, list):
            h5grp.attrs[f"{k}"] = v
        elif is_dataclass(v):
            new_v = asdict(v)
            newgrp = h5grp.create_group(f"{k}")
            save_dict(newgrp, new_v)
        elif isinstance(v, SerializableMixin):
            new_v = v._asdict()
            newgrp = h5grp.create_group(f"{k}")
            save_dict(newgrp, new_v)
        else:
            h5grp[f"{k}"] = v

def load_dict(h5grp):
    """Load dictionary object from HDF5 group recursively."""
    out = dict()
    for k, obj in h5grp.items():
        # convert positive integers into integers if needed
        if k.isdigit():
            k = int(k)
        if isinstance(obj, h5py.Group):
            out[k] = load_dict(obj)
        elif isinstance(obj, h5py.Dataset):
            out[k] = obj[()]
    for k, obj in h5grp.attrs.items():
        out[k] = obj
    return dict(sorted(out.items()))

class SerializableMixin(object):
    """
    Base class for serializable sub-classes.

    This should never be instantiated directly.
    """
    def to_file(self, filename: str):
        """
        Dump all data needed for applying the calibration into an h5 file.

        Args:
          filename: The output file name.

        """
        with h5py.File(filename, "w") as fid:
            all_data = self._asdict()
            save_dict(fid, all_data)

    def _asdict(self):
        """
        Return serializable dict.
        """
        pass

    def _fromdict(self, all_data):
        """
        Rebuild it from dict.
        """
        pass

    @classmethod
    def from_file(cls, filename: str):
        """
        Load setup saved with save previously.
        """
        obj = cls()
        with h5py.File(filename, "r") as fid:
            all_data = load_dict(fid)
            obj._fromdict(all_data)

        return obj

