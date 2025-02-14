
import h5py
from dataclasses import dataclass, asdict, is_dataclass

def save_dict(h5grp, obj: Dict[str, Any]):
    """Save dictionary object into HDF5 group recursively."""
    for k, v in obj.items():
        if isinstance(v, dict):
            newgrp = h5grp.create_group(f"{k}")
            save_dict(newgrp, v)
        elif isinstance(v, str) or isinstance(v, int) or isinstance(v, float):
            h5grp.attrs[f"{k}"] = v.decode('utf-8')
        elif is_dataclass(v):
            new_v = asdict(v)
            h5grp[f"{k}"] = new_v
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

class BaseCalibration(object):
    """
    Base class for all calibration-related sub-classes.

    This should never be instantiated directly.
    """
    def __init__(self):
        raise NotImplementedError("This is a base class implementation. This method must be implemented by subclasses.")

    def setup(self):
        """Setup analysis from previous data."""
        raise NotImplementedError("This is a base class implementation. This method must be implemented by subclasses.")

    def apply(self):
        """Apply into new data"""
        raise NotImplementedError("This is a base class implementation. This method must be implemented by subclasses.")
        

    def to_file(self, filename: str):
        """
        Dump all data needed for applying the calibration into an h5 file.

        Args:
          filename: The output file name.

        """
        with h5py.File(filename, "w") as fid:
            all_data = {k: v for k, v in self.__dict__ if k in self._all_fields}
            save_dict(fid, all_data)

    @classmethod
    def from_file(cls, filename: str):
        """
        Load setup saved with save previously.
        """
        obj = cls()
        with h5py.File(filename, "r") as fid:
            all_data = load_dict(fid)
            for k, v in all_data.items():
                setattr(obj, k, v)

        return obj
