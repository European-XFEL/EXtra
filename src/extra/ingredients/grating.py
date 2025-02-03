from typing import Optional, Union, Dict, List, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from extra_data import open_run, by_id, DataCollection
from scipy.ndimage import rotate
from extra.components import Scan
from scipy.stats import linregress
from functools import partial

import xarray as xr

import h5py

def log(level, msg):
    if level > 0:
        print(msg)

def calc_mean(energy_id: int, scan: Scan, mono_run: DataCollection,
             grating_source: str, grating_key: str):
    energy, train_ids = scan.steps[energy_id]
    print(f"Energy {energy}, energy id {energy_id}")
    data = mono_run.select_trains(by_id[list(train_ids)])[grating_source, grating_key].xarray()
    return data.mean('trainId').to_numpy()

class GratingCalib(object):
    """
    Calibrate a grating spectrometer.

    Arguments:
      bkg_run: The background run.
      run: The calibration run.
      angle: The rotation angle in degrees if the camera is not aligned.
      energy_source: Where to read the undulator energy from.
      log_level: Whether to produce log output. Set to 1 for more log output.

    """
    def __init__(self,
                 bkg_run: DataCollection,
                 run: DataCollection,
                 grating_source: str='SQS_DIAG3_BIU/CAM/CAM_6:daqOutput',
                 angle: float=0.0,
                 energy_source: str='SA3_XTD10_MONO/MDL/PHOTON_ENERGY',
                 log_level: Optional[int]=0,
                ):
        if run is None and bkg_run is None:
            return

        # depending of using DOOCS or MDL device, guess the energy key
        self.energy_source = energy_source
        sd = run[energy_source]
        ev_conv = 1.0
        if 'actualEnergy' in sd:
            self.energy_key = 'actualEnergy'
        elif 'actualPosition' in sd:
            self.energy_key = 'actualPosition'
            ev_conv = 1e3
        else:
            raise ValueError('Unknown energy source.')

        self.grating_source = grating_source
        sd = run[grating_source]
        if 'data.image.pixels' in sd:
            self.grating_key = 'data.image.pixels'
        else:
            raise ValueError('Unknown grating source.')

        self.sources = [self.energy_source,
                        self.grating_source,
                       ]
        self.log_level = log_level
        self.angle = angle

        # select data from the run
        self._run = run.select(self.sources, require_all=True)

        # create scan object
        one_ev = 1.0/ev_conv
        self._scan = Scan(self._run[self.energy_source, self.energy_key], resolution=one_ev)
        self.calibration_energies = self._scan.positions
        self.calibration_energies *= ev_conv

        # outputs
        self.e0 = 0
        self.slope = 0
        self.energy_axis = None

        # background
        log(self.log_level, "Load background ...")
        self.get_background_template(bkg_run)

        # load data
        log(self.log_level, "Load data ...")
        self.load_data()

        # fit
        log(self.log_level, "Fit ...")
        self.fit()

        # now we can use the apply method
        log(self.log_level, "Ready to apply ...")

    def get_background_template(self, bkg_run: DataCollection):
        """Get the background template."""
        self.bkg = bkg_run[self.grating_source, self.grating_key].ndarray().mean(0)

    def load_data(self):
        """Load calibration data."""
        fn = partial(calc_mean,
                     scan=self._scan,
                     mono_run=self._run,
                     grating_source=self.grating_source,
                     grating_key=self.grating_key,
                     )
        energy_ids = np.arange(len(self.calibration_energies))
        with ProcessPoolExecutor() as p:
            data = np.stack(list(p.map(fn, energy_ids)), axis=0)
        self.calibration_data = rotate(data - self.bkg, self.angle, axes=(-1, -2)).mean(1)

    def fit(self):
        """Fit line."""
        sample = np.arange(self.calibration_data.shape[-1])
        sample_mode = np.argmax(self.calibration_data, axis=-1)
        #sample_mode = snp.sum(self.calibration_data*sample, axis=-1)/np.sum(self.calibration_data, axis=-1)
        res = linregress(sample_mode, self.calibration_energies)
        self.slope = res.slope
        self.e0 = res.intercept
        self.energy_axis = self.e0 + self.slope*sample

    def save(self, filename: str):
        """
        Dump all data needed for applying the calibration into an h5 file.
        """
        with h5py.File(filename, "w") as fid:
            # energy axis
            fid["e0"] = self.e0
            fid["slope"] = self.slope
            fid["bkg"] = self.bkg
            fid["angle"] = self.angle
            fid["energy_axis"] = self.energy_axis
            fid["calibration_energies"] = self.calibration_energies
            fid["calibration_data"] = self.calibration_data

            # energy source and key (not really needed to apply it though)
            fid.attrs["energy_source"] = self.energy_source
            fid.attrs["energy_key"] = self.energy_key
            fid.attrs["sources"] = self.sources

            fid.attrs["grating_source"] = self.grating_source
            fid.attrs["grating_key"] = self.grating_key

            fid.attrs["log_level"] = self.log_level


    @classmethod
    def load(cls, filename: str):
        """
        Load setup saved with save previously.
        """
        obj = GratingCalib(None, None)
        with h5py.File(filename, "r") as fid:
            obj.e0 = fid["e0"][()]
            obj.slope = fid["slope"][()]
            obj.bkg = fid["bkg"][()]
            obj.angle = fid["angle"][()]
            obj.energy_axis = fid['energy_axis'][()]
            obj.calibration_energies = fid['calibration_energies'][()]
            obj.calibration_data = fid['calibration_data'][()]

            obj.energy_source = fid.attrs["energy_source"]
            obj.energy_key = fid.attrs["energy_key"]

            obj.grating_source = fid.attrs["grating_source"]
            obj.grating_key = fid.attrs["grating_key"]

            obj.sources = fid.attrs["sources"]
            if "log_level" in fid.attrs:
                obj.log_level = fid.attrs["log_level"]
            else:
                obj.log_level = 1

        return obj
    def apply(self, run: DataCollection) -> xr.Dataset:
        """
        Apply calibration to a new analysis run.
        It is assumed it contains the same settings.
        """
        # do it per train to avoid memory overflow
        trainId = list()
        out_data = list()
        for i, (tid, data) in enumerate(run.trains()):
            #print(f"Train {tid}, idx {i}")
            d = data[self.grating_source][self.grating_key]
            d = rotate(d - self.bkg, self.angle, axes=(-1, -2))
            trainId += [tid]
            out_data += [d.sum(-2)]
        energy = self.energy_axis
        out_data = xr.DataArray(data=np.stack(out_data, axis=0),
                            dims=('trainId', 'energy'),
                            coords=dict(trainId=np.array(trainId),
                                        energy=energy
                                        )
                           )

        return out_data

