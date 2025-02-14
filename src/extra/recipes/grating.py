from typing import Optional, Union, Dict, List, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import numpy as np
import xarray as xr
import h5py

from extra_data import open_run, by_id, DataCollection
from scipy.ndimage import rotate
from extra.components import Scan
from scipy.stats import linregress
from functools import partial

from .base import BaseCalibration

def calc_mean(energy_id: int, scan: Scan, mono_run: DataCollection,
             grating_source: str, grating_key: str):
    energy, train_ids = scan.steps[energy_id]
    print(f"Energy {energy}, energy id {energy_id}")
    data = mono_run.select_trains(by_id[list(train_ids)])[grating_source, grating_key].xarray()
    return data.mean('trainId').to_numpy()

class Grating2DCalibration(BaseCalibration):
    """
    Calibrate a 2D grating spectrometer.

    Args:
      grating_source: Where to read the grating data from.
      angle: The rotation angle in degrees if the camera is not aligned.
      energy_source: Where to read the undulator energy from.
    """
    def __init__(self,
                 grating_source: str='SQS_DIAG3_BIU/CAM/CAM_6:daqOutput',
                 angle: float=0.0,
                 energy_source: str='SA3_XTD10_MONO/MDL/PHOTON_ENERGY',
                ):
        self.energy_source = energy_source
        self.grating_source = grating_source
        self.angle = angle

        self._version = 1
        self._all_fields = ["e0",
                            "slope",
                            "bkg",
                            "angle",
                            "energy_axis",
                            "calibration_energies",
                            "calibration_data",
                            "energy_source",
                            "energy_key",
                            "grating_source",
                            "grating_key",
                            "sources",
                            "_version",
                           ]

    def setup(self,
              bkg_run: DataCollection,
              run: DataCollection):
        """
        Setup calibration.

        Args:
          bkg_run: The background run.
          run: The calibration run.
        """
        # depending of using DOOCS or MDL device, guess the energy key
        sd = run[energy_source]
        ev_conv = 1.0
        if 'actualEnergy' in sd:
            self.energy_key = 'actualEnergy'
        elif 'actualPosition' in sd:
            self.energy_key = 'actualPosition'
            ev_conv = 1e3
        else:
            raise ValueError('Unknown energy source.')

        sd = run[grating_source]
        if 'data.image.pixels' in sd:
            self.grating_key = 'data.image.pixels'
        else:
            raise ValueError('Unknown grating source.')

        self.sources = [self.energy_source,
                        self.grating_source,
                       ]

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
        logging.info("Load background ...")
        self.get_background_template(bkg_run)

        # load data
        logging.info("Load data ...")
        self.load_data()

        # fit
        logging.info("Fit ...")
        self.fit()

        # now we can use the apply method
        logging.info("Ready to apply ...")

    def get_background_template(self, bkg_run: DataCollection):
        """Get the background template.

        Args:
          bkg_run: The background run.
        """
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

