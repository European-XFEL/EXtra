from typing import Optional, Union, Dict, List, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import logging
from functools import partial

import numpy as np
import xarray as xr
import h5py

from extra_data import open_run, by_id, DataCollection, KeyData
from extra.components import Scan

from .base import SerializableMixin

def calc_mean(energy_id: int, scan: Scan,
             grating: KeyData) -> np.ndarray:
    """
    Calculate mean over train IDs with a given energy value.

    Args:
      energy_id: The id of this energy bin in the `scan` object.
      scan: The `Scan` object.
      grating: The camera key object.
    """
    energy, train_ids = scan.steps[energy_id]
    logging.debug(f"Energy {energy}, energy id {energy_id}")
    data = grating.select_trains(by_id[list(train_ids)]).xarray()
    return data.mean('trainId').to_numpy()

class Grating1DCalibration(SerializableMixin):
    """
    Calibrate a 1D grating spectrometer.

    Args:
      offset: Offset of the first pulse.

    Example:
    ```
    bkg_run = open_run(proposal=900485, run=590)
    calib_run = open_run(proposal=900485, run=611)
    scan = Scan(calib_run["SA3_XTD10_MONO/MDL/PHOTON_ENERGY", "actualEnergy"])
    grating_signal = calib_run["SQS_EXP_GH2-2/CORR/RECEIVER:daqOutput", "data.adc"]
    grating_bkg = bkg_run["SQS_EXP_GH2-2/CORR/RECEIVER:daqOutput", "data.adc"]
    pulses = XrayPulses(calib_run)
    grating_calib = Grating1DCalibration()
    grating_calib.setup(grating_signal, grating_bkg, scan, pulses)

    # apply it in new data
    new_run = open_run(...)
    grating_calib.apply(new_run)
    ```
    """
    def __init__(self, offset: int=22):
        self._version = 1
        self.offset = offset
        self._all_fields = [
                            "pulse_period",
                            "offset",
                            "e0",
                            "slope",
                            "bkg",
                            "energy_axis",
                            "calibration_energies",
                            "calibration_data",
                            "grating_source",
                            "grating_key",
                            "sources",
                            "_version",
                           ]

    def setup(self,
              grating_signal: KeyData,
              grating_bkg: KeyData,
              scan: Scan,
              pulses: XrayPulses
              ):
        """
        Setup calibration.

        Args:
          bkg_run: The background run.
          run: The calibration run.
          grating_signal: Where to read the grating data from.
                   Example: `signal_run["SQS_EXP_GH2-2/CORR/RECEIVER:daqOutput", "data.adc"]`
          grating_bkg: Where to read the grating background data from.
               Example: `bkg_run["SQS_EXP_GH2-2/CORR/RECEIVER:daqOutput", "data.adc"]`
          scan: Scan object identfying where to read the undulator energy from.
                Example: `Scan(run["SA3_XTD10_MONO/MDL/PHOTON_ENERGY", "actualEnergy"])`
          pulses: Object with bunch pattern table.
                Example: `XrayPulses(run)`
        """
        self.grating_source = grating_signal.source
        self.grating_key = grating_signal.key
        self._grating_signal = grating_signal
        self._grating_bkg = grating_bkg

        self.sources = [
                        self.grating_source,
                       ]

        # create scan object
        self._scan = scan
        self.calibration_energies = self._scan.positions

        # outputs
        self.e0 = 0
        self.slope = 0
        self.energy_axis = None

        # pulse delta
        logging.info("Extract bunch pattern table")
        self.pulse_period = self.get_pulse_period(pulses)

        # background
        logging.info("Load background ...")
        self.get_background_template()

        # load data
        logging.info("Load data ...")
        self.load_data()

        # fit
        logging.info("Fit ...")
        self.fit()

        # now we can use the apply method
        logging.info("Ready to apply ...")

    def _asdict(self):
        """
        Return serializable dict.
        """
        return {k: v for k, v in self.__dict__.items() if k in self._all_fields}

    def _fromdict(self, all_data):
        """
        Rebuild object from dict.
        """
        for k, v in all_data.items():
            setattr(self, k, v)

    def get_pulse_period(self, pulses: XrayPulses):
        """
        Estimate difference between samples in neighbour pulses.

        Args:
          pulses: XrayPulses element for the run.

        Returns: Sample difference between neighbour pulses.
        """
        pulse_ids = pulses.pulse_ids(labelled=True)
        pids_by_train = pulse_ids.groupby(level=0)
        pulse_period = int(pids_by_train.diff().min())
        return pulse_period

    def get_background_template(self):
        """Get the background template.
        """
        self.bkg = self._grating_bkg.ndarray().mean(0)

    def load_data(self):
        """Load calibration data."""
        from scipy.ndimage import rotate
        fn = partial(calc_mean,
                     scan=self._scan,
                     grating=self._grating_signal,
                     )
        energy_ids = np.arange(len(self.calibration_energies))
        # average data in each mono scan bin
        with ProcessPoolExecutor() as p:
            data = np.stack(list(p.map(fn, energy_ids)), axis=0)
        # subtract the background
        self.calibration_data = data - self.bkg
        # skip offset and collect pulse data each pulse_period samples only
        self.calibration_data = self.calibration_data[:, self.offset::self.pulse_period, :]
        # average over pulses
        self.calibration_data = np.mean(self.calibration_data, axis=1)

    def fit(self):
        """Fit line."""
        from scipy.stats import linregress
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
        pulse_period = XrayPulses(run)
        trainId = list()
        out_data = list()
        for i, (tid, data) in enumerate(run.trains()):
            #print(f"Train {tid}, idx {i}")
            d = data[self.grating_source][self.grating_key]
            d = d - self.bkg
            # skip offset and collect pulse data each pulse_period samples only
            d = d[self.offset::pulse_period, :]
            trainId += [tid]
            out_data += [d]
        energy = self.energy_axis
        out_data = xr.DataArray(data=np.stack(out_data, axis=0),
                            dims=('trainId', 'energy'),
                            coords=dict(trainId=np.array(trainId),
                                        energy=energy
                                        )
                           )

        return out_data

class Grating2DCalibration(SerializableMixin):
    """
    Calibrate a 2D grating spectrometer.

    Args:
      angle: The rotation angle in degrees if the camera is not aligned.
    """
    def __init__(self,
                 angle: float=0.0,
                ):
        self.angle = angle

        self._version = 1
        self._all_fields = ["e0",
                            "slope",
                            "bkg",
                            "angle",
                            "energy_axis",
                            "calibration_energies",
                            "calibration_data",
                            "grating_source",
                            "grating_key",
                            "sources",
                            "_version",
                           ]

    def setup(self,
              grating_signal: KeyData,
              grating_bkg: KeyData,
              scan: Scan):
        """
        Setup calibration.

        Args:
          bkg_run: The background run.
          run: The calibration run.
          grating_signal: Where to read the grating data from.
                   Example: `signal_run["SQS_DIAG3_BIU/CAM/CAM_6:daqOutput, "data.image.pixels"]`
          grating_bkg: Where to read the grating background data from.
               Example: `bkg_run["SQS_DIAG3_BIU/CAM/CAM_6:daqOutput, "data.image.pixels"]`
          scan: Scan object identfying where to read the undulator energy from.
                Example: `Scan(run["SA3_XTD10_MONO/MDL/PHOTON_ENERGY", "actualEnergy"])`
        """
        self.grating_source = grating_signal.source
        self.grating_key = grating_signal.key
        self._grating_signal = grating_signal
        self._grating_bkg = grating_bkg

        self.sources = [
                        self.grating_source,
                       ]

        # create scan object
        self._scan = scan
        self.calibration_energies = self._scan.positions

        # outputs
        self.e0 = 0
        self.slope = 0
        self.energy_axis = None

        # background
        logging.info("Load background ...")
        self.get_background_template()

        # load data
        logging.info("Load data ...")
        self.load_data()

        # fit
        logging.info("Fit ...")
        self.fit()

        # now we can use the apply method
        logging.info("Ready to apply ...")

    def _asdict(self):
        """
        Return serializable dict.
        """
        return {k: v for k, v in self.__dict__.items() if k in self._all_fields}

    def _fromdict(self, all_data):
        """
        Rebuild object from dict.
        """
        for k, v in all_data.items():
            setattr(self, k, v)

    def get_background_template(self):
        """Get the background template.
        """
        self.bkg = self._grating_bkg.ndarray().mean(0)

    def load_data(self):
        """Load calibration data."""
        from scipy.ndimage import rotate
        fn = partial(calc_mean,
                     scan=self._scan,
                     grating=self._grating_signal,
                     )
        energy_ids = np.arange(len(self.calibration_energies))
        with ProcessPoolExecutor() as p:
            data = np.stack(list(p.map(fn, energy_ids)), axis=0)
        self.calibration_data = rotate(data - self.bkg, self.angle, axes=(-1, -2)).mean(1)

    def fit(self):
        """Fit line."""
        from scipy.stats import linregress
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
        from scipy.ndimage import rotate
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

