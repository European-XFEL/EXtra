from typing import Optional, Union, Dict, List, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import logging
from functools import partial

import numpy as np
import xarray as xr
import h5py

from extra_data import open_run, by_id, DataCollection, KeyData
from extra.components import Scan, XrayPulses

from .base import SerializableMixin

def calc_mean(energy_id: int, scan: Scan,
             grating: KeyData,
             ) -> np.ndarray:
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
    def __init__(self, offset: Optional[int]=None, min_pixel: int=500, max_pixel: int=1000):
        self._version = 1
        self.offset = offset
        self.min_pixel = min_pixel
        self.max_pixel = max_pixel
        self.calibration_mask = None
        self._all_fields = [
                            "pulse_period",
                            "offset",
                            "calibration_mask",
                            "min_pixel",
                            "max_pixel",
                            "e0",
                            "slope",
                            "bkg",
                            "bkg_unc",
                            "energy_axis",
                            "calibration_energies",
                            "calibration_data",
                            "calibration_unc",
                            "grating_source",
                            "grating_key",
                            "sources",
                            "_version",
                           ]

    def setup(self,
              grating_signal: KeyData,
              scan: Scan,
              pulses: XrayPulses,
              grating_bkg: Optional[KeyData]=None,
              ):
        """
        Setup calibration.

        Args:
          bkg_run: The background run.
          run: The calibration run.
          grating_signal: Where to read the grating data from.
                   Example: `signal_run["SQS_EXP_GH2-2/CORR/RECEIVER:daqOutput", "data.adc"]`
          scan: Scan object identfying where to read the undulator energy from.
                Example: `Scan(run["SA3_XTD10_MONO/MDL/PHOTON_ENERGY", "actualEnergy"])`
          pulses: Object with bunch pattern table.
                Example: `XrayPulses(run)`
          grating_bkg: Where to read the grating background data from.
               Example: `bkg_run["SQS_EXP_GH2-2/CORR/RECEIVER:daqOutput", "data.adc"]`
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

        if self.offset is None:
            logging.info("Guess offset")
            self.guess_offset()

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

    def guess_offset(self):
        """
        Guess offset.
        """
        I = self._grating_signal.xarray().sel(dim_1=np.s_[self.min_pixel:self.max_pixel]).mean('dim_1').mean('trainId')
        threshold = np.median(I)
        self.offset = np.where(I > threshold)[0][0]
        logging.info(f"Offset estimated at {self.offset}")

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
        logging.info(f"Pulse period estimated at {pulse_period}")
        return pulse_period

    def get_background_template(self):
        """Get the background template.
        """
        if self._grating_bkg is None:
            self.bkg = None
            self.bkg_unc = None
        else:
            self.bkg = self._grating_bkg.ndarray().mean(0)
            self.bkg_unc = self._grating_bkg.ndarray().std(0)

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
        bkg_unc = np.zeros_like(data).mean(0)
        if self.bkg is not None:
            data = data - self.bkg
            bkg_unc = self.bkg_unc
        else:
            self.bkg_unc = bkg_unc
            self.bkg = np.zeros_like(data).mean(0)
        self.calibration_unc = bkg_unc
        # skip offset and collect pulse data each pulse_period samples only
        self.calibration_data = data[:, self.offset::self.pulse_period, self.min_pixel:self.max_pixel]
        self.calibration_unc = self.calibration_unc[self.offset::self.pulse_period, self.min_pixel:self.max_pixel]
        # average over pulses
        self.calibration_data = np.mean(self.calibration_data, axis=1)
        self.calibration_unc = np.mean(self.calibration_unc, axis=0)
        self.calibration_mask = np.ones(self.calibration_data.shape[0], dtype=bool)

    def mask_calibration_point(self, energy: float, mask: bool=False, tol: float=1.0):
        """
        If `mask` is False, the point at a given energy is ignored when
        performing the fit.
        A call to `fit` must be redone after this.

        Args:
          energy: Energy value, in the same units as provided in `scan`.
          mask: If True, keep the point. If False, remove it.
          tol: Tolerance for energy matching.
        """
        self.calibration_mask[np.abs(energy - self.calibration_energies) < tol] = mask

    def fit(self):
        """Fit line."""
        from scipy.stats import linregress
        mask = self.calibration_mask
        sample = np.arange(self.calibration_data.shape[-1])
        sample_mode = np.argmax(self.calibration_data, axis=-1)
        #sample_mode = snp.sum(self.calibration_data*sample, axis=-1)/np.sum(self.calibration_data, axis=-1)
        res = linregress(sample_mode[mask], self.calibration_energies[mask])
        self.slope = res.slope
        self.e0 = res.intercept
        self.energy_axis = self.e0 + self.slope*sample

    def plot(self):
        """
        Plot fit.
        """
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 8))
        sample = np.arange(self.calibration_data.shape[-1])
        sample_mode = np.argmax(self.calibration_data, axis=-1)
        plt.plot(sample, self.energy_axis, lw=2, label="Fit")
        plt.xlabel("Pixel")
        plt.ylabel("Energy [eV]")
        plt.scatter(sample_mode, self.calibration_energies, s=200, marker='x', c='r', label="Data")
        plt.legend(frameon=False)
        plt.grid()
        plt.show()

    def apply(self, run: DataCollection, load_all: bool=True) -> xr.Dataset:
        """
        Apply calibration to a new analysis run.
        It is assumed it contains the same settings.

        Args:
          run: Input run.
          load_all: If True, load all data in memory at once. This is faste, but uses more memory.
                    Disable if not enough memory is available.
        """
        # do it per train to avoid memory overflow
        pulse_period = self.get_pulse_period(XrayPulses(run))
        if load_all:
            out_data = run[self.grating_source, self.grating_key].xarray()
            trainId = out_data.trainId.to_numpy()
            out_data = out_data.to_numpy() - self.bkg
            out_data = out_data[:, self.offset::pulse_period, self.min_pixel:self.max_pixel]
        else:
            trainId = list()
            out_data = list()
            for i, (tid, data) in enumerate(run.trains()):
                #print(f"Train {tid}, idx {i}")
                d = data[self.grating_source][self.grating_key]
                if self.bkg is not None:
                    d = d - self.bkg
                # skip offset and collect pulse data each pulse_period samples only
                d = d[self.offset::pulse_period, self.min_pixel:self.max_pixel]
                trainId += [tid]
                out_data += [d]
            out_data = np.stack(out_data, axis=0)
        energy = self.energy_axis
        out_data = xr.DataArray(data=out_data,
                            dims=('trainId', 'pulseIndex', 'energy'),
                            coords=dict(trainId=np.array(trainId),
                                        pulseIndex=np.arange(out_data.shape[1]),
                                        energy=energy
                                        )
                           )
        out_unc = xr.DataArray(data=self.calibration_unc, dims=('energy'),
                               coords=dict(energy=energy))
        return xr.Dataset(data_vars=dict(data=out_data, unc=out_unc))

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
                            "slope_motor",
                            "bkg",
                            "bkg_unc",
                            "angle",
                            "energy_axis",
                            "calibration_energies",
                            "calibration_data",
                            "calibration_unc",
                            "grating_source",
                            "grating_key",
                            "grating_motor_source",
                            "grating_motor_key",
                            "sources",
                            "i0", "i1", "j0", "j1",
                            "_version",
                           ]

    def setup(self,
              grating_signal: KeyData,
              scan: Scan,
              grating_bkg: Optional[KeyData]=None,
              grating_motor: Optional[KeyData]=None,
              ):
        """
        Setup calibration.

        Args:
          grating_signal: Where to read the grating data from.
                   Example: `signal_run["SQS_DIAG3_BIU/CAM/CAM_6:daqOutput, "data.image.pixels"]`
          scan: Scan object identfying where to read the undulator energy from.
                Example: `Scan(run["SA3_XTD10_MONO/MDL/PHOTON_ENERGY", "actualEnergy"])`
          grating_bkg: Where to read the grating background data from.
               Example: `bkg_run["SQS_DIAG3_BIU/CAM/CAM_6:daqOutput, "data.image.pixels"]`
          grating_motor: KeyData corresponding to the motor position.
               Example: `signal_run['SQS_DIAG3_SCAM/MOTOR/ST_AXIS_X', 'encoderPosition.value']`
        """
        self.grating_source = grating_signal.source
        self.grating_key = grating_signal.key
        self._grating_signal = grating_signal
        self._grating_bkg = grating_bkg

        self.sources = [
                        self.grating_source,
                       ]

        if grating_motor is not None:
            self.grating_motor_source = grating_motor.source
            self.grating_motor_key = grating_motor.key
            self._grating_motor = grating_motor
            self.sources += [self.grating_motor_source]
        else:
            self.grating_motor_source = ""
            self.grating_motor_key = ""
            self._grating_motor = None

        # create scan object
        self._scan = scan
        self.calibration_energies = self._scan.positions

        # outputs
        self.e0 = 0
        self.slope = 0
        self.slope_motor = 0
        self.energy_axis = None

        # estimate pixel positions for cropping
        self.estimate_crop_roi(*self._grating_signal.shape[-2:])

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
        if self._grating_bkg is None:
            self.bkg = None
            self.bkg_unc = None
        else:
            self.bkg = self._grating_bkg.ndarray().mean(0)
            self.bkg_unc = self._grating_bkg.ndarray().std(0)

    def estimate_crop_roi(self, nrows: int, ncols: int):
        """
        Calculate rectangle to be selected to crop the image and avoid
        edge effects.

        Args:
          nrows: Number of pixel rows.
          ncols: Number of pixel columns.

        """
        A = np.array([[np.sin(self.angle), np.cos(self.angle)],
                      [np.cos(self.angle), np.sin(self.angle)]])
        c, d = np.linalg.solve(A, np.array([nrows, ncols]))
        i0 = c*np.sin(2*self.angle)/2
        j0 = d*np.sin(2*self.angle)/2
        i1, j1 = i0 + d, j0 + c
        self.i0, self.i1 = int(i0), int(i1)
        self.j0, self.j1 = int(j0), int(j1)

    def load_data(self):
        """Load calibration data."""
        from scipy.ndimage import rotate
        fn = partial(calc_mean,
                     scan=self._scan,
                     grating=self._grating_signal,
                     )
        fn_motor = partial(calc_mean,
                     scan=self._scan,
                     grating=self._grating_motor,
                     )
        energy_ids = np.arange(len(self.calibration_energies))
        with ProcessPoolExecutor() as p:
            data = np.stack(list(p.map(fn, energy_ids)), axis=0)
            if self.grating_motor_source != "":
                data_motor = np.stack(list(p.map(fn_motor, energy_ids)), axis=0)
            else:
                data_motor = np.zeros((data.shape[0]))
        bkg_unc = np.zeros_like(data).mean(0)
        if self.bkg is not None:
            data = data - self.bkg
            bkg_unc = self.bkg_unc
        else:
            self.bkg_unc = bkg_unc
            self.bkg = np.zeros_like(data).mean(0)
        if self.angle != 0:
            data = self.crop(rotate(data, self.angle, axes=(-1, -2)))
            bkg_unc = self.crop(rotate(bkg_unc, self.angle, axes=(-1, -2)))
        self.calibration_data = data.mean(-2)
        self.calibration_unc = bkg_unc.mean(-2)
        self.calibration_motor = data_motor

    def crop(self, data: np.ndarray) -> np.ndarray:
        """
        Crop picture after rotation to avoid edges.

        Args:
          data: The input data.

        Returns: Cropped data.
        """

        return data[..., self.i0:self.i1, self.j0:self.j1]


    def fit(self):
        """Fit line."""
        from scipy.optimize import least_squares
        sample = np.arange(self.calibration_data.shape[-1])
        sample_mode = np.argmax(self.calibration_data, axis=-1)
        motor_position = self.calibration_motor
        #sample_mode = snp.sum(self.calibration_data*sample, axis=-1)/np.sum(self.calibration_data, axis=-1)
        fun = lambda par, x, y: par[0] + par[1]*x[0] + par[2]*x[1] - y
        par = np.array([np.amin(sample_mode), # e0
                        0.0,                  # slope
                        0.0])                 # slope_motor
        x = np.stack((sample_mode, motor_position), axis=0)
        #res = least_squares(fun, par, loss='soft_l1', f_scale=0.1, args=(x, self.calibration_energies))
        res = least_squares(fun, par, loss='linear', args=(x, self.calibration_energies))
        self.slope_motor = res.x[2]
        self.slope = res.x[1]
        self.e0 = res.x[0]
        self.energy_axis = self.e0 + self.slope*sample + self.slope_motor*motor_position.mean()

    def plot(self):
        """
        Plot fit.
        """
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 8))
        sample = np.arange(self.calibration_data.shape[-1])
        sample_mode = np.argmax(self.calibration_data, axis=-1)
        motor_position = self.calibration_motor
        plt.plot(sample, self.energy_axis, lw=2, label="Fit for mean motor position")
        plt.xlabel("Pixel")
        plt.ylabel("Energy [eV]")
        plt.scatter(sample_mode, self.e0 + self.slope*sample_mode + self.slope_motor*motor_position,
                    s=200, marker='o', facecolors='w', edgecolors="k", label="Prediction (w. motor)")
        plt.scatter(sample_mode, self.calibration_energies, s=200, marker='x', c='r', label="Data")
        plt.legend(frameon=False)
        plt.grid()
        plt.show()

    def apply(self, run: DataCollection, load_all: bool=True) -> xr.Dataset:
        """
        Apply calibration to a new analysis run.
        It is assumed it contains the same settings.

        Args:
          run: Input run.
          load_all: If True, load all data in memory at once. This is faster, but uses more memory.
                    Disable if not enugh memory is available.
        """
        from scipy.ndimage import rotate
        if self.grating_motor_source != "":
            out_data_motor = run[self.grating_motor_source, self.grating_motor_key].xarray()
            logging.info(f"Motor position being assumed fixed at {out_data_motor.mean()}. "
                         f"I detect a rms variation of {out_data_motor.std()}. "
                         f"This should be compatible with zero!")
            motor_position = out_data_motor.mean().to_numpy()
        else:
            motor_position = 0.0
        sample = np.arange(self.calibration_data.shape[-1])
        energy = self.e0 + self.slope*sample + self.slope_motor*motor_position
        if load_all:
            out_data = run[self.grating_source, self.grating_key].xarray()
            trainId = out_data.trainId.to_numpy()
            out_data = out_data.to_numpy() - self.bkg
            if self.angle != 0:
                out_data = self.crop(rotate(out_data, self.angle, axes=(-1, -2)))
            out_data = out_data.sum(-2)
        else:
            # do it per train to avoid memory overflow
            trainId = list()
            out_data = list()
            for i, (tid, data) in enumerate(run.trains()):
                #print(f"Train {tid}, idx {i}")
                d = data[self.grating_source][self.grating_key]
                if self.bkg is not None:
                    d = d - self.bkg
                if self.angle != 0:
                    d = self.crop(rotate(d, self.angle, axes=(-1, -2)))
                trainId += [tid]
                out_data += [d.sum(-2)]
            out_data = np.stack(out_data, axis=0)
        out_data = xr.DataArray(data=out_data,
                            dims=('trainId', 'energy'),
                            coords=dict(trainId=np.array(trainId),
                                        energy=energy
                                        )
                           )
        out_unc = xr.DataArray(data=self.calibration_unc, dims=('energy'),
                               coords=dict(energy=energy))
        return xr.Dataset(data_vars=dict(data=out_data, unc=out_unc))

