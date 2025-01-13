from typing import Optional, Union, Dict, List, Tuple, Any
from dataclasses import dataclass, asdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.colors import LogNorm
import matplotlib.colors as mcolors

from extra_data import open_run, by_id, DataCollection
import h5py
import xarray as xr
import pandas as pd

import itertools
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from extra.components import Scan, AdqRawChannel, XrayPulses, XGM

from lmfit.models import GaussianModel, LinearModel, ConstantModel
from lmfit.model import ModelResult
import scipy
from scipy.stats import linregress
from scipy.optimize import minimize_scalar
from functools import partial
from scipy.signal import fftconvolve
from scipy.interpolate import CubicSpline
from scipy.signal import kaiserord, filtfilt, firwin


@dataclass
class TofFitResult:
    """Class keeping track of fit in a single Tof."""
    energy: np.ndarray
    mu: np.ndarray
    sigma: np.ndarray
    A: np.ndarray
    Aa: np.ndarray
    offset: np.ndarray

def save_dict(h5grp, obj: Dict[str, Any]):
    """Save dictionary object into HDF5 group recursively."""
    for k, v in obj.items():
        if isinstance(v, dict):
            newgrp = h5grp.create_group(f"{k}")
            save_dict(newgrp, v)
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
    return dict(sorted(out.items()))

def log(level, msg):
    if level > 0:
        print(msg)

def search_offset(trace: np.ndarray, sigma: float=20):
    """
    Find highest peaks in the 1D trace.
    """
    axis = np.arange(len(trace))
    gaussian = np.exp(-0.5*(axis - np.mean(axis))**2/sigma**2)
    gaussian /= np.sum(gaussian, axis=0, keepdims=True)
    # apply it to the data
    smoothened = fftconvolve(trace, gaussian, mode="same", axes=0)
    peak_idx = np.argmax(smoothened)
    return peak_idx - 200

def search_roi(roi: np.ndarray):
    """
    Find highest peaks in the 1D trace.
    """
    p, _ = scipy.signal.find_peaks(roi, prominence=(0.25*np.amax(roi), None))
    return p

def model(ts,c,e0,t0):
    """
    Model to fit energy as a function of the sample axis.
    """
    # e = e0 + c/(ts - t0)**2
    # ts = np.sqrt(c/(e - e0)) + t0
    # dt/de = d/de [ sqrt(c/e) ] = d/du sqrt(u) d/de [c/e] = -0.5*c/sqrt(c/e) * 1/(e-e0)^2 
    # dt/de = -0.5*c/(np.sqrt(c/(e - e0)))/(e - e0)**2
    # before transmission, after time -> energy interpolation:
    # multiply peak sizes by dt/de
    return e0+c/(ts-t0)**2

# e = e0 + c/(ts-t0)^2
# 1/(e - e0)
def lin_fit(ts,energies,t0):
    """
    Fit model using linear regression for fixed t0.
    """
    res = linregress(1/np.asarray(ts-t0)**2,np.asarray(energies))
    c = res.slope
    e0 = res.intercept
    return c,e0,t0

def norm_diff_t0(ts,energies,t0):
    #print(ts,'fi',energies)
    c,e0,_ = lin_fit(ts,energies,t0)
    return np.linalg.norm(model(ts,c,e0,t0) - energies)

def fit(peak_ids,energies,t0_bounds):
    peak_ids = np.asarray(peak_ids)
    energies = np.asarray(energies)
    #print(peak_ids)
    #print(energies)

    f = partial(norm_diff_t0,peak_ids,energies)
    res=minimize_scalar(f,method='Bounded',bounds=t0_bounds)
    t0 = res.x
    #print(res)
    c,e0,t0=lin_fit(peak_ids,energies,t0)
    if res.success:
        return c,e0,t0
    else:
        raise Exception('fit did not converge.')


def calc_mean(itr: Tuple[int, int], scan: Scan, xgm_data: xr.DataArray, tof: Dict[int, AdqRawChannel], xgm_threshold: float, filter_length: int) -> xr.DataArray:
    """A function doing the actual calibration, per tof, per energy index."""
    tof_id, energy_id = itr
    energy, train_ids = scan.steps[energy_id]
    #xgm = XGM(run.select_trains(by_id[list(train_ids)]), "SA3_XTD10_XGM/XGM/DOOCS")
    #sel_xgm_data = xgm.pulse_energy().stack(pulse=('trainId', 'pulseIndex'))
    mask = xgm_data.coords["trainId"].isin(list(train_ids))
    sel_xgm_data = xgm_data[mask]
    tof_data = tof[tof_id].select_trains(by_id[list(train_ids)]).pulse_data(pulse_dim='pulseIndex')
    # select XGM
    tof_data = tof_data.loc[sel_xgm_data > xgm_threshold, :]
    tof_xgm_data = sel_xgm_data.loc[sel_xgm_data > xgm_threshold]
    tof_data = tof_data.to_numpy()
    tof_xgm_data = tof_xgm_data.to_numpy()
    out_data = -tof_data.mean(0)
    out_xgm = tof_xgm_data.mean(0)

    # apply filter
    filter_length = filter_length[tof_id]
    if filter_length > 0:
        filtered = apply_filter(out_data, filter_length)
    else:
        filtered = out_data
    return filtered, out_xgm

def apply_filter(data: np.ndarray, filter_length: int) -> np.ndarray:
    """
    Apply a low-pass Kaiser filter on data along its last axis with frequency 1/(2*filter_length) and width 1/(4*filter_length).

    Args:
      data: Data with shape (n_samples, n_energy). Filter is applied on last axis.
      filter_length: Number of samples to consider as fluctuatios to filter out.
    Returns: Filtered data in the same shape as input.
    """
    nyq_rate = 0.5
    width = 1.0/(0.5*filter_length)/0.5
    ripple_db = 20.0
    N, beta = kaiserord(ripple_db, width)
    cutoff = 1.0/(filter_length)/nyq_rate
    taps = firwin(N, cutoff, window=('kaiser', beta))
    return filtfilt(taps, 1.0, data, axis=-1)

class CookieboxCalib(object):
    """
    Calibrate a set of eTOFs read out using an ADQ digitizer device.

    eTOFs provide a 1D trace for all pulses, which can be transformed
    into traces per pulse by the `AdqRawChannel` Extra component.
    However, the trace sample axis is meaningless and needs to be converted
    into energy using a non-linear function. Usually one data analysis run
    is taken in which the undulator energy is scanned to obtain this map,
    which is then applied in the actual analysis run.

    The objective is to take a calibration run (where an energy scan is made)
    and from that run calculate a) the map to be used to interpolate eTOF
    energy axis from "sample number" into eV; and b) calculate the transmission
    correction, which corrects the appropriate intensity of the data,
    given that the eTOF does not measure electrons with equal probability in all energy ranges.

    The concrete steps taken when the object is initialized
    (`obj = CookieboxCalib(calib_run)`) from a calibration run are:
    - From a calibration run, derive a sample number (proportional to time-of-flight)
      to energy non-linear map.
    - Estimate the transformation Jacobian and use it to correct the
      spectrum intensity distribution, so that the probability
      of photo-electrons observed in a range of energies agrees with the
      probability in the corresponding time-of-flight range.
    - Calculate a transmission correction due to the eTOF quantum
      efficiency as "photo-electron integral/Auger+Valence integral".
    - Calculate a normalization correction due to the pulse energy
       as taken from the XGM: "Auger+Valence integral/XGM pulse energy".

    The concrete steps taken when the `obj.apply(other_run)` is called with a run to be calibrated are:
    - Use derived non-linear map to interpolate energy axis per eTOF
    - Subtract per energy offset per eTOF.
    - Scale data by the inverse of the absolute value of the Jacobian per eTOF,
      following the change-of-variable theorem in statistics.
    - Divide by the transmission per eTOF.
    - Divide by the normalization correction.

    The variables calculated can be visualized using `obj.plot_diagnostics()`
    and similar other functions for validation and other cross-checks.

    Arguments:
      run: The calibration run.
      energy_axis: Energy axis in eV to interpolate eTOF to.
      tof_settings: Dictionary with a TOF label as a key (0,1,2, ...).
                    Each value is *either* a) a tuple containing the
                    eTOF source name and channel in the format "1_A";
                    or b) the AdqRawChannel object for that eTOF.
      xgm_threshold: Minimum threshold to ignore dark frames in
                     the calibration data (in uJ).
                     Can be 'median' to use the median over the run.
      first_pulse_offset: Offset to find the first pulse data in digitizers.
                          Use `None` to guess it.
      single_pulse_length: In case of a single pulse, what is the length of the trace to keep.
      auger_start_roi: Start of the Auger and valence RoI in a pulse,
                       relative to the `first_pulse_offset`. Use `None` to guess it.
      start_roi: Start of the RoI in a pulse, relative to the `first_pulse_offset`.
                 Use `None` to guess it.
      stop_roi: End of the RoI, relative to the `first_pulse_offset`. Use `None` to guess it.
      energy_source: Where to read the undulator energy from.
      xgm_source: Where to read the XGM intensity from.
      interleaved: Whether channels are interleaved. If `None`,
                   attempt to auto-detect, but this fails for a union of runs.
      log_level: Whether to produce log output. Set to 1 for more log output.
      filter_length: Number of digital samples from eTOFs to use for the
                     inverse digital frequency. Set to zero to avoid applying it.
                     This is useful to filter ringing.
    """
    def __init__(self, run: DataCollection,
                 energy_axis: np.ndarray,
                 tof_settings: Dict[int, Union[Tuple[str, str], AdqRawChannel]],
                 xgm_threshold: Union[str, float]='median',
                 first_pulse_offset: Optional[int]=None,
                 single_pulse_length: int=400,
                 auger_start_roi: Optional[int]=None,
                 start_roi: Optional[int]=None,
                 stop_roi: Optional[int]=None,
                 energy_source: str='SA3_XTD10_UND/DOOCS/PHOTON_ENERGY',
                 xgm_source: str="SA3_XTD10_XGM/XGM/DOOCS",
                 interleaved: Optional[bool]=None,
                 log_level: Optional[int]=0,
                 filter_length: Union[int, Dict[int, int]]=0,
                ):
        # if no run is set, we are just reading from a fil, so trust
        # `load` to set everything up
        if run is None:
            return
        # only for log output
        self.log_level = log_level

        # base properties
        self._run = run
        self._energy_axis = energy_axis
        self._tof_settings = tof_settings
        self._auger_start_roi = {tof_id: auger_start_roi for tof_id in self._tof_settings.keys()}
        self._start_roi = {tof_id: start_roi for tof_id in self._tof_settings.keys()}
        self._stop_roi = {tof_id: stop_roi for tof_id in self._tof_settings.keys()}
        self._energy_source = energy_source
        self._first_pulse_offset = first_pulse_offset
        self._single_pulse_length = single_pulse_length
        self._interleaved = interleaved
        self._xgm_source = xgm_source
        self._xgm_threshold = xgm_threshold
        self._filter_length = filter_length
        if not isinstance(filter_length, dict):
            self._filter_length = {tof_id: filter_length for tof_id in self._tof_settings.keys()}

        # empty outputs
        self.tof_fit_result = dict()
        self.model_params = dict()
        self.jacobian = dict()
        self.offset = dict()
        self.normalization = dict()
        self.transmission = dict()
        self.int_transmission = dict()

        # now do the full analysis, step by step

        # get XGM and auxiliary data
        self.update_xgm_and_metadata()

        # update eTOF data reading objects
        self.update_tof_settings()

        # find RoI if needed
        self.update_roi()

        # find where the peaks are per energy in each Tof
        self.update_fit_result()

        # calibrate and calculate transmission
        self.update_calibration()

        # now we can use the apply method
        log(self.log_level, "Ready to apply energy calibration and transmission correction to analysis data ...")

    # these setters allow one to update
    # some of the input properties and rerun only the necessary steps
    # for recalibration
    #
    @property
    def run(self):
        return self._run

    @run.setter
    def run(self, value):
        self._run = value
        self.update_xgm_and_metadata()
        self.update_tof_settings()
        self.update_roi()
        self.update_fit_result()
        self.update_calibration()

    @property
    def first_pulse_offset(self):
        return self._first_pulse_offset

    @first_pulse_offset.setter
    def first_pulse_offset(self, value):
        self._first_pulse_offset = value
        self.update_tof_settings()
        self.update_roi()
        self.update_fit_result()
        self.update_calibration()

    @property
    def single_pulse_length(self):
        return self._single_pulse_length

    @single_pulse_length.setter
    def single_pulse_length(self, value):
        self._single_pulse_length = value
        self.update_tof_settings()
        self.update_roi()
        self.update_fit_result()
        self.update_calibration()

    @property
    def interleaved(self):
        return self._interleaved

    @interleaved.setter
    def interleaved(self, value):
        self._interleaved = value
        self.update_tof_settings()
        self.update_roi()
        self.update_fit_result()
        self.update_calibration()

    @property
    def xgm_source(self):
        return self._xgm_source

    @xgm_source.setter
    def xgm_source(self, value):
        self._xgm_source = value
        self.update_xgm_and_metadata()
        self.update_tof_settings()
        self.update_roi()
        self.update_fit_result()
        self.update_calibration()

    @property
    def energy_source(self):
        return self._energy_source

    @energy_source.setter
    def energy_source(self, value):
        self._energy_source = value
        self.update_xgm_and_metadata()
        self.update_tof_settings()
        self.update_roi()
        self.update_fit_result()
        self.update_calibration()

    @property
    def energy_axis(self):
        return self._energy_axis

    @energy_axis.setter
    def energy_axis(self, value):
        self._energy_axis = value
        self.update_calibration()

    @property
    def xgm_threshold(self):
        return self._xgm_threshold

    @xgm_threshold.setter
    def xgm_threshold(self, value):
        self._xgm_threshold = value
        # find median if needed
        if self._xgm_threshold == 'median':
            self._xgm_threshold = np.median(self._xgm_data.to_numpy())
        self.update_roi()
        self.update_fit_result()
        self.update_calibration()

    @property
    def tof_settings(self):
        return self._tof_settings

    @tof_settings.setter
    def tof_settings(self, value):
        self._tof_settings = value
        self.update_tof_settings()
        self.update_roi()
        self.update_fit_result()
        self.update_calibration()

    @property
    def filter_length(self):
        return self._filter_length

    @filter_length.setter
    def filter_length(self, value):
        self._filter_length = value
        if not isinstance(value, dict):
            self._filter_length = {tof_id: value for tof_id in self._tof_settings.keys()}
        self.update_roi()
        self.update_fit_result()
        self.update_calibration()

    @property
    def auger_start_roi(self):
        return self._auger_start_roi

    @auger_start_roi.setter
    def auger_start_roi(self, value):
        self._auger_start_roi = {tof_id: value for tof_id in self._tof_settings.keys()}
        self.update_roi()
        self.update_fit_result()
        self.update_calibration()

    @property
    def start_roi(self):
        return self._start_roi

    @start_roi.setter
    def start_roi(self, value):
        self._start_roi = {tof_id: value for tof_id in self._tof_settings.keys()}
        self.update_roi()
        self.update_fit_result()
        self.update_calibration()

    @property
    def stop_roi(self):
        return self._stop_roi

    @stop_roi.setter
    def stop_roi(self, value):
        self._stop_roi = {tof_id: value for tof_id in self._tof_settings.keys()}
        self.update_roi()
        self.update_fit_result()
        self.update_calibration()

    def update_xgm_and_metadata(self):
        """
        Read calibration XGM and metadata information.
        """
        # set up helper objects
        # pulse structure
        self._pulses = XrayPulses(self.run)

        # depending of using DOOCS or MDL device, guess the energy key
        sd = self.run[self.energy_source]
        if 'actualEnergy' in sd:
            self.energy_key = 'actualEnergy'
        elif 'actualPosition' in sd:
            self.energy_key = 'actualPosition'
        else:
            raise ValueError('Unknown energy source.')

        # keep track of all needed sources and match them
        self.all_digi = list(set([item[0] for item in self._tof_settings.values()]))
        self.all_digi_control = [d.replace(":network", "") for d in self.all_digi]
        self.sources = [self.energy_source,
                        self.xgm_source,
                        f"{self.xgm_source}:output",
                        self._pulses.source.source,
                        *self.all_digi,
                        *self.all_digi_control,
                       ]
        # select data from the run
        self._run = self.run.select(self.sources, require_all=True)

        # get XGM information
        self._xgm = XGM(self._run, self.xgm_source)
        self._xgm_data = self._xgm.pulse_energy().stack(pulse=('trainId', 'pulseIndex'))
        # find median if needed
        if self._xgm_threshold == 'median':
            self._xgm_threshold = np.median(self._xgm_data.to_numpy())

        # recreate pulses object after selection:
        self._pulses = XrayPulses(self._run)

        # create scan object
        self._scan = Scan(self._run[self.energy_source, self.energy_key])
        self.calibration_energies = self._scan.positions*1e3

    def update_tof_settings(self):
        """
        Update position of the first pulse offset if needed and
        create AdqRawChannel.
        """
        # find first peak offset
        if self.first_pulse_offset is None:
            log(self.log_level, "First pulse offset not given: guessing it from data.")
            log(self.log_level, "(This may fail, if it does, please provide a `first_pulse_offset` instead.)")
            self.first_pulse_offset = self.find_offset()
            log(self.log_level, f"Found first pulse offset at {first_pulse_offset}")

        # create tof objects:
        self._tof = dict()
        for tof_id, value in self._tof_settings.items():
            if not isinstance(value, AdqRawChannel):
                digitizer, channel = value
                value = AdqRawChannel(self._run,
                                      channel,
                                      digitizer=digitizer,
                                      first_pulse_offset=self.first_pulse_offset,
                                      single_pulse_length=self.single_pulse_length,
                                      interleaved=self.interleaved,
                                      )
            self._tof[tof_id] = value
        self.mask = {tof_id: True for tof_id in self._tof_settings.keys()}

    def update_roi(self):
        """
        Given calibrated data, apply a selection and find RoI if needed.
        """
        # average data for each energy slice
        log(self.log_level, "Reading calibration data ... (this takes a while)")
        self.select_calibration_data()
        # find RoI if needed
        if (self.auger_start_roi is None
            or self.start_roi is None
            or self.stop_roi is None):
            log(self.log_level, "Finding RoI ...")
            log(self.log_level, "(This may fail. If it does, please provide a `auger_start_roi`, `start_roi` and `stop_roi`.)")
            for tof_id in self._tof_settings.keys():
                self.find_roi(tof_id)
            log(self.log_level, f"Auger start RoIs found: {self.auger_start_roi}")
            log(self.log_level, f"Start RoIs found: {self.start_roi}")
            log(self.log_level, f"Stop RoIs found: {self.stop_roi}")

    def update_fit_result(self):
        """
        Fit TOF data to a Gaussian and collect results.
        """
        log(self.log_level, "Fit peaks to obtain energy calibration ...")
        for tof_id in self._tof_settings.keys():
            log(self.log_level, f"Fitting eTOF {tof_id} ...")
            self.tof_fit_result[tof_id] = self.peak_tof(tof_id)

    def update_calibration(self):
        """
        Calculate calibration maps and transmission.
        """
        # calculate transmission (fills the arrays above)
        log(self.log_level, "Calculate calibration and transmission ...")
        for tof_id in self._tof_settings.keys():
            self.calculate_calibration_and_transmission(tof_id)

        # summarize it for later
        n_tof = len(self._tof_settings.keys())
        n_e = len(self.energy_axis)
        self.e_transmission = np.zeros((n_e, n_tof), dtype=np.float32)
        for idx, tof_id in enumerate(self._tof_settings.keys()):
            self.e_transmission[:,idx] = self.int_transmission[tof_id]

    def save(self, filename: str):
        """
        Dump all data needed for applying the calibration into an h5 file.
        """
        with h5py.File(filename, "w") as fid:
            # energy axis
            fid["energy_axis"] = self.energy_axis
            # keep tof settings as a set of attributes with keys corresponding to tofs
            # the values are tuples with the tof digitizer source and channel name
            save_dict(fid.create_group("tof_settings"), {k: list(v) for k, v in self._tof_settings.items()})
            save_dict(fid.create_group("auger_start_roi"), self.auger_start_roi)
            save_dict(fid.create_group("start_roi"), self.start_roi)
            save_dict(fid.create_group("stop_roi"), self.stop_roi)
            save_dict(fid.create_group("model_params"), self.model_params)
            save_dict(fid.create_group("transmission"), self.transmission)
            save_dict(fid.create_group("int_transmission"), self.int_transmission)
            tof_dict = {k: dict(asdict(v)) for k, v in self.tof_fit_result.items()}
            save_dict(fid.create_group("tof_fit_result"), tof_dict)
            save_dict(fid.create_group("calibration_data"), self.calibration_data)
            save_dict(fid.create_group("calibration_mean_xgm"), self.calibration_mean_xgm)

            save_dict(fid.create_group("mask"), self.mask)
            save_dict(fid.create_group("offset"), self.offset)
            save_dict(fid.create_group("jacobian"), self.jacobian)
            save_dict(fid.create_group("normalization"), self.normalization)
            save_dict(fid.create_group("filter_length"), self._filter_length)


            # energy source and key (not really needed to apply it though)
            fid.attrs["energy_source"] = self.energy_source
            fid.attrs["energy_key"] = self.energy_key
            fid.attrs["xgm_source"] = self.xgm_source
            fid.attrs["sources"] = self.sources

            fid.attrs["first_pulse_offset"] = self.first_pulse_offset
            fid.attrs["single_pulse_length"] = self.single_pulse_length

            fid.attrs["log_level"] = self.log_level
            if self.interleaved is not None:
                fid.attrs["interleaved"] = self.interleaved

            fid["xgm_threshold"] = self._xgm_threshold
            fid["e_transmission"] = self.e_transmission
            fid["tof_ids"] = np.array(list(self._tof_settings.keys()))
            fid["calibration_energies"] = self.calibration_energies


    @classmethod
    def load(cls, filename: str):
        """
        Load setup saved with save previously.
        """
        obj = CookieboxCalib(None, None, None)
        with h5py.File(filename, "r") as fid:
            if "log_level" in fid.attrs:
                obj.log_level = fid.attrs["log_level"]
            else:
                obj.log_level = 1

            obj._energy_axis = fid["energy_axis"][()]
            obj._tof_settings = load_dict(fid["tof_settings"])
            obj._tof_settings = {k: (v[0].decode("utf-8"), v[1].decode("utf-8")) for k, v in obj._tof_settings.items()}
            obj._auger_start_roi = load_dict(fid["auger_start_roi"])
            obj._start_roi = load_dict(fid["start_roi"])
            obj._stop_roi = load_dict(fid["stop_roi"])
            obj._tof_fit_result = {k: TofFitResult(**v) for k, v in load_dict(fid["tof_fit_result"]).items()}
            obj._filter_length = load_dict(fid["filter_length"])
            obj._energy_source = fid.attrs["energy_source"]
            obj._xgm_source = fid.attrs["xgm_source"]
            obj._xgm_threshold = fid["xgm_threshold"][()]
            obj._first_pulse_offset = fid.attrs["first_pulse_offset"]
            obj._single_pulse_length = fid.attrs["single_pulse_length"]
            if "interleaved" in fid.attrs:
                obj._interleaved = fid.attrs["interleaved"]
            else:
                obj._interleaved = None

            # outputs
            obj.model_params = load_dict(fid["model_params"])
            obj.transmission = load_dict(fid["transmission"])
            obj.int_transmission = load_dict(fid["int_transmission"])
            obj.mask = load_dict(fid["mask"])
            obj.offset = load_dict(fid["offset"])
            obj.jacobian = load_dict(fid["jacobian"])
            obj.normalization = load_dict(fid["normalization"])

            obj.energy_key = fid.attrs["energy_key"]
            obj.sources = fid.attrs["sources"]

            obj.e_transmission = fid["e_transmission"][()]
            obj.calibration_data = load_dict(fid["calibration_data"])
            obj.calibration_mean_xgm = load_dict(fid["calibration_mean_xgm"])
            obj.calibration_energies = fid["calibration_energies"][()]
        return obj

    def select_calibration_data(self):
        """
        Select data for calibration.
        """
        tof_ids = list(self._tof.keys())
        energy_ids = np.arange(len(self.calibration_energies))
        data = {tof_id: list() for tof_id in tof_ids}
        mean_xgm = {tof_id: list() for tof_id in tof_ids}
        fn = partial(calc_mean,
                     scan=self._scan,
                     xgm_data=self._xgm_data,
                     tof=self._tof,
                     xgm_threshold=self._xgm_threshold,
                     filter_length=self._filter_length
                     )


        with ProcessPoolExecutor() as p:
            itr_gen = list(itertools.product(tof_ids, energy_ids))
            data_gen = p.map(fn, itr_gen)
            # organize it all in a numpy array
            for (d, x), (tof_id, energy_id) in zip(data_gen, itr_gen):
                data[tof_id] += [d]
                mean_xgm[tof_id] += [x]
            for tof_id in tof_ids:
                data[tof_id] = np.stack(data[tof_id], axis=0)
                mean_xgm[tof_id] = np.stack(mean_xgm[tof_id], axis=0)

        self.calibration_data = data
        self.calibration_mean_xgm = mean_xgm


    def find_offset(self, n_trains: int=1000, n_samples: int=2000) -> int:
        """
        Find global offset.

        Args:
          n_trains: Number of trains to use for speed.
          n_samples: Number of samples to use for samples.

        Returns: the offset.
        """
        tof = {tof_id: AdqRawChannel(self._run,
                                      channel,
                                      digitizer=digitizer,
                                      first_pulse_offset=0,
                                      interleaved=self.interleaved,
                                      )
                     for tof_id, (digitizer, channel) in self._tof_settings.items()}
        all_trace = xr.concat([v.select_trains(np.s_[:n_trains]).train_data(roi=np.s_[:n_samples]) for v in tof.values()],
                              pd.Index(list(tof.keys()), name="tof"))
        data = -all_trace.mean(["tof", "trainId"])
        peaks = search_offset(data)
        # smooth = scipy.ndimage.gaussian_filter1d(data, sigma=50)
        # thr = np.median(smooth)
        # peaks, _ = scipy.signal.find_peaks(smooth, height=thr)
        # if len(peaks) == 0:
        #     raise RuntimeError(f"Unable to find peaks in averaged data.")
        return max(peaks, 0)

    def find_roi(self, tof_id: int):
        """
        Find RoI for the photo-electron peak from calibration data.
        """
        auger = list()
        roi = list()
        for e in range(self.calibration_data[tof_id].shape[0]):
            d = self.calibration_data[tof_id][e, :]
            # idx = np.arange(len(d))
            # D = np.stack((idx, d), axis=0)
            # km = KMeans(n_clusters=2).fit(D)
            peaks = search_roi(d)
            peaks = sorted(peaks)
            if len(peaks) < 2:
                log(self.log_level, f"Failed to find peaks for eTOF {tof_id}, energy index {e}. Check the data quality.")
                continue
            auger += [peaks[0]]
            roi += [peaks[1]]
        if len(auger) == 0 or len(roi) == 0:
            log(self.log_level, f"No peaks found in eTOF {tof_id}. Check the data quality. I will set the RoI to collect non-sense, so this TOF data will be meaningless. It will also be masked.")
            self.auger_start_roi[tof_id] = 0
            self.start_roi[tof_id] = 100
            self.stop_roi[tof_id] = 200
            self.mask[tof_id] = False
            return
        a = min(auger)
        b = min(roi)
        dab = abs(b - a)
        stop_roi = max(roi) + int(dab/2)
        auger_start_roi = int(a - dab/2)
        start_roi = int(b - dab/2)
        self.auger_start_roi[tof_id] = auger_start_roi
        self.start_roi[tof_id] = start_roi
        self.stop_roi[tof_id] = stop_roi

    def plot_calibration_data(self):
        """Plot data for checks.
        """
        data = self.calibration_data
        tof_ids = list(self._tof_settings.keys())
        energies = self.calibration_energies
        n_energies = len(energies)
        fig, axes = plt.subplots(nrows=4, ncols=4, clear=True, figsize=(20,20))
        for tof_id in tof_ids:
            auger_start_roi = self.auger_start_roi[tof_id]
            start_roi = self.start_roi[tof_id]
            stop_roi = self.stop_roi[tof_id]
            ax = axes.flatten()[tof_id]
            temp = data[tof_id][:,auger_start_roi:stop_roi].copy()
            ax.imshow(temp,
                      aspect=(stop_roi - auger_start_roi)/n_energies,
                      norm=Normalize(vmax=np.amax(temp))
                     )
            ax.set_yticklabels((energies+0.5).astype(int))
            ax.set_title(f'TOF {tof_id}')
        plt.show()

    def peak_tof(self, tof_id: int) -> TofFitResult:
        """
        For each energy in tof `tof_id`, fit a Gaussian and store its parameters.

        Returns: Energy vector used, means and std. dev. in the sample axis scale, and integral of Gaussian.
        """
        energies = self.calibration_energies
        data = self.calibration_data[tof_id]
        mu = list()
        sigma = list()
        A = list()
        Aa = list()
        energy = list()
        offset = list()
        auger_start_roi = self.auger_start_roi[tof_id]
        start_roi = self.start_roi[tof_id]
        stop_roi = self.stop_roi[tof_id]
        for e in range(data.shape[0]):
            # fit Auger
            xa = np.arange(auger_start_roi, start_roi)
            ya = data[e, auger_start_roi:start_roi]
            gamodel = GaussianModel() + ConstantModel()
            ii = np.argmax(ya)
            resulta = gamodel.fit(ya, x=xa, center=auger_start_roi+ii, amplitude=ya[ii], sigma=2, c=np.median(ya))
            # fit data
            x = np.arange(start_roi, stop_roi)
            y = data[e, start_roi:stop_roi]
            gmodel = GaussianModel() + ConstantModel()
            ii = np.argmax(y)
            result = gmodel.fit(y, x=x, center=start_roi+ii, amplitude=y[ii], sigma=10, c=np.median(y))
            #result.plot_fit()
            #plt.show()
            # we care about the normalization coefficient, not the normalized amplitude
            #A += [result.best_values["amplitude"]/(result.best_values["sigma"]*np.sqrt(2*np.pi))]
            A += [result.best_values["amplitude"]]
            Aa += [resulta.best_values["amplitude"]]
            mu += [result.best_values["center"]]
            sigma += [result.best_values["sigma"]]
            energy += [energies[e]]
            offset += [result.best_values["c"]]
        energy = np.array(energy)
        mu = np.array(mu)
        sigma = np.array(sigma)
        A = np.array(A)
        Aa = np.array(Aa)
        offset = np.array(offset)
        return TofFitResult(energy, mu, sigma, A, Aa, offset)

    def calculate_calibration_and_transmission(self, tof_id: int):
        """
        Calculate transmissions.
        """
        energy_ids = np.arange(len(self._scan.positions))
        # fit calibration
        c, e0, t0 = fit(self.tof_fit_result[tof_id].mu, self.tof_fit_result[tof_id].energy, t0_bounds=[-500, 500])
        self.model_params[tof_id] = np.array([c, e0, t0], dtype=np.float64)
        self.jacobian[tof_id] = 0.5*c/(np.sqrt(c/(self.energy_axis - e0)))/(self.energy_axis - e0)**2

        # interpolate offset
        eidx = np.argsort(self.tof_fit_result[tof_id].energy)
        ee = self.tof_fit_result[tof_id].energy[eidx]
        eo = self.tof_fit_result[tof_id].offset[eidx]

        self.offset[tof_id] = CubicSpline(ee, eo)(self.energy_axis)

        # self.offset[tof_id] = np.interp(self.energy_axis,
        #                                 ee,
        #                                 eo, left=0, right=0)
        # interpolate amplitude as given by the
        # Auger+Valence (related to the cross section and pulse intensity)
        # normalized by the XGM mean intensity
        en = self.tof_fit_result[tof_id].Aa[eidx]/self.calibration_mean_xgm[tof_id][eidx]
        # self.normalization[tof_id] = np.interp(self.energy_axis,
        #                                        ee,
        #                                        en, left=0, right=0)
        self.normalization[tof_id] = CubicSpline(ee, en)(self.energy_axis)

        # calculate transmission
        self.transmission[tof_id] = self.tof_fit_result[tof_id].A/self.tof_fit_result[tof_id].Aa

        # interpolate transmission for the given energy axis
        et = self.transmission[tof_id][eidx]
        # self.int_transmission[tof_id] = np.interp(self.energy_axis,
        #                                           ee,
        #                                           et, left=0, right=0)
        self.int_transmission[tof_id] = CubicSpline(ee, et)(self.energy_axis)

    def plot_diagnostics(self, tof_id: int):
        """
        Diagnostics plots for finding the energy peaks in a scan.
        """
        auger_start_roi = self.auger_start_roi[tof_id]
        start_roi = self.start_roi[tof_id]
        stop_roi = self.stop_roi[tof_id]
        ts = np.arange(start_roi, stop_roi)
        e = model(ts, *self.model_params[tof_id])
        fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(20,25))
        ax = ax.flatten()
        ax[0].scatter(self.tof_fit_result[tof_id].mu, self.tof_fit_result[tof_id].energy, label="Peak center")
        ax[0].plot(ts, e, lw=2, label="Model fit")
        ax[0].set(xlabel="Samples", ylabel="Energy [eV]")
        ax[1].scatter(self.tof_fit_result[tof_id].energy, self.transmission[tof_id], label="Transmission")
        ax[1].plot(self.energy_axis, self.int_transmission[tof_id], lw=2, label="Interpolated transmission")
        ax[1].set(xlabel="Energy [eV]", ylabel="Transmission [a.u.]")
        ax[2].scatter(self.tof_fit_result[tof_id].energy, self.tof_fit_result[tof_id].offset, label="Offset")
        ax[2].plot(self.energy_axis, self.offset[tof_id], lw=2, label="Interpolated offset")
        ax[2].set(xlabel="Energy [eV]", ylabel="Offset to subtract [a.u.]")
        ax[3].scatter(self.tof_fit_result[tof_id].energy, self.tof_fit_result[tof_id].Aa/self.calibration_mean_xgm[tof_id], label="Normalization")
        ax[3].plot(self.energy_axis, self.normalization[tof_id], lw=2, label="Interpolated normalization")
        ax[3].set(xlabel="Energy [eV]", ylabel="(Auger+valence)/pulse energy [a.u.]")
        ax[4].plot(self.energy_axis, self.jacobian[tof_id], lw=2, label="Interpolated Jacobian")
        ax[4].set(xlabel="Energy [eV]", ylabel="Jacobian [a.u.]")
        for a in ax:
            a.set_title(f"TOF {tof_id}")
        plt.show()

    def plot_all_transmissions(self, tof_ids: Optional[List[int]]=None):
        """
        Plot all transmissions in the same plot.
        """
        fig = plt.figure(figsize=(20, 10))
        for tof_id, transmission in self.int_transmission.items():
            if tof_ids is not None:
                if tof_id not in tof_ids:
                    continue
            plt.plot(self.energy_axis, transmission, lw=2, label=f"eTOF {tof_id}")
        plt.xlabel("Energy [eV]")
        plt.ylabel("Transmission [a.u.]")
        plt.legend()
        plt.show()

    def plot_all_offsets(self, tof_ids: Optional[List[int]]=None):
        """
        Plot all offset in the same plot.
        """
        fig = plt.figure(figsize=(20, 10))
        for tof_id, offset in self.offset.items():
            if tof_ids is not None:
                if tof_id not in tof_ids:
                    continue
            plt.plot(self.energy_axis, offset, lw=2, label=f"eTOF {tof_id}")
        plt.xlabel("Energy [eV]")
        plt.ylabel("Offset [a.u.]")
        plt.legend()
        plt.show()

    def plot_all_jacobians(self, tof_ids: Optional[List[int]]=None):
        """
        Plot all offset in the same plot.
        """
        fig = plt.figure(figsize=(20, 10))
        for tof_id, jacobian in self.jacobian.items():
            if tof_ids is not None:
                if tof_id not in tof_ids:
                    continue
            plt.plot(self.energy_axis, jacobian, lw=2, label=f"eTOF {tof_id}")
        plt.xlabel("Energy [eV]")
        plt.ylabel("Jacobian [1/eV]")
        plt.legend()
        plt.show()

    def plot_all_normalizations(self, tof_ids: Optional[List[int]]=None):
        """
        Plot all normalizations in the same plot.
        """
        fig = plt.figure(figsize=(20, 10))
        for tof_id, normalization in self.normalization.items():
            if tof_ids is not None:
                if tof_id not in tof_ids:
                    continue
            plt.plot(self.energy_axis, normalization, lw=2, label=f"eTOF {tof_id}")
        plt.xlabel("Energy [eV]")
        plt.ylabel("Normalization [a.u.]")
        plt.legend()
        plt.show()

    def plot_fit(self, tof_id: int):
        """
        Diagnostics plots for fit.
        """
        data = self.calibration_data[tof_id]
        auger_start_roi = self.auger_start_roi[tof_id]
        start_roi = self.start_roi[tof_id]
        stop_roi = self.stop_roi[tof_id]
        ts = np.arange(start_roi, stop_roi)
        fig = plt.figure(figsize=(20,10))
        for e, c in zip(range(data.shape[0]), mcolors.CSS4_COLORS.keys()):
            energy = self.calibration_energies[e]
            plt.scatter(ts, data[e,start_roi:stop_roi], c=c, label=f"{energy:.1f} eV")
            mu = self.tof_fit_result[tof_id].mu[e]
            amplitude = self.tof_fit_result[tof_id].A[e]
            #Aa = self.tof_fit_result[tof_id].Aa[e]
            sigma = self.tof_fit_result[tof_id].sigma[e]
            offset = self.tof_fit_result[tof_id].offset[e]
            gmodel = GaussianModel() + ConstantModel()
            gresult = ModelResult(gmodel, gmodel.make_params(center=mu, amplitude=amplitude, sigma=sigma, c=offset))
            plt.plot(ts, gresult.eval(x=ts), c=c, lw=2)
        plt.xlabel("Samples")
        plt.ylabel("Intensty [a.u.]")
        plt.legend()
        plt.title(f"TOF {tof_id}")
        plt.show()

    def apply(self, run: DataCollection) -> xr.Dataset:
        """
        Apply calibration, offset correction and transmission correction to a new analysis run.
        It is assumed it contains the same eTOF settings.
        """

        tof_ids = sorted([idx for idx, tof_id in enumerate(self._tof_settings.keys()) if self.mask[tof_id]])
        n_tof = len(tof_ids)
        n_e = len(self.energy_axis)
        n_trains = len(run.train_ids)
        n_pulses = XrayPulses(run).pulse_counts().max()
        norm = np.stack([v for k, v in self.normalization.items() if k in tof_ids], axis=-1)
        norm *= self.e_transmission[:, tof_ids]
        norm = xr.DataArray(data=norm,
                            dims=('energy', 'tof'),
                            coords=dict(energy=self.energy_axis,
                                        tof=tof_ids))

        def apply_correction(tof_id):
            """
            Apply the energy calibration and transmission correction for a given eTOF.
            """
            log(self.log_level, f"Correcting eTOF {tof_id} ...")
            # the sample axis to use for the calibration
            auger_start_roi = self.auger_start_roi[tof_id]
            start_roi = self.start_roi[tof_id]
            stop_roi = self.stop_roi[tof_id]
            ts = np.arange(start_roi, stop_roi)
            #print(tof_id)
            if self.model_params[tof_id][0] == 0:
                return np.zeros((n_trains, n_pulses, n_e), dtype=np.float32)
            e = model(ts, *self.model_params[tof_id])
            tof = AdqRawChannel(run,
                                self._tof_settings[tof_id][1],
                                digitizer=self._tof_settings[tof_id][0],
                                first_pulse_offset=self.first_pulse_offset,
                                single_pulse_length=self.single_pulse_length)
            pulses = tof.pulse_data(pulse_dim='pulseIndex').unstack('pulse').transpose('trainId', 'pulseIndex', 'sample')
            coords = pulses.coords
            dims = pulses.dims
            pulses = pulses.to_numpy()
            pulses = pulses[:, :, :self.single_pulse_length]
            n_t, n_p, _ = pulses.shape
            assert n_t == n_trains
            assert n_p == n_pulses
            pulses = np.reshape(pulses, (n_t*n_p, -1))
            # apply filter
            filter_length = self._filter_length[tof_id]
            if filter_length > 0:
                filtered = apply_filter(-pulses[:, start_roi:stop_roi], filter_length)
            else:
                filtered = -pulses[:, start_roi:stop_roi]

            # interpolate
            # o = np.apply_along_axis(lambda arr: CubicSpline(e[::-1], arr[::-1])(self.energy_axis),
            #                        axis=1,
            #                        arr=filtered)
            o = np.apply_along_axis(lambda arr: np.interp(self.energy_axis, e[::-1], arr[::-1], left=0, right=0),
                                   axis=1,
                                   arr=filtered)
            o = np.reshape(o, (n_t, n_p, n_e))
            # subtract offset
            o = o - self.offset[tof_id][None, None, :]
            # apply Jacobian
            o = o*self.jacobian[tof_id][None, None, :]
            # regenerate DataArray
            o = xr.DataArray(data=o,
                             dims=('trainId', 'pulseIndex', 'energy'),
                             coords=dict(trainId=coords['trainId'],
                                         pulseIndex=coords['pulseIndex'],
                                         energy=self.energy_axis
                                        )
                            )
            #print(o.shape)
            return o

        outdata = [apply_correction(tof_id)
                   for tof_id in tof_ids]
        outdata = xr.concat(outdata, pd.Index(tof_ids, name="tof"))
        outdata = outdata.transpose('trainId', 'pulseIndex', 'energy', 'tof')
        outdata_corr = outdata/norm.to_numpy()[None, None, :, :]

        xgm_data = XGM(run, self.xgm_source).pulse_energy()

        return xr.Dataset(data_vars=dict(obs=outdata_corr,
                                         obs_notransmission=outdata,
                                         xgm=xgm_data,
                                         transmission=norm
                                         )
                        )

