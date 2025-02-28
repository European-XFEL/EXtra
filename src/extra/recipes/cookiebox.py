from typing import Optional, Union, Dict, List, Tuple, Any
from dataclasses import dataclass, asdict, is_dataclass

import itertools
from functools import partial
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import logging

import numpy as np
from extra_data import open_run, by_id, DataCollection
import h5py
import xarray as xr
import pandas as pd

from .base import SerializableMixin

from extra.components import Scan, AdqRawChannel, XrayPulses, XGM


@dataclass
class TofFitResult:
    """Class keeping track of fit in a single Tof."""
    energy: np.ndarray
    mu: np.ndarray
    sigma: np.ndarray
    A: np.ndarray
    Aa: np.ndarray
    offset: np.ndarray


def search_offset(trace: np.ndarray, sigma: float=20) -> int:
    """
    Find highest peaks in the 1D trace.

    Args:
      trace: Trace.
      sigma: Sigma of the Gaussian to convolve.

    Returns: Offset with some slack.
    """
    # apply it to the data
    from scipy.ndimage import gaussian_filter1d
    smoothened = gaussian_filter1d(trace, sigma=sigma, mode="nearest")
    peak_idx = np.argmax(smoothened)
    return peak_idx - 200

def search_roi(roi: np.ndarray) -> np.ndarray:
    """
    Find highest peaks in the 1D trace.

    Args:
      roi: Trace.

    Returns: Peak position.
    """
    import scipy
    p, _ = scipy.signal.find_peaks(roi, prominence=(0.25*np.amax(roi), None))
    return p

def model(ts: np.ndarray, c: float, e0: float, t0: float) -> np.ndarray:
    """
    Model to fit energy as a function of the sample axis.

    Args:
      ts: Time axis.
      c: Time coefficient.
      e0: Initial energy.
      t0: Zero-time offset.

    Returns: The energy corresponding to the ts axis values.
    """
    return e0+c/(ts-t0)**2

def lin_fit(ts: np.ndarray, energies: np.ndarray, t0: float) -> Tuple[float, float, float]:
    """
    Fit model using linear regression for fixed t0.

    Args:
      ts: Time axis.
      energies: Energy axis.
      t0: Fixed zero-time.

    Returns: Tuple with c, e0 and t0 values after the fit.
    """
    from scipy.stats import linregress
    res = linregress(1/np.asarray(ts-t0)**2,np.asarray(energies))
    c = res.slope
    e0 = res.intercept
    return c, e0, t0

def norm_diff_t0(ts: np.ndarray, energies: np.ndarray, t0: float) -> float:
    """
    Calculates the model mismatch between prediction and observation.

    Args:
      ts: Time axis.
      energies: Energy axis.
      t0: Zero-time.

    Returns: Norm of difference between observation and prediction.
    """
    c,e0,_ = lin_fit(ts,energies,t0)
    return np.linalg.norm(model(ts,c,e0,t0) - energies)

def fit(peak_ids: np.ndarray, energies: np.ndarray, t0_bounds: Tuple[float, float]) -> Tuple[float, float, float]:
    """
    Fit peak IDs to energies withn the zero-time bounds.

    Args:
      peak_ids: The peak IDs.
      energies: The energy values.
      t0_bounds: The minimum and maximum zero-time bounds.

    Returns: Tuple with c, e0, t0.
    """
    from scipy.optimize import minimize_scalar
    peak_ids = np.asarray(peak_ids)
    energies = np.asarray(energies)

    f = partial(norm_diff_t0,peak_ids,energies)
    res=minimize_scalar(f,method='Bounded',bounds=t0_bounds)
    t0 = res.x
    c,e0,t0=lin_fit(peak_ids,energies,t0)
    if res.success:
        return c,e0,t0
    else:
        raise Exception('fit did not converge.')


def calc_mean(itr: Tuple[int, int], scan: Scan, xgm_data: xr.DataArray, tof: Dict[int, AdqRawChannel], xgm_threshold: float, frequencies: Dict[int, List[float]]) -> xr.DataArray:
    """
    Calculate the mean of the ToF data in the given tof and energy bin in `itr`.

    Args:
      itr: The tof ID and energy ID from the Scan object.
      scan: The Scan object.
      xgm_data: All the pulse energy values.
      tof: The Extra component for reading each tof.
      xgm_threshold: The minimum pulse energy to consider.
      frequencies: The digital frequencies to filter.

    Returns: DataArray with mean of data in the energy bin given.
    """
    tof_id, energy_id = itr
    energy, train_ids = scan.steps[energy_id]
    # this does train ID matching, because:
    # - the given run may not have both XGM and eTOF for every train;
    # - and we cannot control the creation of the run object, since we want to receive the ready-made XGM object
    good_ids = sorted(list(set(train_ids).intersection(set(xgm_data.coords["trainId"].to_numpy()))))
    mask = xgm_data.coords["trainId"].isin(good_ids)
    sel_xgm_data = xgm_data[mask]
    tof_data = tof[tof_id].select_trains(by_id[good_ids]).pulse_data(pulse_dim='pulseIndex')
    # select XGM
    tof_data = tof_data.loc[sel_xgm_data > xgm_threshold, :]
    tof_xgm_data = sel_xgm_data.loc[sel_xgm_data > xgm_threshold]
    tof_data = tof_data.to_numpy()
    tof_xgm_data = tof_xgm_data.to_numpy()
    out_data = -tof_data.mean(0)
    out_xgm = tof_xgm_data.mean(0)

    # apply filter
    freq = frequencies[tof_id]
    if len(freq) > 0:
        filtered = apply_filter(out_data, freq)
    else:
        filtered = out_data
    return filtered, out_xgm

def apply_filter(data: np.ndarray, frequencies: List[int]) -> np.ndarray:
    """
    Apply a Kaiser filter on data along its last axis.

    Args:
      data: Data with shape (n_samples, n_energy). Filter is applied on last axis.
      frequencies: Inverse number of samples to consider as fluctuatios to filter out.
    Returns: Filtered data in the same shape as input.
    """
    from scipy.signal import kaiserord, filtfilt, firwin
    nyq_rate = 0.5
    ripple_db = 10.0
    out = data
    for f in frequencies:
        df = min(0.05*f, 0.05)
        N, beta = kaiserord(ripple_db, df)
        if N % 2 == 0:
            N += 1
        taps = firwin(N, [f-df/2, f+df/2], window=('kaiser', beta), pass_zero='bandstop', fs=1)
        out = filtfilt(taps, 1.0, out, axis=-1)
    return out

class CookieboxCalibration(SerializableMixin):
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

    The concrete steps taken when the object is setup
    (`obj = CookieboxCalib()` and `obj.setup(run, energy_axis)`) from a calibration run are:
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

    Args:
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
      interleaved: Whether channels are interleaved. If `None`,
                   attempt to auto-detect, but this fails for a union of runs.
      frequencies: Digital frequencies to filter out.
                     This is useful to filter ringing.
    """
    def __init__(self,
                 xgm_threshold: Union[str, float]='median',
                 first_pulse_offset: Optional[int]=None,
                 single_pulse_length: int=400,
                 auger_start_roi: Optional[int]=None,
                 start_roi: Optional[int]=None,
                 stop_roi: Optional[int]=None,
                 interleaved: Optional[bool]=None,
                 frequencies: Union[float, Dict[int, float]]=0,
                ):
        self._init_auger_start_roi = auger_start_roi
        self._init_start_roi = start_roi
        self._init_stop_roi = stop_roi
        self._init_frequencies = frequencies

        self._init_first_pulse_offset = first_pulse_offset
        self._init_single_pulse_length = single_pulse_length
        self._init_interleaved = interleaved
        self._xgm_threshold = xgm_threshold

        # empty outputs
        self.tof_fit_result = dict()
        self.model_params = dict()
        self.jacobian = dict()
        self.offset = dict()
        self.normalization = dict()
        self.transmission = dict()
        self.int_transmission = dict()

        # what we need to save it all
        self._version = 1
        self._all_fields = ["_energy_axis",
                            "_tof_settings",
                            "_auger_start_roi",
                            "_start_roi",
                            "_stop_roi",
                            "_first_pulse_offset",
                            "_single_pulse_length",
                            "_interleaved",
                            "_xgm_threshold",
                            "_frequencies",
                            "tof_fit_result",
                            "model_params",
                            "jacobian",
                            "offset",
                            "normalization",
                            "transmission",
                            "int_transmission",
                            "tof_fit_result",
                            "calibration_data",
                            "calibration_mean_xgm",
                            "mask",
                            #"sources",
                            "e_transmission",
                            "calibration_energies",
                            "_version",
                           ]
    def _post_load(self):
        """
        Actions to do after loading from file.
        """
        self._tof_settings = {int(k): (v[0].decode("utf-8"), v[1].decode("utf-8")) for k, v in self._tof_settings.items()}
        self.tof_fit_result = {k: TofFitResult(**v) for k, v in self.tof_fit_result.items()}
        self._auger_start_roi = {int(k): v for k, v in self._auger_start_roi.items()}
        self._start_roi = {int(k): v for k, v in self._start_roi.items()}
        self._stop_roi = {int(k): v for k, v in self._stop_roi.items()}
        self.mask = {int(k): v for k, v in self.mask.items()}
        self._frequencies = {int(k): v for k, v in self._frequencies.items()}
        self._first_pulse_offset = {int(k): v for k, v in self._first_pulse_offset.items()}
        self._single_pulse_length = {int(k): v for k, v in self._single_pulse_length.items()}
        self._interleaved = {int(k): v for k, v in self._interleaved.items()}

    def setup(self,
              run: DataCollection,
              energy_axis: np.ndarray,
              tof_settings: Dict[int, Union[Tuple[str, str], AdqRawChannel]],
              scan: Scan,
              xgm: XGM,
              ):
        """
        Derive calibrations.

        Args:
          run: The calibration run.
          energy_axis: Energy axis in eV to interpolate eTOF to.
          tof_settings: Dictionary with a TOF label as a key (0,1,2, ...).
                        Each value is *either* a) a tuple containing the
                        eTOF source name and channel in the format "1_A";
                        or b) the AdqRawChannel object for that eTOF.
          scan: Scan object created with the energy source scan.
                For example: `Scan(calib_run["SA3_XTD10_MONO/MDL/PHOTON_ENERGY", "actualEnergy.value"])`
          xgm: The XGM object used to apply a pulse energy selection.
               For example: `XGM(run, "SQS_DIAG1_XGMD/XGM/DOOCS")`
        """
        # base properties
        self._run = run
        self._energy_axis = energy_axis
        self._tof_settings = tof_settings

        self._xgm = xgm
        self._scan = scan

        self._auger_start_roi = {tof_id: self._init_auger_start_roi for tof_id in self._tof_settings.keys()}
        self._start_roi = {tof_id: self._init_start_roi for tof_id in self._tof_settings.keys()}
        self._stop_roi = {tof_id: self._init_stop_roi for tof_id in self._tof_settings.keys()}
        self._frequencies = self._init_frequencies
        if not isinstance(self._frequencies, dict):
            self._frequencies = {tof_id: self._frequencies for tof_id in self._tof_settings.keys()}
        self._first_pulse_offset = {tof_id: self._init_first_pulse_offset for tof_id in self._tof_settings.keys()}
        self._single_pulse_length = {tof_id: self._init_single_pulse_length for tof_id in self._tof_settings.keys()}
        self._interleaved = {tof_id: self._init_interleaved for tof_id in self._tof_settings.keys()}

        # now do the full analysis, step by step
        # update eTOF data reading objects
        self.update_tof_settings()

        # get XGM and auxiliary data
        self.update_metadata()

        # find RoI if needed
        self.update_roi()

        # find where the peaks are per energy in each Tof
        self.update_fit_result()

        # calibrate and calculate transmission
        self.update_calibration()

        # now we can use the apply method
        logging.info("Ready to apply energy calibration and transmission correction to analysis data ...")

    # these setters allow one to update
    # some of the input properties and rerun only the necessary steps
    # for recalibration
    #
    @property
    def run(self) -> DataCollection:
        return self._run

    def set_run(self, value: DataCollection):
        """
        Update the run object and recompute.

        Args:
          value: The new run.
        """
        self._run = value
        self.update_tof_settings()
        self.update_metadata()
        self.update_roi()
        self.update_fit_result()
        self.update_calibration()

    @property
    def first_pulse_offset(self) -> int:
        return self._first_pulse_offset

    def set_first_pulse_offset(self, value: int):
        """
        Update the first pulse offset and recompute.

        Args:
          value: New first pulse offset value.
        """
        self._first_pulse_offset = value
        if not isinstance(value, dict):
            self._first_pulse_offset = {tof_id: value for tof_id in self._tof_settings.keys()}
        self.update_tof_settings()
        self.update_roi()
        self.update_fit_result()
        self.update_calibration()

    @property
    def single_pulse_length(self) -> int:
        return self._single_pulse_length

    def set_single_pulse_length(self, value: int):
        """
        Update the single pulse length and recompute.

        Args:
          value: New single pulse length.
        """
        self._single_pulse_length = value
        if not isinstance(value, dict):
            self._single_pulse_length = {tof_id: value for tof_id in self._tof_settings.keys()}
        self.update_tof_settings()
        self.update_roi()
        self.update_fit_result()
        self.update_calibration()

    @property
    def interleaved(self) -> bool:
        return self._interleaved

    def set_interleaved(self, value: bool):
        """
        Update interleaved mode and recompute.

        Args:
          value: New value for the interleaved mode.
        """
        self._interleaved = value
        if not isinstance(value, dict):
            self._interleaved = {tof_id: self._interleaved for tof_id in self._tof_settings.keys()}
        self.update_tof_settings()
        self.update_roi()
        self.update_fit_result()
        self.update_calibration()

    @property
    def energy_axis(self) -> np.ndarray:
        return self._energy_axis

    def set_energy_axis(self, value: np.ndarray):
        """
        Update the energy axis and recompute.

        Args:
          value: New value for the energy axis.
        """
        self._energy_axis = value
        self.update_calibration()

    @property
    def xgm_threshold(self) -> float:
        return self._xgm_threshold

    def set_xgm_threshold(self, value: Union[float, str]):
        """
        Update the XGM threshold value and recompute.

        Args:
          value: New XGM threshold value. May be number of "median" to take the median of XGM intensities.
        """
        self._xgm_threshold = value
        # find median if needed
        if self._xgm_threshold == 'median':
            self._xgm_threshold = np.median(self._xgm_data.to_numpy())
        self.update_roi()
        self.update_fit_result()
        self.update_calibration()

    @property
    def tof_settings(self) -> Dict[int, Union[Tuple[str, str], AdqRawChannel]]:
        return self._tof_settings

    def set_tof_settings(self, value: Dict[int, Union[Tuple[str, str], AdqRawChannel]]):
        """
        Update the eTOF settings and recompute.

        Args:
          value: The new eTOF settings as explained in the constructor docstring.
        """
        self._tof_settings = value
        self.update_tof_settings()
        self.update_roi()
        self.update_fit_result()
        self.update_calibration()

    @property
    def frequencies(self) -> Dict[int, int]:
        return self._frequencies

    def set_frequencies(self, value: Dict[int, List[float]]):
        """
        Update the filter length and recompute.

        Args:
          value: The filter length.
        """
        self._frequencies = value
        if not isinstance(value, dict):
            self._frequencies = {tof_id: value for tof_id in self._tof_settings.keys()}
        self.update_roi()
        self.update_fit_result()
        self.update_calibration()

    @property
    def auger_start_roi(self) -> Dict[int, int]:
        return self._auger_start_roi

    def set_auger_start_roi(self, value: int):
        """
        Set the start of the Auger RoI and recompute.

        Args:
          value: The new sample number for the start of the Auger RoI.
        """
        self._auger_start_roi = {tof_id: value for tof_id in self._tof_settings.keys()}
        self.update_roi()
        self.update_fit_result()
        self.update_calibration()

    @property
    def start_roi(self) -> Dict[int, int]:
        return self._start_roi

    def set_start_roi(self, value: int):
        """
        Set the start of the photo-electron RoI and recompute.

        Args:
          value: The new start of the RoI.
        """
        self._start_roi = {tof_id: value for tof_id in self._tof_settings.keys()}
        self.update_roi()
        self.update_fit_result()
        self.update_calibration()

    @property
    def stop_roi(self) -> Dict[int, int]:
        return self._stop_roi

    def set_stop_roi(self, value: int):
        """
        Update the end of the photo-electron RoI and recompute.

        Args:
          value: The end of the photo-electron RoI.
        """
        self._stop_roi = {tof_id: value for tof_id in self._tof_settings.keys()}
        self.update_roi()
        self.update_fit_result()
        self.update_calibration()

    def update_metadata(self):
        """
        Read calibration XGM and metadata information.
        """
        # set up helper objects
        # pulse structure
        self._pulses = XrayPulses(self._run)

        # get XGM information
        self._xgm_data = self._xgm.pulse_energy().stack(pulse=('trainId', 'pulseIndex'))
        # find median if needed
        if self._xgm_threshold == 'median':
            self._xgm_threshold = np.median(self._xgm_data.to_numpy())
        # create scan object
        self.calibration_energies = self._scan.positions

    def update_tof_settings(self):
        """
        Update position of the first pulse offset if needed and
        create AdqRawChannel.
        """
        # find first peak offset
        if self.first_pulse_offset is None:
            need_it = False
            for tof_id in self._tof_settings.keys():
                if not isinstance(self._tof_settings[tof_id], AdqRawChannel):
                    need_it = True
            if need_it:
                logging.info("First pulse offset not given: guessing it from data.")
                logging.info("(This may fail, if it does, please provide a `first_pulse_offset` instead.)")
                self.first_pulse_offset = self.find_offset()
                logging.info(f"Found first pulse offset at {self.first_pulse_offset}")
                self.first_pulse_offset = {tof_id: self.first_pulse_offset for tof_id in self._tof_settings.keys()}

        # create tof objects:
        self._tof = dict()
        for tof_id in self._tof_settings.keys():
            if not isinstance(self._tof_settings[tof_id], AdqRawChannel):
                digitizer, channel = self._tof_settings[tof_id]
                self._tof[tof_id] = AdqRawChannel(self._run,
                                                  channel,
                                                  digitizer=digitizer,
                                                  first_pulse_offset=self.first_pulse_offset[tof_id],
                                                  single_pulse_length=self.single_pulse_length[tof_id],
                                                  interleaved=self.interleaved[tof_id],
                                                  )
            else: # it is an AdqRawChannel, so copy its properties, so we can use it in apply
                self._tof[tof_id] = self._tof_settings[tof_id]
                self._tof_settings[tof_id] = (self._tof[tof_id].instrument_source.source, self._tof[tof_id].name)
                self._first_pulse_offset[tof_id] = self._tof[tof_id].first_pulse_offset
                self._single_pulse_length[tof_id] = self._tof[tof_id].single_pulse_length
                self._interleaved[tof_id] = self._tof[tof_id].interleaved

        self.mask = {tof_id: True for tof_id in self._tof_settings.keys()}

    def update_roi(self):
        """
        Given calibrated data, apply a selection and find RoI if needed.
        """
        # average data for each energy slice
        logging.info("Reading calibration data ... (this takes a while)")
        self.select_calibration_data()
        # find RoI if needed
        if (self.auger_start_roi is None
            or self.start_roi is None
            or self.stop_roi is None):
            logging.info("Finding RoI ...")
            logging.info("(This may fail. If it does, please provide a `auger_start_roi`, `start_roi` and `stop_roi`.)")
            for tof_id in self._tof_settings.keys():
                self.find_roi(tof_id)
            logging.info(f"Auger start RoIs found: {self.auger_start_roi}")
            logging.info(f"Start RoIs found: {self.start_roi}")
            logging.info(f"Stop RoIs found: {self.stop_roi}")

    def update_fit_result(self):
        """
        Fit TOF data to a Gaussian and collect results.
        """
        logging.info("Fit peaks to obtain energy calibration ...")
        for tof_id in self._tof_settings.keys():
            logging.info(f"Fitting eTOF {tof_id} ...")
            self.tof_fit_result[tof_id] = self.peak_tof(tof_id)

    def update_calibration(self):
        """
        Calculate calibration maps and transmission.
        """
        # calculate transmission (fills the arrays above)
        logging.info("Calculate calibration and transmission ...")
        for tof_id in self._tof_settings.keys():
            self.calculate_calibration_and_transmission(tof_id)

        # summarize it for later
        n_tof = len(self._tof_settings.keys())
        n_e = len(self.energy_axis)
        self.e_transmission = np.zeros((n_e, n_tof), dtype=np.float32)
        for idx, tof_id in enumerate(self._tof_settings.keys()):
            self.e_transmission[:,idx] = self.int_transmission[tof_id]

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
                     frequencies=self._frequencies
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
        return max(peaks, 0)

    def find_roi(self, tof_id: int):
        """
        Find RoI for the photo-electron peak from calibration data.

        Args:
          tof_id: The eTOF ID.
        """
        auger = list()
        roi = list()
        for e in range(self.calibration_data[tof_id].shape[0]):
            d = self.calibration_data[tof_id][e, :]
            peaks = search_roi(d)
            peaks = sorted(peaks)
            if len(peaks) < 2:
                logging.info(f"Failed to find peaks for eTOF {tof_id}, energy index {e}. "
                             f"Check the data quality.")
                continue
            auger += [peaks[0]]
            roi += [peaks[1]]
        if len(auger) == 0 or len(roi) == 0:
            logging.info(f"No peaks found in eTOF {tof_id}. "
                         f"Check the data quality. "
                         f"I will set the RoI to collect non-sense,"
                         f" so this TOF data will be meaningless. "
                         f"It will also be masked.")
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
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
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

        Args:
          tof_id: The eTOF ID.

        Returns: Energy vector used, means and std. dev. in the sample axis scale, and integral of Gaussian.
        """
        from lmfit.models import GaussianModel, ConstantModel
        from lmfit.model import ModelResult
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
            #ii = np.sum(xa*(ya-np.amin(ya)))/np.sum(ya-np.amin(ya))
            iisig = 2 #np.sqrt(np.sum((xa -ii)**2*(ya-np.amin(ya)))/np.sum(ya-np.amin(ya)))
            resulta = gamodel.fit(ya, x=xa, center=xa[ii], amplitude=np.amax(ya), sigma=iisig, c=np.median(ya))
            # fit data
            x = np.arange(start_roi, stop_roi)
            y = data[e, start_roi:stop_roi]
            gmodel = GaussianModel() + ConstantModel()
            ii = np.argmax(y)
            #ii = np.sum(x*(y-np.amin(y)))/np.sum(y-np.amin(y))
            iisig = 10 #np.sqrt(np.sum((x -ii)**2*(y-np.amin(y)))/np.sum(y-np.amin(y)))
            result = gmodel.fit(y, x=x, center=x[ii], amplitude=np.amax(y), sigma=iisig, c=np.median(y))
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

        Args:
          tof_id: eTOF ID.
        """
        from scipy.interpolate import CubicSpline

        energy_ids = np.arange(len(self._scan.positions))
        # fit calibration
        c, e0, t0 = fit(self.tof_fit_result[tof_id].mu, self.tof_fit_result[tof_id].energy, t0_bounds=[-3000, 3000])
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
        import matplotlib.pyplot as plt
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

        Args:
          tof_ids: List of eTOF IDs to plot. If none, show all.
        """
        import matplotlib.pyplot as plt
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

        Args:
          tof_ids: List of eTOF IDs to plot. If none, show all.
        """
        import matplotlib.pyplot as plt
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

        Args:
          tof_ids: List of eTOF IDs to plot. If none, show all.
        """
        import matplotlib.pyplot as plt
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

        Args:
          tof_ids: List of eTOF IDs to plot. If none, show all.
        """
        import matplotlib.pyplot as plt
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

        Args:
          tof_id: eTOF ID.
        """
        from lmfit.models import GaussianModel, ConstantModel
        from lmfit.model import ModelResult
        import matplotlib.pyplot as plt
        from matplotlib.colors import CSS4_COLORS
        data = self.calibration_data[tof_id]
        auger_start_roi = self.auger_start_roi[tof_id]
        start_roi = self.start_roi[tof_id]
        stop_roi = self.stop_roi[tof_id]
        ts = np.arange(start_roi, stop_roi)
        fig = plt.figure(figsize=(20,10))
        for e, c in zip(range(data.shape[0]), CSS4_COLORS.keys()):
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

        Args:
          run: The run to calibrate.

        Returns: The xarray Dataset.
        """

        tof_idx = sorted([idx for idx, tof_id in enumerate(self._tof_settings.keys()) if self.mask[tof_id]])
        tof_ids = sorted([tof_id for idx, tof_id in enumerate(self._tof_settings.keys()) if self.mask[tof_id]])
        n_tof = len(tof_ids)
        n_e = len(self.energy_axis)
        n_trains = len(run.train_ids)
        n_pulses = XrayPulses(run).pulse_counts().max()
        norm = np.stack([v for k, v in self.normalization.items() if k in tof_ids], axis=-1)
        norm *= self.e_transmission[:, tof_idx]
        norm = xr.DataArray(data=norm,
                            dims=('energy', 'tof'),
                            coords=dict(energy=self.energy_axis,
                                        tof=tof_ids))

        def apply_correction(tof_id):
            """
            Apply the energy calibration and transmission correction for a given eTOF.
            """
            logging.info(f"Correcting eTOF {tof_id} ...")
            # the sample axis to use for the calibration
            auger_start_roi = self.auger_start_roi[tof_id]
            start_roi = self.start_roi[tof_id]
            stop_roi = self.stop_roi[tof_id]
            ts = np.arange(start_roi, stop_roi)
            if self.model_params[tof_id][0] == 0:
                return np.zeros((n_trains, n_pulses, n_e), dtype=np.float32)
            e = model(ts, *self.model_params[tof_id])
            tof = AdqRawChannel(run,
                                self._tof_settings[tof_id][1],
                                digitizer=self._tof_settings[tof_id][0],
                                first_pulse_offset=self.first_pulse_offset[tof_id],
                                single_pulse_length=self.single_pulse_length[tof_id],
                                interleaved=self.interleaved[tof_id])
            pulses = tof.pulse_data(pulse_dim='pulseIndex').unstack('pulse').transpose('trainId', 'pulseIndex', 'sample')
            coords = pulses.coords
            dims = pulses.dims
            pulses = -pulses.to_numpy()
            n_t, n_p, _ = pulses.shape
            assert n_t == n_trains
            assert n_p == n_pulses
            pulses = np.reshape(pulses, (n_t*n_p, -1))
            # apply filter
            frequencies = self._frequencies[tof_id]
            if len(frequencies) > 0:
                filtered = apply_filter(pulses, frequencies)
            else:
                filtered = pulses

            filtered = filtered[:, start_roi:stop_roi]

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
            return o

        outdata = [apply_correction(tof_id)
                   for tof_id in tof_ids]
        outdata = xr.concat(outdata, pd.Index(tof_ids, name="tof"))
        outdata = outdata.transpose('trainId', 'pulseIndex', 'energy', 'tof')
        outdata_corr = outdata/norm.to_numpy()[None, None, :, :]

        return xr.Dataset(data_vars=dict(obs=outdata_corr,
                                         obs_notransmission=outdata,
                                         transmission=norm
                                         )
                        )

