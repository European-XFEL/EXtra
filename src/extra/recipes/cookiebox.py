from dataclasses import dataclass, asdict, is_dataclass

from typing import Tuple, Dict, Union, List, Optional, Any

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
    mu_auger: np.ndarray


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
    p, _ = scipy.signal.find_peaks(roi, prominence=(0.25*np.max(roi), None))
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
    return np.linalg.norm(model(ts,c,e0,t0) - energies, ord=1)

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


def calc_mean(itr: Tuple[int, int], scan: Scan, xgm_data: xr.DataArray, tof: Dict[int, AdqRawChannel], xgm_threshold: float) -> xr.DataArray:
    """
    Calculate the mean of the ToF data in the given tof and energy bin in `itr`.

    Args:
      itr: The tof ID and energy ID from the Scan object.
      scan: The Scan object.
      xgm_data: All the pulse energy values.
      tof: The Extra component for reading each tof.
      xgm_threshold: The minimum pulse energy to consider.

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
    if len(good_ids) == 0:
        x = tof[tof_id].select_trains(np._[0:1]).pulse_data(pulse_dim='pulseIndex').to_numpy().mean(0)
        return np.zeros_like(x), 0
    tof_data = tof[tof_id].select_trains(by_id[good_ids]).pulse_data(pulse_dim='pulseIndex')
    # select XGM
    tof_data = tof_data.loc[sel_xgm_data > xgm_threshold, :]
    tof_xgm_data = sel_xgm_data.loc[sel_xgm_data > xgm_threshold]
    tof_data = tof_data.to_numpy()
    tof_xgm_data = tof_xgm_data.to_numpy()
    out_data = -tof_data.mean(0)
    out_xgm = tof_xgm_data.mean(0)

    return out_data, out_xgm

def apply_filter(data: np.ndarray, frequencies: List[float]) -> np.ndarray:
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
    order = 5
    for f in frequencies:
        df = 0.1*f
        N, beta = kaiserord(ripple_db, df)
        if N % 2 == 0:
            N -= 1
        a = firwin(N, [f-df/2, f+df/2], window='hamming', pass_zero='bandstop', fs=1)
        b = 1
        out = filtfilt(a, b, out, axis=-1)
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

    Example usage:
    ```
    # select relevant data
    # this is not needed, but useful to be sure all data is correctly matched
    ts = "SQS_RR_UTC/TSYS/TIMESERVER:outputBunchPattern"
    pes1 = 'SQS_DIGITIZER_UTC4/ADC/1:network'
    pes2 = 'SQS_DIGITIZER_UTC5/ADC/1:network'
    xgm_source_ctrl = "SQS_DIAG1_XGMD/XGM/DOOCS"
    xgm_source = "SQS_DIAG1_XGMD/XGM/DOOCS:output"

    energy_source = "SA3_XTD10_MONO/MDL/PHOTON_ENERGY"

    # setup calibration runs
    calib_run = [open_run(proposal=8697, run=r) for r in range(173, 185)]
    calib_run = calib_run[0].union(*calib_run[1:])
    calib_run = calib_run.select([ts,
                                   pes1,
                                   pes2,
                                   xgm_source,
                                   xgm_source_ctrl,
                                   energy_source], require_all=True)
    # set up AdqRawChannel object to read data from each eTOF
    create_channel = lambda digi, ch: AdqRawChannel(calib_run,
                                                    ch,
                                                    digitizer=digi,
                                                    first_pulse_offset=23300,
                                                    single_pulse_length=400,
                                                    interleaved=True,
                                                    baseline=np.s_[:20000],
                                                    )
    tof_settings = {
           0: create_channel(pes1, "1_A"),
           1: create_channel(pes1, "1_C"),
           2: create_channel(pes1, "2_A"),
           3: create_channel(pes1, "2_C"),
           4: create_channel(pes1, "3_A"),
           5: create_channel(pes1, "3_C"),
           6: create_channel(pes1, "4_A"),
           7: create_channel(pes1, "4_C"),
           8: create_channel(pes2, "1_A"),
           9: create_channel(pes2, "1_C"),
           10: create_channel(pes2, "2_A"),
           11: create_channel(pes2, "2_C"),
           12: create_channel(pes2, "3_A"),
           13: create_channel(pes2, "3_C"),
           14: create_channel(pes2, "4_A"),
           15: create_channel(pes2, "4_C"),
           }

    # define energy axis to interpolate to
    energy_axis = np.linspace(968, 1026, 160)

    # defie regions of interest
    cal = CookieboxCalibration(
                    # these were chosen by eye
                    # if set to None, automatic discovery is used
                    # but it may fail
                    auger_start_roi=150,
                    start_roi=200,
                    stop_roi=320,
    )

    # do calibration
    cal.setup(run=calib_run, energy_axis=energy_axis, tof_settings=tof_settings,
              xgm=XGM(calib_run, "SQS_DIAG1_XGMD/XGM/DOOCS"),
              scan=Scan(calib_run["SA3_XTD10_MONO/MDL/PHOTON_ENERGY", "actualEnergy"]))

    # save for later usage
    cal.to_file('cookiebox_calib.h5')

    # re-use later
    cal_read = CookieboxCalibration.from_file('cookiebox_calib.h5')

    # open new run and apply it
    r188 = open_run(proposal=8697, run=188).select(cal_read.sources, require_all=True).select_trains(np.s_[:5])
    r188_cal = cal_read.apply(r188)

    # make a plot
    plt.plot(r188_cal.sel(tof=4).mean('trainId').mean('pulseIndex').to_numpy())

    ```

    Args:
      xgm_threshold: Minimum threshold to ignore dark frames in
                     the calibration data (in uJ).
                     Can be 'median' to use the median over the run.
      auger_start_roi: Start of the Auger and valence RoI in a pulse,
                       relative to the `first_pulse_offset`. Use `None` to guess it.
      start_roi: Start of the RoI in a pulse, relative to the `first_pulse_offset`.
                 Use `None` to guess it.
      stop_roi: End of the RoI, relative to the `first_pulse_offset`. Use `None` to guess it.
    """
    def __init__(self,
                 xgm_threshold: Union[str, float]='median',
                 auger_start_roi: Optional[int]=None,
                 start_roi: Optional[int]=None,
                 stop_roi: Optional[int]=None,
                 interleaved: Optional[bool]=None,
                ):
        self._init_auger_start_roi = auger_start_roi
        self._init_start_roi = start_roi
        self._init_stop_roi = stop_roi

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
        self.all_kwargs_adq = ["first_pulse_offset",
                               "cm_period",
                               "interleaved",
                               "single_pulse_length",
                               "extra_cm_period"]
        self._all_fields = ["_energy_axis",
                            "kwargs_adq",
                            "_auger_start_roi",
                            "_start_roi",
                            "_stop_roi",
                            "_xgm_threshold",
                            "tof_fit_result",
                            "model_params",
                            "jacobian",
                            "offset",
                            "normalization",
                            "transmission",
                            "int_transmission",
                            "calibration_data",
                            "calibration_mean_xgm",
                            "mask",
                            "calibration_mask",
                            #"sources",
                            "e_transmission",
                            "calibration_energies",
                            "_version",
                           ]
    def _asdict(self):
        """
        Return serializable dict.
        """
        return {k: v for k, v in self.__dict__.items() if k in self._all_fields}

    def _fromdict(self, all_data):
        """
        Actions to do after loading from file.
        """
        for k, v in all_data.items():
            setattr(self, k, v)
        self.tof_fit_result = {k: TofFitResult(**v) for k, v in self.tof_fit_result.items()}
        self._auger_start_roi = {int(k): v for k, v in self._auger_start_roi.items()}
        self._start_roi = {int(k): v for k, v in self._start_roi.items()}
        self._stop_roi = {int(k): v for k, v in self._stop_roi.items()}
        self.mask = {int(k): v for k, v in self.mask.items()}
        self.calibration_mask = {int(k): v for k, v in self.calibration_mask.items()}
        self.kwargs_adq = {int(k): v for k, v in self.kwargs_adq.items()}

    def setup(self,
              run: DataCollection,
              energy_axis: np.ndarray,
              tof_settings: Dict[int, AdqRawChannel],
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

        if isinstance(self._init_auger_start_roi, dict):
            self._auger_start_roi = {tof_id: self._init_auger_start_roi[tof_id] for tof_id in self._tof_settings.keys()}
        else:
            self._auger_start_roi = {tof_id: self._init_auger_start_roi for tof_id in self._tof_settings.keys()}
        if isinstance(self._init_start_roi, dict):
            self._start_roi = {tof_id: self._init_start_roi[tof_id] for tof_id in self._tof_settings.keys()}
        else:
            self._start_roi = {tof_id: self._init_start_roi for tof_id in self._tof_settings.keys()}
        if isinstance(self._init_stop_roi, dict):
            self._stop_roi = {tof_id: self._init_stop_roi[tof_id] for tof_id in self._tof_settings.keys()}
        else:
            self._stop_roi = {tof_id: self._init_stop_roi for tof_id in self._tof_settings.keys()}

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
    def auger_start_roi(self) -> Dict[int, int]:
        return self._auger_start_roi

    def set_auger_start_roi(self, value: int):
        """
        Set the start of the Auger RoI and recompute.

        Args:
          value: The new sample number for the start of the Auger RoI.
        """
        self._auger_start_roi = {tof_id: value for tof_id in self.kwargs_adq.keys()}
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
        self._start_roi = {tof_id: value for tof_id in self.kwargs_adq.keys()}
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
        self._stop_roi = {tof_id: value for tof_id in self.kwargs_adq.keys()}
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
        # create tof objects:
        self._tof = dict()
        self.kwargs_adq = dict()
        for tof_id in self._tof_settings.keys():
            self.kwargs_adq[tof_id] = dict()
            self._tof[tof_id] = self._tof_settings[tof_id]
            for k in self.all_kwargs_adq:
                if hasattr(self._tof[tof_id], f"_{k}"):
                    key = f"_{k}"
                else:
                    key = f"{k}"
                self.kwargs_adq[tof_id][k] = getattr(self._tof[tof_id], key)
            self.kwargs_adq[tof_id]["source"] = self._tof[tof_id].instrument_source.source
            self.kwargs_adq[tof_id]["name"] = self._tof[tof_id].name
        self.mask = {tof_id: True for tof_id in self.kwargs_adq.keys()}

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
            for tof_id in self.kwargs_adq.keys():
                self.find_roi(tof_id)
            logging.info(f"Auger start RoIs found: {self.auger_start_roi}")
            logging.info(f"Start RoIs found: {self.start_roi}")
            logging.info(f"Stop RoIs found: {self.stop_roi}")

    def update_fit_result(self):
        """
        Fit TOF data to a Gaussian and collect results.
        """
        logging.info("Fit peaks to obtain energy calibration ...")
        for tof_id in self.kwargs_adq.keys():
            logging.info(f"Fitting eTOF {tof_id} ...")
            self.tof_fit_result[tof_id] = self.peak_tof(tof_id)

    def update_calibration(self):
        """
        Calculate calibration maps and transmission.
        """
        # calculate transmission (fills the arrays above)
        logging.info("Calculate calibration and transmission ...")
        for tof_id in self.kwargs_adq.keys():
            self.calculate_calibration_and_transmission(tof_id)

        # summarize it for later
        n_tof = len(self.kwargs_adq.keys())
        n_e = len(self.energy_axis)
        self.e_transmission = np.zeros((n_e, n_tof), dtype=np.float32)
        for idx, tof_id in enumerate(self.kwargs_adq.keys()):
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
                     )


        with ProcessPoolExecutor(max_workers=10) as p:
            itr_gen = list(itertools.product(tof_ids, energy_ids))
            #data_gen = map(fn, itr_gen)
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
        self.calibration_mask = {tof_id: np.array([True for _ in energy_ids])
                                 for tof_id in tof_ids}

    def mask_calibration_point(self, tof_id: int, energy: float, mask: bool=False, tol: float=0.01):
        """
        If `mask` is False, the point at a given energy and eTOF is ignored when
        performing the fit.

        Args:
          tof_id: eTOF number, as in the `tof_settings`.
          energy: Energy value, in the same units as provided in `scan`.
          mask: If True, keep the point. If False, remove it.
          tol: Tolerance for energy matching.
        """
        self.calibration_mask[tof_id][np.abs(energy - self.tof_fit_result[tof_id].energy) < tol] = mask

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
        tof_ids = list(self.kwargs_adq.keys())
        energies = self.calibration_energies
        idx = np.argsort(self.calibration_energies)
        n_energies = len(energies)
        fig, axes = plt.subplots(nrows=4, ncols=4, clear=True, figsize=(20,20))
        for tof_id in tof_ids:
            auger_start_roi = self.auger_start_roi[tof_id]
            start_roi = self.start_roi[tof_id]
            stop_roi = self.stop_roi[tof_id]
            sample_auger = int(np.min(self.tof_fit_result[tof_id].mu_auger))
            ax = axes.flatten()[tof_id]
            temp = data[tof_id].copy()
            ax.imshow(temp[idx, sample_auger:stop_roi],
                      extent=[sample_auger, stop_roi, np.min(energies), np.max(energies)],
                      aspect="auto",
                      norm=Normalize(vmax=np.max(temp)),
                     )
            ax.set_title(f'TOF {tof_id}')
            ax.set_xlabel("Samples")
            ax.set_ylabel("Energy [eV]")
        plt.tight_layout()

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
        mu_auger = list()
        auger_start_roi = self.auger_start_roi[tof_id]
        start_roi = self.start_roi[tof_id]
        stop_roi = self.stop_roi[tof_id]
        for e in range(data.shape[0]):
            # fit Auger
            xa = np.arange(auger_start_roi, start_roi)
            ya = data[e, auger_start_roi:start_roi]
            gamodel = GaussianModel() + ConstantModel()
            ii = np.argmax(ya)
            iisig = 2 #np.sqrt(np.sum((xa -ii)**2*(ya-np.min(ya)))/np.sum(ya-np.min(ya)))
            resulta = gamodel.fit(ya, x=xa, center=xa[ii], amplitude=np.max(ya), sigma=iisig, c=np.median(ya))
            # fit data
            x = np.arange(start_roi, stop_roi)
            y = data[e, start_roi:stop_roi]
            gmodel = GaussianModel() + ConstantModel()
            ii = np.argmax(y)
            iisig = 10 #np.sqrt(np.sum((x -ii)**2*(y-np.min(y)))/np.sum(y-np.min(y)))
            result = gmodel.fit(y, x=x, center=x[ii], amplitude=np.max(y), sigma=iisig, c=np.median(y))
            # we care about the normalization coefficient, not the normalized amplitude
            A += [result.best_values["amplitude"]]
            Aa += [resulta.best_values["amplitude"]]
            mu += [result.best_values["center"]]
            sigma += [result.best_values["sigma"]]
            energy += [energies[e]]
            offset += [result.best_values["c"]]
            mu_auger += [resulta.best_values["center"]]
        energy = np.array(energy)
        mu = np.array(mu)
        mu_auger = np.array(mu_auger)
        sigma = np.array(sigma)
        A = np.array(A)
        Aa = np.array(Aa)
        offset = np.array(offset)
        return TofFitResult(energy, mu, sigma, A, Aa, offset, mu_auger)

    def calculate_calibration_and_transmission(self, tof_id: int):
        """
        Calculate transmissions.

        Args:
          tof_id: eTOF ID.
        """
        from scipy.interpolate import CubicSpline

        energy_ids = np.arange(len(self._scan.positions))
        mask = self.calibration_mask[tof_id]
        # fit calibration
        c, e0, t0 = fit(self.tof_fit_result[tof_id].mu[mask],
                        self.tof_fit_result[tof_id].energy[mask], t0_bounds=[-3000, 3000])
        self.model_params[tof_id] = np.array([c, e0, t0], dtype=np.float64)
        self.jacobian[tof_id] = 0.5*c/(np.sqrt(c/(self.energy_axis - e0)))/(self.energy_axis - e0)**2

        # interpolate offset
        eidx = np.argsort(self.tof_fit_result[tof_id].energy[mask])
        ee = self.tof_fit_result[tof_id].energy[mask][eidx]
        eo = self.tof_fit_result[tof_id].offset[mask][eidx]

        self.offset[tof_id] = np.interp(self.energy_axis,
                                        ee,
                                        eo)

        # interpolate amplitude as given by the
        # Auger+Valence (related to the cross section and pulse intensity)
        # normalized by the XGM mean intensity
        en = self.tof_fit_result[tof_id].Aa[mask][eidx]/self.calibration_mean_xgm[tof_id][mask][eidx]
        self.normalization[tof_id] = np.interp(self.energy_axis,
                                               ee,
                                               en)

        # calculate transmission
        self.transmission[tof_id] = self.tof_fit_result[tof_id].A[mask]/self.tof_fit_result[tof_id].Aa[mask]

        # interpolate transmission for the given energy axis
        et = self.transmission[tof_id][eidx]
        self.int_transmission[tof_id] = np.interp(self.energy_axis,
                                                  ee,
                                                  et)

    def plot_calibrations(self):
        """
        Diagnostics plots for finding the energy peaks in a scan.
        """
        import matplotlib.pyplot as plt
        from itertools import product
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        lw = 3
        ls = '-'

        fig, ax = plt.subplots(nrows=2, figsize=(12, 20))

        tof_ids = list(self.model_params.keys())
        for i, tof_id in enumerate(tof_ids):
            if not self.mask[tof_id]:
                continue
            a = ax[i//8]
            c = colors[i%8]
            auger_start_roi = self.auger_start_roi[tof_id]
            start_roi = self.start_roi[tof_id]
            stop_roi = self.stop_roi[tof_id]
            ts = np.arange(start_roi, stop_roi)
            e = model(ts, *self.model_params[tof_id])
            a.scatter(self.tof_fit_result[tof_id].mu,
                        self.tof_fit_result[tof_id].energy,
                        c=c)
            a.plot(ts, e, c=c, lw=lw, ls=ls, label=f"eTOF {tof_id}")
        for a in ax:
            a.set(xlabel="Samples",
                  ylabel="Energy [eV]")
            a.legend(frameon=False, ncols=2)

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
        ax[2].plot(self.energy_axis, self.offset[tof_id], lw=2, label="Offset")
        ax[2].set(xlabel="Samples", ylabel="Offset to subtract [a.u.]")
        ax[3].scatter(self.tof_fit_result[tof_id].energy, self.tof_fit_result[tof_id].Aa/self.calibration_mean_xgm[tof_id], label="Normalization")
        ax[3].plot(self.energy_axis, self.normalization[tof_id], lw=2, label="Interpolated normalization")
        ax[3].set(xlabel="Energy [eV]", ylabel="(Auger+valence)/pulse energy [a.u.]")
        ax[4].plot(self.energy_axis, self.jacobian[tof_id], lw=2, label="Interpolated Jacobian")
        ax[4].set(xlabel="Energy [eV]", ylabel="Jacobian [a.u.]")
        for a in ax:
            a.set_title(f"TOF {tof_id}")

    def plot_transmissions(self):
        """
        Plot all transmissions in the same plot.
        """
        import matplotlib.pyplot as plt
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        lw = 3
        ls = '-'

        fig, ax = plt.subplots(nrows=2, figsize=(12, 20))

        tof_ids = list(self.model_params.keys())
        for i, tof_id in enumerate(tof_ids):
            if not self.mask[tof_id]:
                continue
            a = ax[i//8]
            c = colors[i%8]
            a.plot(self.energy_axis, self.int_transmission[tof_id], c=c, lw=lw, ls=ls, label=f"eTOF {tof_id}")
        for a in ax:
            a.set(xlabel="Energy [eV]",
                  ylabel="Transmission [a.u.]")
            a.legend(frameon=False, ncols=2)

    def plot_offsets(self):
        """
        Plot all offset in the same plot.
        """
        import matplotlib.pyplot as plt
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        lw = 3
        ls = '-'

        fig, ax = plt.subplots(nrows=2, figsize=(12, 20))

        tof_ids = list(self.model_params.keys())
        for i, tof_id in enumerate(tof_ids):
            if not self.mask[tof_id]:
                continue
            a = ax[i//8]
            c = colors[i%8]
            a.plot(self.energy_axis, self.offset[tof_id], c=c, lw=lw, ls=ls, label=f"eTOF {tof_id}")
        for a in ax:
            a.set(xlabel="Energy [eV]",
                  ylabel="Offset [a.u.]")
            a.legend(frameon=False, ncols=2)

    def plot_jacobians(self, eV_per_sample: bool=True):
        """
        Plot all jacobians in the same plot.

        """
        import matplotlib.pyplot as plt
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        lw = 3
        ls = '-'

        fig, ax = plt.subplots(nrows=2, figsize=(12, 20))

        tof_ids = list(self.model_params.keys())
        for i, tof_id in enumerate(tof_ids):
            if not self.mask[tof_id]:
                continue
            a = ax[i//8]
            c = colors[i%8]
            if eV_per_sample:
                a.plot(self.energy_axis, 1.0/self.jacobian[tof_id], c=c, lw=lw, ls=ls, label=f"eTOF {tof_id}")
            else:
                a.plot(self.energy_axis, self.jacobian[tof_id], c=c, lw=lw, ls=ls, label=f"eTOF {tof_id}")
        for a in ax:
            if eV_per_sample:
                a.set(xlabel="Energy [eV]",
                      ylabel=r"$\left\vert\frac{dE}{dt}\right\vert$ [eV/samp.]")
            else:
                a.set(xlabel="Energy [eV]",
                      ylabel=r"$\left\vert\frac{dt}{dE}\right\vert$ [samp./eV]")
            a.legend(frameon=False, ncols=2)

    def plot_normalizations(self):
        """
        Plot all normalizations in the same plot.
        """
        import matplotlib.pyplot as plt
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        lw = 3
        ls = '-'

        fig, ax = plt.subplots(nrows=2, figsize=(12, 20))

        tof_ids = list(self.model_params.keys())
        for i, tof_id in enumerate(tof_ids):
            if not self.mask[tof_id]:
                continue
            a = ax[i//8]
            c = colors[i%8]
            a.plot(self.energy_axis, self.normalization[tof_id], c=c, lw=lw, ls=ls, label=f"eTOF {tof_id}")
        for a in ax:
            a.set(xlabel="Energy [eV]",
                  ylabel="Normalization [a.u.]")
            a.legend(frameon=False, ncols=2)

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
            sigma = self.tof_fit_result[tof_id].sigma[e]
            offset = self.tof_fit_result[tof_id].offset[e]
            gmodel = GaussianModel() + ConstantModel()
            gresult = ModelResult(gmodel, gmodel.make_params(center=mu, amplitude=amplitude, sigma=sigma, c=offset))
            plt.plot(ts, gresult.eval(x=ts), c=c, lw=2)
        plt.xlabel("Samples")
        plt.ylabel("Intensty [a.u.]")
        plt.legend()
        plt.title(f"TOF {tof_id}")

    def load_trace(self, run: DataCollection, **extra_kwargs_adq: Dict[str, Any]) -> xr.Dataset:
        """
        See [load_data][extra.recipes.CookieboxCalibration.load_data].
        """
        return self.load_data(run, **extra_kwargs_adq)

    def load_data(self, run: DataCollection, **extra_kwargs_adq: Dict[str, Any]) -> xr.Dataset:
        """
        Only load region of interest for the same settings in a new run and output a Dataset with it.
        This is the recommended way to load data from a new run before applying the calibration.

        Args:
          run: The run to calibrate.
          kwargs_adq: Keyword arguments for the `AdqRawChannel` object if one wishes to override settings.

        Returns: An xarray DataArray with the traces containing axes ('trainId', 'pulseIndex', 'sample', 'tof').
        """

        tof_ids = sorted([tof_id
                          for idx, tof_id in enumerate(self.kwargs_adq.keys())
                          if self.mask[tof_id]])
        def fetch(tof_id):
            """
            Apply the energy calibration and transmission correction for a given eTOF.
            """
            logging.info(f"Fetch data from eTOF {tof_id} ...")
            # the sample axis to use for the calibration
            auger_start_roi = self.auger_start_roi[tof_id]
            start_roi = self.start_roi[tof_id]
            stop_roi = self.stop_roi[tof_id]
            kwargs = {k: v for k, v in self.kwargs_adq[tof_id].items()}
            del kwargs["name"]
            del kwargs["source"]
            kwargs.update(extra_kwargs_adq)
            tof = AdqRawChannel(run,
                                self.kwargs_adq[tof_id]["name"],
                                digitizer=self.kwargs_adq[tof_id]["source"],
                                **kwargs)
            pulses = tof.pulse_data(pulse_dim='pulseIndex').unstack('pulse').transpose('trainId', 'pulseIndex', 'sample')
            pulses = -pulses.isel(sample=slice(start_roi, stop_roi))
            return pulses

        outdata = [fetch(tof_id)
                   for tof_id in tof_ids]
        outdata = xr.concat(outdata, pd.Index(tof_ids, name="tof"))
        outdata = outdata.transpose('trainId', 'pulseIndex', 'sample', 'tof')

        return outdata

    def calibrate(self, trace: xr.DataArray) -> xr.DataArray:
        """
        Takes a trace separated with axes ('trainId', 'pulseIndex', 'sample', 'tof'),
        as given by `load_trace` and applies the calibration.
        The method `apply` provides a direct way to retrieve the trace and calbrate it.
        The methods `load_trace` and `calibrate` allow one to apply an intermediate processing
        step between them.

        Args:
          trace: A pre-processed trace retrieved with the *same* `AdqRawChannel` settings as this calibration object.
                 Its axes are expected to be ('trainId', 'pulseIndex', 'sample', 'tof').
                 It is recommended to use always `load_trace` to obtain this.

        Returns: the calibrated data as an xarray DataArray.
                 The axes of the output are ('trainId', 'pulseIndex', 'energy', 'tof').
        """
        tof_idx = [idx
                   for idx, tof_id in enumerate(self.kwargs_adq.keys())
                   if self.mask[tof_id] and tof_id in trace.tof]
        tof_ids = [tof_id
                   for idx, tof_id in enumerate(self.kwargs_adq.keys())
                   if self.mask[tof_id] and tof_id in trace.tof]
        norm = np.stack([v
                         for k, v in self.normalization.items()
                         if k in tof_ids], axis=-1)
        norm *= self.e_transmission[:, tof_idx]
        norm = xr.DataArray(data=norm,
                            dims=('energy', 'tof'),
                            coords=dict(energy=self.energy_axis,
                                        tof=tof_ids))

        def apply_correction(tof_id, tof_trace):
            """
            Apply the energy calibration and transmission correction for a given eTOF.
            """
            logging.info(f"Correcting eTOF {tof_id} ...")
            # get it in the right order
            pulses = tof_trace.transpose('trainId', 'pulseIndex', 'sample')
            coords = pulses.coords
            pulses = pulses.to_numpy()
            # the sample axis to use for the calibration
            n_t, n_p, n_s = pulses.shape
            pulses = np.reshape(pulses, (n_t*n_p, n_s))

            # read model parameters
            if self.model_params[tof_id][0] == 0:
                return np.zeros((n_trains, n_pulses, n_e), dtype=np.float32)
            # create energy axis
            ts = coords['sample']
            e = model(ts, *self.model_params[tof_id])

            # interpolate
            o = np.apply_along_axis(lambda arr: np.interp(self.energy_axis, e[::-1], arr[::-1], left=0, right=0),
                                   axis=1,
                                   arr=pulses)

            n_e = len(self.energy_axis)
            o = np.reshape(o, (n_t, n_p, n_e))
            # subtract offset
            o = o - self.offset[tof_id][None, None, :]
            # apply Jacobian
            o = o*self.jacobian[tof_id][None, None, :]
            np.nan_to_num(o, copy=False)
            # regenerate DataArray
            return xr.DataArray(data=o,
                             dims=('trainId', 'pulseIndex', 'energy'),
                             coords=dict(trainId=coords['trainId'],
                                         pulseIndex=coords['pulseIndex'],
                                         energy=self.energy_axis
                                        )
                            )

        outdata = [apply_correction(tof_id, trace.sel(tof=tof_id))
                   for tof_id in tof_ids]
        outdata = xr.concat(outdata, pd.Index(tof_ids, name="tof"))
        outdata = outdata.transpose('trainId', 'pulseIndex', 'energy', 'tof')
        outdata_corr = outdata/norm.to_numpy()[None, None, :, :]

        return outdata_corr

    def apply(self, run: DataCollection) -> xr.DataArray:
        """
        Collect trace from run *consistently* with the calibration settings and apply
        calibration, offset correction and transmission correction to a new analysis run.
        It is assumed it contains the same eTOF settings.

        Args:
          run: The run to calibrate.

        Returns: The xarray Dataset.
        """
        # fetch the trace with the same settings
        trace = self.load_trace(run)
        # calibrate the spectrum
        spectrum = self.calibrate(trace)
        return spectrum

