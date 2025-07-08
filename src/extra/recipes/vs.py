from typing import Optional, Union, Dict, List, Tuple, Any
from dataclasses import dataclass, asdict, is_dataclass

import itertools
from functools import partial
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import logging

import numpy as np
from extra_data import open_run, by_id, DataCollection, KeyData
import h5py
import xarray as xr
import pandas as pd

from .base import SerializableMixin

from extra.components import AdqRawChannel, XrayPulses, XGM, Scan

from scipy.signal import csd, fftconvolve, firwin, kaiserord, argrelmin

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

def fwhm(x: np.ndarray, y: np.ndarray) -> float:
    """Return the full width at half maximum of x."""
    # half maximum
    half_max = 0.5*np.amax(y)
    idx = np.where(y > half_max)[0]
    if len(idx) < 2:
        return -1
    left_idx = idx[0]
    right_idx = idx[-1]
    return x[right_idx] - x[left_idx]

def deconv(y: np.ndarray, yhat: np.ndarray, e) -> Dict[str, Any]:
    """Given the grating spectrometer data and the virtual spectrometer data,
    calculate the deconvolution between them.
    """
    # do not subtract the mean! This causes cross terms to appear
    y_s = y
    yhat_s = yhat

    # calculate normalization factor to set mean sum y^2 = 1
    n_bins = float(y.shape[1])
    #A = np.mean(np.sum(y_s**2, axis=1), axis=0)
    #Ahat = np.mean(np.sum(yhat_s**2, axis=1), axis=0)

    # sets mean sum y^2 = 1
    #y_s = y_s/np.sqrt(A)
    #yhat_s = yhat_s/np.sqrt(Ahat)

    nperseg = min(len(e), 400)

    # subtract noise spectral density
    # Fourier transforms
    #Yhat = np.fft.fft(yhat_s)
    #Y = np.fft.fft(y_s)
    ## spectral power of the assumed "true" signal (the grating spectrometer data)
    #Syy = np.mean(np.absolute(Y)**2, axis=0)
    #Syh = np.mean(np.conj(Y)*Yhat, axis=0)
    #Shh = np.mean(np.conj(Yhat)*Yhat, axis=0)
    #Syh[np.absolute(Syh) < beta*np.amax(np.absolute(Syh))] = 0.0

    # Welch method
    _, Syy = csd(y_s, y_s, nperseg=nperseg, window="hamming", scaling="spectrum", return_onesided=False, detrend=False)
    _, Syh = csd(y_s, yhat_s, nperseg=nperseg, window="hamming", scaling="spectrum", return_onesided=False, detrend=False)
    _, Shh = csd(yhat_s, yhat_s, nperseg=nperseg, window="hamming", scaling="spectrum", return_onesided=False, detrend=False)
    Syy = np.mean(Syy, axis=0)
    Syh = np.mean(Syh, axis=0)
    Shh = np.mean(Shh, axis=0)
    Syy = Syy.astype(np.complex128)
    Syh = Syh.astype(np.complex128)
    Shh = Shh.astype(np.complex128)

    # there is aliasing in the grating spectrometer at fs/4 and lots of high frequency noise
    # this makes no difference in the spectrum, but harms the estimate significantly

    # approximate transfer function as the ratio of power spectrum densities
    H = np.where(np.abs(Syy) > 0, Syh/Syy, 0)

    # find out where the effect of signal and noise counter-balance each other
    # freqS = np.fft.fftfreq(len(Syy), d=1.0)
    # smoothener = np.exp(-0.5*(np.fft.fftshift(freqS)/(2/len(Syy)))**2)
    # smoothener /= np.sum(smoothener)
    # abs_H = fftconvolve(smoothener, np.fft.fftshift(np.abs(H)), mode='same', axes=-1)
    # abs_H = np.fft.fftshift(abs_H)
    # idx = argrelmin(abs_H)[0]
    # if len(idx) > 0:
    #     cutoff = freqS[idx[0]]
    # else:
    #     # best alternative guess, based on plots above
    #     cutoff = 0.1
    cutoff = 0.1
    # design FIR filter to cut off all high frequency components
    nyq_rate = 0.5
    width = 0.01
    ripple_db = 40.0
    _, beta = kaiserord(ripple_db, width)
    low_pass_filter = firwin(nperseg, cutoff/nyq_rate, window=('kaiser', beta))
    low_pass_response = np.fft.fft(low_pass_filter)

    H2 = np.absolute(H*low_pass_response)**2
    # inputs are normalized, so the normalization of h tells us how much signal there is
    # Yhat = H Y + N
    # mean sum |Yhat|^2 = mean sum |HY|^2 + mean sum H*Y*N + mean sum HYN* + mean sum |N|^2
    # mean sum HYN* = sum H mean YN* = 0 because noise is uncorrelated
    # mean sum |Yhat|^2 = sum |H|^2 mean |Y|^2 + mean sum |N|^2
    # sum mean |N|^2 = n_bins sigma_n^2, if the noise is white
    # Yhat and Y are normalized, so mean sum |Y|^2 = mean sum |Yhat|^2 = n_bins
    # n_bins = sum |H|^2 mean |Y|^2 + n_bins sigma_n^2
    # n_bins = sum |H|^2 Syy + n_bins sigma_n^2
    # sigma_n^2 = 1 - sum |H|^2 Syy/n_bins
    sigma_n = np.real(np.sqrt(np.sum(Shh)/n_bins - np.sum(H2*Syy)/n_bins))
    sigma_s = np.real(np.sqrt(np.sum(H2*Syy)/n_bins))
    snr = sigma_s/sigma_n

    Hinv = 1.0/H
    Hinv[np.isnan(Hinv)] = 0.0
    Shhinv = 1.0/Shh
    Shhinv[np.isnan(Shhinv)] = 0.0
    H2S = H2*Syy
    #G = Hinv * H2S / (H2S + N)
    G = Hinv * H2S * Shhinv
    g = np.real(np.fft.fftshift(np.fft.ifft(G)))

    h = np.real(np.fft.ifft(H*low_pass_response))

    # mismodelling effect
    residual = np.sqrt(np.mean((fftconvolve(y_s, h[np.newaxis,:], mode="same", axes=-1) - yhat_s)**2, axis=0))

    return dict(h=h,
                H=H,
                H2=H2,
                Syy=Syy,
                Syh=Syh,
                snr=snr,
                g=g,
                residual=residual,
               )

def get_resolution(y: np.ndarray, y_hat: np.ndarray, e: np.ndarray,
                   e_center=None, e_width=None):
    """
    Given the true y and the predicted y, together with the energy axis e,
    estimate the impulse response of the system and return the Gaussian fit to it.
    If e_center and e_width are given, multiply the spectra by a box function with given parameters before the resolution estimate.

    Args:
      y: The true spectrum. Shape (N, K).
      y_hat: The predicted spectrum. Shape (N, K).
      e: The energy axis. Shape (K,).
      e_center: If given the energy value, for which to probe the resolution.
      e_width: The width of the energy neighbourhood to probe if e_center is given.

    Returns: The centered energy axis, the impulse response and the fit result.
    """
    y_sel = y
    y_hat_sel = y_hat
    if e_center is not None and e_width is not None:
        f = np.exp(-np.log(2) * (4*np.fabs((e - e_center) / (2.355*e_width))**2)**2)
        y_sel = y_sel*f[np.newaxis,:]
        y_hat_sel = y_hat_sel*f[np.newaxis,:]
    results = deconv(y_sel, y_hat_sel, e)
    h = results["h"]
    de = e[1] - e[0]
    e_axis = np.linspace(-len(h)//2*de, len(h)//2*de, len(h))
    #sel = np.fabs(e_axis) < 3
    #e_axis = e_axis[sel]
    #h = h[sel]
    results["e_axis"] = e_axis
    results["fwhm"] = fwhm(e_axis, results["h"])
    return results

class VSLight(SerializableMixin):
    """
    Find a function mapping the eTOF spectra to the monochromated spectra.

    eTOFs provide a 1D trace for all pulses, which can be transformed
    into traces per pulse by the `AdqRawChannel` Extra component.
    However, the trace sample axis is meaningless and needs to be converted
    into energy using a non-linear function. Usually one data analysis run
    is taken in which the undulator energy is scanned to obtain this map,
    which is then applied in the actual analysis run.

    While `CookieboxCalibration` uses this run only to interpolate the data and
    apply the transmission correction, the `VSLight`
    attempts to unconvolve the instrument response function.

    The concrete steps taken when the object is initialized
    (`obj = VSLight(); obj.setup(calib_run)`) from a calibration run are:

    - Select principal components from the calibration run per eTOF.
    - Apply Automatic Relevance Determination to the calibration runs,
      setting Gaussians at the undulator energies with given width as targets.

    The concrete steps taken when the `obj.apply(other_run)` is called with a
    run to be calibrated are:

    - Use derived map on the intensity axis.

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
    calib_run = open_run(proposal=900485, run=349).select([ts,
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
                                                    baseline=np.s_[:],
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
    # define the target energy axis to interpolate to
    energy_axis = np.linspace(970, 1050, 160)
    # interleaved is only needed if the eTOF control sources are not available
    cal = VSLight(interleaved=True)
    # calculate calibration factors
    cal.setup(run=calib_run,
              energy_axis=energy_axis,
              tof_settings=deepcopy(tof_settings),
              xgm=XGM(calib_run, xgm_source_c),
              energy=calib_run[energy_source, "actualEnergy"]
             )
    cal.to_file("test_vs_p900485r349.h5")
    # when interested in re-using it:
    cal_read = VSLight.from_file("test_vs_p900485r349.h5")
    # apply it
    new_run = open_run(900485, 349)
    cal_data = cal_read.apply(new_run)
    # plot it
    plt.plot(cal_data.sel(tof=4).mean('trainId').mean('pulseIndex').to_numpy())
    ```

    Arguments:
      energy_width: True energy width of input monochromated data, in eV.
      xgm_threshold: Minimum threshold to ignore dark frames in
                     the calibration data (in uJ).
                     Can be 'median' to use the median over the run.
      counting_mode: Calibration in counting mode?
      counting_threshold: How many counts to use to discover a peak?
    """
    def __init__(self,
                 energy_width: float=1.0,
                 xgm_threshold: Union[str, float]='median',
                 counting_mode: bool=False,
                 counting_threshold: float=-8,
                 n_pca_out: int=100,
                 n_pca_in: int=100,
                ):
        self.energy_width = energy_width
        self._xgm_threshold = xgm_threshold
        self._counting_mode = counting_mode
        self._counting_threshold = counting_threshold
        self.n_pca_in = n_pca_in
        self.n_pca_out = n_pca_out

        # empty outputs
        self.tof_data = dict()
        self.y = dict()
        self.model = dict()
        self.pca_x = dict()
        self.pca_y = dict()
        self.resolution = dict()

        # what we need to save it all
        self._version = 1
        self.all_kwargs_adq = ["first_pulse_offset",
                               "cm_period",
                               "interleaved",
                               "single_pulse_length",
                               "extra_cm_period",
                               ]
        self._all_fields = ["n_pca_in",
                            "n_pca_out",
                            "energy_width",
                            "_energy_axis",
                            "kwargs_adq",
                            "_xgm_threshold",
                            "model_params",
                            "calibration_data",
                            "calibration_mean_xgm",
                            "train_ids",
                            "selection",
                            "mask",
                            "e_probe",
                            "e_width",
                            "resolution",
                            #"sources",
                            "calibration_energies",
                            # "pca_x",
                            # "pca_y",
                            # "model",
                            "_counting_mode",
                            "_counting_threshold",
                            "tof_data",
                            "y",
                            "_version",
                           ]
        self._all_pca_fields = ["components_",
                                "explained_variance_",
                                "explained_variance_ratio_",
                                "singular_values_",
                                "mean_",
                                "var_",
                                "noise_variance_",
                                "n_components_",
                                "n_samples_seen_",
                                #"batch_size_",
                                "n_features_in_",
                                #"feature_names_in_",
                                "whiten"
                               ]
        self._all_model_fields = ["coef_",
                                  "alpha_",
                                  "lambda_",
                                  "sigma_",
                                  "scores_",
                                  "n_iter_",
                                  "intercept_",
                                  "X_offset_",
                                  "X_scale_",
                                  #"n_features_in_",
                                  #"feature_names_in_"
                                 ]
    def _asdict(self):
        """
        Return a serializable dict.
        """
        d = {k: v for k, v in self.__dict__.items() if k in self._all_fields}
        d["pca_x"] = {}
        for tof_id, v in self.pca_x.items():
            d["pca_x"][tof_id] = {name: getattr(v, name) for name in self._all_pca_fields}
        d["pca_y"] = {}
        for tof_id, v in self.pca_y.items():
            d["pca_y"][tof_id] = {name: getattr(v, name) for name in self._all_pca_fields}
        d["model"] = {}
        for tof_id, v in self.model.items():
            d["model"][tof_id] = dict()
            for e, _ in enumerate(self.model[tof_id].estimators_):
                d["model"][tof_id][e] = {name: getattr(self.model[tof_id].estimators_[e], name)
                                         for name in self._all_model_fields}
        return d
    def _fromdict(self, all_data):
        """
        Actions to do after loading from file.
        """
        from sklearn.decomposition import IncrementalPCA
        from sklearn.linear_model import ARDRegression
        from .vs_utils import MultiOutputGenericWithStd
        tof_ids = all_data["kwargs_adq"].keys()
        self.pca_x = dict()
        self.pca_y = dict()
        self.model = dict()
        for k, v in all_data.items():
            if k == "pca_x":
                for i in tof_ids:
                    self.pca_x[i] = IncrementalPCA()
                    for par in self._all_pca_fields:
                        setattr(self.pca_x[i], par, v[i][par])
            elif k == "pca_y":
                for i in tof_ids:
                    self.pca_y[i] = IncrementalPCA()
                    for par in self._all_pca_fields:
                        setattr(self.pca_y[i], par, v[i][par])
            elif k == "model":
                for i in tof_ids:
                    self.model[i] = MultiOutputGenericWithStd(ARDRegression(tol=1e-8, verbose=True), n_jobs=8)
                    self.model[i].estimators_ = list()
                    for e in v[i].keys():
                        self.model[i].estimators_ += [ARDRegression(tol=1e-8, verbose=True)]
                        for par in self._all_model_fields:
                            setattr(self.model[i].estimators_[-1], par, v[i][e][par])
            else:
                setattr(self, k, v)
        # fix some dicts
        self.mask = {int(k): v for k, v in self.mask.items()}
        self.kwargs_adq = {int(k): v for k, v in self.kwargs_adq.items()}
        # for tof_id in self._kwargs_adq.keys():
        #     self.kwargs_adq[tof_id]["source"] = self.kwargs_adq[tof_id]["source"].decode("utf-8")
        #     self.kwargs_adq[tof_id]["name"] = self.kwargs_adq[tof_id]["name"].decode("utf-8")

    def setup(self,
              run: DataCollection,
              energy_axis: np.ndarray,
              tof_settings: Dict[int, AdqRawChannel],
              xgm: XGM,
              energy_for_analog_mode: Optional[KeyData]=None,
              scan_for_counting_mode: Optional[Scan]=None,
              ):
        """
        Derive calibrations and store in this object for later usage wth `apply`.

        Args:
          run: The calibration run.
          energy_axis: Energy axis in eV to interpolate eTOF to.
          tof_settings: Dictionary with a TOF label as a key (0,1,2, ...).
                        Each value is the `AdqRawChannel` object for that eTOF.
          energy_for_analog_mode: In analog mode, KeyData providing the monochromated energy.
                For example: `calib_run["SA3_XTD10_MONO/MDL/PHOTON_ENERGY", "actualEnergy.value"]`
          xgm: The XGM object used to apply a pulse energy selection.
               For example: `XGM(run, "SQS_DIAG1_XGMD/XGM/DOOCS")`
          scan_for_counting_mode: In counting mode, this must be provided to bin a run and average it.
        """
        # base properties
        self._run = run
        self._energy_axis = energy_axis
        self._tof_settings = tof_settings

        self._xgm = xgm
        self._energy = energy_for_analog_mode
        self._scan_for_counting = scan_for_counting_mode

        if self._counting_mode and self._scan_for_counting is None:
            raise ValueError("In counting mode, `scan_for_counting must be passed as a parameter.")

        # now do the full analysis, step by step
        # update eTOF data reading objects
        self.update_tof_settings()

        # get XGM and auxiliary data
        self.update_metadata()

        # get data
        self.update_data()

        # fit
        self.update_fit()

        # now we can use the apply method
        logging.info("Ready to apply energy calibration and transmission correction to analysis data ...")

    # these setters allow one to update
    # some of the input properties and rerun only the necessary steps
    # for recalibration
    #
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
        self.update_data()
        self.update_fit()

    def set_tof_settings(self, value: Dict[int, AdqRawChannel]):
        """
        Update the eTOF settings and recompute.

        Args:
          value: The new eTOF settings as explained in the constructor docstring.
        """
        self._tof_settings = value
        self.update_tof_settings()
        self.update_data()
        self.update_fit()

    @property
    def counting_threshold(self) -> float:
        return self._counting_threshold

    def set_counting_threshold(self, value: float):
        """
        Update counting threshold.

        Args:
          value: Counting threshold.
        """
        self._counting_threshold = value
        self.update_data()
        self.update_fit()

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
        # get energies
        if not self._counting_mode:
            self.calibration_energies = self._energy.xarray()
        else:
            self.calibration_energies = self._scan_for_counting.positions

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

    def plot_calibration_data(self):
        """Plot data for checks.
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
        tof_ids = list(self.kwargs_adq.keys())
        energies = self.calibration_energies
        n_energies = len(energies)
        fig, axes = plt.subplots(nrows=4, ncols=4, clear=True, figsize=(20,20))
        for tof_id in tof_ids:
            data = self.tof_data[tof_id]
            ax = axes.flatten()[tof_id]
            ax.imshow(-data,
                    aspect=data.shape[-1]/n_energies,
                      norm=LogNorm()
                     )
            #ax.set_yticklabels((energies.to_numpy()+0.5).astype(int))
            ax.set_title(f'TOF {tof_id}')
        plt.show()


    def plot_response(self):
        """Plot impulse response function."""
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(nrows=len(self.e_probe), ncols=1, clear=True, figsize=(10, 5*len(self.e_probe)))
        for i, e in enumerate(self.e_probe):
            for tof_id in self.resolution.keys():
                axes[i].plot(self.resolution[tof_id][i]['e_axis'], self.resolution[tof_id][i]['h'], lw=2, label=f"eTOF {tof_id}")
            axes[i].set(xlabel="Energy [eV]",
                        ylabel="Instrument response [a.u.]",
                        title=f"Energy bin at {e} eV")
            axes[i].legend(ncols=2, frameon=False)
        plt.tight_layout()
        plt.show()

    def plot_resolution(self, tof_ids=None):
        """Plot resolution."""
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(10, 8))
        if tof_ids is None:
            tof_ids = self.resolution.keys()
        for tof_id in tof_ids:
            plt.errorbar(self.e_probe, [self.resolution[tof_id][i]['fwhm'] for i, _ in enumerate(self.resolution[tof_id].keys())],
                         xerr=self.e_width/2, alpha=0.5,
                         fmt='o',
                         lw=2, label=f"eTOF {tof_id}")
        plt.xlabel("Energy [eV]")
        plt.ylabel("Resolution [eV]")
        plt.legend(frameon=False, ncols=2)
        plt.tight_layout()
        plt.show()

    def update_data(self):
        """
        Given calibrated data, apply a averaging and define targets.
        """
        # average data for each energy slice
        logging.info("Reading calibration data ... (this takes a while)")
        tof_ids = list(self._tof.keys())

        tof_data = self._tof
        xgm_data = self._xgm_data
        xgm_threshold = self._xgm_threshold

        # select XGM
        sel_xgm = xgm_data > xgm_threshold
        sel_xgm_data = xgm_data[sel_xgm]

        # match them by train ID
        train_ids = np.unique(xgm_data.coords["trainId"].to_numpy())
        n_pulses = self._pulses.pulse_counts().max()

        self.train_ids = train_ids
        self.selection = sel_xgm
        self.calibration_mean_xgm = sel_xgm_data.to_numpy()
        if not self._counting_mode:
            e_data = self.calibration_energies.expand_dims(dim={"pulseIndex": np.arange(n_pulses)}).transpose('trainId', 'pulseIndex')
            e_data = e_data.sel(trainId=train_ids)
            e_data = e_data.stack(pulse=('trainId', 'pulseIndex'))
            e_data = e_data[sel_xgm]
            self.calibration_energies = e_data.to_numpy()

        for tof_id in self.kwargs_adq.keys():
            self.tof_data[tof_id], self.y[tof_id] = self.load_data_for(tof_id)

    def guess_components(self, data, n_min):
        from sklearn.decomposition import IncrementalPCA
        pca_threshold = 0.9
        bare_minimum = min(data.shape[0], min(200, data.shape[-1]))
        if n_min > bare_minimum:
            n_min = bare_minimum
        pca_test = IncrementalPCA(n_components=bare_minimum, whiten=True, batch_size=600)
        perm = np.random.permutation(data.shape[0])
        n_part = min(500, data.shape[0])
        for i, batch_idx in enumerate(np.array_split(perm, data.shape[0]//n_part)[:20]):
            #logging.info(f"Partial fit y {i} ...")
            pca_test.partial_fit(data[batch_idx, :])
        #pca_test.fit(data)
        n_pca_y = np.where(np.cumsum(pca_test.explained_variance_ratio_) > pca_threshold)[0]
        if len(n_pca_y) > 0:
            n_pca_y = n_pca_y[0]
        else:
            n_pca_y = len(pca_test.explained_variance_ratio_)
        n_pca_y = max(n_min, n_pca_y)
        del pca_test
        return n_pca_y

    def load_data_for(self, tof_id):
        if self._counting_mode:
            energy_ids = np.arange(len(self._scan_for_counting.positions))
            tof_data = list()
            y = list()
            for energy_id in energy_ids:
                energy, train_ids = self._scan_for_counting.steps[energy_id]
                train_ids = self.train_ids[np.isin(self.train_ids, train_ids)]
                this_tof_data = self._tof[tof_id].select_trains(by_id[train_ids]).pulse_data(pulse_dim='pulseIndex')
                this_tof_data = this_tof_data.isel(sample=slice(0, self.kwargs_adq[tof_id]["single_pulse_length"]))
                edges = self._tof[tof_id].find_edges(this_tof_data, threshold=self.counting_threshold)
                ts = np.arange(0, this_tof_data.shape[-1]+1)
                hist, _ = np.histogram(edges.edge, bins=ts)
                mu = 1.0/(self.energy_width*np.sqrt(2*np.pi))
                this_y = mu*np.exp(-0.5*(self.energy_axis - energy)**2/((self.energy_width/2.355)**2))
                tof_data += [-hist]
                y += [this_y]
            tof_data = np.stack(tof_data, axis=0)
            y = np.stack(y, axis=0)
        else:
            tof_data = self._tof[tof_id].select_trains(by_id[self.train_ids]).pulse_data(pulse_dim='pulseIndex')
            tof_data = tof_data.isel(sample=slice(0, self.kwargs_adq[tof_id]["single_pulse_length"]))
            tof_data = -tof_data[self.selection].to_numpy()
            mu = self.calibration_mean_xgm[:, None]/(self.energy_width*np.sqrt(2*np.pi))
            y = mu*np.exp(-0.5*(self.energy_axis[None, :] - self.calibration_energies[:, None])**2/((self.energy_width/2.355)**2))
        return tof_data, y

    def update_fit(self):
        """
        Fit TOF data to a Gaussian and collect results.
        """
        from sklearn.decomposition import IncrementalPCA
        from sklearn.linear_model import ARDRegression
        from .vs_utils import MultiOutputGenericWithStd
        logging.info("Fit PCA+ARD...")
        pca_threshold = 0.90

        e = self.calibration_energies
        xgm_data = self.calibration_mean_xgm

        # binning for energy
        e_min = e.min()
        e_max = e.max()
        e_probe = np.linspace(e_min, e_max, 5)
        e_width = (e_max - e_min)/(len(e_probe)-1)
        self.e_width = e_width
        self.e_probe = e_probe

        # how many components in y?
        for tof_id in self.kwargs_adq.keys():
            tof_data, y = self.tof_data[tof_id], self.y[tof_id]
            perm = np.random.permutation(tof_data.shape[0])

            logging.info(f"Transform target for eTOF {tof_id} ...")
            # how many target components?
            n_part = min(500, y.shape[0])

            n_pca_y = min(self.n_pca_out, n_part) #self.guess_components(y, 40)
            logging.info(f"Components for targets in {tof_id}: {n_pca_y}")
            # fit PCA on target
            logging.info(f"Fit PCA for y ...")
            self.pca_y[tof_id] = IncrementalPCA(n_components=n_pca_y, whiten=True, batch_size=5*n_pca_y)
            for i, batch_idx in enumerate(np.array_split(perm, y.shape[0]//n_part)[:20]):
                logging.info(f"Partial fit y {i} ...")
                self.pca_y[tof_id].partial_fit(y[batch_idx, :])
            tr_y = self.pca_y[tof_id].transform(y)

            logging.info(f"Transform source for eTOF {tof_id} ...")
            # how many source components?
            n_part = min(500, tof_data.shape[0])

            n_pca_x = min(self.n_pca_in, n_part) #self.guess_components(tof_data, 100)
            logging.info(f"Components for sources in {tof_id}: {n_pca_x}")
            # pipeline for source
            self.pca_x[tof_id] = IncrementalPCA(n_components=n_pca_x, whiten=True, batch_size=5*n_pca_x)
            self.model[tof_id] = MultiOutputGenericWithStd(ARDRegression(tol=1e-8, verbose=True), n_jobs=8)
            logging.info(f"Fit PCA for x ...")
            for i, batch_idx in enumerate(np.array_split(perm, tof_data.shape[0]//n_part)[:20]):
                logging.info(f"Partial fit x {i} ...")
                self.pca_x[tof_id].partial_fit(tof_data[batch_idx, :])
            tr_x = self.pca_x[tof_id].transform(tof_data)
            logging.info(f"Fit model ...")
            self.model[tof_id].fit(tr_x, tr_y)

            # predict
            logging.info(f"Predict ...")
            tr_y_hat = self.model[tof_id].predict(tr_x)
            y_hat = self.pca_y[tof_id].inverse_transform(tr_y_hat)

            # calculate resolution
            logging.info(f"Calculate resolution ...")
            self.resolution[tof_id] = dict()
            for i, e_center in enumerate(e_probe):
                self.resolution[tof_id][i] = get_resolution(y, y_hat, self.energy_axis,
                                                            e_center, e_width)

    def load_trace(self, run: DataCollection, **extra_kwargs_adq: Dict[str, Any]) -> xr.DataArray:
        """
        Only load region of interest for the same settings in a new run and output a Dataset with it.
        This is the recommended way to load data from a new run before applying the calibration.

        Args:
          run: The run to calibrate.
          extra_kwargs_adq: Extra keyword arguments for the `AdqRawChannel` object if one wishes to override settings.

        Returns:
          A [DataArray][xarray.DataArray] with the traces containing axes `('trainId', 'pulseIndex', 'sample', 'tof')`.
        """

        tof_ids = sorted([tof_id for idx, tof_id in enumerate(self.kwargs_adq.keys()) if self.mask[tof_id]])
        def fetch(tof_id):
            """
            Apply the energy calibration and transmission correction for a given eTOF.
            """
            logging.info(f"Fetch data from eTOF {tof_id} ...")
            kwargs = {k: v for k, v in self.kwargs_adq[tof_id].items()}
            del kwargs["name"]
            del kwargs["source"]
            kwargs.update(extra_kwargs_adq)
            tof = AdqRawChannel(run,
                                self.kwargs_adq[tof_id]["name"],
                                digitizer=self.kwargs_adq[tof_id]["source"],
                                **kwargs)
            pulses = -tof.pulse_data(pulse_dim='pulseIndex').unstack('pulse').transpose('trainId', 'pulseIndex', 'sample')
            return pulses

        outdata = [fetch(tof_id)
                   for tof_id in tof_ids]
        outdata = xr.concat(outdata, pd.Index(tof_ids, name="tof"))
        outdata = outdata.transpose('trainId', 'pulseIndex', 'sample', 'tof')
        return outdata

    def calibrate(self, trace: xr.DataArray) -> xr.DataArray:
        """
        Takes a trace separated with axes `('trainId', 'pulseIndex', 'sample', 'tof')`,
        as given by `load_trace` and applies the calibration.
        The method `apply` provides a direct way to retrieve the trace and calbrate it.
        The methods `load_trace` and `calibrate` allow one to apply an intermediate processing
        step between them.

        Args:
          trace: A pre-processed trace retrieved with the *same* `AdqRawChannel` settings as this calibration object.
                 Its axes are expected to be ('trainId', 'pulseIndex', 'sample', 'tof').
                 It is recommended to use always `load_trace` to obtain this.

        Returns:
          The calibrated data as a [DataArray][xarray.DataArray]. The axes of the output are `('trainId', 'pulseIndex', 'energy', 'tof')`.
        """
        tof_idx = [idx
                   for idx, tof_id in enumerate(self.kwargs_adq.keys())
                   if self.mask[tof_id] and tof_id in trace.tof]
        tof_ids = [tof_id
                   for idx, tof_id in enumerate(self.kwargs_adq.keys())
                   if self.mask[tof_id] and tof_id in trace.tof]

        def apply_correction(tof_id, tof_trace):
            """
            Apply the energy calibration and transmission correction for a given eTOF.
            """
            logging.info(f"Correcting eTOF {tof_id} ...")
            # get it in the right order
            pulses = tof_trace.transpose('trainId', 'pulseIndex', 'sample')
            pulses = pulses.isel(sample=slice(0, self.kwargs_adq[tof_id]["single_pulse_length"]))
            coords = pulses.coords
            pulses = pulses.to_numpy()
            # the sample axis to use for the calibration
            n_t, n_p, n_s = pulses.shape

            # apply model
            pulses = np.reshape(pulses, (n_t*n_p, n_s))
            pca_x = self.pca_x[tof_id].transform(pulses)
            pca_y = self.model[tof_id].predict(pca_x)
            o = self.pca_y[tof_id].inverse_transform(pca_y)
            n_e = len(self.energy_axis)
            o = np.reshape(o, (n_t, n_p, n_e))
            # regenerate DataArray
            o = xr.DataArray(data=o,
                             dims=('trainId', 'pulseIndex', 'energy'),
                             coords=dict(trainId=coords['trainId'],
                                         pulseIndex=coords['pulseIndex'],
                                         energy=self.energy_axis
                                        )
                            )
            return o

        outdata = [apply_correction(tof_id, trace.sel(tof=tof_id))
                   for tof_id in tof_ids]
        outdata = xr.concat(outdata, pd.Index(tof_ids, name="tof"))
        outdata = outdata.transpose('trainId', 'pulseIndex', 'energy', 'tof')

        return outdata

    def apply(self, run: DataCollection) -> xr.DataArray:
        """
        Collect trace from run *consistently* with the calibration settings and apply
        calibration, offset correction and transmission correction to a new analysis run.
        It is assumed it contains the same eTOF settings.

        Args:
          run: The run to calibrate.

        Returns:
          The calibrated data.
        """
        # fetch the trace with the same settings
        trace = self.load_trace(run)
        # calibrate the spectrum
        spectrum = self.calibrate(trace)
        return spectrum
