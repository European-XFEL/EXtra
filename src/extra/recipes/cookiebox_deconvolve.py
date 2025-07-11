from typing import Tuple, Union, Optional, List
from functools import partial
from scipy.linalg import convolution_matrix
from scipy.signal import fftconvolve
import numpy as np
import xarray as xr
from .base import SerializableMixin

import logging

from extra.components import AdqRawChannel, Scan
from extra_data import by_id


def clip_at_zero(x):
    return np.clip(x, a_min=0, a_max=None)

def nn_deconvolution(data: np.ndarray, h: np.ndarray, n_iter: int=4000, n_shift: int=0, nonneg: bool=True):
    r'''
    Chambolle-Pock algorithm for the minimization of the objective function
        $\min_x 1/2||h \otimes x - data||^2 + I(x)$

    This is formulated as the following primal optimization objective:
        $min_x G(x) + I(x)$       (P)

    Where:
        $G(x) = 1/2||A x - b||^2$
        $A =$ matrix representation of the convolution
        $b =$ input data
        $F(y) = I(y)$, which is the non-negatvity constraint
        $y = x $


    References:
    [1] N. Parikh and S. Boyd. Proximal Algorithms.
        Foundations and Trends in Optimization,
        volume 1, issue 3, pp. 127-239, 2014.

    Args:
        h: Impulse response function of system to deconvolve as a 1D array.
        data: Data to deconvolve with shape (n_samples) if a single shot is given or
              (n_shots, n_samples) for multiple shots. It must have n_samples smaller or equal to len(h).
        n_iter: Number of iterations.
        n_shift: At which point the impulse response function start in h.
        nonneg: If True, impose non-negativity constraint.
    '''

    # if data includes multiple shots, it will be (n_shots, n_samples)
    # we need batch multplication with A, so we simply use data.T,
    # which is shape (n_samples, n_shots), so:
    # A is (n_samples, n_samples)
    # x is (n_samples, n_shots)
    # A @ x is (n_samples, n_shots)
    # => the desired result would be trasposed
    is1d = False
    if len(data.shape) == 1:
        is1d = True
        b = data[:, None]
    elif len(data.shape) == 2:
        b = np.transpose(data)
    else:
        raise ValueError(f"Input data must either be 1D or 2D. The shape sent is {data.shape}.")

    N = b.shape[0]

    # make them the same size
    if len(h) < N:
        hl = np.concatenate((h, np.zeros(N - len(h))))
    elif len(h) >= N:
        hl = np.copy(h[:N])
    hl = hl/np.sum(hl)

    # roll over to match the effect of fftcovolve
    #hl = np.roll(hl, -N//2)
    hl = np.roll(hl, -n_shift+len(hl)//2, axis=-1)
    # create the matrix representing the convolution
    # this is expensive, but no easy way comes to mind
    A = convolution_matrix(hl, len(hl), mode='same')

    I = np.eye(N)

    rho = 0.1
    G = np.linalg.pinv(A.T @ A + rho*I)
    x = np.copy(b)
    z = np.copy(b)
    u = np.copy(b)

    for k in range(0, n_iter):
        # consensus ADMM
        x = G @ (2*A.T @ b + rho*(z - u))
        z = np.clip(x + u, a_min=0, a_max=None)
        u = u + (x - z)

        # Calculate norms
        if k%20 == 0:
            fidelity = np.mean(0.5*np.sqrt(np.mean((A @ x - b)**2, axis=0)), axis=-1)
            nn = np.mean(np.sqrt(np.mean(np.clip(-x, a_min=0, a_max=None)**2, axis=0)), axis=-1)
            logging.info("[%4d] : fidelity %e \t mse of negative %e" %(k,fidelity, nn))
    x = z
    if is1d:
        return x[:,0]

    return np.transpose(x)

def tv_deconvolution(data: np.ndarray, h: np.ndarray, Lambda: float=1e-5, n_iter: int=5000, n_shift: int=0, nonneg: bool=True):
    r'''
    Chambolle-Pock algorithm for the minimization of the objective function
        $\min_x 1/2||h \otimes x - data||^2 + Lambda TV(x)$

    This is formulated as the following primal optimization objective:
        $min_x G(x) + Lambda * F(K x)$       (P)

    Where:
        $G(x) = 1/2||A x - b||^2$
        $A =$ matrix representation of the convolution
        $b =$ input data
        $F(y) = ||y||_1$, which is the 1-norm of y
        $y = K x =$ difference between one element of x and the next
        $K =$ gradient matrix

    This is done, so that $TV(x) = F(K(x))$

    The problem is solved using the Chambolle-Pock algorithm, which uss the primal-dual formulation
    of the optimization objective:
        min_x max_y G(x) + <K x, y> - F*(y)

    The algorithm to solve this problem is:
        $x_{i+1} = (I + \tau \partial G)^{-1} (x_i - \tau K^\star y_i)$
        $\bar{x}_{i+1} = x_{i+1} + \omega(x_{i+1} - x_i)$
        $y_{i+1} = (I + \sigma \partial F^\star)^{-1}(y_i + \sigma K \bar{x})$

    It converges if $\tau \sigma L^2 \leq 1$.

    References:
    [1] Chambolle, A., Pock, T. A First-Order Primal-Dual Algorithm for Convex
        Problems with Applications to Imaging.
        J Math Imaging Vis 40, 120–145 (2011).
        https://doi.org/10.1007/s10851-010-0251-1
    [2] N. Parikh and S. Boyd. Proximal Algorithms.
        Foundations and Trends in Optimization,
        volume 1, issue 3, pp. 127-239, 2014.

    Args:
        h: Impulse response function of system to deconvolve as a 1D array.
        data: Data to deconvolve with shape (n_samples) if a single shot is given or
              (n_shots, n_samples) for multiple shots. It must have n_samples smaller or equal to len(h).
        Lambda : Weight of the TV penalty.
        n_iter: Number of iterations.
        n_shift: At which point the impulse response function start in h.
        nonneg: If True, impose non-negativity constraint.
    '''

    # if data includes multiple shots, it will be (n_shots, n_samples)
    # we need batch multplication with A, so we simply use data.T,
    # which is shape (n_samples, n_shots), so:
    # A is (n_samples, n_samples)
    # x is (n_samples, n_shots)
    # A @ x is (n_samples, n_shots)
    # => the desired result would be trasposed
    is1d = False
    if len(data.shape) == 1:
        is1d = True
        b = data[:, None]
    elif len(data.shape) == 2:
        b = np.transpose(data)
    else:
        raise ValueError(f"Input data must either be 1D or 2D. The shape sent is {data.shape}.")

    N = b.shape[0]

    # make them the same size
    if len(h) < N:
        hl = np.concatenate((h, np.zeros(N - len(h))))
    elif len(h) >= N:
        hl = np.copy(h[:N])
    hl = hl/np.sum(hl)

    # roll over to match the effect of fftcovolve
    #hl = np.roll(hl, -N//2)
    hl = np.roll(hl, -n_shift+len(hl)//2, axis=-1)
    # create the matrix representing the convolution
    # this is expensive, but no easy way comes to mind
    A = convolution_matrix(hl, len(hl), mode='same')
    # ensure the convergence criteria is satisfied
    # sigma * tau L^2 <= 1
    L = np.linalg.norm(A)

    sigma = 0.9/L
    tau = 0.9/L

    # we need the proximal operator of F and G
    # The function G(x) is as follows:
    # $G = 1/2 ||A x - b||^2 = 1/2 x^T A^T A x - b^T A x + 1/2 b^T b$
    # Its proximal operator is tabulated (see Ref. [2], sec. 6.1.1):
    # $\text{prox}_{\tau G} (v) = (I + \tau A^T A)^{-1} (v + \tau A^T b)
    I = np.eye(N)
    GG = np.linalg.pinv(I + tau*(A.T @ A))
    ATb = np.ascontiguousarray((A.T @ b))
    if nonneg:
        prox_G = lambda v: np.clip(GG @ (v + tau*ATb), a_min=0, a_max=None)
    else:
        prox_G = lambda v: GG @ (v + tau*ATb)

    # The proximal operator of $F(y) = ||y||_1$ is also tabulated in
    # Ref. [2], sec. 6.5.2.
    # It is the soft thresholding function
    # $\text{prox}_{\sigma F}(v) = (v - \sigma)_+ - (-v - \sigma)_+$
    prox_F = lambda v: np.clip(v - sigma*Lambda, a_min=0, a_max=None) - np.clip(-v - sigma*Lambda, a_min=0, a_max=None)

    # But we need instead the proximal operator of $F^\star$, which we can calculate
    # using the Moreau decomposition (Ref. [2], sec. 2.5)
    # $\text{prox}_{\sigma F^\star}(v) = v - \text{prox}_{\sigma F}(v)
    prox_Fs = lambda v: v - prox_F(v)

    # K is the gradient operator in 1D
    #K = convolution_matrix(np.array([1.0, -1.0])/2.0, N, mode='same')
    gradT = lambda v: np.concatenate([np.diff(v, axis=0)/2.0, np.zeros((1, v.shape[-1]))], axis=0)
    grad = lambda v: np.concatenate([np.zeros((1, v.shape[-1])), -np.diff(v, axis=0)/2.0], axis=0)

    # initialize the temporary variables
    x = np.copy(b)
    x_bar = np.copy(x)

    # the dual variable
    y = np.copy(x)

    for k in range(0, n_iter):
        # no need to cache x_bar and x_old if omega = 1
        x_old = np.copy(x)
        #x = prox_G(x - tau*(K.T @ y))
        x = prox_G(x - tau*gradT(y))
        x_bar = x + 1*(x - x_old)

        #y = prox_Fs(y + sigma*(K @ x_bar))
        y = prox_Fs(y + sigma*grad(x_bar))

        # Calculate norms
        if k % 100 == 0:
            fidelity = np.mean(0.5*np.sqrt(np.mean((A @ x - b)**2, axis=0)), axis=-1)
            tv = np.mean(np.sum(np.abs(grad(x)), axis=0), axis=-1)
            energy = 1.0*fidelity + Lambda*tv
            logging.info("[%4d] : energy %e \t fidelity %e \t TV %e" %(k,energy,fidelity,tv))
    if is1d:
        return x[:,0]

    return np.transpose(x)

def std_deconvolution(data: np.ndarray, h: np.ndarray, snr: float=5.0, n_shift: int=0):
    r'''
    Standard deconvolution.

    Args:
        h: Impulse response function of system to deconvolve as a 1D array.
        data: Data to deconvolve with shape (n_samples) if a single shot is given or
              (n_shots, n_samples) for multiple shots. It must have n_samples smaller or equal to len(h).
        n_shift: At which sample does the impulse response of h start.
    '''

    # if data includes multiple shots, it will be (n_shots, n_samples)
    # we need batch multplication with A, so we simply use data.T,
    # which is shape (n_samples, n_shots), so:
    # A is (n_samples, n_samples)
    # x is (n_samples, n_shots)
    # A @ x is (n_samples, n_shots)
    # => the desired result would be trasposed
    is1d = False
    if len(data.shape) == 1:
        is1d = True
        b = data[:, None]
    elif len(data.shape) == 2:
        b = np.transpose(data)
    else:
        raise ValueError(f"Input data must either be 1D or 2D. The shape sent is {data.shape}.")

    N = b.shape[0]

    # make them the same size
    if len(h) < N:
        hl = np.concatenate((h, np.zeros(N - len(h))))
    elif len(h) >= N:
        hl = np.copy(h[:N])
    hl = hl/np.sum(hl)

    # roll over to match the effect of fftcovolve
    #hl = np.roll(hl, -N//2)
    hl = np.roll(hl, -n_shift, axis=-1)

    H = np.fft.fft(hl)
    if np.isinf(snr):
        W = 1.0/(H+1e-20)
    else:
        snr2 = snr**2
        H2 = np.abs(H)**2
        W = snr2*np.conj(H)/(snr2*H2 + 1)

    x = np.fft.ifft(W[:,None]*np.fft.fft(b, axis=0), axis=0)
    x = np.abs(x)

    if is1d:
        return x[:,0]

    return np.transpose(x)

def comb(period: List[int], amplitude: List[float]):
    c = list()
    for T, A in zip(period, amplitude):
        delta = np.array([1.0]+[0.0]*(T-1))
        c += [A*delta]
    c = np.concatenate(c)
    return np.concatenate((np.zeros_like(c), c))

class TOFResponse(SerializableMixin):
    """
    Given a run with the Cookiebox in counting mode, obtain its impulse response.

    Example usage:
    ```python
    tof_id = 4
    channel = "3_A"
    digitizer = "SQS_DIGITIZER_UTC4/ADC/1:network"
    calib_run = open_run(proposal=900485, run=320)
    calib_run = calib_run.select([digitizer], require_all=True)
    tof = AdqRawChannel(calib_run,
                        channel=channel,
                        digitizer=digitizer,
                        first_pulse_offset=23300,
                        single_pulse_length=600,
                        interleaved=True,
                        baseline=np.s_[:20000],
                        )
    response = TOFResponse()
    response.setup(tof)
    response.to_file("tof4.h5")
    h = response.get_response(type="aligned")
    ```

    This could also be coupled with the
    [CookieboxCalibration][extra.recipes.CookieboxCalibration] object to
    deconvolve and denoise spectra before calibration.

    ```python
    # open a new run
    run = open_run(proposal=900485, run=349)
    # get the calibration constants
    cal = CookieboxCalibration.from_file("my_calibration.h5")
    # load the trace for this new run, exactly as required for calibration
    trace = cal.load_trace(run)
    # apply the deconvolution using this TOFResponse object
    response = TOFResponse.from_file("tof4.h5")
    trace.loc[dict(tof=4)] = response.apply(trace.sel(tof=4))
    # now apply the calibration:
    spectrum = cal.calibrate(trace)
    ```

    Args:
      threshold: Threshold to use when identifying edges.
      n_peaks: Number of peaks to select, if positive. Otherwise no peak selection is made.
      n_samples: Total number of samples use for impulse response.
      n_filter: Require no edges to be found before sample n_filter and after n_samples-n_filter.
      roi: Region of interest. Keep None to use the full trace.
    """
    def __init__(self, threshold: int=-7,
                 n_peaks: Optional[int]=1,
                 n_samples: int=400,
                 n_filter: int=100,
                 roi: Optional[slice]=None):
        self.threshold = threshold
        if n_peaks is None:
            n_peaks = -1
        self.n_peaks = n_peaks
        self.n_samples = n_samples
        self.n_filter = n_filter
        self.h = None
        self.h_digital = None
        self.h_analog = None
        self.roi = roi
        self._version = 1
        self._all_fields = [
                            "h",
                            "h_digital",
                            "h_analog",
                            "threshold",
                            "n_samples",
                            "n_filter",
                            "n_peaks",
                            "_version",
                           ]

    def _asdict(self):
        """
        Return a serializable dict.
        """
        d = {k: v for k, v in self.__dict__.items() if k in self._all_fields}
        return d
    def _fromdict(self, all_data):
        """
        Actions to do after loading from file.
        """
        for k, v in all_data.items():
            setattr(self, k, v)

    def setup(self,
              tof: AdqRawChannel,
              ):
        """
        Given a [AdqRawChannel][extra.components.AdqRawChannel] object,
        calculate this eTOF's impulse response function.

        Args:
          tof: `AdqRawChannel` used to access data for a given eTOF.
        """

        # get pulse data
        logging.info("Get pulse data ...")
        this_tof_data = tof.pulse_data(pulse_dim='pulseIndex')
        if self.roi is not None:
            this_tof_data = this_tof_data.isel(sample=self.roi)

        # get edges
        logging.info("Get edges ...")
        edges_pos, edges_amp = tof.find_edge_array(this_tof_data,
                                                   labelled=False,
                                                   threshold=self.threshold)

        if edges_pos.shape[-1] == 0:
            raise ValueError(f"No peaks found.")

        # require a single edge within the pulse
        edges_sel = ((edges_pos[:,0] >= self.n_filter)
                     & (edges_pos[:,0] <= self.n_samples - self.n_filter))
        if self.n_peaks > 0:
            edges_sel = edges_sel & (np.sum(~np.isnan(edges_pos), axis=-1) == self.n_peaks)

        if np.sum(edges_sel) == 0:
            raise ValueError(f"No peaks found after n_filter={self.n_filter} "
                             f"and before n_samples-n_filter={self.n_samples-self.n_filter} "
                             f"with n_peaks requirement.")

        # build axis
        ts = np.arange(0, self.n_samples)
        h_axis = np.arange(0, self.n_samples)

        # find edges and build both digital and analog response functions
        logging.info("Build digital signal ...")
        edges = tof.find_edges(this_tof_data, threshold=self.threshold)
        ts1 = np.arange(0, self.n_samples+1)
        h_digital, _ = np.histogram(edges.edge, bins=ts1)
        h_digital = h_digital.astype(float)/np.amax(h_digital)
        self.h_digital = h_digital

        logging.info("Build analog signal ...")
        h_analog, _ = np.histogram(edges.edge, bins=ts1, weights=-edges.amplitude)
        h_analog /= np.amax(h_analog)
        self.h_analog = h_analog

        # now align traces
        logging.info("Build aligned signal ...")
        aligned = list()
        for p, e in zip(this_tof_data.pulse[edges_sel].to_numpy(), edges_pos[edges_sel]):
            # select region of interest in the trace
            e0 = int(e[0])
            roi_aligned = -this_tof_data.sel(pulse=p).isel(sample=slice(e0-self.n_filter, e0+self.n_samples-self.n_filter))
            xi = np.arange(self.n_samples) - (e0 - np.floor(e0))
            aligned.append(np.interp(np.arange(self.n_samples), xi, roi_aligned))
        aligned = np.stack(aligned, axis=0).sum(0)
        aligned /= np.amax(aligned)
        self.h = aligned

    def get_response(self, kind: str="aligned", reflection_period: List[int]=list(), reflection_amplitude: List[float]=list()):
        """
        Returns the instrument response.

        Args:
          kind: Which version of the isntrument response to retrieve.
                "aligned" provides the MCP response, which is edge-aligned.
                "digital" is the digital response.
                "analog" is the analog response.
          reflection_period: Simulate reflection with these distances.
          reflection_amplitude: Amplitudes of the reflections.
        """
        if kind == "aligned":
            h = self.h
        elif kind == "digital":
            h = self.h_digital
        elif kind == "analog":
            h = self.h_analog
        else:
            raise ValueError(f"Unknown kind={kind}.")
        if len(reflection_period) > 0:
            h = fftconvolve(h, comb(reflection_period, reflection_amplitude), mode='same')
        return h

    def plot(self, kind: str="aligned"):
        """
        Plot impulse response.
        """
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(10, 8))
        plt.plot(np.arange(-self.n_filter, self.n_samples-self.n_filter), self.get_response(kind), lw=2)
        plt.xlabel("Samples")
        plt.ylabel("Intensity [a.u.]")

    def apply(self, tof_trace: Union[xr.DataArray, np.ndarray],
              n_iter: int=100,
              reflection_period: List[int]=list(),
              reflection_amplitude: List[float]=list(),
              extra_shift: int=2,
              method: str="nn_matrix", **kwargs) -> xr.DataArray:
        """
        Apply TV-deconvolution on TOF data taken in *analog* mode,
        assuming the given parameter setting and using the Lambda parameter
        as a denoising strength.

        This could be coupled with the
        [CookieboxCalibration][extra.recipes.CookieboxCalibration] object to
        deconvolve and denoise spectra before calibration.

        ```python
        # open a new run
        run = open_run(proposal=900485, run=349)
        # get the calibration constants
        cal = CookieboxCalibration.from_file("my_calibration.h5")
        # load the trace for this new run, exactly as required for calibration
        trace = cal.load_trace(run)
        # apply the deconvolution using this TOFResponse object
        response = TOFResponse.from_file("tof4.h5")
        trace.loc[dict(tof=4)] = response.apply(trace.sel(tof=4))
        # now apply the calibration:
        spectrum = cal.calibrate(trace)
        ```

        If the raw trace is desired, this could be used independently from the calibration
        object simply to improve the trace resolution.

        Example of usage without the calibration:
        ```python
        ts = "SQS_RR_UTC/TSYS/TIMESERVER:outputBunchPattern"
        digitizer = 'SQS_DIGITIZER_UTC4/ADC/1:network'
        digitizer_ctl = 'SQS_DIGITIZER_UTC4/ADC/1:network'
        tof_id = 4
        channel = "3_A"
        calib_run = open_run(proposal=900485, run=509).select([ts,
                                                               digitizer, digitizer_ctl
                                                               ], require_all=True)
        # set up AdqRawChannel object to read data from each eTOF
        tof = AdqRawChannel(run,
                            channel=channel,
                            digitizer=digi,
                            first_pulse_offset=23300,
                            single_pulse_length=600,
                            interleaved=True,
                            baseline=np.s_[:20000],
                            )
        # fetch data
        # select only first samples, since we don't care about the full trace
        original = -tof.pulse_data(pulse_dim='pulseIndex').sel(sample=np.s_[0:400])
        # retrieve impulse response from a file
        response = TOFResponse.from_file("tof4.h5")
        # apply it
        result = response.apply(original)
        # plot it
        plt.plot(result.isel(trainId=0, pulseIndex=0), lw=2, label="Deconvolved")
        plt.plot(original.isel(trainId=0, pulseIndex=0), lw=2, label="Original")
        plt.show()
        ```

        Args:
          tof_trace: The eTOF data as a DataArray returned by `AdqRawChannel` with shape (pulses, samples), or
               a numpy array of shape (n_pulses, n_samples)
          n_iter: Number of deconvoluton iterations.
          method: Set to "nn_matrix" for non-negative deconvolution with matrix multiplications.
                  Set to "tv_matrix' for Total Variaton deconvolution.
                  Use "standard" for  standard deconvolution.
          extra_shift: Number of samples to shift the impulse response to additionally for alignment.

        Returns:
          The deconvolved data as a [DataArray][xarray.DataArray].
        """
        norm = 1
        if isinstance(tof_trace, np.ndarray):
            if len(tof_trace.shape) == 1:
                norm = np.sum(tof_trace)
                original = xr.DataArray(tof_trace, dims=('sample',))
                if np.abs(norm) < 1e-6:
                    norm = 1
            else:
                original = xr.DataArray(tof_trace, dims=('pulse', 'sample'))
                norm = np.sum(original.data, axis=-1, keepdims=True)
                norm[np.abs(norm)<1e-6] = 1
                original.data /= norm
        elif isinstance(tof_trace, xr.DataArray):
            original = tof_trace.copy()
            norm = np.sum(original.data, axis=-1, keepdims=True)
            norm[np.abs(norm)<1e-6] = 1
            original.data /= norm
        else:
            raise ValueError("Expect `tof_trace` to be a numpy array or xarray DataArray.")
        h = self.get_response(reflection_period=reflection_period, reflection_amplitude=reflection_amplitude)
        if method == "nn_matrix":
            result_trace = nn_deconvolution(original.data, h=h, n_iter=n_iter, n_shift=self.n_filter+extra_shift, **kwargs)
        elif method == "tv_matrix":
            result_trace = tv_deconvolution(original.data, h=h, n_iter=n_iter, n_shift=self.n_filter+extra_shift, **kwargs)
        elif method == "standard":
            result_trace = std_deconvolution(original.data, h=h, n_shift=self.n_filter+extra_shift, **kwargs)
        else:
            raise ValueError("Unknown method.")
        result_norm = np.sum(result_trace, axis=-1, keepdims=True)
        result_trace /= result_norm
        result = xr.DataArray(result_trace*norm, dims=original.dims, coords=original.coords)

        return result

class TOFAnalogResponse(SerializableMixin):
    """
    Given a run with the Cookiebox in analog mode, obtain its impulse response.

    Example usage:
    ```python
    tof_id = 4
    channel = "3_A"
    digitizer = "SQS_DIGITIZER_UTC4/ADC/1:network"
    energy_source = "SA3_XTD10_MONO/MDL/PHOTON_ENERGY"
    calib_run = open_run(proposal=900485, run=320)
    scan = Scan(calib_run[energy_source, "actualEnergy.value"],
                resolution=2, intra_step_filtering=1)
    calib_run = calib_run.select([digitizer, energy_source], require_all=True)
    tof = AdqRawChannel(calib_run,
                        channel=channel,
                        digitizer=digitizer,
                        first_pulse_offset=23300,
                        single_pulse_length=600,
                        interleaved=True,
                        baseline=np.s_[:20000],
                        )
    response = TOFAnalogResponse(roi=np.s_[75:])
    response.setup(tof, scan)
    response.to_file("tof4.h5")
    h = response.get_response()
    ```

    This could also be coupled with the
    [CookieboxCalibration][extra.recipes.CookieboxCalibration] object to
    deconvolve and denoise spectra before calibration.

    ```python
    # open a new run
    run = open_run(proposal=900485, run=349)
    # get the calibration constants
    cal = CookieboxCalibration.from_file("my_calibration.h5")
    # load the trace for this new run, exactly as required for calibration
    trace = cal.load_trace(run)
    # apply the deconvolution using this TOFResponse object
    response = TOFAnalogResponse.from_file("tof4.h5")
    trace.loc[dict(tof=4)] = response.apply(trace.sel(tof=4))
    # now apply the calibration:
    spectrum = cal.calibrate(trace)
    ```

    Args:
      roi: Region of interest. Keep None to use the full trace.
      n_samples: Total number of samples use for impulse response.
    """
    def __init__(self,
                 roi: Optional[slice]=None,
                 n_samples: int=300,
                 n_filter: int=75,
                 ):
        self.roi = roi
        self.n_samples = n_samples
        self.n_filter = n_filter
        self.h = None
        self._version = 1
        self._all_fields = [
                            "h",
                            "h_unc",
                            "n_samples",
                            "n_filter",
                            "_version",
                           ]

    def _asdict(self):
        """
        Return a serializable dict.
        """
        d = {k: v for k, v in self.__dict__.items() if k in self._all_fields}
        return d
    def _fromdict(self, all_data):
        """
        Actions to do after loading from file.
        """
        for k, v in all_data.items():
            setattr(self, k, v)

    def setup(self,
              tof: AdqRawChannel,
              scan: Scan,
              ):
        """
        Given a [AdqRawChannel][extra.components.AdqRawChannel] object, and an energy scan object
        calculate this eTOF's impulse response function,
        assuming the corresponding run contins a scan over monochromated beams.

        Args:
          tof: `AdqRawChannel` used to access data for a given eTOF.
          scan: `Scan` used to detect which regions cotain which energies.
        """

        # get pulse data
        logging.info("Get pulse data ...")
        this_tof_data = -tof.pulse_data(pulse_dim='pulseIndex')
        if self.roi is not None:
            this_tof_data = this_tof_data.isel(sample=self.roi)
        this_tof_data = this_tof_data.unstack('pulse')

        # get means
        logging.info("Get energy means ...")
        h_axis = np.arange(self.n_samples)
        h = list()
        for k, e in enumerate(scan.positions):
            mono_data = this_tof_data.sel(trainId=scan.positions_train_ids[k]).mean('trainId').mean('pulseIndex').to_numpy()
            mono_data /= np.amax(mono_data)
            m = np.argmax(mono_data)
            xi = np.arange(mono_data.shape[-1]) - m + self.n_filter
            h += [np.interp(h_axis, xi, mono_data)]
        h = np.stack(h, axis=0)
        h_unc = h.std(0)
        h = h.mean(0)
        self.h = h
        self.h_unc = h_unc

    def get_response(self, reflection_period: List[int]=list(), reflection_amplitude: List[float]=list()):
        """
        Returns the instrument response.

        Args:
          reflection_period: Simulate reflection with these distances.
          reflection_amplitude: Amplitudes of the reflections.
        """
        h = self.h
        if len(reflection_period) > 0:
            h = fftconvolve(h, comb(reflection_period, reflection_amplitude), mode='same')
        return h

    def plot(self):
        """
        Plot impulse response.
        """
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(10, 8))
        plt.plot(np.arange(self.n_samples), self.h, lw=2, label="Impulse response")
        plt.fill_between(np.arange(self.n_samples), self.h-self.h_unc, self.h+self.h_unc,
                         facecolor='r', alpha=0.5, label="68% CL")
        plt.xlabel("Samples")
        plt.ylabel("Intensity [a.u.]")
        plt.legend(frameon=False)

    def apply(self, tof_trace: Union[xr.DataArray, np.ndarray],
              n_iter: int=5000,
              reflection_period: List[int]=list(),
              reflection_amplitude: List[float]=list(),
              method: str="tv_matrix",
              extra_shift: int=2,
              **kwargs) -> xr.DataArray:
        """
        Apply TV-deconvolution on TOF data taken in *analog* mode,
        assuming the given parameter setting and using the Lambda parameter
        as a denoising strength.

        This could be coupled with the
        [CookieboxCalibration][extra.recipes.CookieboxCalibration] object to
        deconvolve and denoise spectra before calibration.

        ```python
        # open a new run
        run = open_run(proposal=900485, run=349)
        # get the calibration constants
        cal = CookieboxCalibration.from_file("my_calibration.h5")
        # load the trace for this new run, exactly as required for calibration
        trace = cal.load_trace(run)
        # apply the deconvolution using this TOFResponse object
        response = TOFAnalogResponse.from_file("tof4.h5")
        trace.loc[dict(tof=4)] = response.apply(trace.sel(tof=4))
        # now apply the calibration:
        spectrum = cal.calibrate(trace)
        ```

        If the raw trace is desired, this could be used independetly from the calibration
        object simply to mprove the trace resolution.

        Example of usage without the calibration:
        ```python
        ts = "SQS_RR_UTC/TSYS/TIMESERVER:outputBunchPattern"
        digitizer = 'SQS_DIGITIZER_UTC4/ADC/1:network'
        digitizer_ctl = 'SQS_DIGITIZER_UTC4/ADC/1:network'
        tof_id = 4
        channel = "3_A"
        calib_run = open_run(proposal=900485, run=509).select([ts,
                                                               digitizer, digitizer_ctl
                                                               ], require_all=True)
        # set up AdqRawChannel object to read data from each eTOF
        tof = AdqRawChannel(run,
                            channel=channel,
                            digitizer=digi,
                            first_pulse_offset=23300,
                            single_pulse_length=600,
                            interleaved=True,
                            baseline=np.s_[:20000],
                            )
        # fetch data
        # select only first samples, since we don't care about the full trace
        original = -tof.pulse_data(pulse_dim='pulseIndex').sel(sample=np.s_[0:400])
        # retrieve impulse response from a file
        response = TOFAnalogResponse.from_file("tof4.h5")
        # apply it
        result = response.apply(original)
        # plot it
        plt.plot(result.isel(trainId=0, pulseIndex=0), lw=2, label="Deconvolved")
        plt.plot(original.isel(trainId=0, pulseIndex=0), lw=2, label="Original")
        plt.show()
        ```

        Args:
          tof_trace: The eTOF data as a DataArray returned by `AdqRawChannel` with shape (pulses, samples), or
               a numpy array of shape (n_pulses, n_samples)
          n_iter: Number of deconvoluton iterations.
          method: Set to "nn_matrix" for non-negative matrix multiplications.
                  Set to "tv_matrix" for Total Variatio deconvolution.
                  Use "standard" for  standard deconvolution.
          extra_shift: Shift the impulse response function by this many points before using it to align the data.

        Returns:
          The deconvolved data as a [DataArray][xarray.DataArray].
        """
        norm = 1
        if isinstance(tof_trace, np.ndarray):
            if len(tof_trace.shape) == 1:
                norm = np.sum(tof_trace)
                original = xr.DataArray(tof_trace, dims=('sample',))
                if np.abs(norm) < 1e-6:
                    norm = 1
            else:
                original = xr.DataArray(tof_trace, dims=('pulse', 'sample'))
                norm = np.sum(original.data, axis=-1, keepdims=True)
                norm[np.abs(norm)<1e-6] = 1
                original.data /= norm
        elif isinstance(tof_trace, xr.DataArray):
            original = tof_trace.copy()
            norm = np.sum(original.data, axis=-1, keepdims=True)
            norm[np.abs(norm)<1e-6] = 1
            original.data /= norm
        else:
            raise ValueError("Expect `tof_trace` to be a numpy array or xarray DataArray.")
        h = self.get_response(reflection_period=reflection_period, reflection_amplitude=reflection_amplitude)
        if method == "nn_matrix":
            result_trace = nn_deconvolution(original.data, h=h, n_iter=n_iter, n_shift=self.n_filter+extra_shift, **kwargs)
        elif method == "tv_matrix":
            result_trace = tv_deconvolution(original.data, h=h, n_iter=n_iter, n_shift=self.n_filter+extra_shift, **kwargs)
        elif method == "standard":
            result_trace = std_deconvolution(original.data, h=h, n_shift=self.n_filter+extra_shift, **kwargs)
        else:
            raise ValueError("Unknown method.")
        result_norm = np.sum(result_trace, axis=-1, keepdims=True)
        result_trace /= result_norm
        result = xr.DataArray(result_trace*norm, dims=original.dims, coords=original.coords)

        return result
