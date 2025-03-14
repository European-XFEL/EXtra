from typing import Tuple, Union, Optional
from functools import partial
from scipy.linalg import convolution_matrix
import numpy as np
import xarray as xr
from .base import SerializableMixin

from extra.components import AdqRawChannel, Scan
from extra_data import by_id


def tv_deconvolution(data: np.ndarray, h: np.ndarray, Lambda: float=1.0, n_iter: int=2000):
    '''
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
    assert len(h) <= N

    # make them the same size
    if len(h) < N:
        hl = np.concatenate((h, np.zeros(N - len(h))))/np.sum(h)
    else:
        hl = h/np.sum(h)

    # roll over to match the effect of fftcovolve
    hl = np.roll(hl, -N//2)
    # create the matrix representing the convolution
    # this is expensive, but no easy way comes to mind
    A = convolution_matrix(hl, N, mode='same')
    # ensure the convergence criteria is satisfied
    # sigma * tau L^2 <= 1
    L = np.linalg.norm(A)

    sigma = 1.0/L
    tau = 1.0/L

    # we need the proximal operator of F and G
    # The function G(x) is as follows:
    # $G = 1/2 ||A x - b||^2 = 1/2 x^T A^T A x - b^T A x + 1/2 b^T b$
    # Its proximal operator is tabulated (see Ref. [2], sec. 6.1.1):
    # $\text{prox}_{\tau G} (v) = (I + \tau A^T A)^{-1} (v + \tau A^T b)
    I = np.eye(N)
    GG = np.linalg.pinv(I + tau*(A.T @ A))
    ATb = np.ascontiguousarray((A.T @ b))
    prox_G = lambda v: np.clip(GG @ (v + tau*ATb), a_min=0, a_max=None)

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
    y = 0*b #*(A.T @ b)
    x = 0*b
    x_bar = 0*x
    for k in range(0, n_iter):
        # if k % 50 == 0:
        #     print(f"Iteration {k}/{n_iter}")
        # no need to cache x_bar and x_old if omega = 1
        #x_old = np.copy(x)
        #x = prox_G(x - tau*(K.T @ y))
        ##x = prox_G(x - tau*gradT(y))
        #x_bar = x + 1*(x - x_old)
        x = 2*prox_G(x - tau*gradT(y)) - x

        #y = prox_Fs(y + sigma*(K @ x_bar))
        y = prox_Fs(y + sigma*grad(x_bar))

        # Calculate norms
        # fidelity = 0.5*np.sqrt(np.mean((A @ x - b)**2))
        # tv = np.sum(np.abs(grad(x)))
        # energy = 1.0*fidelity + Lambda*tv
        # if k%20 == 0:
        #    print("[%d] : energy %e \t fidelity %e \t TV %e" %(k,energy,fidelity,tv))
    if is1d:
        return x[:,0]

    return np.transpose(x)
    
def tv_deconvolution_fft(data: np.ndarray, h: np.ndarray, Lambda: float=1.0, n_iter: int=2000):
    '''
    Chambolle-Pock algorithm for the minimization of the objective function
        $\min_x 1/2||h \otimes x - data||^2 + Lambda TV(x)$

    This is an almost equivalent version of the code above (except for a hift of a sample) using FFT.
    This is only more efficient for large h (len(h) > 500 samples), which is not often the conditions one is
    interested in.
    
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
    if len(h) > N:
        h = h[:N]

    # make them the same size
    if len(h) < N:
        hl = np.concatenate((h, np.zeros(N - len(h))))/np.sum(h)
    else:
        hl = h/np.sum(h)

    # roll over to match the effect of fftcovolve
    #hl = np.roll(hl, -N//2)
    # create the matrix representing the convolution
    # this is expensive, but no easy way comes to mind
    #A = convolution_matrix(hl, N, mode='same')
    # ensure the convergence criteria is satisfied
    # sigma * tau L^2 <= 1
    #L = np.linalg.norm(A)
    
    # Frobenius norm is sqrt(sum |a_{ij}|^2) = sqrt(N*sum |h|^2),
    # since every column of the convolution matrix is just h rolled over
    L = np.sqrt(np.sum(hl**2)*len(hl))
    sigma = 1.0/L
    tau = 1.0/L

    # we need the proximal operator of F and G
    # The function G(x) is as follows:
    # $G = 1/2 ||A x - b||^2 = 1/2 x^T A^T A x - b^T A x + 1/2 b^T b$
    # Its proximal operator is tabulated (see Ref. [2], sec. 6.1.1):
    # $\text{prox}_{\tau G} (v) = (I + \tau A^T A)^{-1} (v + \tau A^T b)
    # I = np.eye(N)
    # GG = np.linalg.pinv(I + tau*(A.T @ A))
    # ATb = np.ascontiguousarray((A.T @ b))
    # prox_G = lambda v: np.clip(GG @ (v + tau*ATb), a_min=0, a_max=None)

    # prox_G using FFT should be more efficient
    # A.T == A for Toeplitz matrices
    # so (I + tau*A^T A)^{-1} is the deconvolution in Fourier domain of 1 + |H|^2
    # Therefore (I + tau*A^T A)^{-1} z can be implemented as ifft(1/(1 + tau*|H|^2)* fft(z))
    # additionally A^T b is the convolution of h and b, or, in the Fourier domain: ifft(H, fft(b))
    # sums and products by scalars are trivially represented in the Fourier domain
    H = np.fft.fft(hl)
    FATb = H[:, None] * np.fft.fft(b, axis=0)
    H2 = np.abs(H)**2
    W = 1.0/(tau*H2 + 1)
    prox_G = lambda v: np.clip(np.fft.ifft(W[:, None] * (np.fft.fft(v, axis=0) + tau*FATb), axis=0), a_min=0, a_max=None)

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
    y = 0*b #*(A.T @ b)
    x = 0*b
    x_bar = 0*x
    for k in range(0, n_iter):
        if k % 50 == 0:
            print(f"Iteration {k}/{n_iter}")
        # no need to cache x_bar and x_old if omega = 1
        #x_old = np.copy(x)
        #x = prox_G(x - tau*(K.T @ y))
        ##x = prox_G(x - tau*gradT(y))
        #x_bar = x + 1*(x - x_old)
        x = 2*prox_G(x - tau*gradT(y)) - x

        #y = prox_Fs(y + sigma*(K @ x_bar))
        y = prox_Fs(y + sigma*grad(x_bar))

        # Calculate norms
        #fidelity = 0.5*np.sqrt(np.mean((A @ x - b)**2))
        #tv = np.sum(np.abs(K @ x))
        #energy = 1.0*fidelity + Lambda*tv
        #if k%50 == 0:
        #    print("[%d] : energy %e \t fidelity %e \t TV %e" %(k,energy,fidelity,tv))
    if is1d:
        return x[:,0]

    return np.transpose(x)

class TOFResponse(SerializableMixin):
    """
    Given a run with the Cookiebox in counting mode, obtain its MCP impulse response.
    The impulse response is calculated in bins of the MCP Scan object provided,
    so that one can characterize the impulse response as a functon of the MCP voltage.

    Example usage:
    ```
    tof_id = 4
    channel = "3_A"
    digitizer = "SQS_DIGITIZER_UTC4/ADC/1:network"
    mpod_source = "SQS_RACK_MPOD-2/MDL/MPOD_MAPPER"
    calib_run = open_run(proposal=900485, run=320)
    calib_run = calib_run.select([digitizer, mpod_source], require_all=True)
    tof = AdqRawChannel(calib_run,
                        channel=channel,
                        digitizer=digitizer,
                        first_pulse_offset=23300,
                        single_pulse_length=600,
                        interleaved=True,
                        baseline=np.s_[:20000],
                        )
    scan = Scan(calib_run[mpod_source, f"TOF_{tof_id}.channelVoltage.value"],
                resolution=10, intra_step_filtering=1)
    response = TOFResponse()
    response.setup(tof, scan)
    response.to_file("tof4.h5")
    h = response.get_response(mcp_voltage=3000)
    ```

    This could also be coupled with the `CookieboxCalibration` object to deconvolve
    and denoise spectra before calibration.

    ```
    # open a new run
    run = open_run(proposal=900485, run=349)
    # get the calibration constants
    cal = CookieboxCalibration.from_file("my_calibration.h5")
    # load the trace for this new run, exactly as required for calibration
    trace = cal.load_trace(run)
    # apply the deconvolution using this TOFResponse object
    response = TOFResponse.from_file("tof4.h5")
    trace.loc[dict(tof=4)] = response.apply(trace.sel(tof=4), mcp_voltage=3600)
    # now apply the calibration:
    spectrum = cal.calibrate(trace)
    ```

    Args:
      counting_threshold: Threshold to use when identifying edges.
      n_samples: Total number of samples use for impulse response.
      n_filter: Require no edges to be found before sample n_filter and after n_samples-n_filter.
    """
    def __init__(self, counting_threshold: int=-10,
                 n_samples: int=400,
                 n_filter: int=100):
        self.counting_threshold = counting_threshold
        self.n_samples = n_samples
        self.n_filter = n_filter
        self.h = None
        self.h_digital = None
        self.h_analog = None
        self.mcp_v = None
        self._version = 1
        self._all_fields = [
                            "h",
                            "h_digital",
                            "h_analog",
                            "mcp_v",
                            "counting_threshold",
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
        Given a `AdqRawChannel` object, and a `Scan` object over the corresponding MCP,
        calculate this eTOF's impulse response function.

        Args:
          tof: `AdqRawChannel` used to access data for a given eTOF.
          scan: `Scan` object on
        """
        self.mcp_v = list()
        self.h = list()
        self.h_digital = list()
        self.h_analog = list()
        for step_nr, (mcp_v, train_ids) in enumerate(scan.steps):
            # get pulse data
            this_tof_data = tof.select_trains(by_id[train_ids]).pulse_data(pulse_dim='pulseIndex')
            # get edges
            edges_pos, edges_amp = tof.find_edge_array(this_tof_data,
                                                       labelled=False,
                                                       threshold=self.counting_threshold)

            if edges_pos.shape[-1] == 0:
                continue

            # require a single edge within the pulse
            edges_sel = ((np.sum(~np.isnan(edges_pos), axis=-1) == 1)
                         & (edges_pos[:,0] >= self.n_filter)
                         & (edges_pos[:,0] <= self.n_samples - self.n_filter))

            if np.sum(edges_sel) == 0:
                continue

            # build axis
            ts = np.arange(0, this_tof_data.shape[-1])
            h_axis = np.arange(0, self.n_samples)

            # find edges and build both digital and analog response functions
            edges = tof.find_edges(this_tof_data, threshold=self.counting_threshold)
            ts1 = np.arange(0, this_tof_data.shape[-1]+1)
            h_digital, _ = np.histogram(edges.edge, bins=ts1)
            h_digital = h_digital.astype(float)/np.amax(h_digital)
            self.h_digital += [h_digital]

            h_analog, _ = np.histogram(edges.edge, bins=ts1, weights=-edges.amplitude)
            h_analog /= np.amax(h_analog)
            self.h_analog += [h_analog]

            # now align traces
            aligned = list()
            for p, e in zip(this_tof_data.pulse[edges_sel].to_numpy(), edges_pos[edges_sel]):
                # select region of interest in the trace
                e0 = int(e[0])
                roi_aligned = -this_tof_data.sel(pulse=p).isel(sample=slice(e0-self.n_filter, e0+self.n_samples-self.n_filter))
                xi = np.arange(self.n_samples) - (e0 - np.floor(e0))
                aligned.append(np.interp(np.arange(self.n_samples), xi, roi_aligned))
            aligned = np.stack(aligned, axis=0).sum(0)
            aligned /= np.amax(aligned)
            self.h += [aligned]

            self.mcp_v += [mcp_v]
        self.h = np.stack(self.h, axis=0)[:, self.n_filter:]
        self.h_analog = np.stack(self.h_analog, axis=0)[:, self.n_filter:]
        self.h_digital = np.stack(self.h_digital, axis=0)[:, self.n_filter:]
        self.mcp_v = np.array(self.mcp_v)

    def get_response(self, mcp_voltage: float, tol: float=1.0):
        """
        Returns the instrument response for a given MCP voltage setting

        Args:
          mcp_voltage: The MCP potential difference in Volts.
          tol: Maximum acceptable difference in MCP voltage, in Volts.
        """
        diff = np.fabs(self.mcp_v - mcp_voltage)
        if np.amin(diff) > tol:
            raise ValueError(f"No instrument response available for MCP voltage setting {mcp_voltage}. "
                             f"Only the following voltage settings are available: {self.mcp_v}")
        idx = np.argmin(diff)
        return self.h[idx]

    def plot(self):
        """
        Plot impulse response.
        """
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(10, 8))
        for k, mcp_v in enumerate(self.mcp_v):
            plt.plot(self.h[k], lw=2, label=f"{mcp_v} V")
        plt.xlabel("Samples")
        plt.ylabel("Intensity [a.u.]")
        plt.legend(frameon=False, ncols=2)

    def apply(self, tof_trace: Union[xr.DataArray, np.ndarray],
              mcp_voltage: float,
              Lambda: float=0.01,
              n_iter: int=2000,
              method: str="matrix") -> xr.DataArray:
        """
        Apply TV-deconvolution on TOF data taken n *analog* mode,
        assuming the given MCP voltage and using the Lambda parameter
        as a denoising strength.

        This could be coupled with the `CookieboxCalibration` object to deconvolve
        and denoise spectra before calibration.

        ```
        # open a new run
        run = open_run(proposal=900485, run=349)
        # get the calibration constants
        cal = CookieboxCalibration.from_file("my_calibration.h5")
        # load the trace for this new run, exactly as required for calibration
        trace = cal.load_trace(run)
        # apply the deconvolution using this TOFResponse object
        response = TOFResponse.from_file("tof4.h5")
        trace.loc[dict(tof=4)] = response.apply(trace.sel(tof=4), mcp_voltage=3600)
        # now apply the calibration:
        spectrum = cal.calibrate(trace)
        ```

        If the raw trace is desired, this could be used independetly from the calibration
        object simply to mprove the trace resolution.

        Example of usage without the calibration:
        ```
        ts = "SQS_RR_UTC/TSYS/TIMESERVER:outputBunchPattern"
        digitizer = 'SQS_DIGITIZER_UTC4/ADC/1:network'
        digitizer_ctl = 'SQS_DIGITIZER_UTC4/ADC/1:network'
        mpod_source = "SQS_RACK_MPOD-2/MDL/MPOD_MAPPER"
        tof_id = 4
        channel = "3_A"
        calib_run = open_run(proposal=900485, run=509).select([ts,
                                                               digitizer, digitizer_ctl
                                                               mpod_source], require_all=True)
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
        mcp_voltage = run[mpod_source, f"TOF_{tof_id}.channelVoltage.value"].ndarray().mean()
        # retrieve impulse response from a file
        response = TOFResponse.from_file("tof4.h5")
        # apply it
        result = response.apply(original, mcp_voltage)
        # plot it
        plt.plot(result.isel(trainId=0, pulseIndex=0), lw=2, label="Deconvolved")
        plt.plot(original.isel(trainId=0, pulseIndex=0), lw=2, label="Original")
        plt.show()
        ```
        
        Args:
          tof_trace: The eTOF data as a DataArray returned by `AdqRawChannel` with shape (pulses, samples), or
               a numpy array of shape (n_pulses, n_samples)
          mcp_voltage: The MCP voltage set in this data.
          Lambda: The regularization strength.
          n_iter: Number of deconvoluton iterations.
          method: Set to "matrix" for matrix multplications (faster for shorter traces, ie len(trace) < 400).
                  Use "fft" for the FFT-based method. Warning: the two methods provide similar, but *not identcal* results.

        Returns: The deconvolved data as a DataArray.
        """
        norm = 1
        if isinstance(tof_trace, np.ndarray):
            if len(tof_trace.shape) == 1:
                norm = np.max(tof_trace)
                original = xr.DataArray(tof_trace/norm, dims=('sample',))
            else:
                original = xr.DataArray(tof_trace, dims=('pulse', 'sample'))
                norm = np.max(original.data, axis=0, keepdims=True)
                original.data /= norm
        elif isinstance(tof_trace, xr.DataArray):
            original = tof_trace.copy()
            norm = np.max(original.data, axis=0, keepdims=True)
            original.data /= norm
        else:
            raise ValueError("Expect `tof_trace` to be a numpy array or xarray DataArray.")
        h = self.get_response(mcp_voltage)[self.n_filter:]
        if method == "matrix":
            result_trace = tv_deconvolution(original.data, h=h, Lambda=Lambda, n_iter=n_iter)
        else:
            result_trace = tv_deconvolution_fft(original.data, h=h, Lambda=Lambda, n_iter=n_iter)
        result = original
        result.data = result_trace*norm
        return result
