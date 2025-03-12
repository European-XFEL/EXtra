from typing import Tuple, Any
from functools import partial
import numpy as np

from .base import SerializableMixin

from extra.components import AdqRawChannel, Scan

def tv_deconvolution(data: np.ndarray, h: np.ndarray, Lambda: float=1.0, n_iter: int=4000):
    '''
    Chambolle-Pock algorithm for the minimization of the objective function
        $\min_x 1/2||h \otimes x - data||^2 + Lambda*TV(x)$
    
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
        h: Impulse response function of system to deconvolve.
        data: Data to deconvolve. It must have length smaller or equal to h.
        Lambda : Weight of the TV penalty.
        n_iter: Number of iterations.
    '''

    assert len(h) <= len(data)
    N = len(data)
    # make them the same size
    if len(h) < len(data):
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
    # $\text{prox}_{\tau G} (v) = (I + \tau A^T A)^{-1} (v + \tau b^T A)
    I = np.eye(N)
    GG = np.linalg.pinv(I + tau*(A.T @ A))
    prox_G = lambda v: GG @ (v + tau*(data.T @ A))

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
    K = convolution_matrix(np.array([1.0, -1.0])/2.0, N, mode='same')

    # initialize the temporary variables
    y = 0*(A.T @ data)
    x = 0*data
    x_bar = 0*x
    omega = 1.0
    for k in range(0, n_iter):
        x_old = np.copy(x)
        x = prox_G(x - tau*(K.T @ y))
        x_bar = x + omega*(x - x_old)
        y = prox_Fs(y + sigma*(K @ x_bar))
    
        # Calculate norms
        #fidelity = 0.5*np.sqrt(np.mean((A @ x - data)**2))
        #tv = np.sum(np.abs(K @ x))
        #energy = 1.0*fidelity + Lambda*tv
        #if k%50 == 0:
        #    print("[%d] : energy %e \t fidelity %e \t TV %e" %(k,energy,fidelity,tv))
    return x

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

    Args:
      counting_threshold: Threshold to use when identifying edges.
      n_samples: Total number of samples use for impulse response.
      n_filter: Require no edges to be found before sample n_filter and after n_samples-n_filter.
    """
    def __init__(self, counting_threshold: int=-10,
                 n_samples: int=1000,
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
              scan: Optional[Scan]=None,
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
        self.h = np.stack(self.h, axis=0)
        self.h_analog = np.stack(self.h_analog, axis=0)
        self.h_digital = np.stack(self.h_digital, axis=0)
        self.mcp_v = np.array(self.mcp_v)

    def get_response(self, mcp_voltage: float, tol: float=1.0):
        """
        Returns the instrument response for a given MCP voltage setting

        Args:
          mcp_voltage: The MCP potential difference in Volts.
          tol: Maximum acceptable difference in MCP voltage, in Volts.
        """
        diff = np.fabs(self.mcp_v - mcp_voltage)
        if np.amin(diff) > tolerance:
            raise ValueError(f"No instrument response available for MCP voltage setting {mcp_voltage}. "
                             f"Only the following voltage settings are available: {self.mcp_v}")
        idx = np.argmin(diff)
        return self.h[idx]

    def apply(self, tof: AdqRawChannel, mcp_voltage: float, Lambda: float=0.01) -> Tuple[xr.DataArray, xr.DataArray]:
        """
        Apply TV-deconvolution on TOF data taken n *analog* mode,
        assuming the given MCP voltage and using the Lambda parameter
        as a denoising strength.

        Example:
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
        mcp_voltage = run[mpod_source, f"TOF_{tof_id}.channelVoltage.value"].ndarray().mean()
        # retrieve impulse response from a file
        response = TOFResponse.from_file("tof4.h5")
        # apply it
        result, original = response.apply(tof, mcp_voltage)
        # plot it
        plt.plot(result.isel(trainId=0, pulseIndex=0), lw=2, label="Deconvolved")
        plt.plot(original.isel(trainId=0, pulseIndex=0), lw=2, label="Original")
        plt.show()
        ```
        
        Args:
          tof: The eTOF data.
          mcp_voltage: The MCP voltage set in this data.
          Lambda: The regularization strength.

        Returns: The deconvolved data and the original trace.
        """
        original = -tof.pulse_data(pulse_dim='pulseIndex')
        h = self.get_response(mcp_voltage)
        #result = tv_deconvolution(h, original, Lambda=Lambda)
        dec = partial(tv_deconvolution, h=h, Lambda=Lambda)
        result_trace = np.apply_along_axis(tv_deconvolution, axis=0, arr=original.data)
        result = original.copy()
        result.data = result_trace
        return result, original
