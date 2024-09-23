import numpy as np


def gaussian(x, y0, A, μ, σ, norm=True):
    r"""Gaussian profile.

    If `norm=True` the profile is normalized in the sense that:

    $$
    \int_{-\infty}^{\infty} gaussian(x, 0, A, \mu, \sigma > 0) \; dx = A
    $$

    Args:
        x (array_like, float): Function argument
        y0 (float): Vertical offset
        A (float): Amplitude
        μ (float): Expected value
        σ (float): Standard deviation
        norm (bool): Whether to normalize the Gaussian

    Returns:
        (array_like): Function value(s)
    """
    norm_factor = σ * np.sqrt(2*np.pi) if norm else 1
    return y0 + (A / norm_factor) * np.exp(-(x - μ)**2 / (2 * σ**2))


def fit_gaussian(ydata, xdata=None, p0=None, norm=False, A_sign=0, **kwargs):
    """Fit a Gaussian to some data.

    This uses [curve_fit()][scipy.optimize.curve_fit] to fit a Gaussian (from
    [gaussian()][extra.utils.gaussian]) to `ydata`. If `p0` is not passed the
    function will set them to reasonable defaults. It will return `None` if
    fitting fails, or if there are no finite values in `ydata`.

    !!! note
        By default this will only return the `popt` array from
        [curve_fit()][scipy.optimize.curve_fit], if you want `pcov` or any other
        output you must pass `full_output=True`.

    !!! note
        When visualizing the fit results with [gaussian()][extra.utils.gaussian]
        make sure the `norm` parameters match. i.e. if you're using the default
        of fitting an unnormalized Gaussian: `gaussian(xdata, *popt, norm=False)`.

    Args:
        ydata (array_like): The data to fit. NaN's and infs will automatically
            be masked before fitting.
        xdata (array_like): Optional x-values corresponding to `ydata`.
        p0 (list): A list of `[y0, A, μ, σ]` to match the arguments to
            [gaussian()][extra.utils.gaussian].
        norm (bool): Whether to fit a normalized or unnormalized Gaussian.
        A_sign (int): Sign of the amplitude (A) parameter for the Gaussian.
            1 for an upwards peak, -1 for downwards. 0 (default) allows either,
            using a faster algorithm. Passing `bounds=` overrides this.
        **kwargs (): All other keyword arguments will be passed to
            [curve_fit()][scipy.optimize.curve_fit].
    """
    if xdata is None:
        xdata = np.arange(len(ydata))

    from scipy.optimize import curve_fit

    full_output_requested = kwargs.get("full_output", False)

    # Filter nans and infs
    finite_mask = np.isfinite(ydata)
    xdata = xdata[finite_mask]
    ydata = ydata[finite_mask]

    if len(ydata) == 0:
        return None

    if p0 is None:
        if A_sign >= 0:  # Peak upwards (or not specified)
            μ_idx = np.argmax(ydata)
            A = max(1, ydata[μ_idx])
            y0 = np.min(ydata)
        else:  # Peak downwards
            μ_idx = np.argmin(ydata)
            A = min(-1, ydata[μ_idx])
            y0 = np.max(ydata)

        μ = xdata[μ_idx]
        σ = abs(np.max(xdata) - np.min(xdata)) / 4
        p0 = [y0, A, μ, σ]

    if A_sign != 0 and 'bounds' not in kwargs:
        Amin, Amax = (0, np.inf) if A_sign > 0 else (-np.inf, 0)
        kwargs['bounds'] = (
            # y0      A     μ        σ
            [-np.inf, Amin, -np.inf, 0],
            [np.inf,  Amax, np.inf, np.inf],
        )

    func = lambda *args: gaussian(*args, norm=norm)
    try:
        result = curve_fit(func, xdata, ydata, p0=p0, **kwargs)
        return result if full_output_requested else result[0]
    except RuntimeError:
        return None


def gaussian2d(x, y, z0, A, μ_x, μ_y, σ_x, σ_y):
    r"""Normalized 2D Gaussian profile.

    The profile is normalized in the sense that

    $$
    \iint_{-\infty}^{\infty} gaussian2d(x, y, 0, A, \mu_x, \mu_y, \sigma_x > 0, \sigma_y > 0) \; dx \; dy = A
    $$

    Args:
        x (array_like, float): Function arguments
        y (array_like, float): Function arguments
        z0 (float): Vertical offset
        μ_x (float): Expected x value
        μ_y (float): Expected y value
        σ_x (float): Standard deviation for x
        σ_y (float): Standard deviation for y

    Returns:
        (array_like): Function value(s)
    """
    return z0 + (A/(σ_x * σ_y * 2*np.pi)) \
        * np.exp(- (x[None, :] - μ_x)**2 / (2 * σ_x**2)
                 - (y[:, None] - μ_y)**2 / (2 * σ_y**2))


def lorentzian(x, y0, A, x0, γ):
    r"""Normalized Lorentzian profile.

    The profile is normalized in the sense that:

    $$
    \int_{-\infty}^{\infty} lorentzian(x, 0, A, x0 \in \mathbb{R}, y > 0) \; dx = A
    $$

    Args:
        x (array_like, float): Function argument
        y0 (float): Vertical offset
        A (float): Amplitude
        x0 (float): Location parameter
        γ (float): Scale parameter

    Returns:
       (array_like): Function value
    """
    return y0 + (A/(np.pi*γ)) * (γ**2/((x - x0)**2 + γ**2))
