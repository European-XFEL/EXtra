
import numpy as np


def gaussian(x, y0, A, μ, σ):
    """Normalized Gaussian profile.

    The profile is normalized in the sense that  

    ∞
    ∫ gaussian(x, 0, A, μ, σ > 0) dx = A
    -∞

    Args:
        x (array_like, real): Function argument
        y0 (real): Vertical offset
        A (real): Amplitude
        μ (real): Expected value
        σ (real): Standard deviation

    Returns:
        (array_like) Function value(s)

    """

    return y0 + (A / (σ * np.sqrt(2*np.pi))) * np.exp(-(x - μ)**2 / (2 * σ**2))


def gaussian2d(x, y, z0, A, μ_x, μ_y, σ_x, σ_y):
    """Normalized 2d Gaussian profile.

    The profile is normalized in the sense that  

    ∞
    ∬ gaussian(x, y, 0, A, μ_x, μ_y, σ_x > 0, σ_y > 0) dx dy = A
    -∞

    Args:
        x, y (array_like, real): Function arguments
        z0 (real): Vertical offset
        μ_x, μ_y (real): Expected values
        σ_x, σ_y (real): Standard deviations

    Returns:
        (array_like) Function value(s)

    """

    return z0 + (A/(σ_x * σ_y * 2*np.pi)) \
        * np.exp(- (x[None, :] - μ_x)**2 / (2 * σ_x**2)
                 - (y[:, None] - μ_y)**2 / (2 * σ_y**2))


def lorentzian(x, y0, A, x0, γ):
    """Normalized Lorentzian profile.

    The profile is normalized in the sense that

    ∞
    ∫ lorentzian(x, 0, A, x0 ∈ ℝ, γ > 0) dx = A
    -∞

    Args:
        x (array_like, real): Function argument
        y0 (real): Vertical offset
        A (real): Amplitude
        x0 (real): Location parameter
        γ (real): Scale parameter

   Returns:
       (array_like) Function value

    """

    return y0 + (A/(np.pi*γ)) * (γ**2/((x - x0)**2 + γ**2))
