import numpy as np


def gaussian(x, y0, A, μ, σ):
    r"""Normalized Gaussian profile.

    The profile is normalized in the sense that:

    $$
    \int_{-\infty}^{\infty} gaussian(x, 0, A, \mu, \sigma > 0) \; dx = A
    $$

    Args:
        x (array_like, float): Function argument
        y0 (float): Vertical offset
        A (float): Amplitude
        μ (float): Expected value
        σ (float): Standard deviation

    Returns:
        (array_like): Function value(s)
    """
    return y0 + (A / (σ * np.sqrt(2*np.pi))) * np.exp(-(x - μ)**2 / (2 * σ**2))


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
