import numpy as np
from scipy.stats import norm


# ----------------- with tau ------- Rookley + Haerdle (Applied Quant. Finance)
def gaussian_kernel(x, Xi, h):
    """Gaussian Kernel.

    :math:`K(u) = \\frac{1}{\\sqrt{2\\pi}} e^{-\\frac{1}{2} u^2}` where :math:`u = \\frac {x - X_i}{h}`

    Args:
        x (float): point of interest
        Xi (array): data points, surrounding of x
        h (float): bandwidth

    Returns:
        ndarray: Kernel function for the point x
    """
    u = (x - Xi) / h
    return norm.pdf(u)


def epanechnikov_kernel(x, Xi, h):
    """Epanechnikov kernel

    :math:`K(u) = \\frac{3}{4} \\left( 1 - u^2 \\right)` where :math:`u = \\frac {x - X_i}{h}`

    Args:
        x (float): point of interest
        Xi (array): data points, surrounding of x
        h (float): bandwidth

    Returns:
        ndarray: Kernel function for the point x
    """
    u = (x - Xi) / h
    indicator = np.where(abs(u) <= 1, 1, 0)
    k = 0.75 * (1 - u ** 2)
    return k * indicator


kernel_dict = {"gaussian": gaussian_kernel, "epanechnikov": epanechnikov_kernel}
