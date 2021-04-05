import numpy as np
from scipy.stats import norm


# ----------------- with tau ------- Rookley + Haerdle (Applied Quant. Finance)
def gaussian_kernel(x, Xi, h):
    u = (x - Xi) / h
    return norm.pdf(u)


def epanechnikov_kernel(x, Xi, h):
    u = (x - Xi) / h
    indicator = np.where(abs(u) <= 1, 1, 0)
    k = 0.75 * (1 - u ** 2)
    return k * indicator
