import numpy as np
import pandas as pd
import math
import random
import scipy.interpolate as interpolate  # B-Spline

from local_polynomial_regression.config.core import config


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def sort_values_by_X(X, y):
    df = pd.DataFrame(y, X)
    df = df.sort_index()
    X_sorted = np.array(df.index)
    y_sorted = np.array(df[0])
    return X_sorted, y_sorted


def create_partitions(X, y, n_sections, sampling_type):
    n = X.shape[0]
    if sampling_type == "random":
        idx = list(range(0, n))
        random.seed(config.model_config.random_state)  # not np.random.seed !!!
        random.shuffle(idx)
    elif sampling_type == "slicing":
        idx = list(range(0, n))
    X_partition_idxs = list(chunks(idx, math.ceil(n / n_sections)))
    return X_partition_idxs


def bspline(x, y, sections, degree=3):
    idx = np.linspace(0, len(x) - 1, sections + 1, endpoint=True).round(0).astype("int")
    x = x[idx]
    y = y[idx]

    t, c, k = interpolate.splrep(x, y, s=0, k=degree)
    spline = interpolate.BSpline(t, c, k, extrapolate=True)
    pars = {"t": t, "c": c, "deg": k}
    points = {"x": x, "y": y}
    return pars, spline, points
