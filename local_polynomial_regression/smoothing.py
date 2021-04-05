import math
import random
import pandas as pd
import numpy as np

from .config.core import config
from .utils.helpers import chunks
from .utils.kernels import gaussian_kernel, epanechnikov_kernel


def choose_kernel(kernel_str):
    if kernel_str == "gaussian_kernel":
        return gaussian_kernel
    elif kernel_str == "epanechnikov_kernel":
        return epanechnikov_kernel
    else:
        print("kernel unknow, use Gaussian Kernel")
        return gaussian_kernel


def local_polynomial_estimation(X, y, x, h, kernel):
    n = X.shape[0]
    K_i = 1 / h * kernel(x, X, h)
    f_i = 1 / n * sum(K_i)

    if f_i == 0:  # doesnt really happen, but in order to avoid possible errors
        W_hi = np.zeros(n)
    else:
        W_hi = K_i / f_i

    X1 = np.ones(n)
    X2 = X - x
    X3 = X2 ** 2

    X = np.array([X1, X2, X3]).T
    W = np.diag(W_hi)  # (n,n)

    XTW = (X.T).dot(W)  # (3,n)
    XTWX = XTW.dot(X)  # (3,3)
    XTWy = XTW.dot(y)  # (3,1)

    beta = np.linalg.pinv(XTWX).dot(XTWy)  # (3,1)
    return beta[0], beta[1], beta[2], W_hi


def bandwidth_cv_slicing(
    X,
    y,
    x_bandwidth,
    smoothing=local_polynomial_estimation,
    kernel=config.model_config.kernel,
    n_slices=config.model_config.n_slices,
):
    np.random.seed(1)
    df = pd.DataFrame(data=y, index=X)
    df = df.sort_index()
    X = np.array(df.index)
    y = np.array(df[0])
    n = X.shape[0]
    idx = list(range(0, n))
    slices = list(chunks(idx, math.ceil(n / n_slices)))
    if len(slices[0]) > 30:
        samples = 30
    else:
        samples = len(slices[0])

    num = len(x_bandwidth)
    mse_bw = np.zeros(num)  # for each bandwidth have mse - loss function

    for b, h in enumerate(x_bandwidth):
        mse_slice = np.zeros(n_slices)
        for i, chunk in enumerate(slices):
            X_train, X_test = np.delete(X, chunk), X[chunk]
            y_train, y_test = np.delete(y, chunk), y[chunk]

            runs = min(samples, len(chunk))
            y_true = np.zeros(runs)
            y_pred = np.zeros(runs)
            mse_test = np.zeros(runs)
            for j, idx_test in enumerate(random.sample(list(range(0, len(chunk))), runs)):
                y_hat = smoothing(X_train, y_train, X_test[idx_test], h, kernel)[0]
                y_true[j] = y_test[idx_test]
                y_pred[j] = y_hat
                mse_test[j] = (y_test[idx_test] - y_hat) ** 2
            mse_slice[i] = 1 / runs * sum((y_true - y_pred) ** 2)
        mse_bw[b] = 1 / n_slices * sum(mse_slice)

    results = {
        "bandwidths": x_bandwidth,
        "MSE": mse_bw,
        "h": x_bandwidth[mse_bw.argmin()],
    }
    return results


def bandwidth_cv_random(
    X,
    y,
    x_bandwidth,
    smoothing=local_polynomial_estimation,
    kernel=config.model_config.kernel,
    n_slices=config.model_config.n_slices,
):
    np.random.seed(1)
    df = pd.DataFrame(data=y, index=X)
    df = df.sort_index()
    X = np.array(df.index)
    y = np.array(df[0])
    n = X.shape[0]
    idx = list(range(0, n))
    random.shuffle(idx)
    slices = list(chunks(idx, math.ceil(n / n_slices)))
    if len(slices[0]) > 30:
        samples = 30
    else:
        samples = len(slices[0])

    num = len(x_bandwidth)
    mase = np.zeros(num)

    for b, h in enumerate(x_bandwidth):
        mse = np.zeros(n_slices)
        for i, chunk in enumerate(slices):
            X_train, X_test = np.delete(X, chunk), X[chunk]
            y_train, y_test = np.delete(y, chunk), y[chunk]

            runs = min(samples, len(chunk))
            mse_tmp = np.zeros(runs)
            for j, idx_test in enumerate(random.sample(list(range(0, len(chunk))), runs)):
                y_pred = smoothing(X_train, y_train, X_test[idx_test], h, kernel)[0]
                mse_tmp[j] = (y_test[idx_test] - y_pred) ** 2
            mse[i] = 1 / runs * sum(mse_tmp)
        mase[b] = 1 / n_slices * sum(mse)

    results = {
        "bandwidths": x_bandwidth,
        "MSE": mase,
        "h": x_bandwidth[mase.argmin()],
    }
    return results


def bandwidth_cv(
    X,
    y,
    bandwidths_1,
    smoothing=local_polynomial_estimation,
    kernel=config.model_config.kernel,
):
    # 1) coarse parameter search
    coarse_results = bandwidth_cv_slicing(X, y, bandwidths_1)

    # 2) fine parameter search, around minimum of first search
    bandwidths_1 = coarse_results["bandwidths"]
    h = coarse_results["h"]
    stepsize = bandwidths_1[1] - bandwidths_1[0]
    bandwidths_2 = np.linspace(h - (stepsize * 1.1), h + (stepsize * 1.1), 10)

    fine_results = bandwidth_cv_slicing(X, y, bandwidths_2)

    return {
        "fine results": fine_results,
        "coarse results": coarse_results,
    }


def create_fit(
    X,
    y,
    h,
    gridsize=100,
    smoothing=local_polynomial_estimation,
    kernel_str=config.model_config.kernel,
):
    kernel = choose_kernel(kernel_str)

    X_domain = np.linspace(X.min(), X.max(), gridsize)
    fit = np.zeros(len(X_domain))
    first = np.zeros(len(X_domain))
    second = np.zeros(len(X_domain))
    for i, x in enumerate(X_domain):
        b0, b1, b2, W_hi = smoothing(X, y, x, h, kernel)
        fit[i] = b0
        first[i] = b1
        second[i] = b2
    return X_domain, fit, first, second, h
