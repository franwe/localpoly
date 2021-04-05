import math
import random
import pandas as pd
import numpy as np

from .config.core import config
from .utils.helpers import chunks
from .utils.kernels import kernel_dict


class LocalPolynomialRegression:
    """Local polynomial regression.

    LocalPolynomialRegression fits a polynomial of degree 3 in to the sourrounding of each point.
    The surrounding is realized by a kernel with bandwidth h. The regression returns the fit, as
    well as its first and second derivative.

    Attributes:
        X: X-values of data that is to be fitted (explanatory variable)
        y: y-values of data that is to be fitted (observations)
        h: bandwidth for the kernel
        gridsize: desired size of the fit (granularity)
        kernel_str: the name of the kernel as a string "gaussian"
    """

    def __init__(self, X, y, h, kernel=config.model_config.kernel, gridsize=100):
        self.X = X
        self.y = y
        self.h = h
        self.kernel_str = kernel
        self.kernel = kernel_dict[self.kernel_str]
        self.gridsize = gridsize

    def local_polynomial_estimation(self, x):
        n = self.X.shape[0]
        K_i = 1 / self.h * self.kernel(x, self.X, self.h)
        f_i = 1 / n * sum(K_i)

        if f_i == 0:  # doesnt really happen, but in order to avoid possible errors
            W_hi = np.zeros(n)
        else:
            W_hi = K_i / f_i

        X1 = np.ones(n)
        X2 = self.X - x
        X3 = X2 ** 2

        X = np.array([X1, X2, X3]).T
        W = np.diag(W_hi)  # (n,n)

        XTW = (X.T).dot(W)  # (3,n)
        XTWX = XTW.dot(X)  # (3,3)
        XTWy = XTW.dot(self.y)  # (3,1)

        beta = np.linalg.pinv(XTWX).dot(XTWy)  # (3,1)
        return beta[0], beta[1], beta[2], W_hi

    def fit(self):
        X_domain = np.linspace(self.X.min(), self.X.max(), self.gridsize)
        fit = np.zeros(len(X_domain))
        first = np.zeros(len(X_domain))
        second = np.zeros(len(X_domain))
        for i, x in enumerate(X_domain):
            b0, b1, b2, W_hi = self.local_polynomial_estimation(x)
            fit[i] = b0
            first[i] = b1
            second[i] = b2
        return X_domain, fit, first, second, self.h


class LocalPolynomialRegressionCV(LocalPolynomialRegression):
    def __init__(
        self,
        X,
        y,
        h,
        kernel=config.model_config.kernel,
        gridsize=100,
        n_sections=config.model_config.n_sections,
        loss=config.model_config.loss,
        sampling=config.model_config.sampling,
    ):
        self.n_sections = n_sections
        self.loss = loss
        self.sampling = sampling

        # invoking the __init__ of the parent class
        LocalPolynomialRegression.__init__(self, X, y, h, kernel, gridsize)

    def bandwidth_cv(
        self,
        bandwidths_1,
    ):
        # 1) coarse parameter search
        coarse_results = self.bandwidth_cv_slicing(X, y, bandwidths_1)

        # 2) fine parameter search, around minimum of first search
        bandwidths_1 = coarse_results["bandwidths"]
        h = coarse_results["h"]
        stepsize = bandwidths_1[1] - bandwidths_1[0]
        bandwidths_2 = np.linspace(h - (stepsize * 1.1), h + (stepsize * 1.1), 10)

        fine_results = self.bandwidth_cv_slicing(X, y, bandwidths_2)

        return {
            "fine results": fine_results,
            "coarse results": coarse_results,
        }

    def bandwidth_cv_slicing(self, list_of_bandwidths):
        np.random.seed(1)
        df = pd.DataFrame(data=self.y, index=self.X)
        df = df.sort_index()
        X = np.array(df.index)
        y = np.array(df[0])
        n = X.shape[0]
        idx = list(range(0, n))
        slices = list(chunks(idx, math.ceil(n / self.n_sections)))
        if len(slices[0]) > 30:
            samples = 30
        else:
            samples = len(slices[0])

        num = len(list_of_bandwidths)
        mse_bw = np.zeros(num)  # for each bandwidth have mse - loss function

        # for bandwidth h in list_of_bandwidths
        for b, h in enumerate(list_of_bandwidths):
            mse_slice = np.zeros(self.n_sections)
            # take out chunks of our data and do leave-out-prediction
            for i, chunk in enumerate(slices):
                X_train, X_test = np.delete(X, chunk), X[chunk]
                y_train, y_test = np.delete(y, chunk), y[chunk]

                model = LocalPolynomialRegression(X_train, y_train, h, self.kernel_str, self.gridsize)
                runs = min(samples, len(chunk))
                y_true = np.zeros(runs)
                y_pred = np.zeros(runs)
                mse_test = np.zeros(runs)
                for j, idx_test in enumerate(random.sample(list(range(0, len(chunk))), runs)):
                    y_hat = model.local_polynomial_estimation(X_test[idx_test])[
                        0
                    ]  # use model.local_polynomial_estimation instead of model.fit, because we need the estimate for one point of X_test, not projected to X_est
                    y_true[j] = y_test[idx_test]
                    y_pred[j] = y_hat
                    mse_test[j] = (y_test[idx_test] - y_hat) ** 2
                mse_slice[i] = 1 / runs * sum((y_true - y_pred) ** 2)
            mse_bw[b] = 1 / self.n_sections * sum(mse_slice)

        results = {
            "bandwidths": list_of_bandwidths,
            "MSE": mse_bw,
            "h": list_of_bandwidths[mse_bw.argmin()],
        }
        return results

    def bandwidth_cv_random(self, list_of_bandwidths):
        np.random.seed(1)
        df = pd.DataFrame(data=self.y, index=self.X)
        df = df.sort_index()
        X = np.array(df.index)
        y = np.array(df[0])
        n = X.shape[0]
        idx = list(range(0, n))
        random.shuffle(idx)
        slices = list(chunks(idx, math.ceil(n / self.n_sections)))
        if len(slices[0]) > 30:
            samples = 30
        else:
            samples = len(slices[0])

        num = len(list_of_bandwidths)
        mase = np.zeros(num)

        for b, h in enumerate(list_of_bandwidths):
            mse = np.zeros(self.n_sections)
            for i, chunk in enumerate(slices):
                X_train, X_test = np.delete(X, chunk), X[chunk]
                y_train, y_test = np.delete(y, chunk), y[chunk]
                model = LocalPolynomialRegression(X_train, y_train, h, self.kernel_str, self.gridsize)

                runs = min(samples, len(chunk))
                mse_tmp = np.zeros(runs)
                for j, idx_test in enumerate(random.sample(list(range(0, len(chunk))), runs)):

                    y_pred = model.local_polynomial_estimation(X_test[idx_test])[
                        0
                    ]  # use model.local_polynomial_estimation instead of model.fit, because we need the estimate for one point of X_test, not projected to X_est

                    mse_tmp[j] = (y_test[idx_test] - y_pred) ** 2
                mse[i] = 1 / runs * sum(mse_tmp)
            mase[b] = 1 / self.n_sections * sum(mse)

        results = {
            "bandwidths": list_of_bandwidths,
            "MSE": mase,
            "h": list_of_bandwidths[mase.argmin()],
        }
        return results
