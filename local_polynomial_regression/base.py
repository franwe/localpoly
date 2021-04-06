import random
import numpy as np

from .utils.helpers import sort_values_by_X, create_partitions
from .config.core import config
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
        self.prediction_interval = (self.X.min(), self.X.max())

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

    def fit(self, prediction_interval):
        X_min, X_max = prediction_interval
        X_domain = np.linspace(X_min, X_max, self.gridsize)
        fit = np.zeros(len(X_domain))
        first = np.zeros(len(X_domain))
        second = np.zeros(len(X_domain))
        for i, x in enumerate(X_domain):
            b0, b1, b2, W_hi = self.local_polynomial_estimation(x)
            fit[i] = b0
            first[i] = b1
            second[i] = b2
        return X_domain, fit, first, second, self.h  # TODO: remove h ?


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
        coarse_list_of_bandwidths,
    ):
        """[summary]

        Args:
            bandwidths (list): coarse list of bandwidths, it is suggested to give values around the Silverman bandwidth

        Returns:
            [dict]: fine results and coarse results of bandwidth search. results as in bandwdith_cv_sampling
        """
        # 1) coarse parameter search
        coarse_results = self.bandwidth_cv_sampling(coarse_list_of_bandwidths)

        # 2) fine parameter search, around minimum of first search
        coarse_h = coarse_results["h"]
        stepsize = coarse_list_of_bandwidths[1] - coarse_list_of_bandwidths[0]
        fine_list_of_bandwidths = np.linspace(coarse_h - (stepsize * 1.1), coarse_h + (stepsize * 1.1), 10)

        fine_results = self.bandwidth_cv_sampling(fine_list_of_bandwidths)

        return {
            "fine results": fine_results,
            "coarse results": coarse_results,
        }

    def bandwidth_cv_sampling(self, list_of_bandwidths):
        X_sorted, y_sorted = sort_values_by_X(self.X, self.y)
        X_sections = create_partitions(X_sorted, y_sorted, self.n_sections, self.sampling)
        max_comparisons_per_section = min(len(X_sections[0]), 30)

        num = len(list_of_bandwidths)
        mse_bw = np.zeros(num)  # for each bandwidth have mse - loss function

        # for bandwidth h in list_of_bandwidths
        for b, h in enumerate(list_of_bandwidths):
            # take out chunks of our data and do leave-out-prediction
            runs, mse = 0, 0
            for i, section in enumerate(X_sections):
                X_train, X_test = np.delete(X_sorted, section), X_sorted[section]
                y_train, y_test = np.delete(y_sorted, section), y_sorted[section]
                model = LocalPolynomialRegression(X_train, y_train, h, self.kernel_str, self.gridsize)
                X_est, y_est, first, second, h = model.fit(self.prediction_interval)

                max_section_comparison_length = min(len(X_test), max_comparisons_per_section)
                random.seed(config.model_config.random_state)
                for random_section_element in random.sample(range(len(X_test)), max_section_comparison_length):
                    x, y = X_test[random_section_element], y_test[random_section_element]
                    # compare to y_est to closest y_test
                    nearest_x_idx = min(range(len(X_est)), key=lambda i: abs(X_est[i] - x))
                    mse += (y - y_est[nearest_x_idx]) ** 2
                    runs += 1

            mse_bw[b] = 1 / runs * mse

        results = {
            "bandwidths": list_of_bandwidths,
            "MSE": mse_bw,
            "h": list_of_bandwidths[mse_bw.argmin()],
        }
        return results
