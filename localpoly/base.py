import random
import numpy as np

from .utils.helpers import sort_values_by_X, create_partitions
from .utils.kernels import kernel_dict

RANDOM_STATE = 1


class LocalPolynomialRegression:
    """Local polynomial regression.

    LocalPolynomialRegression fits a polynomial of degree 3 in to the sourrounding of each point.
    The surrounding is realized by a kernel with bandwidth h. The regression returns the fit, as
    well as its first and second derivative.

    Parameters:
        X: X-values of data that is to be fitted (explanatory variable)
        y: y-values of data that is to be fitted (observations)
        h: bandwidth for the kernel
        gridsize: desired size of the fit (granularity)
        kernel_str: the name of the kernel as a string "gaussian"
    """

    def __init__(self, X, y, h, kernel="gaussian", gridsize=100):
        self.X = X
        self.y = y
        self.h = h
        self.kernel_str = kernel
        self.kernel = kernel_dict[self.kernel_str]
        self.gridsize = gridsize

    def localpoly(self, x):
        """Calculates estimate for position x via Local Polynomial Regression.

        The usage of Local Polynomial Regression allows to not only calculate the estimate, but also its first and
        second derivative in this point. Data (X, y) and regression settings (kernel, h) are saved in self.

        Args:
            x (float): Position for which to calculate the estimated value.

        Returns:
            dict: Results of regression. The estimated value for point x, its first and second derivative in this point
            and the weight vector of the influence of the surrounding points.::

                {"fit": beta[0], "first": beta[1], "second": beta[2], "weight": W_hi}
        """
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
        return {"fit": beta[0], "first": beta[1], "second": beta[2], "weight": W_hi}

    def fit(self, prediction_interval):
        """Fit the Local Polynomial Regression model for the prediction interval.

        Args:
            prediction_interval (tuple): interval for which the prediction is calculated

        Returns:
            dict: Results of the fit. The estimated function (fit) in the prediction interval (X) and its first and
            second derivative::

                {
                    'X' : X_domain,    # prediction interval of fit
                    'fit': fit,        # fit of the function at point x
                    'first': first,    # first derivative at point x
                    'second': second,  # second derivative at point x
                }
        """
        X_min, X_max = prediction_interval
        X_domain = np.linspace(X_min, X_max, self.gridsize)
        fit = np.zeros(len(X_domain))
        first = np.zeros(len(X_domain))
        second = np.zeros(len(X_domain))
        for i, x in enumerate(X_domain):
            results = self.localpoly(x)
            fit[i] = results["fit"]
            first[i] = results["first"]
            second[i] = results["second"]
        return {"X": X_domain, "fit": fit, "first": first, "second": second}


class LocalPolynomialRegressionCV(LocalPolynomialRegression):
    """Bandwidth Selection via Cross Validation for Local Polynomial Regression.

    LocalPolynomialRegressionCV performs the parameter optimization for LocalPolynomialRegression. The optimal Bandwidth
    highly depends on the data (X, y) and the kernel.

    Args:
        X (np.array): X-values of data that is to be fitted (explanatory variable)
        y (np.array): y-values of data that is to be fitted (observations)
        kernel (str, optional): Name of the kernel. Defaults to "gaussian".
        gridsize (int, optional): Desired size of the fit - granularity. Defaults to 100.
        n_sections (int, optional): Amount of sections to devide the dataset in cross validation (k-folds). Defaults to 10.
        loss (str, optional): Loss function for optimization. Defaults to "MSE".
        sampling (str, optional): Whether the dataset should be partitioned "random" or as "slicing". Defaults to "random".

    Attributes:
        prediction_interval: Interval in which to calculate the estimates, automatically set to (X.min(), X.max())
    """

    def __init__(
        self,
        X,
        y,
        kernel="gaussian",
        gridsize=100,
        n_sections=10,
        loss="MSE",
        sampling="random",
    ):
        self.n_sections = n_sections
        self.loss = loss
        self.sampling = sampling
        self.prediction_interval_ = (X.min(), X.max())

        # invoking the __init__ of the parent class
        LocalPolynomialRegression.__init__(self, X=X, y=y, h=None, kernel=kernel, gridsize=gridsize)

    def bandwidth_cv(
        self,
        coarse_list_of_bandwidths,
    ):
        """Cross Validation for Bandwidth optimization.

        The CV Routine is performed twice. First, for a ``coarse_list_of_bandwidths``, then on a finer grid which spans
        around the first optimal value, ``fine_list_of_bandwidths``.

        Args:
            coarse_list_of_bandwidths (list): coarse list of bandwidths, it is suggested to give values around the Silverman bandwidth

        Returns:
            dict: fine results and coarse results of bandwidth search::

                {
                    "fine results": {
                        "bandwidths": fine_list_of_bandwidths,
                        "MSE": # mean squared errors for bandwidths,
                        "h": # optimal bandwidth within fine_list_of_bandwidths,
                    },
                    "coarse results": {
                        # ... same as above but with coarse_list_of_bandwidths
                    },
                }
        """
        # 1) coarse parameter search
        coarse_results = self._bandwidth_cv_sampling(coarse_list_of_bandwidths)

        # 2) fine parameter search, around minimum of first search
        coarse_h = coarse_results["h"]
        stepsize = coarse_list_of_bandwidths[1] - coarse_list_of_bandwidths[0]
        fine_list_of_bandwidths = np.linspace(coarse_h - (stepsize * 1.1), coarse_h + (stepsize * 1.1), 10)

        fine_results = self._bandwidth_cv_sampling(fine_list_of_bandwidths)

        return {
            "fine results": fine_results,
            "coarse results": coarse_results,
        }

    def _bandwidth_cv_sampling(self, list_of_bandwidths):
        """The actual CV.

        First, the data is sorted by X, which is important in case the sampling type "slicing" was selected. Then the
        partitions for the CV are created (slices or random). Finally the CV is performed.
        Args:
            list_of_bandwidths (list): list of bandwidths that are evaluated

        Returns:
            dict: Results of CV. List of bandwidth that were evaluated. MSE for each bandwidth. Optimal bandwidth "h".
        """
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
                max_section_comparison_length = min(len(X_test), max_comparisons_per_section)
                model = LocalPolynomialRegression(X_train, y_train, h, self.kernel_str, self.gridsize)
                random.seed(RANDOM_STATE)
                # for random y in y_test, extimate y_hat and calculate mse
                for idx_test in random.sample(range(len(X_test)), max_section_comparison_length):
                    y_hat = model.localpoly(X_test[idx_test])["fit"]
                    y_true = y_test[idx_test]
                    mse += (y_true - y_hat) ** 2
                    runs += 1

            mse_bw[b] = 1 / runs * mse

        return {
            "bandwidths": list_of_bandwidths,
            "MSE": mse_bw,
            "h": list_of_bandwidths[mse_bw.argmin()],
        }
