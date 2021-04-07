import numpy as np
from matplotlib import pyplot as plt

from local_polynomial_regression.base import LocalPolynomialRegressionCV


np.random.seed(1)
X = np.linspace(-np.pi, np.pi, num=150)
y_real = np.sin(X)
y = np.random.normal(0, 0.3, len(X)) + y_real

# Bandwidth selection for local polynomial regression
model_cv = LocalPolynomialRegressionCV(
    X=X,
    y=y,
    kernel="gaussian",
    n_sections=15,
    loss="MSE",
    sampling="random",
)


results = model_cv.bandwidth_cv(np.linspace(0.5, 1.5, 10))
print(f"Optimal bandwidth: {results['fine results']['h']}")
plt.plot(results["coarse results"]["bandwidths"], results["coarse results"]["MSE"])
plt.plot(results["fine results"]["bandwidths"], results["fine results"]["MSE"])
plt.show()
