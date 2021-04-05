import numpy as np
from matplotlib import pyplot as plt

from local_polynomial_regression.base import LocalPolynomialRegression

# simulate data
np.random.seed(1)
X = np.linspace(-np.pi, np.pi, num=150)
y_real = np.sin(X)
y = np.random.normal(0, 0.5, len(X)) + y_real

# local polynomial regression
model = LocalPolynomialRegression(X=X, y=y, h=1.011, kernel="gaussian", gridsize=100)
X_est, y_est, first, second, h = model.fit()

# plot
plt.scatter(X, y)
plt.plot(X, y_real, "grey", ls="--")
plt.plot(X_est, y_est, "r")
plt.show()