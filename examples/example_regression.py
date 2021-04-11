import numpy as np
from matplotlib import pyplot as plt

from local_polynomial_regression.base import LocalPolynomialRegression

# simulate data
np.random.seed(1)
X = np.linspace(-np.pi, np.pi, num=150)
y_real = np.sin(X)
y = np.random.normal(0, 0.3, len(X)) + y_real

# local polynomial regression
model = LocalPolynomialRegression(X=X, y=y, h=0.8469, kernel="gaussian", gridsize=100)
prediction_interval = (X.min(), X.max())
results = model.fit(prediction_interval)

# plot
plt.scatter(X, y)
plt.plot(X, y_real, "grey", ls="--", alpha=0.5, label="function")
plt.plot(results["X"], results["fit"], "r", alpha=0.9, label="fit")
plt.legend()
plt.show()
