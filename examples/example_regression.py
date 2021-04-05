import numpy as np
from local_polynomial_regression.smoothing import create_fit
from matplotlib import pyplot as plt

# simulate data
np.random.seed(1)
X = np.linspace(-np.pi, np.pi, num=150)
y_real = np.sin(X)
y = np.random.normal(0, 0.5, len(X)) + y_real

# local polynomial regression
X_est, y_est, first, second, h = create_fit(X, y, h=0.4)

# plot
plt.scatter(X, y)
plt.plot(X, y_real, "grey", ls="--")
plt.plot(X_est, y_est, "r")
plt.show()