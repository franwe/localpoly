# localpoly

This package uses Local Polynomial Regression to create a fit to the data. The model conveniently also estimates the first and second derivative of the fit.

The main functions are LocalPolynomialRegression.fit, which creates the fit and LocalPolynomialRegressionCV.bandwidth_cv, which finds the optimal bandwidth for the kernel which is used for the fit.

Read the documentation for the theoretical background and examples. 

## Installation

Via pip
```
    $ pip install localpoly
```

Or via download from git:

```
    $ pip install git+https://github.com/franwe/localpoly#egg=localpoly
```

Note that in order to avoid potential conflicts with other packages it is strongly recommended to use a virtual environment (venv) or a conda environment.

![alt text][logo]

[logo]: docs/_static/example_fit.png "Example fit"