This package uses Local Polynomial Regression to create a fit to the data. 
The model conveniently also estimates the first and second derivative of the fit. 

The main functions are ``LocalPolynomialRegression.fit``, which creates the fit 
and ``LocalPolynomialRegressionCV.bandwidth_cv``, which finds the optimal bandwidth for the kernel which is used for the fit. 