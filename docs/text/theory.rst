Theoretical Background
================================================================

Assume that the function of interest :math:`y` is a real-valued function and is twice differentiable at :math:`z=x` for all 
:math:`x`, then there exists a linear approximation at this point. The second degree **Taylor expansion** of the function :math:`y` 
centered around :math:`z` in a neighborhood of :math:`x` is given by:

.. math::

    \begin{equation}
        y(z) \approx y(x) + \frac{\partial y(x)}{\partial X}
            + \frac{1}{2} \frac{\partial^2 y(x)}{\partial X^2} (z-x)^2
    \end{equation}

This representation is a linear combination and can be reformulated as a linear model, in order to solve it with 
**Least Squared Regression**.
Translating the representation of the Taylor expansion to the Linear Model leads to: 

.. math::

    \begin{align*}
        y_i &= \beta_0 + \beta_1  x_{i1} + \beta_2  x_{i2} + \varepsilon_i\;\;\;\;\; 
        \rightarrow  \;\; y = \mathbf{X} \beta + \varepsilon \;\;\;\;\; \text{(matrix notation)}\\
        \text{where}\\
        \mathbf{X} &= \begin{pmatrix}
            1      & (X_1 - x) & (X_1 - x)^2 \\
            1      & (X_2 - x) & (X_2 - x)^2 \\
            \vdots & \vdots      & \vdots        \\
            1      & (X_n - x) & (X_n - x)^2 \\
        \end{pmatrix}      , \;\; 
        y = \begin{pmatrix}
            y_1 \\
            y_2 \\
            \vdots   \\
            y_n \\
        \end{pmatrix}       , \;\;
        \beta = \begin{pmatrix}
            \beta_0 \\
            \beta_1 \\
            \beta_2 \\
        \end{pmatrix}   
        = \begin{pmatrix}
            y(x)\\
            \frac{\partial y(x)}{\partial X}\\
            \frac{\partial^2 y(x)}{\partial X^2}\\
        \end{pmatrix}\\
    \end{align*}

Where :math:`y` is the target (observed values :math:`y_i, i = 1, \ldots, n`), 
the :math:`(z-x)`-terms are the regressors (:math:`n \times 3` dimensional matrix :math:`\mathbf{X}`, containing the explanatory variable :math:`X_i`) and the 
approximated fit and its derivatives will be found in the vector of coefficients :math:`\beta`.

.. math::
    \begin{equation}
        \widetilde{\beta}(x) = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top y 
    \end{equation}

| In order to achieve an estimation that is dependant on the neighborhood of :math:`x`, a **Kernel** is added to the least-squares minimization problem. 
| Unlike in ordinary least suqares regression and linear regression, the errors covariance matrix is therefore different from the identity matrix :math:`\mathbb{1}`, but is a diagonal matrix of the Kernel: 

.. math::
    \begin{equation}
        \mathbf{W} = diag\{Ker(X_i-x)\}
    \end{equation}

The minimization problem is solved by the weighted least squares estimator :math:`\hat{\beta}`:

.. math::
    \begin{equation}
        \hat{\beta}(x) = (\mathbf{X}^\top  \mathbf{W} \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{W}y 
    \end{equation}

The combination of these three key components lead the the term **Local Polynomial Regression**
(local: Weights/Kernel, polynomial: Taylor expansion, where the function is represented in polynomials, regression: Least Squared Regression).

Resources:
----------

.. [1] Nonparametric and Semiparametric Models - Chapter: Nonparametric Regression - W. K. Haerdle - Springer Berlin Heidelberg - 2004