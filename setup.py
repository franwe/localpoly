from setuptools import setup, find_packages

setup(
    name="local_polynomial_regression",
    version="0.0.1",
    url="https://github.com/franwe/local_polynomial_regression",
    author="Franziska Wehrmann",
    author_email="franziska.wehrmann@gmail.com",
    description="Performs local polynomial regression. Returns the fit and its first and second derivative.",
    packages=find_packages(),
    install_requires=["numpy >= 1.11.1"],
)
