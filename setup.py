from setuptools import setup, find_packages

setup(
    name="localpoly",
    version="0.0.1",
    url="https://github.com/franwe/localpoly",
    author="Franziska Wehrmann",
    author_email="franziska.wehrmann@gmail.com",
    description="Performs local polynomial regression. Returns the fit and its first and second derivative. Includes cross validation for bandwidth optimization",
    packages=find_packages(),
    install_requires=["numpy >= 1.11.1", "pandas >= 1.2.4", "scipy >= 1.6.2"],
)
