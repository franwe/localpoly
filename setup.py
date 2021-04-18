from setuptools import setup, find_packages
import pathlib

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

setup(
    name="localpoly",
    version="0.1.0",
    url="https://github.com/franwe/localpoly",
    project_urls={
        "Documentation" : "https://localpoly.readthedocs.io/en/latest/",
        "Code" : "https://github.com/franwe/localpoly"
    }
    author="Franziska Wehrmann",
    author_email="franziska.wehrmann@gmail.com",
    description="Performs Local Polynomial Regression.",
    long_description=README,
    long_description_content_type="text/markdown",
    license="MIT",
    packages=["localpoly"],
    install_requires=["numpy >= 1.11.1", "pandas >= 1.2.4", "scipy >= 1.6.2"],
)
