from setuptools import setup, find_packages
import pathlib

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

setup(
    name="localpoly",
    version="0.1.3",
    url="https://github.com/franwe/localpoly",
    project_urls={
        "Documentation": "https://localpoly.readthedocs.io/en/latest/",
        "Code": "https://github.com/franwe/localpoly",
    },
    author="franwe",
    author_email="franziska.wehrmann@gmail.com",
    description="Performs Local Polynomial Regression.",
    long_description=README,
    long_description_content_type="text/markdown",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3",
    install_requires=["numpy", "pandas", "scipy"],
)
