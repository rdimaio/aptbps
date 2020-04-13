from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="aptbps",
    version=1.0,
    description="Density-adaptive distance encoding for machine learning on point clouds",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rdimaio/aptbps",
    setup_requires=["numpy", "sklearn", "tqdm", "scipy", "KDEpy"],
    install_requires=[
        "sklearn",
        "tqdm",
        "numpy",
        "scipy",
        "KDEpy"
    ],
    author="Riccardo Di Maio",
    license="MIT",
    keywords="aptbps",
    author_email="riccardodimaio11@gmail.com",
    packages=["aptbps"]
)