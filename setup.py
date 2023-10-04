from setuptools import setup, find_packages

setup(
    name="meeb",
    version="0.2",
    description="Mesoscale Explicit Ecogeomorphic Barrier model",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Ian Reeves",
    author_email="ireeves@usgs.gov",
    url="https://github.com/irbreeves/meeb",
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GPLv3 License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    keywords=["barrier", "dune", "overwash", "coastal", "aeolian", "vegetation", "earth science"],
    install_requires=open("requirements.txt", "r").read().splitlines(),
    include_package_data=True,
    packages=find_packages(),
    python_requires=">=3.6, <3.12.*",
)
