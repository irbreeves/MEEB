from setuptools import setup, find_packages

setup(
    name="beem",
    version="0.1",
    description="Barrier Explicit Evolution Model",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Ian Reeves",
    author_email="ireeves@whoi.edu",
    url="https://github.com/irbreeves/beem",
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GPLv3 License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    keywords=["barrier", "dune", "overwash", "coastal", "aeolian", "vegetation", "earth science"],
    install_requires=open("requirements.txt", "r").read().splitlines(),
    include_package_data=True,
    packages=find_packages(),
    python_requires=">=3.6",
)
