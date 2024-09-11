# *MEEB*
Mesoscale Explicit Ecogeomorphic Barrier model v1.0

Reeves, I. R. B., Ashton, A. D., Lentz, E. L., Sherwood, C. R., Passeri, D. L., & Zeigler., S. L. (in review). Projecting 
management-relevant change of undeveloped coastal barriers with the Mesoscale Explicit Ecogeomorphic Barrier model (MEEB) v1.0. 

## About
The Mesoscale Explicit Ecogeomorphic Barrier model (*MEEB*) resolves cross-shore and alongshore changes in topography and ecology 
to simulate the ecogeomorphic evolution of an undeveloped barrier or barrier segment. The model is designed to operate over 
spatiotemporal scales most relevant to coastal management practices (decades and kilometers). *MEEB* uses weekly timesteps and 
meter-scale cell size.

*MEEB* explicitly yet efficiently simulates coupled aeolian, marine, vegetation, and shoreline components of barrier 
ecogeomorphology, including dune growth, vegetation expansion and mortality, beach and foredune erosion, barrier overwash, 
and shoreline and shoreface change processes.

## Requirements
*MEEB* requires Python 3 and the libraries listed in the project's `requirements.txt` file.

## Installation

First, download the source code for *MEEB* into a new project directory. To get the source code, download the zip file:

    https://github.com/irbreeves/MEEB/archive/refs/heads/main.zip

Then, run the following from the top-level folder (the one that contains `setup.py`) to install into the current environment:

    pip install -e .

## Input Files, Parameters, and Run Scripts

Input files are located in the `/Input` directory and include initial topography and storm timeseries. A main set of 
commonly-manipulated parameters can be adjusted in the initializarion of the *MEEB* class in a model run script.

Two run scripts for *MEEB* are available:
    
`/Tools/run_MEEB.py` runs a single deterministic simulation

`/Tools/run_MEEB_Probabilistic.py` produces probabilistic projections that account for uncertainties related to future forcing
conditions and the inherent randomness of natural phenomena

