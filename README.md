# *MEEB*
Mesoscale Explicit Ecogeomorphic Barrier model v1.1

Journal Article: Reeves, I. R. B., Ashton, A. D., Lentz, E. L., Sherwood, C. R., Passeri, D. L., and Zeigler., S. L., in review,
Projecting management-relevant change of undeveloped coastal barriers with the Mesoscale Explicit Ecogeomorphic Barrier model (MEEB) 
v1.0: Geoscientific Model Development.

Software Package: Reeves, I. R. B., 2025, Mesoscale Explicit Ecogeomorphic Barrier Model (MEEB) v1.0: U.S. Geological Survey software release, https://doi.org/10.5066/P13N6RHA.

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

First, download the source code for *MEEB* into a new project directory.

Then, run the following from the top-level folder (the one that contains `setup.py`) to install into the current environment:

    pip install -e .

## Input Files & Parameters Values

Input files are located in the `/Input` directory and include initial topography and storm timeseries. A set of 
commonly-manipulated parameters can be adjusted in the initializarion of the *MEEB* class in a model run script (see examples below).

## Running *MEEB*

### Run Scripts

Run scripts are separate files that run model simulations and produce model output (i.e., plot results and/or save data). These 
scripts can be manipulated to alter model parameters and simulation specifications according to the needs of the modeler. Two run 
scripts for *MEEB* are available:
    
`/Tools/run_MEEB.py` runs a single deterministic simulation

`/Tools/run_MEEB_Probabilistic.py` produces probabilistic projections that account for uncertainties related to future forcing
conditions and the inherent randomness of natural phenomena

### Jupyter Notebook Tutorials

In-depth tutorials for running *MEEB* deterministically and probabilistically are stored in the `/Notebooks` folder. These Jupyter
Notebooks step through an example deterministic or probabilistic run script with descriptions and explanations for each block of code.
To run a Notebook, navigate to the main `/MEEB` directory in a *terminal* window and type:

      jupyter notebook

This will launch an interactive Notebook environment in your default web browser. In the Jupyter Notebook web browser, open on the 
`/Notebooks` folder, and double click to open one of two Notebook tutorials:

   `run_MEEB_Deterministic.ipynb` for a simple deterministic run. Start here!

   `run_MEEB_Probabilistic.ipynb` for a much more complex probabilistic projection.

Progress through the tutorials, cell-by-cell, by clicking the play button on the top ribbon. You can also edit cells to explore different
parameter values. [Click here](https://jupyter-notebook.readthedocs.io/en/stable/user-documentation.html "Jupyter Notebook Documentation")
for more documentation on Jupyter Notebooks.

### Basic approach for running a *MEEB* simulation

1) Import *MEEB* and dependencies:

        from meeb import MEEB
        import matplotlib.pyplot as plt

2) Create an instance of the *MEEB* class, here specifying a 3 yr, 500-m-long simulation with a relative sea-level rise rate of 3 mm/yr:

         meeb = MEEB(
             simulation_time_yr=3,  # [yr]
             RSLR=0.003,  # [m/yr]
             alongshore_domain_boundary_min=500,  # [m]
             alongshore_domain_boundary_max=1000,  # [m]
         )

3) Loop through time with *MEEB's* `update()` function:

        for time_step in range(int(meeb.iterations)):
            meeb.update(time_step)

4) Once the simulation finishes, plot results such as the elevation at the last time step:

        plt.matshow(meeb.topo, cmap='terrain', vmin=-1, vmax=6)
        plt.show()

Refer to the Notebook tutorials and provided run scripts referenced above for additional complexity.