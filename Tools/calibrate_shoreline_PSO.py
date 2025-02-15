"""
Script for calibrating MEEB shoreline parameters using Particle Swarms Optimization.

Calibrates based on fitess score for shoreline change between two timesteps.

IRBR 13 February 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import routines_meeb as routine
import copy
import time
import gc
from tabulate import tabulate
import pyswarms as ps
from joblib import Parallel, delayed
from tqdm import tqdm

from meeb import MEEB


# ___________________________________________________________________________________________________________________________________
# ___________________________________________________________________________________________________________________________________
# FUNCTIONS FOR RUNNING MODEL HINDCASTS AND CALCULATING SKILL

def model_skill(obs, sim, t0):
    """
    Perform suite of model skill assesments and return scores.
    Mask is boolean array with same size as change maps, with cells to be excluded from skill analysis set to FALSE.
    """
    if np.isnan(np.sum(sim)):
        nse = -1e10
        rmse = -1e10
        nmae = -1e10
        mass = -1e10
        bss = -1e10

    else:
        # _____________________________________________
        # Nash-Sutcliffe Model Efficiency
        """The closer the score is to 1, the better the agreement. If the score is below 0, the mean observed value is a better predictor than the model."""
        A = np.nanmean(np.square(np.subtract(obs, sim)))
        B = np.nanmean(np.square(np.subtract(obs, np.nanmean(obs))))
        nse = 1 - A / B

        # _____________________________________________
        # Root Mean Square Error
        rmse = np.sqrt(np.nanmean(np.square(sim - obs)))

        # _____________________________________________
        # Normalized Mean Absolute Error
        nmae = np.nanmean(np.abs(sim - obs)) / (np.nanmax(obs) - np.nanmin(obs))  # (np.nanstd(np.abs(obs[mask])))

        # _____________________________________________
        # Mean Absolute Skill Score
        mass = 1 - np.nanmean(np.abs(sim - obs)) / np.nanmean(np.abs(t0 - obs))

        # _____________________________________________
        # Brier Skill Score
        """A skill score value of zero means that the score for the predictions is merely as good as that of a set of baseline or reference or default predictions, 
        while a skill score value of one (100%) represents the best possible score. A skill score value less than zero means that the performance is even worse than 
        that of the baseline or reference predictions (i.e., the baseline matches the final field profile more closely than the simulation output)."""
        MSE = np.nanmean(np.square(np.abs(np.subtract(sim, obs))))
        MSEref = np.nanmean(np.square(np.abs(np.subtract(t0, obs))))
        bss = 1 - MSE / MSEref

    return nse, rmse, nmae, mass, bss


def shoreline_fitness(solution):
    """Run a hindcast this particular combintion of parameter values, and return fitness value of simulated to observed."""

    # Create an instance of the BMI class
    meeb = MEEB(
        name=name,
        simulation_time_yr=hindcast_duration,
        init_filename=start,
        hindcast=True,
        simulation_start_date=startdate,
        alongshore_domain_boundary_min=ymin,
        alongshore_domain_boundary_max=ymax,
        cellsize=cellsize,
        RSLR=0.006,
        seeded_random_numbers=False,
        storm_timeseries_filename='StormTimeSeries_1979-2020_NCB-CE_Beta0pt039_BermEl1pt78.npy',
        storm_list_filename='SyntheticStorms_NCB-CE_10k_1979-2020_Beta0pt039_BermEl1pt78.npy',
        # --- Aeolian --- #
        saltation_length=2,
        saltation_length_rand_deviation=1,
        slabheight=0.02,
        p_dep_sand=0.09,  # Q = hs * L * n * pe/pd
        p_dep_sand_VegMax=0.17,
        p_ero_sand=0.08,
        entrainment_veg_limit=0.09,
        saltation_veg_limit=0.37,
        repose_threshold=0.37,
        shadowangle=12,
        repose_bare=20,
        repose_veg=30,
        wind_rose=(0.91, 0.04, 0.01, 0.04),  # (right, down, left, up)
        # --- Storms --- #
        Rin=232,
        Cs=0.0235,
        MaxUpSlope=1.5,
        marine_flux_limit=1,
        Kow=0.0003615,
        mm=1.05,
        overwash_substeps=25,
        beach_equilibrium_slope=0.021,
        swash_erosive_timescale=1.51,
        beach_substeps=1,
        flow_reduction_max_spec1=0.02,
        flow_reduction_max_spec2=0.05,
        # --- Shoreline --- #
        wave_asymmetry=0.6,
        wave_high_angle_fraction=0.39,
        mean_wave_height=0.98,
        mean_wave_period=6.6,
        alongshore_section_length=25,
        shoreline_diffusivity_coefficient=solution[0],
        estimate_shoreface_parameters=True,
        # --- Veg --- #
        sp1_lateral_probability=0.2,
        sp2_lateral_probability=0.2,
        sp1_pioneer_probability=0.05,
        sp2_pioneer_probability=0.03,
        # MY GRASS
        sp1_a=-1.2,
        sp1_b=-0.2,  # Mullins et al. (2019)
        sp1_c=0.5,
        sp1_d=1.2,
        sp1_e=2.1,
        sp1_peak=0.2,
        # MY SHRUB
        sp2_a=-1.0,
        sp2_b=-0.2,  # Conn and Day (1993)
        sp2_c=0.0,
        sp2_d=0.2,
        sp2_e=2.1,
        sp2_peak=0.05,
    )

    # Loop through time
    for time_step in range(int(meeb.iterations)):
        # Run time step
        meeb.update(time_step)

    # __________________________________________________________________________________________________________________________________
    # ASSESS MODEL SKILL

    # Simulated shorelines
    shoreline_sim_start = meeb.x_s_TS[0, :]
    shoreline_sim_end = meeb.x_s_TS[-1, :]

    # Observed shorelines
    shoreline_obs_start = routine.ocean_shoreline(topo_start, meeb.MHW_init)
    shoreline_obs_end = routine.ocean_shoreline(topo_end_obs, meeb.MHW)

    # Shoreline changes
    shoreline_change_sim = shoreline_sim_end - shoreline_sim_start
    shoreline_change_obs = shoreline_obs_end - shoreline_obs_start

    # Model Skill
    nse, rmse, nmae, mass, bss = model_skill(shoreline_change_obs, shoreline_change_sim, np.zeros(shoreline_change_sim.shape))  # Vegetation skill based on percent cover change

    score = bss

    return score


def opt_func(X):
    """Runs a parallelized batch of hindcast simulations and returns a fitness result for each"""

    with routine.tqdm_joblib(tqdm(desc="Iteration Progress", total=swarm_size)) as progress_bar:
        solutions = Parallel(n_jobs=n_jobs)(delayed(shoreline_fitness)(X[i, :]) for i in range(swarm_size))

    return np.array(solutions) * -1


# ___________________________________________________________________________________________________________________________________
# ___________________________________________________________________________________________________________________________________
# SET UP CALIBRATION

start_time = time.time()  # Record time at start of calibration

# __________________________________________________________________________________________________________________________________
# VARIABLES AND INITIALIZATIONS

# 2014 - 2018
start = "Init_NCB-NewDrum-Ocracoke_2014_PostSandy_NCFMP-Planet_2m_HighDensity.npy"
stop = "Init_NCB-2200-34200_2018_USACE_PostFlorence_2m.npy"
startdate = '20140406'
hindcast_duration = 4.34
cellsize = 2  # [m]

MHW = 0.39  # [m NAVD88] Initial
name = '1000-12500, 2014-2018, BSS'

# Define Alongshore Coordinates of Domain
ymin = 1000
ymax = 12500
xmin = 450
xmax = 1350

# Resize According to Cellsize
ymin = int(ymin / cellsize)  # Alongshore
ymax = int(ymax / cellsize)  # Alongshore
xmin = int(xmin / cellsize)  # Cross-shore
xmax = int(xmax / cellsize)  # Cross-shore

# ____________________________________

# Initial Observed Topo
Init = np.load("Input/" + start)
# Final Observed
End = np.load("Input/" + stop)

# Transform Initial Observed Topo
topo_i = Init[0, ymin: ymax, xmin: xmax].copy()  # [m]
topo_start = copy.deepcopy(topo_i)  # [m] INITIAL TOPOGRPAHY

# Transform Final Observed Topo
topo_e = End[0, ymin: ymax, xmin: xmax].copy()  # [m]
topo_end_obs = copy.deepcopy(topo_e)  # [m] FINAL OBSERVED TOPOGRAPHY

del Init, End
gc.collect()

# ___________________________________________________________________________________________________________________________________
# ___________________________________________________________________________________________________________________________________
# PARTICLE SWARMS OPTIMIZATION

# _____________________________________________
# Prepare Particle Swarm Parameters

iterations = 7
swarm_size = 7
dimensions = 1  # Number of free paramters
n_jobs = 7
options = {'c1': 1.5, 'c2': 1.5, 'w': 0.5}
"""
w: Inertia weight constant. [0-1] Determines how much the particle keeps on with its previous velocity (i.e., speed and direction of the search). 
c1 & c2: Cognitive and the social coefficients, respectively. Control how much weight should be given between refining the search result of the
particle itself and recognizing the search result of the swarm; Control the trade off between exploration and exploitation.
"""

bounds = (
    # Minimum
    np.array([-0.001]),    # pioneer_probability
    # Maximum
    np.array([0.12])      # pioneer_probability
)


# _____________________________________________
# Call an instance of PSO
optimizer = ps.single.GlobalBestPSO(n_particles=swarm_size,
                                    dimensions=dimensions,
                                    options=options,
                                    bounds=bounds,
                                    oh_strategy={"w": 'exp_decay', 'c1': 'lin_variation'})

# _____________________________________________
# Perform optimization
solution_fitness, best_solution = optimizer.optimize(opt_func, iters=iterations)


# _____________________________________________
# Print Results

SimDuration = time.time() - start_time
print()
print("Elapsed Time: ", SimDuration, "sec")
print()
print("Complete.")

best_solution = np.expand_dims(best_solution, axis=1)
print()
print(tabulate({
    "BEST SOLUTION": ["BSS"],
    "sp1_b": [best_solution[0]],
    "Score": [solution_fitness]
    }, headers="keys", floatfmt=(None, ".3f", ".3f"))
)

# _____________________________________________
# Plot Results
plt.plot(np.array(optimizer.cost_history) * -1)
plt.ylabel('Fitness (BSS)')
plt.xlabel('Iteration')
