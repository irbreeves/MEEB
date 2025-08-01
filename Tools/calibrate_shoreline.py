"""
Script for calibrating MEEB shoreline parameters.

Calibrates based on fitess score for shoreline change between two timesteps.

IRBR 22 July 2025
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

    return [nse, rmse, nmae, mass, bss]


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
        crossshore_domain_boundary_min=xmin,
        crossshore_domain_boundary_max=xmax,
        cellsize=cellsize,
        RSLR=0.006,
        seeded_random_numbers=True,
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
        Rin=245,
        Cs=0.0235,
        MaxUpSlope=1.5,
        marine_flux_limit=1,
        Kow=0.0003615,
        mm=1.05,
        overwash_substeps=25,
        beach_equilibrium_slope=0.021,
        swash_erosive_timescale=1.51,
        beach_substeps=1,
        # --- Shoreline --- #
        wave_asymmetry=0.6,
        wave_high_angle_fraction=0.39,
        mean_wave_height=0.98,
        mean_wave_period=6.6,
        alongshore_section_length=25,
        shoreline_diffusivity_coefficient=solution,
        estimate_shoreface_parameters=True,
        # --- Veg --- #
    )

    # Loop through time
    for time_step in range(int(meeb.iterations)):
        # Run time step
        meeb.update(time_step)

    # __________________________________________________________________________________________________________________________________
    # ASSESS MODEL SKILL

    # Beginning shoreline location
    shoreline_start = routine.ocean_shoreline(meeb.topo_TS[:, :, 0], meeb.MHW_init)

    # Simulated shoreline location
    shoreline_sim_end = routine.ocean_shoreline(meeb.topo_TS[:, :, -1], meeb.MHW)

    # Observed shoreline location
    shoreline_obs_end = routine.ocean_shoreline(topo_end_obs, meeb.MHW)

    # Shoreline changes
    shoreline_change_sim = shoreline_sim_end - shoreline_start
    shoreline_change_obs = shoreline_obs_end - shoreline_start

    # Model Skill
    score = model_skill(shoreline_change_obs, shoreline_change_sim, np.zeros(shoreline_change_sim.shape))

    return score


def opt_func(X):
    """Runs a parallelized batch of hindcast simulations and returns a fitness result for each"""

    with routine.tqdm_joblib(tqdm(desc="Iteration Progress", total=num_values)):
        solutions = Parallel(n_jobs=n_jobs)(delayed(shoreline_fitness)(X[i]) for i in range(num_values))

    return np.array(solutions)


# ___________________________________________________________________________________________________________________________________
# ___________________________________________________________________________________________________________________________________
# SET UP CALIBRATION

start_time = time.time()  # Record time at start of calibration

# __________________________________________________________________________________________________________________________________
# VARIABLES AND INITIALIZATIONS

# 2014 - 2018
start = "Init_NCB-NewDrum-Ocracoke_2005_2m.npy"
stop = "Init_NCB-NewDrum-Ocracoke_2018_PostFlorence_2m.npy"
startdate = '20051126'
hindcast_duration = 12.35
cellsize = 2  # [m]

MHW = 0.39  # [m NAVD88] Initial
name = '200-27200, 2005-2018, BSS'

# Define Alongshore Coordinates of Domain
ymin = 200
ymax = 27200
xmin = 250
xmax = 1550

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
# CALIBRATION

values = np.arange(0.0, 0.11, 0.01)

num_values = len(values)
n_jobs = num_values

# _____________________________________________
# Perform optimization
solution_fitness = opt_func(values)

nse = solution_fitness[:, 0]
rmse = solution_fitness[:, 1]
nmae = solution_fitness[:, 2]
mass = solution_fitness[:, 3]
bss = solution_fitness[:, 4]

# _____________________________________________
# Print Results

SimDuration = time.time() - start_time
print()
print("Elapsed Time: ", SimDuration, "sec")
print()
print("Complete.")


# _____________________________________________
# Plot Results
fig = plt.figure()
ax1 = fig.add_subplot(221)
ax1.plot(values, bss)
plt.ylabel('Fitness (BSS)')
plt.xlabel('Kd Value')
plt.title('BSS')

ax2 = fig.add_subplot(222)
ax2.plot(values, nse)
plt.ylabel('Fitness (NSE)')
plt.xlabel('Kd Value')
plt.title('NSE')

ax3 = fig.add_subplot(223)
ax3.plot(values, nmae)
plt.ylabel('Fitness (NMAE)')
plt.xlabel('Kd Value')
plt.title('NMAE')

ax4 = fig.add_subplot(224)
ax4.plot(values, rmse)
plt.ylabel('Fitness (RMSE)')
plt.xlabel('Kd Value')
plt.title('RMSE')

plt.tight_layout()
plt.show()
