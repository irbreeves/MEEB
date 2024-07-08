"""
Script for calibrating MEEB vegetation parameters using Particle Swarms Optimization.

Calibrates based on fitess score for morphologic and ecologic change between two timesteps.

IRBR 8 July 2024
"""

import numpy as np
import matplotlib.pyplot as plt
import routines_meeb as routine
import copy
import time
from tabulate import tabulate
import pyswarms as ps
from joblib import Parallel, delayed
from tqdm import tqdm

from meeb import MEEB


# ___________________________________________________________________________________________________________________________________
# ___________________________________________________________________________________________________________________________________
# FUNCTIONS FOR RUNNING MODEL HINDCASTS AND CALCULATING SKILL

def model_skill_categorical(obs, sim, catmask):
    """
    Perform categorical skill assesment and return scores.
    Mask is boolean array with same size as change maps, with cells to be excluded from skill analysis set to FALSE.
    """

    threshold = 0.02
    sim_loss = sim < -threshold
    sim_gain = sim > threshold
    sim_no_change = np.logical_and(sim <= threshold, -threshold <= sim)
    obs_loss = obs < -threshold
    obs_gain = obs > threshold
    obs_no_change = np.logical_and(obs <= threshold, -threshold <= obs)

    cat = np.zeros(obs.shape)
    cat[np.logical_and(sim_loss, obs_loss)] = 1  # Hit
    cat[np.logical_and(sim_gain, obs_gain)] = 1  # Hit
    cat[np.logical_and(sim_loss, ~obs_loss)] = 2  # False Alarm
    cat[np.logical_and(sim_gain, ~obs_gain)] = 2  # False Alarm
    cat[np.logical_and(sim_no_change, obs_no_change)] = 3  # Correct Reject
    cat[np.logical_and(sim_no_change, ~obs_no_change)] = 4  # Miss

    hits = np.count_nonzero(cat[catmask] == 1)
    false_alarms = np.count_nonzero(cat[catmask] == 2)
    correct_rejects = np.count_nonzero(cat[catmask] == 3)
    misses = np.count_nonzero(cat[catmask] == 4)
    J = hits + false_alarms + correct_rejects + misses

    if J > 0:
        # Percentage Correct
        """Ratio of correct predictions as a fraction of the total number of forecasts. Scores closer to 1 (100%) are better."""
        PC = (hits + correct_rejects) / J

        # Heidke Skill Score
        """The percentage correct, corrected for the number expected to be correct by chance. Scores closer to 1 (100%) are better."""
        G = ((hits + false_alarms) * (hits + misses) / J ** 2) + ((misses + correct_rejects) * (false_alarms + correct_rejects) / J ** 2)  # Fraction of predictions of the correct categories (H and C) that would be expected from a random choice
        if G < 1:
            HSS = (PC - G) / (1 - G)  # The percentage correct, corrected for the number expected to be correct by chance
        else:
            HSS = 1
    else:
        PC = -1e10
        HSS = -1e10

    return PC, HSS


def veg_fitness(solution):
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
        saltation_length=5,
        saltation_length_rand_deviation=2,
        p_dep_sand=0.22,
        p_dep_sand_VegMax=0.54,
        p_ero_sand=0.10,
        entrainment_veg_limit=0.10,
        saltation_veg_limit=0.35,
        shadowangle=12,
        repose_bare=20,
        repose_veg=30,
        wind_rose=(0.81, 0.04, 0.06, 0.09),  # (right, down, left, up)
        groundwater_depth=0.4,
        # --- Storms --- #
        Rin=249,
        Cs=0.0283,
        MaxUpSlope=1.5,
        marine_flux_limit=1,
        Kow=0.0001684,
        mm=1.04,
        overwash_substeps=50,
        beach_equilibrium_slope=0.022,
        swash_erosive_timescale=1.48,
        beach_substeps=25,
        flow_reduction_max_spec1=0.02,
        flow_reduction_max_spec2=0.05,
        # --- Shoreline --- #
        wave_asymetry=0.6,
        wave_high_angle_fraction=0.39,
        mean_wave_height=0.98,
        mean_wave_period=6.6,
        alongshore_section_length=25,
        estimate_shoreface_parameters=True,
        # --- Veg Params --- #
        sp1_a=-1.5,  # Vertice a, spec1. vegetation growth based on Nield and Baas (2008)ion[0],
        sp1_b=solution[0],
        sp1_c=solution[1],
        sp1_d=solution[2],
        sp1_e=2.2,
        sp2_a=-1.6,
        sp2_b=solution[3],
        sp2_c=solution[4],
        sp2_d=solution[5],
        sp2_e=2.1,
        sp1_peak=solution[6],
        sp2_peak=solution[7],
        sp1_lateral_probability=solution[8],
        sp1_pioneer_probability=solution[9],
    )

    # Loop through time
    for time_step in range(int(meeb.iterations)):
        # Run time step
        meeb.update(time_step)

    # __________________________________________________________________________________________________________________________________
    # ASSESS MODEL SKILL

    # Veg change
    veg_end_sim = meeb.veg
    veg_change_sim = veg_end_sim - veg_start  # [m]
    veg_change_obs = veg_end_obs - veg_start  # [m]
    veg_present_sim = veg_end_sim > 0.05  # [bool]
    veg_present_obs = veg_end_obs > 0.05  # [bool]

    # Topo changes
    topo_end_sim = meeb.topo  # [m MHW]
    mhw_end_sim = meeb.slabheight  # [m NAVD88]

    # Subaerial mask
    subaerial_mask = np.logical_and(topo_end_sim > mhw_end_sim, topo_end_obs > mhw_end_sim)  # [bool] Mask for every cell above water

    # Cross-shore range mask
    range_mask = np.ones(veg_end_sim.shape)  # [bool] Mask for every cell between two cross-shore locations
    range_mask[:, :835] = False
    range_mask[:, 950:] = False

    # Elevation mask
    elev_mask = veg_end_sim > 2.0  # [bool] Mask for every cell above water

    # Choose combination of masks to use
    mask = subaerial_mask.copy()  # np.logical_and(subaerial_mask)  # [bool] Combined mask used for analysis

    # Optional: Reduce Resolutions
    if ResReduc:
        veg_change_obs = routine.reduce_raster_resolution(veg_change_obs, reduc)
        veg_change_sim = routine.reduce_raster_resolution(veg_change_sim, reduc)
        veg_present_obs = routine.reduce_raster_resolution(veg_present_obs, reduc)
        veg_present_sim = routine.reduce_raster_resolution(veg_present_sim, reduc)
        mask = (routine.reduce_raster_resolution(mask, reduc)) == 1

        veg_present_obs = np.rint(veg_present_obs).astype(bool)  # Set back to bool type by rounding
        veg_present_sim = np.rint(veg_present_sim).astype(bool)  # Set back to bool type by rounding

    # Model Skill
    pcc, hssc = model_skill_categorical(veg_change_obs, veg_change_sim, mask)  # Vegetation skill based on percent cover change
    pcp, hssp = model_skill_categorical(veg_present_obs, veg_present_sim, mask)  # Vegetation skill based on presence or absence

    score = hssp

    return score


def opt_func(X):
    """Runs a parallelized batch of hindcast simulations and returns a fitness result for each"""

    with routine.tqdm_joblib(tqdm(desc="Iteration Progress", total=swarm_size)) as progress_bar:
        solutions = Parallel(n_jobs=n_jobs)(delayed(veg_fitness)(X[i, :]) for i in range(swarm_size))

    return np.array(solutions) * -1


# ___________________________________________________________________________________________________________________________________
# ___________________________________________________________________________________________________________________________________
# SET UP CALIBRATION

start_time = time.time()  # Record time at start of calibration

# __________________________________________________________________________________________________________________________________
# VARIABLES AND INITIALIZATIONS

# 2014 - 2018
start = "Init_NCB-NewDrum-Ocracoke_2014_PostSandy-NCFMP-Plover.npy"
stop = "Init_NCB-NewDrum-Ocracoke_2018_PostFlorence-Plover.npy"
startdate = '20140406'
hindcast_duration = 4.5
cellsize = 1  # [m]

MHW = 0.39  # [m NAVD88] Initial
ResReduc = True  # Option to reduce raster resolution for skill assessment
reduc = 5  # Raster resolution reduction factor
name = '18950-19250, 2014-2018, HSS mean(present/absent, change)'

# Define Alongshore Coordinates of Domain
ymin = 18950
ymax = 19250

# Resize According to Cellsize
ymin = int(ymin / cellsize)  # Alongshore
ymax = int(ymax / cellsize)  # Alongshore

# ____________________________________

# Initial Observed Topo
Init = np.load("Input/" + start)
# Final Observed
End = np.load("Input/" + stop)

# Transform Initial Observed Topo
topo_i = Init[0, ymin: ymax, :]  # [m]
topo_start = copy.deepcopy(topo_i)  # [m] INITIAL TOPOGRPAHY

# Transform Final Observed Topo
topo_e = End[0, ymin: ymax, :]  # [m]
topo_end_obs = copy.deepcopy(topo_e)  # [m] FINAL OBSERVED TOPOGRAPHY

# Set Veg Domain
spec1_i = Init[1, ymin: ymax, :]
spec2_i = Init[2, ymin: ymax, :]
veg_start = spec1_i + spec2_i  # INITIAL VEGETATION COVER
veg_start[veg_start > 1] = 1
veg_start[veg_start < 0] = 0

spec1_e = End[1, ymin: ymax, :]
spec2_e = End[2, ymin: ymax, :]
veg_end = spec1_e + spec2_e
veg_end_obs = veg_end.copy()  # FINAL OBSERVED VEGETATION COVER
veg_end[veg_end > 1] = 1
veg_end[veg_end < 0] = 0


# ___________________________________________________________________________________________________________________________________
# ___________________________________________________________________________________________________________________________________
# PARTICLE SWARMS OPTIMIZATION

# _____________________________________________
# Prepare Particle Swarm Parameters

iterations = 14
swarm_size = 14
dimensions = 10  # Number of free paramters
n_jobs = 14
options = {'c1': 1.5, 'c2': 1.5, 'w': 0.5}
"""
w: Inertia weight constant. [0-1] Determines how much the particle keeps on with its previous velocity (i.e., speed and direction of the search). 
c1 & c2: Cognitive and the social coefficients, respectively. Control how much weight should be given between refining the search result of the
particle itself and recognizing the search result of the swarm; Control the trade off between exploration and exploitation.
"""

bounds = (
    # Minimum
    np.array([-0.7,     # sp1_b
              0.3,     # sp1_c
              1.0,     # sp1_d
              -1.15,   # sp2_b
              -0.35,   # sp2_c
              0.1,   # sp2_d
              0.1,    # sp1_peak
              0.01,    # sp2_peak
              0.10,    # lateral_probability
              0.01]),    # pioneer_probability
    # Maximum
    np.array([0.3,     # sp1_b
              1.0,  # sp1_c
              1.85,  # sp1_d
              -0.35,  # sp2_b
              0.1,  # sp2_c
              1.15,  # sp2_d
              0.55,     # sp1_peak
              0.10,    # sp2_peak
              0.3,     # lateral_probability
              0.15])    # pioneer_probability
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
    "BEST SOLUTION": ["HSSp"],
    "sp1_b": [best_solution[0]],
    "sp1_c": [best_solution[1]],
    "sp1_d": [best_solution[2]],
    "sp2_b": best_solution[3],
    "sp2_c": best_solution[4],
    "sp2_d": best_solution[5],
    "sp1_peak": [best_solution[6]],
    "sp2_peak": [best_solution[7]],
    "lateral_probability": [best_solution[8]],
    "pioneer_probability": [best_solution[9]],
    "Score": [solution_fitness]
    }, headers="keys", floatfmt=(None, ".3f", ".3f", ".3f", ".3f", ".3f", ".3f", ".3f", ".3f", ".3f", ".3f", ".3f"))
)

# _____________________________________________
# Plot Results
plt.plot(np.array(optimizer.cost_history) * -1)
plt.ylabel('Fitness (HSSp)')
plt.xlabel('Iteration')
