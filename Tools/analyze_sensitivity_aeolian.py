"""
Script for running sensitivity analyses of MEEB aeolian parameters using SALib.

Model output used in sensitivity analysis is the Brier Skill Score for elevation.

IRBR 9 November 2023
"""

import numpy as np
import matplotlib.pyplot as plt
import routines_meeb as routine
import copy
import time
from tabulate import tabulate
from tqdm import tqdm
from SALib.sample import sobol as sobol_sample
from SALib.sample import morris as morris_sample
from SALib.analyze import sobol as sobol
from SALib.analyze import morris as morris
from joblib import Parallel, delayed

from meeb import MEEB


# ___________________________________________________________________________________________________________________________________
# ___________________________________________________________________________________________________________________________________
# FUNCTIONS FOR RUNNING MODEL HINDCASTS AND CALCULATING SKILL

def model_skill(obs, sim, t0, mask):
    """
    Perform suite of model skill assesments and return scores.
    Mask is boolean array with same size as change maps, with cells to be excluded from skill analysis set to FALSE.
    """

    sim_change = sim - t0  # [m]
    obs_change = obs - t0  # [m]

    # _____________________________________________
    # Nash-Sutcliffe Model Efficiency
    """The closer the score is to 1, the better the agreement. If the score is below 0, the mean observed value is a better predictor than the model."""
    A = np.nanmean(np.square(np.subtract(obs[mask], sim[mask])))
    B = np.nanmean(np.square(np.subtract(obs[mask], np.nanmean(obs[mask]))))
    NSE = 1 - A / B

    # _____________________________________________
    # Root Mean Square Error
    RMSE = np.sqrt(np.nanmean(np.square(sim[mask] - obs[mask])))

    # _____________________________________________
    # Normalized Mean Absolute Error
    NMAE = np.nanmean(np.abs(sim[mask] - obs[mask])) / (np.nanmax(obs[mask]) - np.nanmin(obs[mask]))  # (np.nanstd(np.abs(obs[mask])))

    # _____________________________________________
    # Mean Absolute Skill Score
    MASS = 1 - np.nanmean(np.abs(sim[mask] - obs[mask])) / np.nanmean(np.abs(t0[mask] - obs[mask]))

    # _____________________________________________
    # Brier Skill Score
    """A skill score value of zero means that the score for the predictions is merely as good as that of a set of baseline or reference or default predictions, 
    while a skill score value of one (100%) represents the best possible score. A skill score value less than zero means that the performance is even worse than 
    that of the baseline or reference predictions (i.e., the baseline matches the final field profile more closely than the simulation output)."""
    BSS = routine.brier_skill_score(sim, obs, t0, mask)

    # _____________________________________________
    # Categorical
    threshold = 0.02
    sim_erosion = sim_change < -threshold
    sim_deposition = sim_change > threshold
    sim_no_change = np.logical_and(sim_change <= threshold, -threshold <= sim_change)
    obs_erosion = obs_change < -threshold
    obs_deposition = obs_change > threshold
    obs_no_change = np.logical_and(obs_change <= threshold, -threshold <= obs_change)

    cat_Mask = np.zeros(obs_change.shape)
    cat_Mask[np.logical_and(sim_erosion, obs_erosion)] = 1  # Hit
    cat_Mask[np.logical_and(sim_deposition, obs_deposition)] = 1  # Hit
    cat_Mask[np.logical_and(sim_erosion, ~obs_erosion)] = 2  # False Alarm
    cat_Mask[np.logical_and(sim_deposition, ~obs_deposition)] = 2  # False Alarm
    cat_Mask[np.logical_and(sim_no_change, obs_no_change)] = 3  # Correct Reject
    cat_Mask[np.logical_and(sim_no_change, ~obs_no_change)] = 4  # Miss

    hits = np.count_nonzero(cat_Mask[mask] == 1)
    false_alarms = np.count_nonzero(cat_Mask[mask] == 2)
    correct_rejects = np.count_nonzero(cat_Mask[mask] == 3)
    misses = np.count_nonzero(cat_Mask[mask] == 4)
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

    return NSE, RMSE, NMAE, MASS, BSS, PC, HSS


def aeolian_fitness(solution):
    """Run a hindcast this particular combintion of parameter values, and return fitness value of simulated to observed."""

    # Construct wind rose
    wind_axis_ratio = solution[8]  # Proportion of cross-shore winds (right/left, i.e. onshore/offshore) versus alongshore winds (up/down)
    wind_dir_1 = max(0, solution[9] * wind_axis_ratio)  # Onshore (proportion towards right)
    wind_dir_2 = max(0, (1 - solution[9]) * wind_axis_ratio)  # Offshore (proportion towards left)
    wind_dir_3 = max(0, solution[10] * (1 - wind_axis_ratio))  # Alongshore (proportion towards down)
    wind_dir_4 = max(0, 1 - (wind_dir_1 + wind_dir_2 + wind_dir_3))  # Alongshore (proportion towards up)
    rose = (wind_dir_1, wind_dir_2, wind_dir_3, wind_dir_4)  # Tuple that sums to 1.0

    # Create an instance of the BMI class
    meeb = MEEB(
        name=name,
        simulation_time_yr=hindcast_duration,
        alongshore_domain_boundary_min=xmin,
        alongshore_domain_boundary_max=xmax,
        RSLR=0.000,
        MHW=MHW,
        seeded_random_numbers=True,
        init_filename=start,
        hindcast=True,
        simulation_start_date=startdate,
        storm_timeseries_filename='StormTimeSeries_1979-2020_NCB-CE_Beta0pt039_BermEl1pt78.npy',
        # --- Aeolian --- #
        p_dep_sand=solution[0],
        p_dep_sand_VegMax=solution[0] + solution[1],
        p_ero_sand=solution[2],
        entrainment_veg_limit=solution[3],
        saltation_veg_limit=solution[4],
        shadowangle=int(round(solution[5])),
        repose_bare=int(round(solution[6])),
        repose_veg=int(round(solution[6] + solution[7])),
        wind_rose=rose,
        # --- Storms --- #
        Rin_ru=164,
        Cx=47,
        MaxUpSlope=1.32,
        K_ru=0.0000468,
        substep_ru=3,
        beach_equilibrium_slope=0.027,
        swash_transport_coefficient=0.00086,
        wave_period_storm=9.4,
        beach_substeps=20,
        flow_reduction_max_spec1=0.25,
        flow_reduction_max_spec2=0.14,
        # --- Shoreline --- #
        wave_asymetry=0.6,
        wave_high_angle_fraction=0.39,
        mean_wave_height=0.98,
        mean_wave_period=6.6,
        alongshore_section_length=25,
        estimate_shoreface_parameters=True,
        # --- Veg --- #
        # sp1_c=1.20,
        # sp2_c=-0.47,
        # sp1_peak=0.307,
        # sp2_peak=0.148,
        # lateral_probability=0.34,
        # pioneer_probability=0.11,
        # Spec1_elev_min=0.60,
        # Spec2_elev_min=0.13,
    )

    # Loop through time
    for time_step in range(int(meeb.iterations)):
        # Run time step
        meeb.update(time_step)

    # __________________________________________________________________________________________________________________________________
    # ASSESS MODEL SKILL

    topo_end_sim = meeb.topo  # [m NAVDD88]
    mhw_end_sim = meeb.MHW  # [m NAVD88]
    topo_change_sim = topo_end_sim - topo_start  # [m]
    topo_change_obs = topo_end_obs - topo_start  # [m]

    # Subaerial mask
    subaerial_mask = topo_end_sim > mhw_end_sim  # [bool] Mask for every cell above water

    # Beach mask
    dune_crest = routine.foredune_crest(topo_start, mhw_end_sim)
    beach_duneface_mask = np.zeros(topo_end_sim.shape)
    for l in range(topo_start.shape[0]):
        beach_duneface_mask[l, :dune_crest[l]] = True
    beach_duneface_mask = np.logical_and(beach_duneface_mask, subaerial_mask)  # [bool] Map of every cell seaward of dune crest

    # Cross-shore range mask
    range_mask = np.ones(topo_end_sim.shape)  # [bool] Mask for every cell between two cross-shore locations
    range_mask[:, :835] = False
    range_mask[:, 950:] = False
    # range_mask[:, :1100] = False
    # range_mask[:, 1350:] = False

    # Elevation mask
    elev_mask = topo_end_sim > 2.0  # [bool] Mask for every cell above water

    # Choose masks to use
    mask = np.logical_and(range_mask, subaerial_mask)  # [bool] Combined mask used for analysis

    # Dune crest locations and heights
    crest_loc_obs_start = routine.foredune_crest(topo_start, mhw_end_sim)
    crest_loc_obs = routine.foredune_crest(topo_end_obs, mhw_end_sim)
    crest_loc_sim = routine.foredune_crest(topo_end_sim, mhw_end_sim)
    crest_loc_change_obs = crest_loc_obs - crest_loc_obs_start
    crest_loc_change_sim = crest_loc_sim - crest_loc_obs_start

    crest_height_obs_start = topo_start[np.arange(topo_start.shape[0]), crest_loc_obs_start]
    crest_height_obs = topo_end_obs[np.arange(topo_end_obs.shape[0]), crest_loc_obs]
    crest_height_sim = topo_end_sim[np.arange(topo_end_obs.shape[0]), crest_loc_sim]
    crest_height_change_obs = crest_height_obs - crest_height_obs_start
    crest_height_change_sim = crest_height_sim - crest_height_obs_start

    # Optional: Reduce Resolutions
    if ResReduc:
        topo_change_obs = routine.reduce_raster_resolution(topo_change_obs, reduc)
        topo_change_sim = routine.reduce_raster_resolution(topo_change_sim, reduc)
        mask = (routine.reduce_raster_resolution(mask, reduc)) == 1

    # Model Skill
    nse, rmse, nmae, mass, bss, pc, hss = model_skill(topo_change_obs, topo_change_sim, np.zeros(topo_change_obs.shape), mask)  # All cells (excluding masked areas)
    nse_dl, rmse_dl, nmae_dl, mass_dl, bss_dl, pc_dl, hss_dl = model_skill(crest_loc_obs.astype('float32'), crest_loc_sim.astype('float32'), crest_loc_obs_start.astype('float32'), np.full(crest_loc_obs.shape, True))  # Foredune location
    nse_dh, rmse_dh, nmae_dh, mass_dh, bss_dh, pc_dh, hss_dh = model_skill(crest_height_obs, crest_height_sim, crest_height_obs_start, np.full(crest_height_change_obs.shape, True))  # Foredune elevation

    # Combine Skill Scores (Multi-Objective Optimization)
    score = np.average([nmae, nmae_dl, nmae_dh], weights=[1, 1, 1])  # This is the skill score used in particle swarms optimization

    return score


def run_model(X):
    """Runs a parallelized batch of hindcast simulations and returns a fitness result for each"""

    with routine.tqdm_joblib(tqdm(desc="Progress", total=X.shape[0])) as progress_bar:
        solutions = Parallel(n_jobs=20)(delayed(aeolian_fitness)(X[q, :]) for q in range(X.shape[0]))

    return np.array(solutions)


# ___________________________________________________________________________________________________________________________________
# ___________________________________________________________________________________________________________________________________
# SET UP CALIBRATION

start_time = time.time()  # Record time at start of calibration

# __________________________________________________________________________________________________________________________________
# VARIABLES AND INITIALIZATIONS

# 2014 - 2017
start = "Init_NCB-NewDrum-Ocracoke_2014_PostSandy-NCFMP-Plover.npy"
stop = "Init_NCB-NewDrum-Ocracoke_2017_PreFlorence.npy"
hindcast_duration = 3.44
startdate = '20140406'

# # 2014 - 2018
# start = "Init_NCB-NewDrum-Ocracoke_2014_PostSandy-NCFMP-Plover.npy"
# stop = "Init_NCB-NewDrum-Ocracoke_2018_PostFlorence-Plover.npy"
# startdate = '20140406'
# hindcast_duration = 4.5

# Define Alongshore Coordinates of Domain
xmin = 6300  # 18950
xmax = 6600  # 19250

MHW = 0.39  # [m NAVD88] Initial
ResReduc = False  # Option to reduce raster resolution for skill assessment
reduc = 5  # Raster resolution reduction factor
name = '6300-6600, 2014-2017, NMAE multi-objective, N=30'

# ____________________________________

# Initial Observed Topo
Init = np.load("Input/" + start)
# Final Observed
End = np.load("Input/" + stop)

# Transform Initial Observed Topo
topo_i = Init[0, xmin: xmax, :]  # [m]
topo_start = copy.deepcopy(topo_i)  # [m] INITIAL TOPOGRPAHY

# Transform Final Observed Topo
topo_e = End[0, xmin: xmax, :]  # [m]
topo_end_obs = copy.deepcopy(topo_e)  # [m] FINAL OBSERVED TOPOGRAPHY

# Set Veg Domain
spec1_i = Init[1, xmin: xmax, :]
spec2_i = Init[2, xmin: xmax, :]
veg_start = spec1_i + spec2_i  # INITIAL VEGETATION COVER
veg_start[veg_start > 1] = 1
veg_start[veg_start < 0] = 0

spec1_e = End[1, xmin: xmax, :]
spec2_e = End[2, xmin: xmax, :]
veg_end = spec1_e + spec2_e  # FINAL OBSERVED VEGETATION COVER
veg_end[veg_end > 1] = 1
veg_end[veg_end < 0] = 0

# ___________________________________________________________________________________________________________________________________
# ___________________________________________________________________________________________________________________________________
# SENSITIVITY ANALYSIS

# _____________________________________________
# Prepare Particle Swarm Parameters


inputs = {
    'num_vars': 11,
    'names': ['p_dep_sand', 'p_dep_VegMax', 'p_ero_sand', 'entr_veg_limit', 'salt_veg_limit', 'shadowangle', 'repose_bare', 'repose_veg', 'wind_axis_ratio', 'wind_cs_right', 'wind_as_down'],
    'bounds': [[0.02, 0.5],
               [0.05, 0.5],
               [0.02, 0.5],
               [0.05, 0.55],
               [0.05, 0.4],
               [5, 15],
               [15, 30],
               [5, 10],
               [0, 1],  # Proportion of cross-shore winds versus alongshore
               [0, 1],  # Proportion of cross-shore winds towards right (onshore)
               [0, 1],  # Proportion of alongshore winds towards down
               ]
}
N = 30  # Number of samples: sobol = N * (2 * num_vars + 2), morris = N * (num_vars + 2)

# _____________________________________________
# Generate Samples

# param_values_sobol = sobol_sample.sample(inputs, N)
param_values_morris = morris_sample.sample(inputs, N, num_levels=4)

# _____________________________________________
# Run Model Simulations

# outputs_sobol = run_model(param_values_sobol)
outputs_morris = run_model(param_values_morris)

SimDuration = time.time() - start_time

# _____________________________________________
# Perform Analysis

# Si_sobol = sobol.analyze(inputs, outputs_sobol)  # Sobol' method (variance-based sensitivity analysis)
Si_morris = morris.analyze(inputs, param_values_morris, outputs_morris)  # Morris method (Elementary Effects Test)

# _____________________________________________
# Print Results

print()
print("Elapsed Time: ", SimDuration, "sec")
print()

# print()
# print(tabulate({
#     "Sobol'": ["1st-Order", "Total"],
#     "p_dep_sand": [Si_sobol['S1'][0], Si_sobol['ST'][0]],
#     "p_dep_VegMax": [Si_sobol['S1'][1], Si_sobol['ST'][1]],
#     "p_ero_sand": [Si_sobol['S1'][2], Si_sobol['ST'][2]],
#     "entr_veg_limit": [Si_sobol['S1'][3], Si_sobol['ST'][3]],
#     "salt_veg_limit": [Si_sobol['S1'][4], Si_sobol['ST'][4]],
#     "shadowangle": [Si_sobol['S1'][5], Si_sobol['ST'][5]],
#     "repose_bare": [Si_sobol['S1'][6], Si_sobol['ST'][6]],
#     "repose_veg": [Si_sobol['S1'][7], Si_sobol['ST'][7]],
#     "wind_axis_ratio": [Si_sobol['S1'][8], Si_sobol['ST'][8]],
#     "wind_cs_right": [Si_sobol['S1'][9], Si_sobol['ST'][9]],
#     "wind_as_down": [Si_sobol['S1'][10], Si_sobol['ST'][10]],
# }, headers="keys", floatfmt=(None, ".4f", ".4f", ".4f", ".4f", ".4f", ".4f", ".4f"))
# )
# print()

print()
print(tabulate({
    "Morris": ["mu*", "sigma"],
    "p_dep_sand": [Si_morris['mu_star'][0], Si_morris['sigma'][0]],
    "p_dep_VegMax": [Si_morris['mu_star'][1], Si_morris['sigma'][1]],
    "p_ero_sand": [Si_morris['mu_star'][2], Si_morris['sigma'][2]],
    "entr_veg_limit": [Si_morris['mu_star'][3], Si_morris['sigma'][3]],
    "salt_veg_limit": [Si_morris['mu_star'][4], Si_morris['sigma'][4]],
    "shadowangle": [Si_morris['mu_star'][5], Si_morris['sigma'][5]],
    "repose_bare": [Si_morris['mu_star'][6], Si_morris['sigma'][6]],
    "repose_veg": [Si_morris['mu_star'][7], Si_morris['sigma'][7]],
    "wind_axis_ratio": [Si_morris['mu_star'][8], Si_morris['sigma'][8]],
    "wind_cs_right": [Si_morris['mu_star'][9], Si_morris['sigma'][9]],
    "wind_as_down": [Si_morris['mu_star'][10], Si_morris['sigma'][10]],
}, headers="keys", floatfmt=(None, ".4f", ".4f", ".4f", ".4f", ".4f", ".4f", ".4f"))
)
print()

# _____________________________________________
# Print & Plot Results

# axes_sobol = Si_sobol.plot()
# axes_sobol[0].set_yscale('linear')
# fig_sobol = plt.gcf()  # get current figure
# fig_sobol.set_size_inches(10, 4)
# plt.tight_layout()

axes_morris = Si_morris.plot()
axes_morris.set_yscale('linear')
fig_morris = plt.gcf()  # get current figure
fig_morris.set_size_inches(10, 4)
plt.tight_layout()

plt.show()
