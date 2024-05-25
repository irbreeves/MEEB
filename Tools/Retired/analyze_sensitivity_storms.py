"""
Script for running sensitivity analyses of MEEB storm parameters using SALib.

Model output used in sensitivity analysis is the Brier Skill Score for elevation.

IRBR 8 February 2024
"""

import numpy as np
import matplotlib.pyplot as plt
import routines_meeb as routine
import copy
import time
from tqdm import tqdm
from tabulate import tabulate
from joblib import Parallel, delayed
from SALib.sample import sobol as sobol_sample
from SALib.sample import morris as morris_sample
from SALib.analyze import sobol as sobol
from SALib.analyze import morris as morris


# ___________________________________________________________________________________________________________________________________
# ___________________________________________________________________________________________________________________________________
# FUNCTIONS FOR RUNNING MODEL HINDCASTS AND CALCULATING SKILL

def model_skill(obs, sim, t0, mask):
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
        A = np.nanmean(np.square(np.subtract(obs[mask], sim[mask])))
        B = np.nanmean(np.square(np.subtract(obs[mask], np.nanmean(obs[mask]))))
        nse = 1 - A / B

        # _____________________________________________
        # Root Mean Square Error
        rmse = np.sqrt(np.nanmean(np.square(sim[mask] - obs[mask])))

        # _____________________________________________
        # Normalized Mean Absolute Error
        nmae = np.nanmean(np.abs(sim[mask] - obs[mask])) / (np.nanmax(obs[mask]) - np.nanmin(obs[mask]))  # (np.nanstd(np.abs(obs[mask])))

        # _____________________________________________
        # Mean Absolute Skill Score
        mass = 1 - np.nanmean(np.abs(sim[mask] - obs[mask])) / np.nanmean(np.abs(t0[mask] - obs[mask]))

        # _____________________________________________
        # Brier Skill Score
        """A skill score value of zero means that the score for the predictions is merely as good as that of a set of baseline or reference or default predictions, 
        while a skill score value of one (100%) represents the best possible score. A skill score value less than zero means that the performance is even worse than 
        that of the baseline or reference predictions (i.e., the baseline matches the final field profile more closely than the simulation output)."""
        MSE = np.nanmean(np.square(np.abs(np.subtract(sim[mask], obs[mask]))))
        MSEref = np.nanmean(np.square(np.abs(np.subtract(t0[mask], obs[mask]))))
        bss = 1 - MSE / MSEref

    return nse, rmse, nmae, mass, bss


def storm_fitness(solution):
    """Run a storm with this particular combintion of parameter values, and return fitness value of simulated to observed."""

    # Run Model
    topo_start_copy = copy.deepcopy(topo_start)
    topo_end_sim, topo_change_overwash, OWflux, netDischarge, inundated, Qbe = routine.storm_processes_OLD(
        topo_start_copy,
        Rhigh,
        Rlow,
        dur,
        Rin=int(round(solution[0])),
        Cs=int(round(solution[1])),
        AvgSlope=2 / 200,
        nn=0.5,
        MaxUpSlope=solution[2],
        fluxLimit=1,
        Qs_min=1,
        Kow=solution[3],
        mm=solution[8],
        MHW=MHW,
        Cbb=0.7,
        Qs_bb_min=1,
        substep=4,
        beach_equilibrium_slope=solution[4],
        swash_transport_coefficient=solution[5],
        wave_period_storm=9.4,
        beach_substeps=20,
        x_s=x_s,
        cellsize=1,
        spec1=spec1,
        spec2=spec2,
        flow_reduction_max_spec1=solution[6],  # Grass
        flow_reduction_max_spec2=solution[7],  # Shrub
    )

    topo_start_obs = topo_start.copy()
    topo_change_sim = topo_end_sim - topo_start_obs  # [m] Simulated change
    topo_change_obs = copy.deepcopy(topo_final) - topo_start_obs  # [m] Observed change

    # _____________________________________________
    # Model Skill: Comparisons to Observations

    subaerial_mask = np.logical_and(topo_end_sim > MHW, topo_final > MHW)  # [bool] Map of every cell above water

    beach_duneface_mask = np.zeros(topo_end_sim.shape)
    for q in range(topo_start_obs.shape[0]):
        beach_duneface_mask[q, :dune_crest[q]] = True
    beach_duneface_mask = np.logical_and(beach_duneface_mask, subaerial_mask)  # [bool] Map of every cell seaward of dune crest

    overwash_mask = np.logical_and(OW_Mask, subaerial_mask)
    overwash_mask_all = np.logical_and(np.logical_or(OW_Mask, inundated), ~beach_duneface_mask) * subaerial_mask  # [bool] Map of every cell involved in observed or simulated overwash (landward of dune crest)

    mask_all = np.logical_or(overwash_mask, inundated, beach_duneface_mask.copy()) * subaerial_mask  # [bool] Map of every subaerial cell that was inundated in simulation or observation or both
    mask_obs = np.logical_or(overwash_mask, beach_duneface_mask.copy()) * subaerial_mask  # [bool] Map of every subaerial cell that was inundated in observation
    topo_change_obs[~mask_obs] = 0

    nse, rmse, nmae, mass, bss = model_skill(topo_change_obs, topo_change_sim, np.zeros(topo_change_obs.shape), mask_all)
    nse_ow, rmse_ow, nmae_ow, mass_ow, bss_ow = model_skill(topo_change_obs, topo_change_sim, np.zeros(topo_change_obs.shape), overwash_mask_all)  # Skill scores for just overwash
    nse_bd, rmse_bd, nmae_bd, mass_bd, bss_bd = model_skill(topo_change_obs, topo_change_sim, np.zeros(topo_change_obs.shape), beach_duneface_mask)  # Skill scores for just beach/dune

    score = bss  # This is the skill score used in particle swarms optimization

    return score


def run_model(X):
    """Runs a parallelized batch of hindcast simulations and returns a fitness result for each"""

    with routine.tqdm_joblib(tqdm(desc="Progress", total=X.shape[0])) as progress_bar:
        solutions = Parallel(n_jobs=15)(delayed(storm_fitness)(X[q, :]) for q in range(X.shape[0]))

    return np.array(solutions)


# ___________________________________________________________________________________________________________________________________
# ___________________________________________________________________________________________________________________________________
# SET UP CALIBRATION

start_time = time.time()  # Record time at start of calibration

# _____________________________________________
# Define Variables
Rhigh = 3.32
Rlow = 1.90
dur = 83
MHW = 0.39  # [m NAVD88]

# Initial Observed Topo
Init = np.load("Input/Init_NCB-NewDrum-Ocracoke_2017_PreFlorence.npy")
# Final Observed
End = np.load("Input/Init_NCB-NewDrum-Ocracoke_2018_PostFlorence-Plover.npy")

# Observed Overwash Mask
Florence_Overwash_Mask = np.load("Input/Mask_NCB-NewDrum-Ocracoke_2018_Florence.npy")  # Load observed overwash mask

# Define Alongshore Coordinates of Domain
xmin = 18950  # 19825, 20375, 20525 20975
xmax = 19250  # 20275, 20525, 20725 21125
# small flat, large flat, small gap, large gap, tall ridge

name = '18950-19250, weighted BSS, Morris'

# _____________________________________________
# Conversions & Initializations

# Transform Initial Observed Topo
topo_init = Init[0, xmin: xmax, :]  # [m NAVD88]
topo_start = copy.deepcopy(topo_init)  # [m NAVD88]
topo = copy.deepcopy(topo_start)  # [m] Initialise the topography map

# Transform Final Observed Topo
topo_final = End[0, xmin:xmax, :]  # [m]
OW_Mask = Florence_Overwash_Mask[xmin: xmax, :]  # [bool]

# Set Veg Domain
spec1 = Init[1, xmin: xmax, :]
spec2 = Init[2, xmin: xmax, :]
veg = spec1 + spec2  # Determine the initial cumulative vegetation effectiveness
veg[veg > 1] = 1  # Cumulative vegetation effectiveness cannot be negative or larger than one
veg[veg < 0] = 0

# Find Dune Crest, Shoreline Positions
dune_crest = routine.foredune_crest(topo_start, MHW)
x_s = routine.ocean_shoreline(topo_start, MHW)

# Transform water levels to vectors
Rhigh = Rhigh * np.ones(topo_final.shape[0])
Rlow = Rlow * np.ones(topo_final.shape[0])

topo_prestorm = copy.deepcopy(topo)

# ___________________________________________________________________________________________________________________________________
# ___________________________________________________________________________________________________________________________________
# SENSITIVITY ANALYSIS

# _____________________________________________
# Define Model Inputs

inputs = {
    'num_vars': 9,
    'names': ['Rin', 'Cx', 'MaxUpSlope', 'Kr', 'beq', 'Kc', 'frms1', 'frms2', 'mm'],
    'bounds': [[100, 320],
               [10, 80],
               [0.5, 2.0],
               [8e-06, 1e-04],
               [0.01, 0.04],  # Should be larger, e.g. 0.04 or 0.05
               [0.0005, 0.0035],
               [0.02, 0.4],
               [0.05, 0.5],
               [1.0, 1.15]]
}
N = 500  # Number of samples = N * (2 * num_vars + 2) (Sobol') or N * (num_vars + 1) (Morris)

# _____________________________________________
# Generate Samples

# param_values_sobol = sobol_sample.sample(inputs, N)
param_values_morris = morris_sample.sample(inputs, N, num_levels=6)

# _____________________________________________
# Run Model Simulations

# outputs_sobol = run_model(param_values_sobol)
outputs_morris = run_model(param_values_morris)

SimDuration = int(time.time() - start_time)

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
#     "Rin": [Si_sobol['S1'][0], Si_sobol['ST'][0]],
#     "Cx": [Si_sobol['S1'][1], Si_sobol['ST'][1]],
#     "MUS": [Si_sobol['S1'][2], Si_sobol['ST'][2]],
#     "Kr": [Si_sobol['S1'][3], Si_sobol['ST'][3]],
#     "Beq": [Si_sobol['S1'][4], Si_sobol['ST'][4]],
#     "Kc": [Si_sobol['S1'][5], Si_sobol['ST'][5]],
#     "Qreduc_s1": [Si_sobol['S1'][6], Si_sobol['ST'][6]],
#     "Qreduc_s2": [Si_sobol['S1'][7], Si_sobol['ST'][7]],
# }, headers="keys", floatfmt=(None, ".4f", ".4f", ".4f", ".4f", ".4f", ".4f", ".4f", ".4f"))
# )
# print()

print()
print(tabulate({
    "Morris": ["mu*", "sigma"],
    "Rin": [Si_morris['mu_star'][0], Si_morris['sigma'][0]],
    "Cx": [Si_morris['mu_star'][1], Si_morris['sigma'][1]],
    "MUS": [Si_morris['mu_star'][2], Si_morris['sigma'][2]],
    "Kr": [Si_morris['mu_star'][3], Si_morris['sigma'][3]],
    "Beq": [Si_morris['mu_star'][4], Si_morris['sigma'][4]],
    "Kc": [Si_morris['mu_star'][5], Si_morris['sigma'][5]],
    "Qreduc_s1": [Si_morris['mu_star'][6], Si_morris['sigma'][6]],
    "Qreduc_s2": [Si_morris['mu_star'][7], Si_morris['sigma'][7]],
}, headers="keys", floatfmt=(None, ".4f", ".4f", ".4f", ".4f", ".4f", ".4f", ".4f", ".4f"))
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
