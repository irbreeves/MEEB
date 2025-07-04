"""
Script for running sensitivity analyses of MEEB storm parameters using SALib.

Model output used in sensitivity analysis is the Brier Skill Score for elevation.

IRBR 27 June 2025
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


def storm_fitness(solution, topo_start_obs, topo_end_obs, Rhigh, dur, OW_Mask, spec1, spec2):
    """Run a storm with this particular combintion of parameter values, and return fitness value of simulated to observed."""

    # Find Dune Crest, Shoreline Positions
    dune_crest, not_gap = routine.foredune_crest(topo_start_obs, MHW, cellsize)
    x_s = routine.ocean_shoreline(topo_start_obs, MHW)

    veg = spec1 + spec2  # Determine the initial cumulative vegetation effectiveness

    # Run Model
    topo_start_copy = copy.deepcopy(topo_start_obs)
    sim_topo_final, topo_change_overwash, OWflux, inundated, Qbe = routine.storm_processes(
        topo_start_copy,
        Rhigh,
        dur,
        Rin=int(round(solution[0])),
        Cs=round(solution[1], 4),
        nn=0.5,
        MaxUpSlope=solution[2],
        fluxLimit=1,
        Qs_min=1,
        Kow=solution[3],
        mm=round(solution[8], 2),
        MHW=MHW,
        Cbb=0.7,
        Qs_bb_min=1,
        substep=25,
        beach_equilibrium_slope=solution[4],
        swash_erosive_timescale=solution[5],
        beach_substeps=1,
        x_s=x_s,
        cellsize=cellsize,
        herbaceous_cover=spec1,
        woody_cover=spec2,
        flow_reduction_max_spec1=solution[6],  # Grass
        flow_reduction_max_spec2=solution[7],  # Shrub
    )

    topo_end_sim = routine.enforceslopes(sim_topo_final, veg, sh=0.02, anglesand=20, angleveg=30, th=0.3, MHW=MHW, cellsize=cellsize, RNG=RNG)

    topo_change_sim = topo_end_sim - topo_start_obs  # [m] Simulated change
    topo_change_obs = topo_end_obs - topo_start_obs  # [m] Observed change

    # _____________________________________________
    # Model Skill: Comparisons to Observations

    subaerial_mask = np.logical_and(topo_end_sim > MHW, topo_end_obs > MHW)  # [bool] Map of every cell above water

    beach_duneface_mask = np.zeros(topo_end_sim.shape)
    for l in range(topo_start_obs.shape[0]):
        beach_duneface_mask[l, :dune_crest[l]] = True
    beach_duneface_mask = np.logical_and(beach_duneface_mask, subaerial_mask)  # [bool] Map of every cell seaward of dune crest

    OW_Mask = np.logical_and(OW_Mask, subaerial_mask)

    mask_all = np.logical_or(OW_Mask, inundated, beach_duneface_mask.copy()) * subaerial_mask  # [bool] Map of every subaerial cell that was inundated in simulation or observation or both
    mask_obs = np.logical_or(OW_Mask, beach_duneface_mask.copy()) * subaerial_mask  # [bool] Map of every subaerial cell that was inundated in observation
    topo_change_obs[~mask_obs] = 0

    nse, rmse, nmae, mass, bss = model_skill(topo_change_obs, topo_change_sim, np.zeros(topo_change_obs.shape), mask_all)

    score = bss  # This is the skill score used in particle swarms optimization

    return score


def multi_fitness(solution):
    """Runs particular parameter combinations for multiple storms and locations and returns average skill score."""

    global BestScore
    global BestScores

    score_list = []

    for s in range(len(storm_start)):
        for x in range(len(ymin)):

            # Initial Observed Topo
            Init = np.load(storm_start[s])
            # Final Observed
            End = np.load(storm_stop[s])

            # Transform Initial Observed Topo
            topo_init = Init[0, ymin[x]: ymax[x], :]  # [m]
            topo_start = copy.deepcopy(topo_init)  # [m] Initialise the topography map

            # Transform Final Observed Topo
            topo_final = End[0, ymin[x]: ymax[x], :]  # [m]
            OW_Mask = overwash_mask_file[ymin[x]: ymax[x], :]  # [bool]

            # Set Veg Domain
            spec1 = Init[1, ymin[x]: ymax[x], :]
            spec2 = Init[2, ymin[x]: ymax[x], :]

            # Initialize storm stats
            Rhigh = storm_Rhigh[s] * np.ones(topo_final.shape[0])
            dur = storm_dur[s]

            score = storm_fitness(solution, topo_start, topo_final, Rhigh, dur, OW_Mask, spec1, spec2)
            score_list.append(score)

    # Take mean of scores from all locations
    multi_score = np.nanmean(score_list)

    if multi_score > BestScore:
        BestScore = multi_score
        BestScores = score_list

    return multi_score


def run_model(X):
    """Runs a parallelized batch of hindcast simulations and returns a fitness result for each"""

    with routine.tqdm_joblib(tqdm(desc="Progress", total=X.shape[0])) as progress_bar:
        solutions = Parallel(n_jobs=n_cores)(delayed(multi_fitness)(X[q, :]) for q in range(X.shape[0]))

    return np.array(solutions)


# ___________________________________________________________________________________________________________________________________
# ___________________________________________________________________________________________________________________________________
# SET UP

start_time = time.time()  # Record time at start of calibration
RNG = np.random.default_rng(seed=13)  # Seeded random numbers for reproducibility

name = 'Multi-loc, Morris, N=115, 28May24'
print(name)

# Choose sensitivity test(s) to run
test_sobol = False
test_morris = True

# Choose number of cores to run parallel simulations on
n_cores = 21

# _____________________________________________
# Define Variables
MHW = 0.39  # [m NAVD88]
cellsize = 2  # [m]

# Observed Overwash Mask
overwash_mask_file = np.load("Input/Mask_NCB-NewDrum-Ocracoke_2018_Florence_2m.npy")  # Load observed overwash mask

BestScore = -1e10
BestScores = []

# _____________________________________________
# Define Event(s)

storm_name = ['Florence']
storm_start = ["Input/Init_NCB-NewDrum-Ocracoke_2017_PreFlorence_2m.npy"]
storm_stop = ["Input/Init_NCB-NewDrum-Ocracoke_2018_USACE_PostFlorence_2m.npy"]
storm_Rhigh = [3.32]
storm_dur = [83]


# _____________________________________________
# Define Location(s)

ymin = [18950, 19825, 20375, 20525, 6300]
ymax = [19250, 20275, 20525, 20725, 6600]
# small flat, large flat, small gap, large gap, tall ridge

# Resize according to cellsize
ymin = [int(i / cellsize) for i in ymin]  # Alongshore
ymax = [int(i / cellsize) for i in ymax]  # Alongshore


# ___________________________________________________________________________________________________________________________________
# ___________________________________________________________________________________________________________________________________
# SENSITIVITY ANALYSIS

# _____________________________________________
# Define Model Inputs

inputs = {
    'num_vars': 9,
    'names': ['Rin', 'Cs', 'MaxUpSlope', 'Kow', 'Beq', 'Te', 'FlowReduc_sp1', 'FlowReduc_s2', 'mm'],
    'bounds': [[50, 280],
               [0.010, 0.040],
               [0.5, 2.0],
               [0.00005, 0.01],
               [0.01, 0.04],
               [1.00, 3.00],
               [0.02, 0.4],
               [0.05, 0.5],
               [1.01, 1.12]]
}
N_sobol = 115  # Number of trajectories = N * (2 * num_vars + 2)
N_morris = 115  # Number of trajectories = N * (num_vars + 1)

# _____________________________________________
if test_sobol:
    # Generate samples
    param_values_sobol = sobol_sample.sample(inputs, N_sobol)

    # Run model simulations
    outputs_sobol = run_model(param_values_sobol)

    # Perform analysis
    Si_sobol = sobol.analyze(inputs, outputs_sobol)  # Sobol' method (variance-based sensitivity analysis)

    # Print results
    print()
    print(tabulate({
        "Sobol'": ["1st-Order", "Total"],
        "Rin": [Si_sobol['S1'][0], Si_sobol['ST'][0]],
        "Cs": [Si_sobol['S1'][1], Si_sobol['ST'][1]],
        "MUS": [Si_sobol['S1'][2], Si_sobol['ST'][2]],
        "Kow": [Si_sobol['S1'][3], Si_sobol['ST'][3]],
        "Beq": [Si_sobol['S1'][4], Si_sobol['ST'][4]],
        "Te": [Si_sobol['S1'][5], Si_sobol['ST'][5]],
        "FlowReduc_s1": [Si_sobol['S1'][6], Si_sobol['ST'][6]],
        "FlowReduc_s2": [Si_sobol['S1'][7], Si_sobol['ST'][7]],
        "mm": [Si_sobol['S1'][8], Si_sobol['ST'][8]],
    }, headers="keys", floatfmt=(None, ".4f", ".4f", ".4f", ".4f", ".4f", ".4f", ".4f", ".4f", ".4f"))
    )
    print()

    # Plot results
    axes_sobol = Si_sobol.plot()
    axes_sobol[0].set_yscale('linear')
    fig_sobol = plt.gcf()  # get current figure
    fig_sobol.set_size_inches(10, 4)
    plt.tight_layout()

# _____________________________________________
if test_morris:
    # Generate Samples
    param_values_morris = morris_sample.sample(inputs, N_morris, num_levels=6)

    # Run Model Simulations
    outputs_morris = run_model(param_values_morris)

    # Perform Analysis
    Si_morris = morris.analyze(inputs, param_values_morris, outputs_morris)  # Morris method (Elementary Effects Test)

    # Print Results
    print()
    print(tabulate({
        "Morris": ["mu*", "sigma"],
        "Rin": [Si_morris['mu_star'][0], Si_morris['sigma'][0]],
        "Cs": [Si_morris['mu_star'][1], Si_morris['sigma'][1]],
        "MUS": [Si_morris['mu_star'][2], Si_morris['sigma'][2]],
        "Kr": [Si_morris['mu_star'][3], Si_morris['sigma'][3]],
        "Beq": [Si_morris['mu_star'][4], Si_morris['sigma'][4]],
        "Te": [Si_morris['mu_star'][5], Si_morris['sigma'][5]],
        "FlowReduc_s1": [Si_morris['mu_star'][6], Si_morris['sigma'][6]],
        "FlowReduc_s2": [Si_morris['mu_star'][7], Si_morris['sigma'][7]],
        "mm": [Si_morris['mu_star'][8], Si_morris['sigma'][8]],
    }, headers="keys", floatfmt=(None, ".4f", ".4f", ".4f", ".4f", ".4f", ".4f", ".4f", ".4f", ".4f"))
    )
    print()

    # Plot Results
    axes_morris = Si_morris.plot()
    axes_morris.set_yscale('linear')
    fig_morris = plt.gcf()  # get current figure
    fig_morris.set_size_inches(10, 4)
    plt.tight_layout()

# _____________________________________________
SimDuration = int(time.time() - start_time)
print()
print("Elapsed Time: ", SimDuration, "sec")
print()

plt.show()
