"""
Script for calibrating MEEB storm parameters using Particle Swarms Optimization.

Calibrates based on fitess score for all beach/dune/overwash morphologic change, and incorporates multiple
storm events and/or locations into each fitness score.

IRBR 9 November 2023
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


def storm_fitness(solution, topo_start_obs, topo_end_obs, Rhigh, Rlow, dur, OW_Mask, spec1, spec2):
    """Run a storm with this particular combintion of parameter values, and return fitness value of simulated to observed."""

    # Find Dune Crest, Shoreline Positions
    dune_crest = routine.foredune_crest(topo_start_obs, MHW)
    x_s = routine.ocean_shoreline(topo_start_obs, MHW)

    # Run Model
    topo_start_copy = copy.deepcopy(topo_start_obs)
    topo_end_sim, topo_change_overwash, OWflux, netDischarge, inundated, Qbe = routine.storm_processes_2(
        topo_start_copy,
        Rhigh,
        Rlow,
        dur,
        threshold_in=0.25,
        Rin_i=5,
        Rin_r=int(round(solution[0])),
        Cx=int(round(solution[1])),
        AvgSlope=2 / 200,
        nn=0.5,
        MaxUpSlope=solution[2],
        fluxLimit=1,
        Qs_min=1,
        Kr=solution[4],
        Ki=5e-06,
        mm=1,
        MHW=MHW,
        Cbb_i=0.85,
        Cbb_r=0.7,
        Qs_bb_min=1,
        substep_i=6,
        substep_r=int(round(solution[6])),
        beach_equilibrium_slope=solution[7],
        swash_transport_coefficient=solution[8],
        wave_period_storm=9.4,
        beach_substeps=int(round(solution[9])),
        x_s=x_s,
        cellsize=1,
        spec1=spec1,
        spec2=spec2,
        flow_reduction_max_spec1=solution[3],  # Grass
        flow_reduction_max_spec2=solution[5],  # Shrub
    )

    topo_change_sim = topo_end_sim - topo_start_obs  # [m] Simulated change
    topo_change_obs = topo_end_obs - topo_start_obs  # [m] Observed change

    # _____________________________________________
    # Model Skill: Comparisons to Observations

    subaerial_mask = topo_end_sim > MHW  # [bool] Map of every cell above water

    beach_duneface_mask = np.zeros(topo_end_sim.shape)
    for l in range(topo_start_obs.shape[0]):
        beach_duneface_mask[l, :dune_crest[l]] = True
    beach_duneface_mask = np.logical_and(beach_duneface_mask, subaerial_mask)  # [bool] Map of every cell seaward of dune crest

    OW_Mask = np.logical_and(OW_Mask, subaerial_mask)
    mask_overwash_all = np.logical_and(np.logical_or(OW_Mask, inundated), ~beach_duneface_mask) * subaerial_mask  # [bool] Map of every cell involved in observed or simulated overwash (landward of dune crest)

    mask_all = np.logical_or(OW_Mask, inundated, beach_duneface_mask.copy()) * subaerial_mask  # [bool] Map of every subaerial cell that was inundated in simulation or observation or both
    mask_obs = np.logical_or(OW_Mask, beach_duneface_mask.copy()) * subaerial_mask  # [bool] Map of every subaerial cell that was inundated in observation
    topo_change_obs[~mask_obs] = 0

    nse, rmse, nmae, mass, bss = model_skill(topo_change_obs, topo_change_sim, np.zeros(topo_change_obs.shape), mask_all)
    nse_ow, rmse_ow, nmae_ow, mass_ow, bss_ow = model_skill(topo_change_obs, topo_change_sim, np.zeros(topo_change_obs.shape), mask_overwash_all)  # Skill scores for just overwash
    nse_bd, rmse_bd, nmae_bd, mass_bd, bss_bd = model_skill(topo_change_obs, topo_change_sim, np.zeros(topo_change_obs.shape), beach_duneface_mask)  # Skill scores for just beach/dune

    if np.isinf(bss_bd) or np.isinf(bss_ow):
        weighted_bss = -10e6
    else:
        weighted_bss = np.average([bss_ow, bss_bd], weights=[2, 1])

    score = bss  # This is the skill score used in particle swarms optimization

    return score


def multi_fitness(solution):
    """Runs particular parameter combinations for multiple storms and locations and returns average skill score."""

    global BestScore
    global BestScores

    score_list = []

    for s in range(len(storm_start)):
        for x in range(len(x_min)):

            # Initial Observed Topo
            Init = np.load(storm_start[s])
            # Final Observed
            End = np.load(storm_stop[s])

            # Transform Initial Observed Topo
            topo_init = Init[0, x_min[x]: x_max[x], :]  # [m]
            topo_start = copy.deepcopy(topo_init)  # [m] Initialise the topography map

            # Transform Final Observed Topo
            topo_final = End[0, x_min[x]: x_max[x], :]  # [m]
            OW_Mask = Florence_Overwash_Mask[x_min[x]: x_max[x], :]  # [bool]

            # Set Veg Domain
            spec1 = Init[1, x_min[x]: x_max[x], :]
            spec2 = Init[2, x_min[x]: x_max[x], :]

            # Initialize storm stats
            Rhigh = storm_Rhigh[s] * np.ones(topo_final.shape[0])
            Rlow = storm_Rlow[s] * np.ones(topo_final.shape[0])
            dur = storm_dur[s]

            score = storm_fitness(solution, topo_start, topo_final, Rhigh, Rlow, dur, OW_Mask, spec1, spec2)
            # print("  > score:", score)
            score_list.append(score)

    # Take mean of scores from all locations
    multi_score = np.nanmean(score_list)

    if multi_score > BestScore:
        BestScore = multi_score
        BestScores = score_list

    return multi_score


def opt_func(X):
    """Runs a parallelized batch of hindcast simulations and returns a fitness result for each"""

    with routine.tqdm_joblib(tqdm(desc="Iteration Progress", total=swarm_size)) as progress_bar:
        solutions = Parallel(n_jobs=5)(delayed(multi_fitness)(X[i, :]) for i in range(swarm_size))

    print("  >>> solutions:", solutions)

    return np.array(solutions) * -1


# ___________________________________________________________________________________________________________________________________
# ___________________________________________________________________________________________________________________________________
# SET UP CALIBRATION

start_time = time.time()  # Record time at start of calibration

# _____________________________________________
# Define Variables
MHW = 0.39  # [m NAVD88]

# Observed Overwash Mask
Florence_Overwash_Mask = np.load("Input/Mask_NCB-NewDrum-Ocracoke_2018_Florence.npy")  # Load observed overwash mask
name = 'Multi-Location Storm (Florence), Particle Swarms Optimization, NON-Weighted Score, Nswarm 16, Iterations 40, 2Nov23'

BestScore = -1e10
BestScores = []

# _____________________________________________
# Define Multiple Events

# storm_name = ['Sandy', 'Florence']
# storm_start = ["Init_NCB-NewDrum-Ocracoke_2011_PostIrene.npy", "Input/Init_NorthernNCB_2017_PreFlorence.npy"]
# storm_stop = ["Init_NCB-NewDrum-Ocracoke_2012_PostSandyUSGS_MinimalThin.npy", "Input/Init_NorthernNCB_2018_PostFlorence.npy"]
# storm_Rhigh = [2.06, 3.32]
# storm_Rlow = [1.12, 1.90]
# storm_dur = [49, 83]

storm_name = ['Florence']
storm_start = ["Input/Init_NCB-NewDrum-Ocracoke_2017_PreFlorence.npy"]
storm_stop = ["Input/Init_NCB-NewDrum-Ocracoke_2018_PostFlorence-Plover.npy"]
storm_Rhigh = [3.32]
storm_Rlow = [1.90]
storm_dur = [83]


# _____________________________________________
# Define Multiple Locations

x_min = [18950, 19825, 20375, 20525, 6300]  # 20975
x_max = [19250, 20275, 20525, 20725, 6600]  # 21125
# small flat, large flat, small gap, large gap, tall ridge  # tall ridge


# ___________________________________________________________________________________________________________________________________
# ___________________________________________________________________________________________________________________________________
# PARTICLE SWARMS OPTIMIZATION

# _____________________________________________
# Prepare Particle Swarms Parameters

iterations = 40
swarm_size = 20
dimensions = 10  # Number of free paramters
options = {'c1': 1.5, 'c2': 1.5, 'w': 0.5}
"""
w: Inertia weight constant. [0-1] Determines how much the particle keeps on with its previous velocity (i.e., speed and direction of the search). 
c1 & c2: Cognitive and the social coefficients, respectively. Control how much weight should be given between refining the search result of the
particle itself and recognizing the search result of the swarm; Control the trade off between exploration and exploitation.
"""

bounds = (
    # Minimum
    np.array([50,  # Rin
              1,  # Cx
              0.5,  # MaxUpSlope
              0.02,  # flow_reduction_max_spec1
              8e-06,  # Kr
              0.05,  # flow_reduction_max_spec2
              1,  # substep_r (overwash)
              0.002,  # beach_equilibrium_slope
              0.00025,  # swash_transport_coefficient
              10]),  # beach_substep
    # Maximum
    np.array([450,  # Rin
              100,  # Cx
              2.5,  # MaxUpSlope
              0.4,  # flow_reduction_max_spec1
              1e-04,  # Kr
              0.5,  # flow_reduction_max_spec2
              12,  # substep_r (overwash)
              0.03,  # beach_equilibrium_slope
              0.0035,  # swash_transport_coefficient
              80])  # beach_substep
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

print()
print(tabulate({
    "BEST SOLUTION": ["BSS"],
    "Rin": [best_solution[0]],
    "Cx": [best_solution[1]],
    "MUS": [best_solution[2]],
    "Qreduc_s1": [best_solution[3]],
    "Kr": [best_solution[4]],
    "Qreduc_s2": [best_solution[5]],
    "SSo": [best_solution[6]],
    "Beq": [best_solution[7]],
    "Kc": [best_solution[8]],
    "SSb": [best_solution[9]],
    "Score": [solution_fitness * -1]
}, headers="keys", floatfmt=(None, ".0f", ".0f", ".2f", ".2f", ".7f", ".2f", ".0f", ".3f", ".7f", ".0f", ".4f"))
)

# _____________________________________________
# Plot Results
plt.plot(np.array(optimizer.cost_history) * -1)
plt.ylabel('Fitness (BSS)')
plt.xlabel('Iteration')

plt.show()
