"""
Script for calibrating MEEB storm parameters using Particle Swarms Optimization.

Calibrates based on fitess score for all beach/dune/overwash morphologic change, and incorporates multiple
storm events and/or locations into each fitness score.

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
    dune_crest, not_gap = routine.foredune_crest(topo_start_obs, MHW, cellsize)
    x_s = routine.ocean_shoreline(topo_start_obs, MHW)

    # Run Model
    topo_start_copy = copy.deepcopy(topo_start_obs)
    veg = spec1 + spec2  # Determine the initial cumulative vegetation effectiveness

    sim_topo_final, topo_change_overwash, OWflux, inundated, Qbe = routine.storm_processes(
        topo_start_copy,
        Rhigh,
        Rlow,
        dur,
        Rin=int(round(solution[0])),
        Cs=round(solution[1], 4),
        nn=0.5,
        MaxUpSlope=1.5,
        fluxLimit=1,
        Qs_min=1,
        Kow=solution[2],
        mm=round(solution[5], 2),
        MHW=MHW,
        Cbb=0.7,
        Qs_bb_min=1,
        substep=50,
        beach_equilibrium_slope=solution[3],
        swash_erosive_timescale=solution[4],
        beach_substeps=25,
        x_s=x_s,
        cellsize=cellsize,
        spec1=spec1,
        spec2=spec2,
        flow_reduction_max_spec1=0.02,  # Grass
        flow_reduction_max_spec2=0.05,  # Shrub
    )

    topo_end_sim = routine.enforceslopes(sim_topo_final, veg, sh=0.02, anglesand=20, angleveg=30, th=0.3, MHW=MHW, cellsize=cellsize, RNG=RNG)[0]

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
            Rlow = storm_Rlow[s] * np.ones(topo_final.shape[0])
            dur = storm_dur[s]

            score = storm_fitness(solution, topo_start, topo_final, Rhigh, Rlow, dur, OW_Mask, spec1, spec2)
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
        solutions = Parallel(n_jobs=n_cores)(delayed(multi_fitness)(X[i, :]) for i in range(swarm_size))

    print("  >>> solutions:", solutions)

    return np.array(solutions) * -1  # Return negative because optimization algorithm tries to find minimum score, whereas better BSS are larger


# ___________________________________________________________________________________________________________________________________
# ___________________________________________________________________________________________________________________________________
# SET UP CALIBRATION

start_time = time.time()  # Record time at start of calibration
RNG = np.random.default_rng(seed=13)  # Seeded random numbers for reproducibility

# _____________________________________________
# Define Variables
MHW = 0.39  # [m NAVD88]
cellsize = 1  # [m]

# Observed Overwash Mask
overwash_mask_file = np.load("Input/Mask_NCB-NewDrum-Ocracoke_2018_Florence.npy")  # Load observed overwash mask
name = 'Multi-Location Storm (Florence), NON-Weighted BSS PSO, Nswarm 18, Iterations 50, SS=50/25, 24May24'

BestScore = -1e10
BestScores = []

# _____________________________________________
# Define Event(s)

storm_name = ['Florence']
storm_start = ["Input/Init_NCB-NewDrum-Ocracoke_2017_PreFlorence.npy"]
storm_stop = ["Input/Init_NCB-NewDrum-Ocracoke_2018_PostFlorence-Plover.npy"]
storm_Rhigh = [3.32]
storm_Rlow = [1.90]
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
# PARTICLE SWARMS OPTIMIZATION

# Choose number of cores to run parallel simulations on
n_cores = 18

# _____________________________________________
# Prepare Particle Swarms Parameters

iterations = 50
swarm_size = 18
dimensions = 6  # Number of free paramters
options = {'c1': 1.5, 'c2': 1.5, 'w': 0.5}
"""
w: Inertia weight constant. [0-1] Determines how much the particle keeps on with its previous velocity (i.e., speed and direction of the search). 
c1 & c2: Cognitive and the social coefficients, respectively. Control how much weight should be given between refining the search result of the
particle itself and recognizing the search result of the swarm; Control the trade off between exploration and exploitation.
"""

bounds = (
    # Minimum
    np.array([50,  # Rin
              0.010,  # Cs
              0.00005,  # Kr
              0.01,  # beach_equilibrium_slope
              1.0,  # Swash erosive timescale
              1.01]),  # mm
    # Maximum
    np.array([280,  # Rin
              0.040,  # Cs
              0.01,  # Kr
              0.04,  # beach_equilibrium_slope
              3.0,  # Swash erosive timescale
              1.12])  # mm
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
    "Kr": [best_solution[2]],
    "Beq": [best_solution[3]],
    "Te": [best_solution[4]],
    "mm": [best_solution[5]],
    "Score": [solution_fitness * -1]
}, headers="keys", floatfmt=(None, ".0f", ".4f", ".7f", ".3f", ".2f", ".2f", ".4f"))
)

# _____________________________________________
# Plot Results
plt.plot(np.array(optimizer.cost_history) * -1)
plt.ylabel('Fitness (BSS)')
plt.xlabel('Iteration')

plt.show()
