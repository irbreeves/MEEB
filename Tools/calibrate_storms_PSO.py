"""
Script for calibrating MEEB storm parameters using Particle Swarms Optimization.

Calibrates based on fitess score for all beach/dune/overwash morphologic change.

IRBR 13 July 2023
"""

import numpy as np
import matplotlib.pyplot as plt
import routines_meeb as routine
import copy
import time
from tabulate import tabulate
import pyswarms as ps
from joblib import Parallel, delayed


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

    sim_topo_final, topo_change_overwash, OWflux, netDischarge, inundated, Qbe = routine.storm_processes(
        copy.deepcopy(topo_prestorm),
        Rhigh,
        Rlow,
        dur,
        slabheight_m=slabheight_m,
        threshold_in=0.25,
        Rin_i=5,
        Rin_r=int(round(solution[0])),
        Cx=int(round(solution[1])),
        AvgSlope=2 / 200,
        nn=0.5,
        MaxUpSlope=solution[2],
        fluxLimit=solution[3],
        Qs_min=1,
        Kr=solution[4],
        Ki=5e-06,
        mm=solution[5],
        MHW=MHW,
        Cbb_i=0.85,
        Cbb_r=0.7,
        Qs_bb_min=1,
        substep_i=6,
        substep_r=int(round(solution[6])),
        beach_equilibrium_slope=solution[7],
        beach_erosiveness=solution[8],
        beach_substeps=int(round(solution[9])),
        x_s=x_s,
    )

    # Final Elevations
    obs_final_m = topo_final * slabheight_m  # [m] Observed final topo
    sim_final_m = sim_topo_final * slabheight_m  # [m] Simulated final topo
    obs_change_m = (topo_final - topo_prestorm) * slabheight_m  # [m] Observed change
    sim_change_m = (sim_topo_final - topo_prestorm) * slabheight_m  # [m] Simulated change

    # Masking
    subaerial_mask = sim_final_m > (MHW * slabheight_m)  # [bool] Map of every cell above water

    beach_duneface_mask = np.zeros(sim_final_m.shape)
    for l in range(sim_final_m.shape[0]):
        beach_duneface_mask[l, :dune_crest[l]] = True
    beach_duneface_mask = np.logical_and(beach_duneface_mask, subaerial_mask)  # [bool] Map of every cell seaward of dune crest

    mask_all = np.logical_or(OW_Mask, inundated, beach_duneface_mask) * subaerial_mask  # [bool] Map of every cell landward of dune crest that was inundated in simulation or observation or both

    # Find Scores
    if np.isnan(np.sum(sim_change_m)):
        score = -1e10
    else:
        # nse, rmse, bss, pc, hss = model_skill(obs_change_masked, sim_change_masked, obs_change_mean_masked, Sim_Obs_All_Mask)
        nse, rmse, nmae, mass, bss = model_skill(obs_change_m, sim_change_m, np.zeros(obs_change_m.shape), mask_all)
        score = bss  # This is the skill score used in genetic algorithm

    return score


def opt_func(X):
    """Runs a parallelized batch of hindcast simulations and returns a fitness result for each"""

    solutions = Parallel(n_jobs=10)(delayed(storm_fitness)(X[i, :]) for i in range(swarm_size))

    return np.array(solutions) * -1


# ___________________________________________________________________________________________________________________________________
# ___________________________________________________________________________________________________________________________________
# SET UP CALIBRATION

start_time = time.time()  # Record time at start of calibration

# _____________________________________________
# Define Variables
Rhigh = 3.32
Rlow = 1.93
dur = 70
slabheight_m = 0.1
MHW = 0

# Initial Observed Topo
Init = np.load("Input/Init_NorthernNCB_2017_PreFlorence.npy")
# Final Observed
End = np.load("Input/Init_NorthernNCB_2018_PostFlorence.npy")

# Observed Overwash Mask
Florence_Overwash_Mask = np.load("Input/NorthernNCB_FlorenceOverwashMask.npy")  # Load observed overwash mask

# Define Alongshore Coordinates of Domain
xmin = 575  # 575, 2000, 2150, 2000, 3800  # 2650
xmax = 825  # 825, 2125, 2350, 2600, 4450  # 2850

name = '575-825, KQ(S+C), 150 itr'

# _____________________________________________
# Conversions & Initializations

# Transform Initial Observed Topo
topo_init = Init[0, xmin: xmax, :]  # [m]
topo0 = topo_init / slabheight_m  # [slabs] Transform from m into number of slabs
topo = copy.deepcopy(topo0)  # [slabs] Initialise the topography map

# Transform Final Observed Topo
topo_final = End[0, xmin:xmax, :] / slabheight_m  # [slabs] Transform from m into number of slabs
OW_Mask = Florence_Overwash_Mask[xmin: xmax, :]  # [bool]

# Set Veg Domain
spec1 = Init[2, xmin: xmax, :]
spec2 = Init[3, xmin: xmax, :]
veg = spec1 + spec2  # Determine the initial cumulative vegetation effectiveness
veg[veg > 1] = 1  # Cumulative vegetation effectiveness cannot be negative or larger than one
veg[veg < 0] = 0

# Find Dune Crest, Shoreline Positions
dune_crest = routine.foredune_crest(topo, MHW)
x_s = routine.ocean_shoreline(topo, MHW)

# Transform water levels to vectors
Rhigh = Rhigh * np.ones(topo_final.shape[0])
Rlow = Rlow * np.ones(topo_final.shape[0])

topo_prestorm = copy.deepcopy(topo)

# ___________________________________________________________________________________________________________________________________
# ___________________________________________________________________________________________________________________________________
# PARTICLE SWARMS OPTIMIZATION

# _____________________________________________
# Prepare Particle Swarm Parameters

iterations = 5
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
              1,  # fluxLimit
              8e-06,  # Kr
              2,  # mm
              1,  # OW substep
              0.002,  # Beq
              0.25,  # Et
              10]),  # BD substep
    # Maximum
    np.array([450,
              100,
              2.5,
              1.00001,
              1e-04,
              2.00001,
              12,
              0.03,
              3,
              80])
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
    "FLim": [best_solution[3]],
    "Kr": [best_solution[4]],
    "mm": [best_solution[5]],
    "SSo": [best_solution[6]],
    "Beq": [best_solution[7]],
    "Et": [best_solution[8]],
    "SSb": [best_solution[9]],
    "Score": [solution_fitness * -1]
}, headers="keys", floatfmt=(None, ".0f", ".0f", ".2f", ".2f", ".7f", ".2f", ".0f", ".3f", ".2f", ".0f", ".4f"))
)

# _____________________________________________
# Plot Results
plt.plot(np.array(optimizer.cost_history) * -1)
plt.ylabel('Fitness (BSS)')
plt.xlabel('Iteration')
