"""
Script for calibrating MEEB storm parameters using Particle Swarms Optimization.

Calibrates based on fitess score for all beach/dune/overwash morphologic change.

IRBR 8 February 2024
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


def storm_fitness(solution):
    """Run a storm with this particular combintion of parameter values, and return fitness value of simulated to observed."""

    # Run Model
    topo_start_copy = copy.deepcopy(topo_start)
    topo_end_sim, topo_change_overwash, OWflux, netDischarge, inundated, Qbe = routine.storm_processes(
        topo_start_copy,
        Rhigh,
        Rlow,
        dur,
        Rin=int(round(solution[0])),
        Cx=int(round(solution[1])),
        AvgSlope=2 / 200,
        nn=0.5,
        MaxUpSlope=1,
        fluxLimit=1,
        Qs_min=1,
        Kow=solution[2],
        mm=round(solution[5], 2),
        MHW=MHW,
        Cbb=0.7,
        Qs_bb_min=1,
        substep=4,
        beach_equilibrium_slope=solution[3],
        swash_transport_coefficient=solution[4],
        wave_period_storm=9.4,
        beach_substeps=20,
        x_s=x_s,
        cellsize=1,
        spec1=spec1,
        spec2=spec2,
        flow_reduction_max_spec1=0.2,  # solution[3],  # Grass
        flow_reduction_max_spec2=0.3,  # solution[5],  # Shrub
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


def opt_func(X):
    """Runs a parallelized batch of hindcast simulations and returns a fitness result for each"""

    with routine.tqdm_joblib(tqdm(desc="Iteration Progress", total=swarm_size)) as progress_bar:
        solutions = Parallel(n_jobs=6)(delayed(storm_fitness)(X[i, :]) for i in range(swarm_size))

    return np.array(solutions) * -1


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
xmin = 19220  # 19825, 20375, 20525 20975
xmax = 21220  # 20275, 20525, 20725 21125
# small flat, large flat, small gap, large gap, tall ridge

name = '19220-21220, NON-weighted BSS, 18 particles, 50 iter'

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
# PARTICLE SWARMS OPTIMIZATION

# _____________________________________________
# Prepare Particle Swarm Parameters

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
    np.array([100,  # Rin
              10,  # Cx
              8e-06,  # Kr
              0.01,  # beach_equilibrium_slope
              0.0005,  # swash_transport_coefficient
              1.0]),  # mm
    # Maximum
    np.array([320,  # Rin
              80,  # Cx
              1e-04,  # Kr
              0.04,  # beach_equilibrium_slope
              0.0035,  # swash_transport_coefficient
              1.15])  # mm
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
    "Kc": [best_solution[4]],
    "mm": [best_solution[5]],
    "Score": [solution_fitness * -1]
}, headers="keys", floatfmt=(None, ".0f", ".0f", ".7f", ".3f", ".5f", ".2f", ".4f"))
)

# _____________________________________________
# Plot Results
plt.plot(np.array(optimizer.cost_history) * -1)
plt.ylabel('Fitness (BSS)')
plt.xlabel('Iteration')

plt.show()
