"""
Script for calibrating MEEB storm parameters using genetic algorithm.

Calibrates based on fitess score for all morphologic change (i.e., beach/dune and overwash together).

IRBR 14 Mar 2023
"""

import numpy as np
import matplotlib.pyplot as plt
import routines_meeb as routine
import copy
import time
from tabulate import tabulate
import pygad


# ___________________________________________________________________________________________________________________________________
# ___________________________________________________________________________________________________________________________________

def model_skill(obs_change, sim_change, obs_change_mean, mask):
    """Perform suite of model skill assesments and return scores."""

    if np.isnan(np.sum(sim_change)):
        nse = -1e10
        rmse = -1e10
        bss = -1e10
        pc = -1e10
        hss = -1e10

    else:
        # _____________________________________________
        # Nash-Sutcliffe Model Efficiency
        """The closer the score is to 1, the better the agreement. If the score is below 0, the mean observed value is a better predictor than the model."""
        A = np.mean(np.square(np.subtract(obs_change[mask], sim_change[mask])))
        B = np.mean(np.square(np.subtract(obs_change[mask], obs_change_mean)))
        nse = 1 - A / B

        # _____________________________________________
        # Root Mean Square Error
        rmse = np.sqrt(np.mean(np.square(sim_change[mask] - obs_change[mask])))

        # _____________________________________________
        # Brier Skill Score
        """A skill score value of zero means that the score for the predictions is merely as good as that of a set of baseline or reference or default predictions, 
        while a skill score value of one (100%) represents the best possible score. A skill score value less than zero means that the performance is even worse than 
        that of the baseline or reference predictions (i.e., the baseline matches the final field profile more closely than the simulation output)."""
        bss = routine.brier_skill_score(sim_change, obs_change, np.zeros(sim_change.shape), mask)

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
            pc = (hits + correct_rejects) / J

            # Heidke Skill Score
            """The percentage correct, corrected for the number expected to be correct by chance. Scores closer to 1 (100%) are better."""
            G = ((hits + false_alarms) * (hits + misses) / J ** 2) + ((misses + correct_rejects) * (false_alarms + correct_rejects) / J ** 2)  # Fraction of predictions of the correct categories (H and C) that would be expected from a random choice
            hss = (pc - G) / (1 - G)  # The percentage correct, corrected for the number expected to be correct by chance

        else:
            pc = -1e10
            hss = -1e10

    return nse, rmse, bss, pc, hss


def storm_fitness(solution, solution_idx):
    """Run a storm with this particular combintion of parameter values, and return fitness value of simulated to observed."""

    topof = copy.deepcopy(topo_prestorm)

    topof, topo_change_overwash, OWflux, netDischarge, inundated, Qbe = routine.storm_processes(
        topof,
        Rhigh,
        Rlow,
        dur,
        slabheight_m=slabheight_m,
        threshold_in=0.25,
        Rin_i=5,
        Rin_r=solution[0],
        Cx=solution[1],
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
        substep_r=solution[6],
        beach_equilibrium_slope=solution[7],
        beach_erosiveness=solution[8],
        beach_substeps=solution[9],
        x_s=x_s,
    )

    sim_topo_final = topof * slabheight_m  # [m]
    obs_topo_final_m = topo_final * slabheight_m  # [m]
    topo_pre_m = topo_prestorm * slabheight_m  # [m]
    topo_change_prestorm = sim_topo_final - topo_pre_m

    # _____________________________________________
    # Model Skill: Comparisons to Observations

    subaerial_mask = sim_topo_final > MHW  # [bool] Map of every cell above water

    beach_duneface_mask = np.zeros(sim_topo_final.shape)
    for l in range(topo_prestorm.shape[0]):
        beach_duneface_mask[l, :dune_crest[l]] = True
    beach_duneface_mask = np.logical_and(beach_duneface_mask, subaerial_mask)  # [bool] Map of every cell seaward of dune crest

    Sim_Obs_All_Mask = np.logical_or(OW_Mask, inundated, beach_duneface_mask) * subaerial_mask  # [bool] Map of every cell landward of dune crest that was inundated in simulation or observation or both

    # Final Elevation Changes
    obs_change_m = (obs_topo_final_m - topo_pre_m)  # [m] Observed change
    sim_change_m = topo_change_prestorm  # [m] Simulated change

    # Mask
    obs_change_masked = obs_change_m * Sim_Obs_All_Mask  # [m]
    sim_change_masked = sim_change_m * Sim_Obs_All_Mask  # [m]
    obs_change_mean_masked = np.mean(obs_change_m[Sim_Obs_All_Mask])  # [m] Average beach change of observations, masked

    if np.isnan(np.sum(sim_change_masked)):
        score = -1e10
    else:
        nse, rmse, bss, pc, hss = model_skill(obs_change_masked, sim_change_masked, obs_change_mean_masked, Sim_Obs_All_Mask)
        score = bss  # This is the skill score used in genetic algorithm

    return score


# ___________________________________________________________________________________________________________________________________
# ___________________________________________________________________________________________________________________________________

start_time = time.time()  # Record time at start of calibration

# _____________________________________________
# Define Variables
Rhigh = 3.32
Rlow = 0.9  # Actual Florence: 1.93
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

name = '575-825, KQ(S+C)'


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


# _____________________________________________
# Prepare Genetic Algoritm Parameters

# Genes: Free parameters
num_genes = 10
gene_type = [int,
             int,
             [float, 2],
             [float, 2],
             [float, 7],
             [float, 2],
             int,
             [float, 3],
             [float, 2],
             int]
gene_space = [{'low': 50, 'high': 450},  # Rin
              {'low': 1, 'high': 100},  # Cx
              {'low': 0.5, 'high': 2.5},  # MaxUpSlope
              {'low': 1, 'high': 1},  # fluxLimit
              {'low': 8e-06, 'high': 1e-04},  # Kr
              {'low': 2, 'high': 2},  # mm
              {'low': 1, 'high': 12},  # OW Substep
              {'low': 0.002, 'high': 0.03},  # Beq
              {'low': 0.25, 'high': 3},  # Et
              {'low': 10, 'high': 80}]  # BD Substep

# Generations
num_generations = 75
sol_per_pop = 5  # Solutions for each population

mutation_type = "adaptive"
mutation_percent_genes = [15, 5]

num_parents_mating = 4
parent_selection_type = "sss"
keep_parents = 1
crossover_type = "single_point"


# ___________________________________________________________________________________________________________________________________
# ___________________________________________________________________________________________________________________________________

# _____________________________________________
# Find Best Set of Parameter Values With PyGAD

# Create instance of GA class
ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=storm_fitness,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       gene_type=gene_type,
                       gene_space=gene_space,
                       parent_selection_type=parent_selection_type,
                       keep_parents=keep_parents,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       mutation_percent_genes=mutation_percent_genes,
                       suppress_warnings=True,
                       parallel_processing=3)

# Run genetic algorithm
print(name)
print()
print("  Working...")
ga_instance.run()


# _____________________________________________
SimDuration = time.time() - start_time
print()
print("Elapsed Time: ", SimDuration, "sec")
print()
print("Complete.")


# _____________________________________________
# Plot & Print

best_solution, solution_fitness, solution_idx = ga_instance.best_solution()

print()
print(tabulate({
    "BEST SOLUTION": ["BSS"],
    "Rin":  [best_solution[0]],
    "Cx":   [best_solution[1]],
    "MUS":  [best_solution[2]],
    "FLim": [best_solution[3]],
    "Kr":   [best_solution[4]],
    "mm":   [best_solution[5]],
    "SSo":  [best_solution[6]],
    "Beq": [best_solution[7]],
    "Et": [best_solution[8]],
    "SSb": [best_solution[9]],
    "Score": [solution_fitness]
    }, headers="keys", floatfmt=(None, ".0f", ".0f", ".2f", ".2f", ".7f", ".2f", ".0f", ".3f", ".2f", ".0f", ".4f"))
)

ga_instance.plot_fitness(title=name)
