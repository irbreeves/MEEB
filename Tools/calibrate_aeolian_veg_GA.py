"""
Script for calibrating MEEB aeolian and vegetation parameters using a genetic algorithm.

Calibrates based on fitess score for all morphologic and ecologic change between two timesteps.

Can choose to calibrate for just aeolian, just veg, or both.

IRBR 16 June 2023
"""

import numpy as np
import matplotlib.pyplot as plt
import routines_meeb as routine
import copy
import time
from tabulate import tabulate
import pygad

from meeb import MEEB


# ___________________________________________________________________________________________________________________________________
# ___________________________________________________________________________________________________________________________________

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


def storm_fitness(solution, solution_idx):
    """Run a hindcast this particular combintion of parameter values, and return fitness value of simulated to observed."""

    # Create an instance of the BMI class
    meeb = MEEB(
        name="SLR 0 mm/yr, 2004-2009 Hindcast",
        simulation_time_yr=5.1,
        alongshore_domain_boundary_min=xmin,
        alongshore_domain_boundary_max=xmax,
        RSLR=0.000,
        MHW=MHW,
        seeded_random_numbers=True,
        p_dep_sand=solution[0],
        p_dep_sand_VegMax=solution[0] + solution[1],
        p_ero_sand=solution[2],
        entrainment_veg_limit=solution[3],
        shadowangle=solution[4],
        repose_bare=solution[5],
        repose_veg=solution[5] + solution[6],
        direction2=solution[7],
        direction4=solution[8],
        init_filename=start,
        hindcast=True,
        simulation_start_date='20040716',
        storm_timeseries_filename='StormTimeSeries_1980-2020_NCB-CE_Beta0pt039_BermEl2pt03.npy',
    )


    # Loop through time
    for time_step in range(int(meeb.iterations)):
        # Run time step
        meeb.update(time_step)

    # __________________________________________________________________________________________________________________________________
    # ASSESS MODEL SKILL

    topo_end_sim = meeb.topo * meeb.slabheight  #  [m NAVDD88]
    mhw_end_sim = meeb.MHW * meeb.slabheight  # [m NAVD88]
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

    # Temp limit interior in analysis to dunes
    subaerial_mask[:, :835] = False
    subaerial_mask[:, 950:] = False

    # Limit interior in analysis by elevation
    elev_mask = topo_end_sim > 2.0  # [bool] Mask for every cell above water

    # # Optional: Reduce Resolutions
    # reduc = 5  # Reduction factor
    # topo_change_obs = routine.reduce_raster_resolution(topo_change_obs, reduc)
    # topo_change_sim = routine.reduce_raster_resolution(topo_change_sim, reduc)
    # subaerial_mask = (routine.reduce_raster_resolution(subaerial_mask, reduc)) == 1

    # Model Skill
    nse, rmse, nmae, mass, bss, pc, hss = model_skill(topo_change_obs, topo_change_sim, np.zeros(topo_change_obs.shape), subaerial_mask)  # All cells (excluding masked areas)
    nse_dl, rmse_dl, nmae_dl, mass_dl, bss_dl, pc_dl, hss_dl = model_skill(crest_loc_obs.astype('float32'), crest_loc_sim.astype('float32'), crest_loc_obs_start.astype('float32'), np.full(crest_loc_obs.shape, True))  # Foredune location
    nse_dh, rmse_dh, nmae_dh, mass_dh, bss_dh, pc_dh, hss_dh = model_skill(crest_height_obs, crest_height_sim, crest_height_obs_start, np.full(crest_height_change_obs.shape, True))  # Foredune elevation

    # Combine skill scores, given relative weights for each score
    # score = -1 * (nmae + nmae_dl + nmae_dh)  # This is the skill score used in genetic algorithm

    score = bss  # This is the skill score used in genetic algorithm

    return score


# ___________________________________________________________________________________________________________________________________
# ___________________________________________________________________________________________________________________________________

start_time = time.time()  # Record time at start of calibration

# __________________________________________________________________________________________________________________________________
# VARIABLES AND INITIALIZATIONS
# # 0.92, 20161012
# start = "Init_NCB-NewDrum-Ocracoke_2016_PostMatthew.npy"
# stop = "Init_NCB-NewDrum-Ocracoke_2017_PreFlorence.npy"

# 5.1, 20040716
start = "Init_NCB-NewDrum-Ocracoke_2004_PostIsabel.npy"
stop = "Init_NCB-NewDrum-Ocracoke_2009_PreIrene.npy"

# # 3.44, 20140406
# start = "Init_NCB-NewDrum-Ocracoke_2014_PostSandyNCFMP.npy"
# stop = "Init_NCB-NewDrum-Ocracoke_2017_PreFlorence.npy"

# # 4.78, 20121129
# start = "Init_NCB-NewDrum-Ocracoke_2012_PostSandyUSGS_NoThin.npy"
# stop = "Init_NCB-NewDrum-Ocracoke_2017_PreFlorence.npy"

# Define Alongshore Coordinates of Domain
xmin = 6500  # 575, 2000, 2150, 2000, 3800  # 2650
xmax = 6600  # 825, 2125, 2350, 2600, 4450  # 2850

MHW = 0.4  # [m NAVD88]
name = '6500-6600, 2004-2009, BSS'

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



# _____________________________________________
# Prepare Genetic Algoritm Parameters

# Genes: Free parameters
num_genes = 9
gene_type = [[float, 2],
             [float, 2],
             [float, 2],
             [float, 2],
             int,
             int,
             int,
             int,
             int]
gene_space = [{'low': 0.02, 'high': 0.5},  # p_dep_sand
              {'low': 0.05, 'high': 0.5},  # p_dep_sand_VegMax
              {'low': 0.02, 'high': 0.5},  # p_ero_sand
              {'low': 0.05, 'high': 0.55},  # entrainment_veg_limit
              {'low': 5, 'high': 15},  # shadowangle
              {'low': 15, 'high': 30},  # repose_bare
              {'low': 5, 'high': 10},  # repose_veg
              {'low': 1, 'high': 4},  # direction2
              {'low': 1, 'high': 4}]  # direction4

# Generations
num_generations = 50
sol_per_pop = 5  # Solutions for each population, AKA individuals

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
                       save_best_solutions=True,
                       parallel_processing=4)

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
    "BEST SOLUTION": ["NMAE"],
    "p_dep_sand":  [best_solution[0]],
    "p_dep_sand_VegMax":  [best_solution[0] + best_solution[1]],
    "p_ero_sand":   [best_solution[2]],
    "entrainment_veg_limit":  [best_solution[3]],
    "shadowangle":  [best_solution[4]],
    "repose_bare": [best_solution[5]],
    "repose_veg": [best_solution[5] + best_solution[6]],
    "direction2":   [best_solution[7]],
    "direction4":   [best_solution[8]],
    "Score": [solution_fitness]
    }, headers="keys", floatfmt=(None, ".2f", ".2f", ".2f", ".2f", ".0f", ".0f", ".0f", ".0f", ".0f", ".4f"))
)

ga_instance.plot_fitness(title=name)
