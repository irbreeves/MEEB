"""
Script for calibrating MEEB aeolian parameters using Particle Swarms Optimization.

Calibrates based on fitess score for morphologic change between two observation, and incorporates multiple
timeframes and/or locations into each fitness score.

IRBR 14 March 2024
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


def aeolian_fitness(solution, topo_name, topo_start, topo_end_obs, xmin, xmax, y_min, y_max, startdate, dur):
    """Run a hindcast this particular combintion of parameter values, and return fitness value of simulated to observed."""

    # Construct wind rose
    wind_axis_ratio = solution[6]  # Proportion of cross-shore winds (right/left, i.e. onshore/offshore) versus alongshore winds (up/down)
    wind_dir_1 = max(0, solution[7] * wind_axis_ratio)  # Onshore (proportion towards right)
    wind_dir_2 = max(0, (1 - solution[7]) * wind_axis_ratio)  # Offshore (proportion towards left)
    wind_dir_3 = max(0, solution[8] * (1 - wind_axis_ratio))  # Alongshore (proportion towards down)
    wind_dir_4 = max(0, 1 - (wind_dir_1 + wind_dir_2 + wind_dir_3))  # Alongshore (proportion towards up)
    rose = (wind_dir_1, wind_dir_2, wind_dir_3, wind_dir_4)  # Tuple that sums to 1.0

    # Create an instance of the BMI class
    meeb = MEEB(
        name=name,
        simulation_time_yr=dur,
        alongshore_domain_boundary_min=xmin,
        alongshore_domain_boundary_max=xmax,
        RSLR=0.004,
        MHW=MHW,
        seeded_random_numbers=True,
        init_filename=topo_name,
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
        repose_bare=20,
        repose_veg=30,
        wind_rose=rose,
        # --- Storms --- #
        Rin=213,
        Cx=36,
        MaxUpSlope=1.57,
        Kow=0.0000501,
        mm=1.02,
        overwash_substeps=4,
        beach_equilibrium_slope=0.027,
        swash_transport_coefficient=0.001,
        wave_period_storm=9.4,
        beach_substeps=20,
        flow_reduction_max_spec1=0.02,
        flow_reduction_max_spec2=0.05,
        # --- Veg --- #
        sp1_b=-0.05,
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
    subaerial_mask = np.logical_and(topo_end_sim > mhw_end_sim, topo_end_obs > mhw_end_sim)  # [bool] Mask for every cell above water

    # Cross-shore range mask
    range_mask = np.ones(topo_end_sim.shape)  # [bool] Mask for every cell between two cross-shore locations
    range_mask[:, :y_min] = False
    range_mask[:, y_max:] = False

    # Choose masks to use
    mask = np.logical_and(range_mask, subaerial_mask)  # [bool] Combined mask used for analysis

    # Dune crest locations and heights
    crest_loc_obs_start, not_gap_obs_start = routine.foredune_crest(topo_start, mhw_end_sim)
    crest_loc_obs, not_gap_obs = routine.foredune_crest(topo_end_obs, mhw_end_sim)
    crest_loc_sim, not_gap_sim = routine.foredune_crest(topo_end_sim, mhw_end_sim)

    crest_height_obs_start = topo_start[np.arange(topo_start.shape[0]), crest_loc_obs_start]
    crest_height_obs = topo_end_obs[np.arange(topo_end_obs.shape[0]), crest_loc_obs]
    crest_height_sim = topo_end_sim[np.arange(topo_end_obs.shape[0]), crest_loc_sim]
    crest_height_change_obs = crest_height_obs - crest_height_obs_start

    # Optional: Reduce Resolutions
    if ResReduc:
        topo_change_obs = routine.reduce_raster_resolution(topo_change_obs, reduc)
        topo_change_sim = routine.reduce_raster_resolution(topo_change_sim, reduc)
        mask = (routine.reduce_raster_resolution(mask, reduc)) == 1

    # Model Skill
    nse, rmse, nmae, mass, bss, pc, hss = model_skill(topo_change_obs, topo_change_sim, np.zeros(topo_change_obs.shape), mask)  # All cells (excluding masked areas)
    nse_dh, rmse_dh, nmae_dh, mass_dh, bss_dh, pc_dh, hss_dh = model_skill(crest_height_obs, crest_height_sim, crest_height_obs_start, np.full(crest_height_change_obs.shape, True))  # Foredune elevation

    # Combine Skill Scores (Multi-Objective Optimization)
    score = np.average([bss, bss_dh])  # This is the skill score used in particle swarms optimization

    return score


def multi_fitness(solution):
    """Runs particular parameter combinations for multiple storms and locations and returns average skill score."""

    score_list = []

    for s in range(len(sim_start)):
        for x in range(len(x_min)):
            # Initial Observed Topo
            Init = np.load("Input/" + sim_start[s])
            # Final Observed
            End = np.load("Input/" + sim_stop[s])

            # Transform Initial Observed Topo
            topo_i = Init[0, x_min[x]: x_max[x], :]  # [m]
            topo_start_x = copy.deepcopy(topo_i)  # [m] INITIAL TOPOGRPAHY

            # Transform Final Observed Topo
            topo_e = End[0, x_min[x]: x_max[x], :]  # [m]
            topo_end_obs_x = copy.deepcopy(topo_e)  # [m] FINAL OBSERVED TOPOGRAPHY

            # Set Veg Domain
            spec1_i = Init[1, x_min[x]: x_max[x], :]
            spec2_i = Init[2, x_min[x]: x_max[x], :]
            veg_start = spec1_i + spec2_i  # INITIAL VEGETATION COVER
            veg_start[veg_start > 1] = 1
            veg_start[veg_start < 0] = 0

            spec1_e = End[1, x_min[x]: x_max[x], :]
            spec2_e = End[2, x_min[x]: x_max[x], :]
            veg_end = spec1_e + spec2_e  # FINAL OBSERVED VEGETATION COVER
            veg_end[veg_end > 1] = 1
            veg_end[veg_end < 0] = 0

            score = aeolian_fitness(solution, sim_start[s], topo_start_x, topo_end_obs_x, x_min[x], x_max[x], y_range_min[x], y_range_max[x], sim_startdate[s], sim_dur[s])
            score_list.append(score)

    # Take mean of scores from all locations
    multi_score = np.nanmean(score_list)

    return multi_score


def opt_func(X):
    """Runs a parallelized batch of hindcast simulations and returns a fitness result for each"""

    with routine.tqdm_joblib(tqdm(desc="Iteration Progress", total=swarm_size)) as progress_bar:
        solutions = Parallel(n_jobs=n_cores)(delayed(multi_fitness)(X[i, :]) for i in range(swarm_size))

    return np.array(solutions) * -1


# ___________________________________________________________________________________________________________________________________
# ___________________________________________________________________________________________________________________________________
# SET UP CALIBRATION

start_time = time.time()  # Record time at start of calibration

n_cores = 16  # Number of cores to run parallel simulations on

# __________________________________________________________________________________________________________________________________
# VARIABLES AND INITIALIZATIONS

# _____________________________________________
# Define Multiple Timeframes

sim_start = ["Init_NCB-NewDrum-Ocracoke_2014_PostSandy-NCFMP-Plover.npy"]
sim_stop = ["Init_NCB-NewDrum-Ocracoke_2017_PreFlorence.npy"]
sim_startdate = ['20140406']  # [yyyymmdd]
sim_dur = [3.44]  # [yr]


# _____________________________________________
# Define Multiple Locations

x_min = [6300, 21000, 19030]
x_max = [6500, 21200, 19230]
# tall ridge 1, tall ridge 2, overwash fan, overwash gap

# Cross-shore range of foredune system for skill assessment
y_range_min = [838, 1068, 1135]
y_range_max = [880, 1100, 1185]


# ____________________________________
# Other Parameters & Variables
MHW = 0.39  # [m NAVD88] Initial
ResReduc = False  # Option to reduce raster resolution for skill assessment
reduc = 5  # Raster resolution reduction factor
name = '2014-2017, 3 Locations, Multi-Objective BSS, 3Mar24'


# ___________________________________________________________________________________________________________________________________
# ___________________________________________________________________________________________________________________________________
# PARTICLE SWARMS OPTIMIZATION

# _____________________________________________
# Prepare Particle Swarm Parameters

iterations = 50
swarm_size = 16
dimensions = 9  # Number of free paramters
options = {'c1': 1.5, 'c2': 1.5, 'w': 0.5}
"""
w: Inertia weight constant. [0-1] Determines how much the particle keeps on with its previous velocity (i.e., speed and direction of the search). 
c1 & c2: Cognitive and the social coefficients, respectively. Control how much weight should be given between refining the search result of the
particle itself and recognizing the search result of the swarm; Control the trade off between exploration and exploitation.
"""

bounds = (
    # Minimum
    np.array([0.02,  # p_dep_sand
              0.05,  # p_dep_sand_VegMax
              0.02,  # p_ero_sand
              0.05,  # entrainment_veg_limit
              0.05,  # saltveglimit
              5,     # shadowangle
              0.5,   # Proportion of cross-shore winds versus alongshore
              0.5,   # Proportion of cross-shore winds towards right (onshore)
              0]),   # Proportion of alongshore winds towards down
    # Maximum
    np.array([0.5,
              0.5,
              0.5,
              0.55,
              0.4,
              15,
              1,
              1,
              1])
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

best_wind_axis_ratio = best_solution[6]  # Proportion of cross-shore winds (right/left, i.e. onshore/offshore) versus alongshore winds (up/down)
best_wind_dir_1 = best_solution[7] * best_wind_axis_ratio  # Onshore (proportion towards right)
best_wind_dir_2 = (1 - best_solution[7]) * best_wind_axis_ratio  # Offshore (proportion towards left)
best_wind_dir_3 = best_solution[8] * (1 - best_wind_axis_ratio)  # Alongshore (proportion towards down)
best_wind_dir_4 = 1 - (best_wind_dir_1 + best_wind_dir_2 + best_wind_dir_3)  # Alongshore (proportion towards up)

print()
print(tabulate({
    "BEST SOLUTION": ["Multi-Objective BSS"],
    "p_dep_sand": [best_solution[0]],
    "p_dep_sand_VegMax": [best_solution[0] + best_solution[1]],
    "p_ero_sand": [best_solution[2]],
    "entrainment_veg_limit": [best_solution[3]],
    "salt_veg_limit": [best_solution[4]],
    "shadowangle": [best_solution[5]],
    "direction1": [best_wind_dir_1],
    "direction2": [best_wind_dir_2],
    "direction3": [best_wind_dir_3],
    "direction4": [best_wind_dir_4],
    "Score": [solution_fitness]
    }, headers="keys", floatfmt=(None, ".2f", ".2f", ".2f", ".2f", ".2f", ".0f", ".3f", ".3f", ".3f", ".4f"))
)

# _____________________________________________
# Plot Results
plt.plot(np.array(optimizer.cost_history))
plt.ylabel('Fitness (Multi-Objective BSS)')
plt.xlabel('Iteration')

plt.show()
