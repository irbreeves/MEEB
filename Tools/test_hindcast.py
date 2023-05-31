"""
Script for testing MEEB hindcast simulations.

Calculates fitess score for all morphologic and ecologic change between two observations.

IRBR 31 May 2023
"""

import numpy as np
import matplotlib.pyplot as plt
import routines_meeb as routine
import copy
import time
import os
import imageio
from tabulate import tabulate

from meeb import MEEB


# __________________________________________________________________________________________________________________________________
# MODEL SKILL

def model_skill(obs_change, sim_change, obs_change_mean, mask):
    """Perform suite of model skill assesments and return scores.

    Mask is boolean array with same size as change maps, with cells to be excluded from skill analysis set to FALSE.
    """

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


# __________________________________________________________________________________________________________________________________
# VARIABLES AND INITIALIZATIONS

# # 5.1, 20040716
# start = "Init_NCB-NewDrum-Ocracoke_2004_PostIsabel.npy"
# stop = "Init_NCB-NewDrum-Ocracoke_2009_PreIrene.npy"

# # 0.92, 20161012
# start = "Init_NCB-NewDrum-Ocracoke_2016_PostMatthew.npy"
# stop = "Init_NCB-NewDrum-Ocracoke_2017_PreFlorence.npy"

# # 1.96, 20161012
# start = "Init_NCB-NewDrum-Ocracoke_2016_PostMatthew.npy"
# stop = "Init_NCB-NewDrum-Ocracoke_2018_PostFlorence.npy"

# 9.7, 20040716
start = "Init_NCB-NewDrum-Ocracoke_2004_PostIsabel.npy"
stop = "Init_NCB-NewDrum-Ocracoke_2009_PreIrene.npy"

# Initial Observed Topo
Init = np.load("Input/" + start)
# Final Observed
End = np.load("Input/" + stop)

# Define Alongshore Coordinates of Domain
xmin = 6100  # 575, 2000, 2150, 2000, 3800  # 2650 #6500
xmax = 6200  # 825, 2125, 2350, 2600, 4450  # 2850 #6600

slabheight_m = 0.1  # [m]
MHW = 0.4  # [m NAVD88]
name = '6100-6200, 2004-2014, 48-74-41-9-4-17-27-3-1'
ResReduc = False  # Option to reduce raster resolution for skill assessment
reduc = 5  # Raster resolution reduction factor

# Transform Initial Observed Topo
topo_i = Init[0, xmin: xmax, :]  # [m]
topo_start = copy.deepcopy(topo_i)  # [m] Initialise the topography

# Transform Final Observed Topo
topo_e = End[0, xmin: xmax, :]  # [m]
topo_end_obs = copy.deepcopy(topo_e)  # [m] Initialise the topography

# Set Veg Domain
spec1_i = Init[1, xmin: xmax, :]
spec2_i = Init[2, xmin: xmax, :]
veg_start = spec1_i + spec2_i  # Determine the initial cumulative vegetation effectiveness
veg_start[veg_start > 1] = 1  # Cumulative vegetation effectiveness cannot be negative or larger than one
veg_start[veg_start < 0] = 0

spec1_e = End[1, xmin: xmax, :]
spec2_e = End[2, xmin: xmax, :]
veg_end = spec1_e + spec2_e  # Determine the initial cumulative vegetation effectiveness
veg_end[veg_end > 1] = 1  # Cumulative vegetation effectiveness cannot be negative or larger than one
veg_end[veg_end < 0] = 0


# __________________________________________________________________________________________________________________________________
# RUN MODEL

start_time = time.time()  # Record time at start of simulation

# Create an instance of the MEEB class
meeb = MEEB(
    name=name,
    simulation_time_yr=9.7,
    RSLR=0.00,
    MHW=MHW,
    seeded_random_numbers=True,
    p_dep_sand=0.48,  # Q = hs * L * n * pe/pd
    p_dep_sand_VegMax=0.74,
    p_ero_sand=0.41,
    entrainment_veg_limit=0.09,
    shadowangle=4,
    repose_bare=17,
    repose_veg=27,
    direction2=3,
    direction3=1,
    wave_asymetry=0.5,
    init_filename=start,  # !!! Currently, the min and max have to be modified in meeb.py too
    hindcast=True,
    simulation_start_date='20040716',
    storm_timeseries_filename='StormTimeSeries_1980-2020_NCB-CE_Beta0pt039_BermEl2pt03.npy',
)

print(meeb.name)

# Loop through time
for time_step in range(int(meeb.iterations)):
    # Print time step to screen
    print("\r", "Time Step: ", time_step / meeb.iterations_per_cycle, "years", end="")

    # Run time step
    meeb.update(time_step)

# Print elapsed time of simulation
print()
SimDuration = time.time() - start_time
print()
print("Elapsed Time: ", SimDuration, "sec")


# __________________________________________________________________________________________________________________________________
# ASSESS MODEL SKILL

topo_end_sim = meeb.topo * meeb.slabheight
topo_change_sim = topo_end_sim - topo_start  # [m]
topo_change_obs = topo_end_obs - topo_start  # [m]

# Subaerial mask
subaerial_mask_sim = topo_end_sim > MHW  # [bool] Mask for every cell above water
subaerial_mask_obs = topo_end_obs > MHW  # [bool] Mask for every cell above water

# Beach mask
dune_crest = routine.foredune_crest(topo_start, MHW)
beach_duneface_mask = np.zeros(topo_end_sim.shape)
for l in range(topo_start.shape[0]):
    beach_duneface_mask[l, :dune_crest[l]] = True
beach_duneface_mask = np.logical_and(beach_duneface_mask, subaerial_mask_sim)  # [bool] Map of every cell seaward of dune crest

# Temp limit interior in analysis to dunes
subaerial_mask = copy.copy(subaerial_mask_sim)
subaerial_mask[:, :820] = False
subaerial_mask[:, 950:] = False

if ResReduc:
    # Reduce Resolutions
    topo_change_obs = routine.reduce_raster_resolution(topo_change_obs, reduc)
    topo_change_sim = routine.reduce_raster_resolution(topo_change_sim, reduc)
    subaerial_mask = (routine.reduce_raster_resolution(subaerial_mask, reduc)) == 1
    subaerial_mask_obs = (routine.reduce_raster_resolution(subaerial_mask_obs, reduc)) == 1
    subaerial_mask_sim = (routine.reduce_raster_resolution(subaerial_mask_sim, reduc)) == 1

# Model Skill
nse, rmse, bss, pc, hss = model_skill(topo_change_obs, topo_change_sim, np.mean(topo_change_obs), subaerial_mask)

# Prin scores
print()
print(tabulate({
    "Scores": [""],
    "NSE": [nse],
    "RMSE": [rmse],
    "BSS": [bss],
    "PC": [pc],
    "HSS": [hss],
    }, headers="keys", floatfmt=(None, ".3f", ".3f", ".3f", ".3f", ".3f"))
)


# __________________________________________________________________________________________________________________________________
# PLOT RESULTS

xmin = 700
xmax = xmin + 600  #topo_start.shape[1]

# Final Elevation & Vegetation
Fig = plt.figure(figsize=(14, 7.5))
Fig.suptitle(meeb.name, fontsize=13)
MHW = meeb.RSLR * meeb.simulation_time_yr
topo = meeb.topo[:, xmin: xmax] * meeb.slabheight
topo = np.ma.masked_where(topo <= MHW, topo)  # Mask cells below MHW
cmap1 = routine.truncate_colormap(copy.copy(plt.cm.get_cmap("terrain")), 0.5, 0.9)  # Truncate colormap
cmap1.set_bad(color='dodgerblue', alpha=0.5)  # Set cell color below MHW to blue
ax1 = Fig.add_subplot(211)
cax1 = ax1.matshow(topo, cmap=cmap1, vmin=0, vmax=5.0)
cbar = Fig.colorbar(cax1)
cbar.set_label('Elevation [m]', rotation=270, labelpad=20)
ax2 = Fig.add_subplot(212)
veg = meeb.veg[:, xmin: xmax]
veg = np.ma.masked_where(topo <= MHW, veg)  # Mask cells below MHW
cmap2 = copy.copy(plt.cm.get_cmap("YlGn"))
cmap2.set_bad(color='dodgerblue', alpha=0.5)  # Set cell color below MHW to blue
cax2 = ax2.matshow(veg, cmap=cmap2, vmin=0, vmax=1)
cbar = Fig.colorbar(cax2)
cbar.set_label('Vegetation [%]', rotation=270, labelpad=20)
plt.tight_layout()

# Topo Change, Observed vs Simulated
if ResReduc:
    # Reduced Resolutions
    xmin_reduc = int(xmin / reduc)
    xmax_reduc = int(xmax / reduc)
    tco = topo_change_obs[:, xmin_reduc: xmax_reduc] * subaerial_mask_obs[:, xmin_reduc: xmax_reduc]
    tcs = topo_change_sim[:, xmin_reduc: xmax_reduc] * subaerial_mask_sim[:, xmin_reduc: xmax_reduc]
    to = topo_end_obs[:, xmin: xmax]
    ts = topo_end_sim[:, xmin: xmax]
else:
    tco = topo_change_obs[:, xmin: xmax] * subaerial_mask_obs[:, xmin: xmax]
    tcs = topo_change_sim[:, xmin: xmax] * subaerial_mask_sim[:, xmin: xmax]
    to = topo_end_obs[:, xmin: xmax] * subaerial_mask_obs[:, xmin: xmax]
    ts = topo_end_sim[:, xmin: xmax] * subaerial_mask_sim[:, xmin: xmax]

maxx = max(abs(np.min(tco)), abs(np.max(tco)))
maxxx = max(abs(np.min(tcs)), abs(np.max(tcs)))
maxxxx = max(maxx, maxxx)

Fig = plt.figure(figsize=(14, 7.5))
Fig.suptitle(meeb.name, fontsize=13)
ax1 = Fig.add_subplot(221)
cax1 = ax1.matshow(to, cmap='terrain', vmin=-1, vmax=5)
plt.title("Observed")

ax2 = Fig.add_subplot(222)
cax2 = ax2.matshow(ts, cmap='terrain', vmin=-1, vmax=5)
plt.title("Simulated")

ax3 = Fig.add_subplot(223)
cax3 = ax3.matshow(tco, cmap='bwr', vmin=-maxxxx, vmax=maxxxx)

ax4 = Fig.add_subplot(224)
cax4 = ax4.matshow(tcs, cmap='bwr', vmin=-maxxxx, vmax=maxxxx)
plt.tight_layout()

# Animation: Elevation and Vegetation Over Time
for t in range(0, int(meeb.simulation_time_yr / meeb.writeyear) + 1):
    Fig = plt.figure(figsize=(14, 8))

    MHW = meeb.RSLR * t
    topo = meeb.topo_TS[:, xmin: xmax, t] * meeb.slabheight  # [m]
    topo = np.ma.masked_where(topo <= MHW, topo)  # Mask cells below MHW
    cmap1 = routine.truncate_colormap(copy.copy(plt.cm.get_cmap("terrain")), 0.5, 0.9)  # Truncate colormap
    cmap1.set_bad(color='dodgerblue', alpha=0.5)  # Set cell color below MHW to blue
    ax1 = Fig.add_subplot(211)
    cax1 = ax1.matshow(topo, cmap=cmap1, vmin=0, vmax=5.0)
    cbar = Fig.colorbar(cax1)
    cbar.set_label('Elevation [m]', rotation=270, labelpad=20)
    timestr = "Year " + str(t * meeb.writeyear)
    plt.text(2, meeb.topo.shape[0] - 2, timestr, c='white')

    veg = meeb.veg_TS[:, xmin: xmax, t]
    veg = np.ma.masked_where(topo <= MHW, veg)  # Mask cells below MHW
    cmap2 = copy.copy(plt.cm.get_cmap("YlGn"))
    cmap2.set_bad(color='dodgerblue', alpha=0.5)  # Set cell color below MHW to blue
    ax2 = Fig.add_subplot(212)
    cax2 = ax2.matshow(veg, cmap=cmap2, vmin=0, vmax=1)
    cbar = Fig.colorbar(cax2)
    cbar.set_label('Vegetation [%]', rotation=270, labelpad=20)
    timestr = "Year " + str(t * meeb.writeyear)
    plt.text(2, meeb.veg.shape[0] - 2, timestr, c='darkblue')
    plt.tight_layout()
    if not os.path.exists("Output/SimFrames/"):
        os.makedirs("Output/SimFrames/")
    name = "Output/SimFrames/meeb_elev_" + str(t)
    plt.savefig(name)  # dpi=200
    plt.close()

frames = []
for filenum in range(0, int(meeb.simulation_time_yr / meeb.writeyear) + 1):
    filename = "Output/SimFrames/meeb_elev_" + str(filenum) + ".png"
    frames.append(imageio.imread(filename))
imageio.mimwrite("Output/SimFrames/meeb_elev.gif", frames, fps=3)
print()
print("[ * GIF successfully generated * ]")

plt.show()
