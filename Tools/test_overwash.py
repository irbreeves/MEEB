"""
Script for testing BEEM overwash function.
IRBR 13 Feb 2023
"""

import numpy as np
import matplotlib.pyplot as plt
import routines_beem as routine
import copy
import time
from matplotlib import colors
import math

start_time = time.time()  # Record time at start of run


# _____________________________________________
# Define Variables
Rhigh = 3.32
Rlow = 0.9  # 1.93
dur = 70
slabheight_m = 0.1
MHW = 0

# Initial Observed Topo
Init = np.load("Input/Init_NorthernNCB_2017_PreFlorence.npy")
# Final Observed
End = np.load("Input/Init_NorthernNCB_2018_PostFlorence.npy")

# Define Alongshore Coordinates of Domain
xmin = 4000  # 575, 2000, 2150, 2000, 3800
xmax = 4500  # 825, 2125, 2350, 2600, 4450


# _____________________________________________
# Conversions & Initializations

# Transform Initial Observed Topo
topo_init = Init[0, xmin: xmax, :]  # [m]
topo0 = topo_init / slabheight_m  # [slabs] Transform from m into number of slabs
topo = copy.deepcopy(topo0)  # np.round(topo0)  # [slabs] Initialise the topography map

# Transform Final Observed Topo
topo_final = End[0, xmin:xmax, :] / slabheight_m  # [slabs] Transform from m into number of slabs

# Set Veg Domain
spec1 = Init[2, xmin: xmax, :]
spec2 = Init[3, xmin: xmax, :]
veg = spec1 + spec2  # Determine the initial cumulative vegetation effectiveness
veg[veg > 1] = 1  # Cumulative vegetation effectiveness cannot be negative or larger than one
veg[veg < 0] = 0

# Find Dune Crest, Beach Slopes
dune_crest = routine.foredune_crest(topo * slabheight_m)
# dune_crest[245: 299] = 171  # 1715-1845  # 2000-2600 TEMP!!!

# Transform water levels to vectors
Rhigh = Rhigh * np.ones(topo_final.shape[0])
Rlow = Rlow * np.ones(topo_final.shape[0])


# _____________________________________________
# Overwash, Beach, & Dune Change
topo_prestorm = copy.deepcopy(topo)

name = "2150-2350, KQ(S+C)"
print(name)

sim_topo_final, topo_change_overwash, OWflux, netDischarge, inundated = routine.storm_processes(
    topo,
    Rhigh,
    Rlow,
    dur,
    slabheight_m,
    threshold_in=0.25,
    Rin_i=5,
    Rin_r=339,
    Cx=24,
    AvgSlope=2/200,
    nn=0.5,
    MaxUpSlope=0.89,
    fluxLimit=1,
    Qs_min=1,
    Kr=5.15e-05,
    Ki=5e-06,
    mm=2,
    MHW=MHW,
    Cbb_i=0.85,
    Cbb_r=0.7,
    Qs_bb_min=1,
    substep_i=6,
    substep_r=5,
    beach_equilibrium_slope=0.02,
    beach_erosiveness=1.74,
    beach_substeps=17,
)

topo_change_prestorm = sim_topo_final - topo_prestorm

SimDuration = time.time() - start_time
print()
print("Elapsed Time: ", SimDuration, "sec")


# _____________________________________________
# Model Skill: Comparisons to Observations

longshore, crossshore = sim_topo_final.shape

beach_duneface_mask = np.zeros(sim_topo_final.shape).astype(bool)  # [bool] Map of every cell seaward of dune crest
for l in range(longshore):
    beach_duneface_mask[l, :dune_crest[l]] = True

subaerial_mask = sim_topo_final > MHW  # [bool] Map of every cell above water

Florence_Overwash_Mask = np.load("Input/NorthernNCB_FlorenceOverwashMask.npy")  # Load observed overwash mask
OW_Mask = Florence_Overwash_Mask[xmin: xmax, :]

Sim_Obs_OW_Mask = np.logical_or(OW_Mask, inundated) * (~beach_duneface_mask)  # [bool] Map of every cell landward of dune crest that was inundated in simulation or observation or both

# Final Elevations
obs_final_m = topo_final * slabheight_m  # [m] Observed final topo
sim_final_m = sim_topo_final * slabheight_m  # [m] Simulated final topo
obs_mean_m = np.mean(obs_final_m[Sim_Obs_OW_Mask])

# Final Elevation Changes
obs_change_m = (topo_final - topo_prestorm) * slabheight_m  # [m] Observed change
sim_change_m = topo_change_prestorm * slabheight_m  # [m] Simulated change
obs_change_masked = obs_change_m * OW_Mask * ~beach_duneface_mask  # [m]
sim_change_masked = sim_change_m * ~beach_duneface_mask  # [m]
obs_change_masked_beach = obs_change_m * OW_Mask * subaerial_mask  # [m] Includes beach, exclues water
obs_change_mean_masked = np.mean(obs_change_m[Sim_Obs_OW_Mask])  # [m] Average change of observations, masked
obs_change_mean = np.mean(obs_change_m)  # [m] Average change of observations

# _____________________________________________
# Nash-Sutcliffe Model Efficiency
A = np.mean(np.square(np.subtract(obs_change_masked[Sim_Obs_OW_Mask], sim_change_masked[Sim_Obs_OW_Mask])))
B = np.mean(np.square(np.subtract(obs_change_masked[Sim_Obs_OW_Mask], obs_change_mean_masked)))
NSE = 1 - A / B
print("  --> NSE mask", NSE)

A2 = np.mean(np.square(np.subtract(obs_change_m, sim_change_m)))
B2 = np.mean(np.square(np.subtract(obs_change_m, obs_change_mean)))
NSE2 = 1 - A2 / B2
# print("  --> NSE no mask", NSE2)

# _____________________________________________
# Root Mean Square Error
RMSE = np.sqrt(np.mean(np.square(sim_change_masked[Sim_Obs_OW_Mask] - obs_change_masked[Sim_Obs_OW_Mask])))
print("  --> RMSE mask", RMSE)

RMSE2 = np.sqrt(np.mean(np.square(sim_change_m - obs_change_m)))
# print("  --> RMSE no mask", RMSE2)

# _____________________________________________
# Brier Skill Score
BSS = routine.brier_skill_score(sim_change_masked, obs_change_masked, np.zeros(sim_change_m.shape), Sim_Obs_OW_Mask)
print("  --> BSS mask", BSS)

BSS2 = routine.brier_skill_score(sim_change_m, obs_change_m, np.zeros(sim_change_m.shape), np.ones(sim_change_m.shape).astype(bool))
# print("  --> BSS no mask", BSS2)

# _____________________________________________
# Categorical

# ----------
# No Mask
threshold = 0.02
sim_erosion = sim_change_m < -threshold
sim_deposition = sim_change_m > threshold
sim_no_change = np.logical_and(sim_change_m <= threshold, -threshold <= sim_change_m)
obs_erosion = obs_change_m < -threshold
obs_deposition = obs_change_m > threshold
obs_no_change = np.logical_and(obs_change_m <= threshold, -threshold <= obs_change_m)

cat_NoMask = np.zeros(obs_change_m.shape)
cat_NoMask[np.logical_and(sim_erosion, obs_erosion)] = 1          # Hit
cat_NoMask[np.logical_and(sim_deposition, obs_deposition)] = 1    # Hit
cat_NoMask[np.logical_and(sim_erosion, ~obs_erosion)] = 2         # False Alarm
cat_NoMask[np.logical_and(sim_deposition, ~obs_deposition)] = 2   # False Alarm
cat_NoMask[np.logical_and(sim_no_change, obs_no_change)] = 3      # Correct Reject
cat_NoMask[np.logical_and(sim_no_change, ~obs_no_change)] = 4     # Miss

hits = np.count_nonzero(cat_NoMask == 1)
false_alarms = np.count_nonzero(cat_NoMask == 2)
correct_rejects = np.count_nonzero(cat_NoMask == 3)
misses = np.count_nonzero(cat_NoMask == 4)
J = hits + false_alarms + correct_rejects + misses

# ----------
# Mask
threshold = 0.02
sim_erosion = sim_change_m < -threshold
sim_deposition = sim_change_m > threshold
sim_no_change = np.logical_and(sim_change_m <= threshold, -threshold <= sim_change_m)
obs_erosion = obs_change_masked_beach < -threshold
obs_deposition = obs_change_masked_beach > threshold
obs_no_change = np.logical_and(obs_change_masked_beach <= threshold, -threshold <= obs_change_masked_beach)

cat_Mask = np.zeros(obs_change_m.shape)
cat_Mask[np.logical_and(sim_erosion, obs_erosion)] = 1          # Hit
cat_Mask[np.logical_and(sim_deposition, obs_deposition)] = 1    # Hit
cat_Mask[np.logical_and(sim_erosion, ~obs_erosion)] = 2         # False Alarm
cat_Mask[np.logical_and(sim_deposition, ~obs_deposition)] = 2   # False Alarm
cat_Mask[np.logical_and(sim_no_change, obs_no_change)] = 3      # Correct Reject
cat_Mask[np.logical_and(sim_no_change, ~obs_no_change)] = 4     # Miss

hits_m = np.count_nonzero(cat_Mask[Sim_Obs_OW_Mask] == 1)
false_alarms_m = np.count_nonzero(cat_Mask[Sim_Obs_OW_Mask] == 2)
correct_rejects_m = np.count_nonzero(cat_Mask[Sim_Obs_OW_Mask] == 3)
misses_m = np.count_nonzero(cat_Mask[Sim_Obs_OW_Mask] == 4)
J_m = hits_m + false_alarms_m + correct_rejects_m + misses_m

# ----------
# Percentage Correct
PC = (hits + correct_rejects) / J
PC_m = (hits_m + correct_rejects_m) / J_m
print("  --> PC mask", PC_m)

# Heidke Skill Score
G_m = ((hits_m + false_alarms_m) * (hits_m + misses_m) / J_m ** 2) + ((misses_m + correct_rejects_m) * (false_alarms_m + correct_rejects_m) / J_m ** 2)  # Fraction of predictions of the correct categories (H and C) that would be expected from a random choice
HSS2 = (PC_m - G_m) / (1 - G_m)   # The percentage correct, corrected for the number expected to be correct by chance
print("  --> HSS mask", HSS2)

G = ((hits + false_alarms) * (hits + misses) / J ** 2) + ((misses + correct_rejects) * (false_alarms + correct_rejects) / J ** 2)  # Fraction of predictions of the correct categories (H and C) that would be expected from a random choice
HSS = (PC - G) / (1 - G)   # The percentage correct, corrected for the number expected to be correct by chance
# print("  --> HSS no mask", HSS)

# _____________________________________________
# Plot

pxmin = 0
pxmax = 600

# Categorical
catfig = plt.figure(figsize=(14, 7.5))
cmap2 = colors.ListedColormap(['green', 'yellow', 'gray', 'red'])
bounds = [0.5, 1.5, 2.5, 3.5, 4.5]
norm = colors.BoundaryNorm(bounds, cmap2.N)
ax1 = catfig.add_subplot(211)
cax1 = ax1.matshow(cat_Mask[:, pxmin: pxmax], cmap=cmap2, norm=norm)
cbar1 = plt.colorbar(cax1, boundaries=bounds, ticks=[1, 2, 3, 4])
cbar1.set_ticklabels(['Hit', 'False Alarm', 'Correct Reject', 'Miss'])
ax2 = catfig.add_subplot(212)
cax2 = ax2.matshow(cat_NoMask[:, pxmin: pxmax], cmap=cmap2, norm=norm)
cbar2 = plt.colorbar(cax2, boundaries=bounds, ticks=[1, 2, 3, 4])
cbar2.set_ticklabels(['Hit', 'False Alarm', 'Correct Reject', 'Miss'])

# Change Comparisons
cmap1 = routine.truncate_colormap(copy.copy(plt.cm.get_cmap("terrain")), 0.5, 0.9)  # Truncate colormap
cmap1.set_bad(color='dodgerblue', alpha=0.5)  # Set cell color below MHW to blue

# Pre Storm (Observed) Topo
Fig = plt.figure(figsize=(14, 7.5))
ax1 = Fig.add_subplot(221)
topo1 = topo_prestorm[:, pxmin: pxmax] * 0.1  # [m]
topo1 = np.ma.masked_where(topo1 < MHW, topo1)  # Mask cells below MHW
cax1 = ax1.matshow(topo1, cmap=cmap1, vmin=0, vmax=5.0)
ax1.plot(dune_crest, np.arange(len(dune_crest)), c='black', alpha=0.6)
plt.title(name)

# Post-Storm (Simulated) Topo
ax2 = Fig.add_subplot(222)
topo2 = sim_topo_final[:, pxmin: pxmax] * 0.1  # [m]
topo2 = np.ma.masked_where(topo2 < MHW, topo2)  # Mask cells below MHW
cax2 = ax2.matshow(topo2, cmap=cmap1, vmin=0, vmax=5.0)
# cbar = Fig.colorbar(cax2)
# cbar.set_label('Elevation [m MHW]', rotation=270, labelpad=20)

# Simulated Topo Change
maxx = max(abs(np.min(obs_change_m)), abs(np.max(obs_change_m)))
maxxx = max(abs(np.min(sim_change_m)), abs(np.max(sim_change_m)))
maxxxx = 1  # max(maxx, maxxx)
ax3 = Fig.add_subplot(223)
cax3 = ax3.matshow(sim_change_m[:, pxmin: pxmax], cmap='bwr', vmin=-maxxxx, vmax=maxxxx)
ax3.plot(dune_crest, np.arange(len(dune_crest)), c='black', alpha=0.6)
# cbar = Fig.colorbar(cax3)
# cbar.set_label('Change [m]', rotation=270, labelpad=20)

# Observed Topo Change
ax4 = Fig.add_subplot(224)
cax4 = ax4.matshow(obs_change_masked_beach[:, pxmin: pxmax], cmap='bwr', vmin=-maxxxx, vmax=maxxxx)
ax4.plot(dune_crest, np.arange(len(dune_crest)), c='black', alpha=0.6)
# cbar = Fig.colorbar(cax4)
# cbar.set_label('Elevation Change [m]', rotation=270, labelpad=20)
plt.tight_layout()

# Cumulative Discharge
plt.figure(figsize=(14, 7.5))
plt.plot(np.sum(netDischarge, axis=0))
plt.ylabel("Cumulative Discharge")

# Cumulative Discharge
plt.matshow(inundated)
plt.title("Inundated")

print()
print("Complete.")
plt.show()
