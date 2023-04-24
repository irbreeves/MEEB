"""
Script for testing MEEB dune/beach erosion function.
IRBR 1 Mar 2023
"""

import numpy as np
import matplotlib.pyplot as plt
import routines_meeb as routine
import copy
import time
from matplotlib import colors

start_time = time.time()  # Record time at start of run

# _____________________________________________
# Define Variables
Rhigh = 3.32
Rlow = 0.9
dur = 70
slabheight_m = 0.1
MHW = 0
repose_bare = 20  # [deg] - orig:30
repose_veg = 30  # [deg] - orig:35
repose_threshold = 0.3

# Initial Observed Topo
Init = np.load("Input/Init_NorthernNCB_2017_PreFlorence.npy")
# Final Observed
End = np.load("Input/Init_NorthernNCB_2018_PostFlorence.npy")

# Define Alongshore Coordinates of Domain
xmin = 2600  # 575, 2000, 2150, 2000, 3800  # 2650
xmax = 3300  # 825, 2125, 2350, 2600, 4450  # 2850

# _____________________________________________
# Conversions & Initializations

# Transform Initial Observed Topo
topo_init = Init[0, xmin: xmax, :]  # [m]
topo0 = topo_init  # [m]
topo = copy.deepcopy(topo0)  # [m] Initialise the topography map

# Transform Final Observed Topo
topo_final = End[0, xmin:xmax, :]  # [m]

# Set Veg Domain
spec1 = Init[2, xmin: xmax, :]
spec2 = Init[3, xmin: xmax, :]
veg = spec1 + spec2  # Determine the initial cumulative vegetation effectiveness
veg[veg > 1] = 1  # Cumulative vegetation effectiveness cannot be negative or larger than one
veg[veg < 0] = 0

# Find Dune Crest, Beach Slopes
dune_crest = routine.foredune_crest(topo, veg)
# dune_crest[245: 299] = 171  # 1715-1845  # 2000-2600 TEMP!!!

# Transform water levels to vectors
Rhigh = Rhigh * np.ones(topo_final.shape[0])
Rlow = Rlow * np.ones(topo_final.shape[0])

dune_beach_volume_change = np.zeros([topo.shape[0]])
RNG = np.random.default_rng(seed=13)


# _____________________________________________
# Overwash, Beach, & Dune Change
topo_prestorm = copy.deepcopy(topo)

name = "Calibrated: Beq = 0.02, SS = 40, Et = 0.5"
print(name)

for t in range(dur):
    topo, dV, wetMap = routine.calc_dune_erosion_TS(
        topo=topo,
        dx=1,
        crestline=dune_crest,
        MHW=0,
        Rhigh=Rhigh,
        Beq=0.02,
        Et=0.5,
        substeps=40,
    )

    dune_beach_volume_change += dV

# topo = routine.enforceslopes2(topo, veg, slabheight_m, repose_bare, repose_veg, repose_threshold, RNG)[0]

sim_topo_final = topo
topo_change_prestorm = sim_topo_final - topo_prestorm

SimDuration = time.time() - start_time
print()
print("Elapsed Time: ", SimDuration, "sec")

# _____________________________________________
# Model Skill: Comparisons to Observations

longshore, crossshore = sim_topo_final.shape

subaerial_mask = sim_topo_final > MHW  # [bool] Map of every cell above water

beach_duneface_mask = np.zeros(sim_topo_final.shape).astype(bool)  # [bool] Map of every cell seaward of dune crest
for l in range(longshore):
    beach_duneface_mask[l, :dune_crest[l]] = True
beach_duneface_mask = np.logical_and(beach_duneface_mask, subaerial_mask)

# Final Elevations
obs_final_m = topo_final  # [m] Observed final topo
sim_final_m = sim_topo_final  # [m] Simulated final topo
obs_mean_m = np.mean(obs_final_m[beach_duneface_mask])

# # Final Elevation Changes
obs_change_m = (topo_final - topo_prestorm)  # [m] Observed change
sim_change_m = topo_change_prestorm  # [m] Simulated change
obs_change_masked = obs_change_m * beach_duneface_mask  # [m]
sim_change_masked = sim_change_m * beach_duneface_mask  # [m]
obs_change_mean_masked = np.mean(obs_change_m[beach_duneface_mask])  # [m] Average change of observations, masked
obs_change_mean = np.mean(obs_change_m)  # [m] Average change of observations

# _____________________________________________
# Nash-Sutcliffe Model Efficiency
"""The closer the score is to 1, the better the agreement. If the score is below 0, the mean observed value is a better predictor than the model."""

A = np.mean(np.square(np.subtract(obs_change_masked[beach_duneface_mask], sim_change_masked[beach_duneface_mask])))
B = np.mean(np.square(np.subtract(obs_change_masked[beach_duneface_mask], obs_change_mean_masked)))
NSE = 1 - A / B
print("  --> NSE mask", NSE)

A2 = np.mean(np.square(np.subtract(obs_change_m, sim_change_m)))
B2 = np.mean(np.square(np.subtract(obs_change_m, obs_change_mean)))
NSE2 = 1 - A2 / B2
# print("  --> NSE no mask", NSE2)

# _____________________________________________
# Root Mean Square Error
RMSE = np.sqrt(np.mean(np.square(sim_change_masked[beach_duneface_mask] - obs_change_masked[beach_duneface_mask])))
print("  --> RMSE mask", RMSE)

RMSE2 = np.sqrt(np.mean(np.square(sim_change_m - obs_change_m)))
# print("  --> RMSE no mask", RMSE2)

# _____________________________________________
# Brier Skill Score
"""A skill score value of zero means that the score for the predictions is merely as good as that of a set of baseline or reference or default predictions, 
while a skill score value of one (100%) represents the best possible score. A skill score value less than zero means that the performance is even worse than 
that of the baseline or reference predictions."""

BSS = routine.brier_skill_score(sim_change_masked, obs_change_masked, np.zeros(sim_change_m.shape), beach_duneface_mask)
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
cat_NoMask[np.logical_and(sim_erosion, obs_erosion)] = 1  # Hit
cat_NoMask[np.logical_and(sim_deposition, obs_deposition)] = 1  # Hit
cat_NoMask[np.logical_and(sim_erosion, ~obs_erosion)] = 2  # False Alarm
cat_NoMask[np.logical_and(sim_deposition, ~obs_deposition)] = 2  # False Alarm
cat_NoMask[np.logical_and(sim_no_change, obs_no_change)] = 3  # Correct Reject
cat_NoMask[np.logical_and(sim_no_change, ~obs_no_change)] = 4  # Miss

hits = np.count_nonzero(cat_NoMask == 1)
false_alarms = np.count_nonzero(cat_NoMask == 2)
correct_rejects = np.count_nonzero(cat_NoMask == 3)
misses = np.count_nonzero(cat_NoMask == 4)
J = hits + false_alarms + correct_rejects + misses

# ----------
# Mask
threshold = 0.02
sim_erosion = sim_change_masked < -threshold
sim_deposition = sim_change_masked > threshold
sim_no_change = np.logical_and(sim_change_masked <= threshold, -threshold <= sim_change_masked)
obs_erosion = obs_change_masked < -threshold
obs_deposition = obs_change_masked > threshold
obs_no_change = np.logical_and(obs_change_masked <= threshold, -threshold <= obs_change_masked)

cat_Mask = np.zeros(obs_change_masked.shape)
cat_Mask[np.logical_and(sim_erosion, obs_erosion)] = 1  # Hit
cat_Mask[np.logical_and(sim_deposition, obs_deposition)] = 1  # Hit
cat_Mask[np.logical_and(sim_erosion, ~obs_erosion)] = 2  # False Alarm
cat_Mask[np.logical_and(sim_deposition, ~obs_deposition)] = 2  # False Alarm
cat_Mask[np.logical_and(sim_no_change, obs_no_change)] = 3  # Correct Reject
cat_Mask[np.logical_and(sim_no_change, ~obs_no_change)] = 4  # Miss

hits_m = np.count_nonzero(cat_Mask[beach_duneface_mask] == 1)
false_alarms_m = np.count_nonzero(cat_Mask[beach_duneface_mask] == 2)
correct_rejects_m = np.count_nonzero(cat_Mask[beach_duneface_mask] == 3)
misses_m = np.count_nonzero(cat_Mask[beach_duneface_mask] == 4)
J_m = hits_m + false_alarms_m + correct_rejects_m + misses_m

# ----------
# Percentage Correct
"""Ratio of correct predictions as a fraction of the total number of forecasts. Scores closer to 1 (100%) are better."""

PC = (hits + correct_rejects) / J
PC_m = (hits_m + correct_rejects_m) / J_m
print("  --> PC mask", PC_m)
# print("  --> PC no mask", PC)


# Heidke Skill Score
"""The percentage correct, corrected for the number expected to be correct by chance. Scores closer to 1 (100%) are better."""

G_m = ((hits_m + false_alarms_m) * (hits_m + misses_m) / J_m ** 2) + ((misses_m + correct_rejects_m) * (false_alarms_m + correct_rejects_m) / J_m ** 2)  # Fraction of predictions of the correct categories (H and C) that would be expected from a random choice
HSS2 = (PC_m - G_m) / (1 - G_m)  # The percentage correct, corrected for the number expected to be correct by chance
print("  --> HSS mask", HSS2)

G = ((hits + false_alarms) * (hits + misses) / J ** 2) + ((misses + correct_rejects) * (false_alarms + correct_rejects) / J ** 2)  # Fraction of predictions of the correct categories (H and C) that would be expected from a random choice
HSS = (PC - G) / (1 - G)  # The percentage correct, corrected for the number expected to be correct by chance
# print("  --> HSS no mask", HSS)


# _____________________________________________
# Plot

pxmin = 0
pxmax = 600

# Categorical
catfig = plt.figure(figsize=(11, 7.5))
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
Fig = plt.figure(figsize=(11, 7.5))
ax1 = Fig.add_subplot(221)
topo1 = topo_prestorm[:, pxmin: pxmax]  # [m]
topo1 = np.ma.masked_where(topo1 < MHW, topo1)  # Mask cells below MHW
cax1 = ax1.matshow(topo1, cmap=cmap1, vmin=0, vmax=5.0)
ax1.plot(dune_crest, np.arange(len(dune_crest)), c='black', alpha=0.6)
plt.title(name)

# Post-Storm (Simulated) Topo
ax2 = Fig.add_subplot(222)
topo2 = sim_topo_final[:, pxmin: pxmax]  # [m]
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
cax4 = ax4.matshow(obs_change_m[:, pxmin: pxmax], cmap='bwr', vmin=-maxxxx, vmax=maxxxx)
ax4.plot(dune_crest, np.arange(len(dune_crest)), c='black', alpha=0.6)
# cbar = Fig.colorbar(cax4)
# cbar.set_label('Elevation Change [m]', rotation=270, labelpad=20)
plt.tight_layout()

xx = 10
proffig = plt.figure(figsize=(11, 7.5))
plt.plot(topo1[xx, 70:250], c='black')
plt.plot(obs_final_m[xx, 70:250], c='green')
plt.plot(topo2[xx, 70:250], c='red')
plt.legend(["Pre", "Post Obs", "Post Sim"])

plt.text(3,
         4,
         'NSE = ' + str(np.round(NSE, 3)) + '\n' + \
         'RMSE =' + str(np.round(RMSE, 3)) + '\n' + \
         'BSS = ' + str(np.round(BSS, 3)) + '\n' + \
         'PC =  ' + str(np.round(PC, 3)) + '\n' + \
         'HSS = ' + str(np.round(HSS, 3))
         )

plt.title(name)

print()
print("Complete.")
plt.show()
