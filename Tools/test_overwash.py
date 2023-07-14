"""
Script for testing MEEB overwash function.
IRBR 14 July 2023
"""

import numpy as np
import matplotlib.pyplot as plt
import routines_meeb as routine
import copy
import time
from matplotlib import colors
from tabulate import tabulate


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


start_time = time.time()  # Record time at start of run

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

# Define Alongshore Coordinates of Domain
xmin = 575  # 575, 2000, 2150, 2000, 3800
xmax = 825  # 825, 2125, 2350, 2600, 4450


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

# Find Dune Crest, Shoreline Positions
dune_crest = routine.foredune_crest(topo, MHW)
x_s = routine.ocean_shoreline(topo, MHW)

# Transform water levels to vectors
Rhigh = Rhigh * np.ones(topo_final.shape[0])
Rlow = Rlow * np.ones(topo_final.shape[0])


# _____________________________________________
# Overwash, Beach, & Dune Change
topo_prestorm = copy.deepcopy(topo)

name = "575-825, KQ(S+C)"
print(name)

sim_topo_final, topo_change_overwash, OWflux, netDischarge, inundated, Qbe = routine.storm_processes(
    topo,
    Rhigh,
    Rlow,
    dur,
    slabheight_m,
    threshold_in=0.25,
    Rin_i=5,
    Rin_r=344,
    Cx=38,
    AvgSlope=2/200,
    nn=0.5,
    MaxUpSlope=0.54,
    fluxLimit=1,
    Qs_min=1,
    Kr=0.0000454,
    Ki=5e-06,
    mm=2,
    MHW=MHW,
    Cbb_i=0.85,
    Cbb_r=0.7,
    Qs_bb_min=1,
    substep_i=6,
    substep_r=6,
    beach_equilibrium_slope=0.018,
    beach_erosiveness=2.49,
    beach_substeps=71,
    x_s=x_s,
)

sim_topo_change = sim_topo_final - topo_prestorm

SimDuration = time.time() - start_time
print()
print("Elapsed Time: ", SimDuration, "sec")


# _____________________________________________
# Model Skill: Comparisons to Observations

longshore, crossshore = sim_topo_final.shape

Florence_Overwash_Mask = np.load("Input/NorthernNCB_FlorenceOverwashMask.npy")  # Load observed overwash mask
OW_Mask = Florence_Overwash_Mask[xmin: xmax, :]

# Final Elevations
obs_final_m = topo_final * slabheight_m  # [m] Observed final topo
sim_final_m = sim_topo_final * slabheight_m  # [m] Simulated final topo
obs_change_m = (topo_final - topo_prestorm) * slabheight_m  # [m] Observed change
sim_change_m = sim_topo_change * slabheight_m  # [m] Simulated change

subaerial_mask = sim_final_m > (MHW * slabheight_m)  # [bool] Map of every cell above water

beach_duneface_mask = np.zeros(sim_final_m.shape)
for l in range(sim_final_m.shape[0]):
    beach_duneface_mask[l, :dune_crest[l]] = True
beach_duneface_mask = np.logical_and(beach_duneface_mask, subaerial_mask)  # [bool] Map of every cell seaward of dune crest

mask_all = np.logical_or(OW_Mask, inundated, beach_duneface_mask) * subaerial_mask  # [bool] Map of every cell landward of dune crest that was inundated in simulation or observation or both

nse, rmse, nmae, mass, bss = model_skill(obs_change_m, sim_change_m, np.zeros(obs_change_m.shape), mask_all)

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
obs_erosion = np.logical_and(obs_change_m < -threshold, mask_all)
obs_deposition = np.logical_and(obs_change_m > threshold, mask_all)
obs_no_change = np.logical_and(obs_change_m <= threshold, np.logical_and(-threshold <= obs_change_m, mask_all))

cat_Mask = np.zeros(obs_change_m.shape)
cat_Mask[np.logical_and(sim_erosion, obs_erosion)] = 1          # Hit
cat_Mask[np.logical_and(sim_deposition, obs_deposition)] = 1    # Hit
cat_Mask[np.logical_and(sim_erosion, ~obs_erosion)] = 2         # False Alarm
cat_Mask[np.logical_and(sim_deposition, ~obs_deposition)] = 2   # False Alarm
cat_Mask[np.logical_and(sim_no_change, obs_no_change)] = 3      # Correct Reject
cat_Mask[np.logical_and(sim_no_change, ~obs_no_change)] = 4     # Miss

hits_m = np.count_nonzero(cat_Mask[mask_all] == 1)
false_alarms_m = np.count_nonzero(cat_Mask[mask_all] == 2)
correct_rejects_m = np.count_nonzero(cat_Mask[mask_all] == 3)
misses_m = np.count_nonzero(cat_Mask[mask_all] == 4)
J_m = hits_m + false_alarms_m + correct_rejects_m + misses_m

# ----------
# Percentage Correct
PC = (hits + correct_rejects) / J
PC_m = (hits_m + correct_rejects_m) / J_m

# Heidke Skill Score
G_m = ((hits_m + false_alarms_m) * (hits_m + misses_m) / J_m ** 2) + ((misses_m + correct_rejects_m) * (false_alarms_m + correct_rejects_m) / J_m ** 2)  # Fraction of predictions of the correct categories (H and C) that would be expected from a random choice
HSS_m = (PC_m - G_m) / (1 - G_m)   # The percentage correct, corrected for the number expected to be correct by chance

G = ((hits + false_alarms) * (hits + misses) / J ** 2) + ((misses + correct_rejects) * (false_alarms + correct_rejects) / J ** 2)  # Fraction of predictions of the correct categories (H and C) that would be expected from a random choice
HSS = (PC - G) / (1 - G)   # The percentage correct, corrected for the number expected to be correct by chance

# _____________________________________________
# Print

# Print scores
print()
print(tabulate({
    "Scores": ["Mask", "No Mask"],
    "NSE": [nse],
    "RMSE": [rmse],
    "NMAE": [nmae],
    "MASS": [mass],
    "BSS": [bss],
    "PC": [PC_m, PC],
    "HSS": [HSS_m, HSS],
}, headers="keys", floatfmt=(None, ".3f", ".3f", ".3f", ".3f", ".3f", ".3f", ".3f"))
)

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
cmap1 = routine.truncate_colormap(copy.copy(plt.colormaps.get_cmap("terrain")), 0.5, 0.9)  # Truncate colormap
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
obs_change_masked = obs_change_m.copy()
obs_change_masked[~mask_all] = 0
cax4 = ax4.matshow(obs_change_masked[:, pxmin: pxmax], cmap='bwr', vmin=-maxxxx, vmax=maxxxx)
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
