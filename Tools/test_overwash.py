"""
Script for testing MEEB overwash function.
IRBR 1 April 2025
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
Rhigh = 3.32  # [m NAVD88]
Rlow = 1.90  # [m NAVD88]
dur = 83  # [hr]
MHW = 0.39  # [m NAVD88]
cellsize = 2  # [m]

# Initial Observed Topo
Init = np.load("Input/Init_NCB-NewDrum-Ocracoke_2017_PreFlorence_2m.npy")
# Final Observed
End = np.load("Input/Init_NCB-NewDrum-Ocracoke_2018_USACE_PostFlorence_2m.npy")
# Load observed overwash mask
overwash_mask_file = np.load("Input/Mask_NCB-NewDrum-Ocracoke_2018_Florence_2m.npy")

# Define Alongshore Coordinates of Domain
ymin = 19250  # 19250 8230 22400 6855
ymax = 19750  # 19750 8730 22900 7355

plot_xmin = 900
plot_xmax = plot_xmin + 600

name = "19250-19750, Florence Rh=3.32"
print(name)


# _____________________________________________
# Conversions & Initializations

# Resize according to cellsize
ymin = int(ymin / cellsize)  # Alongshore
ymax = int(ymax / cellsize)  # Alongshore
plot_xmin = int(plot_xmin / cellsize)  # Cross-shore plotting
plot_xmax = int(plot_xmax / cellsize)  # Cross-shore plotting

# Initial Observed Topo
topo0 = Init[0, ymin: ymax, :]  # [m NAVD88]
topo = copy.deepcopy(topo0)  # [m NAVD88] Initialise the topography map

# Final Observed Topo
obs_topo_final = End[0, ymin:ymax, :]  # [m NAVD88]

# Set Veg Domain
spec1 = Init[1, ymin: ymax, :]
spec2 = Init[2, ymin: ymax, :]
veg = spec1 + spec2  # Determine the initial cumulative vegetation effectiveness
veg[veg > 1] = 1  # Cumulative vegetation effectiveness cannot be negative or larger than one
veg[veg < 0] = 0

# Find Dune Crest, Shoreline Positions
dune_crest, not_gap = routine.foredune_crest(topo, MHW, cellsize)
x_s = routine.ocean_shoreline(topo, MHW)

# Transform water levels to vectors
Rhigh = Rhigh * np.ones(obs_topo_final.shape[0])
Rlow = Rlow * np.ones(obs_topo_final.shape[0])

RNG = np.random.default_rng(seed=13)


# _____________________________________________
# Overwash, Beach, & Dune Change
topo_prestorm = copy.deepcopy(topo)  # [m NAVD88]

sim_topo_post_storm, OWflux, inundated, Qbe = routine.storm_processes(
    topo,
    Rhigh,
    dur,
    Rin=245,
    Cs=0.0235,
    nn=0.5,
    MaxUpSlope=1.5,
    fluxLimit=1,
    Qs_min=1,
    Kow=0.0003615,
    mm=1.05,
    MHW=MHW,
    Cbb=0.7,
    Qs_bb_min=1,
    substep=25,
    beach_equilibrium_slope=0.021,
    swash_erosive_timescale=1.51,
    beach_substeps=1,
    x_s=x_s,
    cellsize=cellsize,
    spec1=spec1,
    spec2=spec2,
    flow_reduction_max_spec1=0.002,
    flow_reduction_max_spec2=0.02,
)

sim_topo_final = routine.enforceslopes(sim_topo_post_storm, veg, sh=0.02, anglesand=20, angleveg=30, th=0.37, MHW=MHW, cellsize=cellsize, RNG=RNG)  # Enforce angles of repose

SimDuration = time.time() - start_time
print()
print("Elapsed Time: ", SimDuration, "sec")


# _____________________________________________
# Model Skill: Comparisons to Observations

longshore, crossshore = sim_topo_final.shape

# Final Elevation Change
obs_change_m = obs_topo_final - topo_prestorm  # [m] Observed change
sim_change_m = sim_topo_final - topo_prestorm  # [m] Simulated change

# Masks for skill scoring
subaerial_mask = np.logical_and(sim_topo_final > MHW, obs_topo_final > MHW)  # [bool] Map of every cell above water

overwash_mask = np.logical_and(overwash_mask_file[ymin: ymax, :], subaerial_mask)

beach_duneface_mask = np.zeros(sim_topo_final.shape)
for l in range(sim_topo_final.shape[0]):
    beach_duneface_mask[l, :dune_crest[l]] = True
beach_duneface_mask = np.logical_and(beach_duneface_mask, subaerial_mask)  # [bool] Map of every cell seaward of dune crest

mask_overwash_all = np.logical_and(np.logical_or(overwash_mask, inundated), ~beach_duneface_mask) * subaerial_mask  # [bool] Map of every cell involved in observed or simulated overwash (landward of dune crest)

mask_all = np.logical_or(overwash_mask, inundated, beach_duneface_mask.copy()) * subaerial_mask  # [bool] Map of every subaerial cell that was inundated in simulation or observation or both
mask_obs = np.logical_or(overwash_mask, beach_duneface_mask.copy()) * subaerial_mask  # [bool] Map of every subaerial cell that was inundated in observation
obs_change_m[~mask_obs] = 0

# Determine Skill
nse, rmse, nmae, mass, bss = model_skill(obs_change_m, sim_change_m, np.zeros(obs_change_m.shape), mask_all)  # Skill scores for all
nse_ow, rmse_ow, nmae_ow, mass_ow, bss_ow = model_skill(obs_change_m, sim_change_m, np.zeros(obs_change_m.shape), mask_overwash_all)  # Skill scores for just overwash
nse_bd, rmse_bd, nmae_bd, mass_bd, bss_bd = model_skill(obs_change_m, sim_change_m, np.zeros(obs_change_m.shape), beach_duneface_mask)  # Skill scores for just beach/dune


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
    "Scores": ["Beach/Dune/Overwash", "Beach/Dune", "Overwash", "No Mask"],
    "NSE": [nse, nse_bd, nse_ow],
    "RMSE": [rmse, rmse_bd, rmse_ow],
    "NMAE": [nmae, nmae_bd, nmae_ow],
    "MASS": [mass, mass_bd, mass_ow],
    "BSS": [bss, bss_bd, bss_ow],
    "PC": [PC_m, None, None, PC],
    "HSS": [HSS_m, None, None, HSS],
}, headers="keys", floatfmt=(None, ".3f", ".3f", ".3f", ".3f", ".3f", ".3f", ".3f"))
)


# _____________________________________________
# Plot

# ----------
# # Categorical
# catfig = plt.figure(figsize=(14, 7.5))
# cmap2 = colors.ListedColormap(['green', 'yellow', 'gray', 'red'])
# bounds = [0.5, 1.5, 2.5, 3.5, 4.5]
# norm = colors.BoundaryNorm(bounds, cmap2.N)
# ax1 = catfig.add_subplot(211)
# cax1 = ax1.matshow(cat_Mask[:, ymin: ymax], cmap=cmap2, norm=norm)
# cbar1 = plt.colorbar(cax1, boundaries=bounds, ticks=[1, 2, 3, 4])
# cbar1.set_ticklabels(['Hit', 'False Alarm', 'Correct Reject', 'Miss'])
# ax2 = catfig.add_subplot(212)
# cax2 = ax2.matshow(cat_NoMask[:, ymin: ymax], cmap=cmap2, norm=norm)
# cbar2 = plt.colorbar(cax2, boundaries=bounds, ticks=[1, 2, 3, 4])
# cbar2.set_ticklabels(['Hit', 'False Alarm', 'Correct Reject', 'Miss'])

# ----------
# Change Comparisons

# Set topo colormap
cmap1 = routine.truncate_colormap(copy.copy(plt.colormaps.get_cmap("terrain")), 0.5, 0.9)  # Truncate colormap
cmap1.set_bad(color='dodgerblue', alpha=0.5)  # Set cell color below MHW to blue

dune_crest_end, not_gap = routine.foredune_crest(sim_topo_final, MHW, cellsize)

# Set topo change colormap limits
max_change_obs = max(abs(np.min(obs_change_m)), abs(np.max(obs_change_m)))  # [m]
max_change_sim = max(abs(np.min(sim_change_m)), abs(np.max(sim_change_m)))  # [m]
cmap_lim = 1.5  # max(max_change_obs, max_change_sim)

# Post-Storm (Observed) Topo
Fig = plt.figure(figsize=(14, 7.5))
ax1 = Fig.add_subplot(221)
topo1 = obs_topo_final[:, plot_xmin: plot_xmax]  # [m] Post-storm
topo1 = np.ma.masked_where(topo1 < MHW, topo1)  # Mask cells below MHW
cax1 = ax1.matshow(topo1, cmap=cmap1, vmin=0, vmax=6.0)
# ax1.plot(dune_crest - plot_xmin, np.arange(len(dune_crest)), c='black', alpha=0.6)
# ax1.plot(dune_crest_end - plot_xmin, np.arange(len(dune_crest_end)), c='green', alpha=0.6)
plt.title('Observed')
plt.suptitle(name)

# Post-Storm (Simulated) Topo
ax2 = Fig.add_subplot(222)
topo2 = sim_topo_final[:, plot_xmin: plot_xmax]  # [m]
topo2 = np.ma.masked_where(topo2 < MHW, topo2)  # Mask cells below MHW
cax2 = ax2.matshow(topo2, cmap=cmap1, vmin=0, vmax=6.0)
plt.title('Simulated')
# cbar = Fig.colorbar(cax2)
# cbar.set_label('Elevation [m MHW]', rotation=270, labelpad=20)

# Simulated Topo Change
ax3 = Fig.add_subplot(224)
cax3 = ax3.matshow(sim_change_m[:, plot_xmin: plot_xmax], cmap='bwr_r', vmin=-cmap_lim, vmax=cmap_lim)
# ax3.plot(dune_crest - plot_xmin, np.arange(len(dune_crest)), c='black', alpha=0.6)
# ax3.plot(dune_crest_end - plot_xmin, np.arange(len(dune_crest_end)), c='green', alpha=0.6)
# cbar = Fig.colorbar(cax3)
# cbar.set_label('Change [m]', rotation=270, labelpad=20)

# Observed Topo Change
ax4 = Fig.add_subplot(223)
obs_change_masked = obs_change_m.copy()
obs_change_masked[~mask_obs] = 0
cax4 = ax4.matshow(obs_change_masked[:, plot_xmin: plot_xmax], cmap='bwr_r', vmin=-cmap_lim, vmax=cmap_lim)
# ax4.plot(dune_crest - plot_xmin, np.arange(len(dune_crest)), c='black', alpha=0.6)
# cbar = Fig.colorbar(cax4)
# cbar.set_label('Elevation Change [m]', rotation=270, labelpad=20)
plt.tight_layout()

# ----------
# # Cumulative Discharge
# plt.figure(figsize=(14, 7.5))
# plt.plot(np.sum(netDischarge, axis=0))
# plt.ylabel("Cumulative Discharge")

# ----------
# # Inundation
# plt.matshow(inundated)
# plt.title("Inundated")

# ----------
# Profile Change
profx = int(94 / cellsize)
proffig2 = plt.figure(figsize=(11, 7.5))
plt.plot(topo_prestorm[profx, plot_xmin: plot_xmin + 415], c='black')
plt.plot(obs_topo_final[profx, plot_xmin: plot_xmin + 415], c='green')
plt.plot(sim_topo_final[profx, plot_xmin: plot_xmin + 415], c='red')
plt.legend(["Pre", "Post Obs", "Post Sim"])
plt.title(name + ", x =" + str(profx))

print()
print("Complete.")
plt.show()
