"""
Script for calibrating beach/dune change parameters based on best fit to observations.
IRBR 1 Mar 2023
"""

import numpy as np
import matplotlib.pyplot as plt
import routines_beem as routine
import copy
import time


def model_skill(obs_change, sim_change, obs_change_mean, mask):
    """Perform suite of model skill assesments and return scores."""

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
    that of the baseline or reference predictions."""
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
        pc = np.nan
        hss = np.nan

    return nse, rmse, bss, pc, hss


def SkillSuite2(topo_pre, v1, v2, Rh, crest):

    Scores = np.empty((5, len(v1), len(v2)))

    for b in range(len(v1)):
        for e in range(len(v2)):

            # _____________________________________________
            # Run Beach/Dune Model For This Particular Combintion of Parameter Values

            topof = topo_pre.copy()  # Reset initial topo for each simulation
            dune_beach_volume_change = np.zeros(topof.shape[0])

            for t in range(dur):
                topof, dV, wetMap = routine.calc_dune_erosion_TS(
                    topo=topof,
                    dx=1,
                    crestline=crest,
                    MHW=0,
                    Rhigh=Rh,
                    Beq=v1[b],
                    Et=v2[e],
                    substeps=10,
                )

                dune_beach_volume_change += dV

            sim_topo_final = topof
            topo_change_prestorm = sim_topo_final - topo_pre

            # _____________________________________________
            # Model Skill: Comparisons to Observations

            subaerial_mask = sim_topo_final > MHW  # [bool] Map of every cell above water

            beach_duneface_mask = np.zeros(sim_topo_final.shape)
            for l in range(longshore):
                beach_duneface_mask[l, :crest[l]] = True
            beach_duneface_mask = np.logical_and(beach_duneface_mask, subaerial_mask)  # [bool] Map of every cell seaward of dune crest

            # # Final Elevation Changes
            obs_change_m = (topo_final - topo_pre)  # [m] Observed change
            sim_change_m = topo_change_prestorm  # [m] Simulated change
            obs_change_masked = obs_change_m * beach_duneface_mask  # [m]
            sim_change_masked = sim_change_m * beach_duneface_mask  # [m]
            obs_change_mean_masked = np.mean(obs_change_m[beach_duneface_mask])  # [m] Average change of observations, masked

            # Calculate Skill Scores
            NSE, RMSE, BSS, PC, HSS = model_skill(obs_change_masked, sim_change_masked, obs_change_mean_masked, beach_duneface_mask)

            # Store Skill Scores
            Scores[0, b, e] = NSE
            Scores[1, b, e] = RMSE
            Scores[2, b, e] = BSS
            Scores[3, b, e] = PC
            Scores[4, b, e] = HSS

    return Scores


def SkillSuite3(topo_pre, v1, v2, v3, Rh, crest):

    Scores = np.empty((5, len(v1), len(v2), len(v3)))

    for b in range(len(v1)):
        for e in range(len(v2)):
            for s in range(len(v3)):

                # _____________________________________________
                # Run Beach/Dune Model For This Particular Combintion of Parameter Values

                topof = topo_pre.copy()  # Reset initial topo for each simulation
                dune_beach_volume_change = np.zeros(topof.shape[0])

                for t in range(dur):
                    topof, dV, wetMap = routine.calc_dune_erosion_TS(
                        topo=topof,
                        dx=1,
                        crestline=crest,
                        MHW=0,
                        Rhigh=Rh,
                        Beq=v1[b],
                        Et=v2[e],  # Larger = more erosiveness
                        substeps=v3[s],
                    )

                    dune_beach_volume_change += dV

                sim_topo_final = topof
                topo_change_prestorm = sim_topo_final - topo_pre

                # _____________________________________________
                # Model Skill: Comparisons to Observations

                subaerial_mask = sim_topo_final > MHW  # [bool] Map of every cell above water

                beach_duneface_mask = np.zeros(sim_topo_final.shape)
                for l in range(longshore):
                    beach_duneface_mask[l, :crest[l]] = True
                beach_duneface_mask = np.logical_and(beach_duneface_mask, subaerial_mask)  # [bool] Map of every cell seaward of dune crest

                # # Final Elevation Changes
                obs_change_m = (topo_final - topo_pre)  # [m] Observed change
                sim_change_m = topo_change_prestorm  # [m] Simulated change
                obs_change_masked = obs_change_m * beach_duneface_mask  # [m]
                sim_change_masked = sim_change_m * beach_duneface_mask  # [m]
                obs_change_mean_masked = np.mean(obs_change_m[beach_duneface_mask])  # [m] Average change of observations, masked

                # Calculate Skill Scores
                NSE, RMSE, BSS, PC, HSS = model_skill(obs_change_masked, sim_change_masked, obs_change_mean_masked, beach_duneface_mask)

                # Store Skill Scores
                Scores[0, b, e, s] = NSE
                Scores[1, b, e, s] = RMSE
                Scores[2, b, e, s] = BSS
                Scores[3, b, e, s] = PC
                Scores[4, b, e, s] = HSS

    return Scores


start_time = time.time()  # Record time at start of calibration

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

# PARAMETER RANGES

# Best
# var1 = np.arange(start=0.005, stop=0.0325, step=0.0025)  # Beq
# var2 = np.arange(start=0.5, stop=3.5, step=0.5)  # dT
# var3 = np.arange(start=10, stop=50, step=5)  # substeps

# Shortened
var1 = np.arange(start=0.005, stop=0.030, step=0.005)  # Beq
var2 = np.arange(start=0.5, stop=2.5, step=0.5)  # Et
var3 = np.arange(start=10, stop=50, step=10)  # substeps

# Tiny
# var1 = np.arange(start=0.01, stop=0.030, step=0.01)  # Beq
# var2 = np.arange(start=1, stop=3, step=1)  # dT
# var3 = np.arange(start=10, stop=20, step=10)  # substeps

name = '2600-3300'

# _____________________________________________
# Conversions & Initializations

# Transform Initial Observed Topo
topo_init = Init[0, xmin: xmax, :]  # [m]
topo0 = topo_init  # [m]
topo = copy.deepcopy(topo0)  # [m] Initialise the topography map
longshore, crossshore = topo.shape

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

topo_prestorm = copy.deepcopy(topo)


# _____________________________________________
# Loop Over Parameter Ranges

print(name)

SkillScores = SkillSuite3(topo_prestorm, var1, var2, var3, Rhigh, dune_crest)

SimDuration = time.time() - start_time
print()
print("Elapsed Time: ", SimDuration, "sec")


# _____________________________________________
# Identify Best Parameter Values
NSE_max = np.unravel_index(np.nanargmax(SkillScores[0, :, :, :]), SkillScores[0, :, :, :].shape)
RMSE_min = np.unravel_index(np.nanargmin(SkillScores[1, :, :, :]), SkillScores[1, :, :, :].shape)
BSS_max = np.unravel_index(np.nanargmax(SkillScores[2, :, :, :]), SkillScores[2, :, :, :].shape)
PC_max = np.unravel_index(np.nanargmax(SkillScores[3, :, :, :]), SkillScores[3, :, :, :].shape)
HSS_max = np.unravel_index(np.nanargmax(SkillScores[4, :, :, :]), SkillScores[4, :, :, :].shape)

max0 = np.nanmax(SkillScores[0, :, :, :])
min1 = np.nanmin(SkillScores[1, :, :, :])
max2 = np.nanmax(SkillScores[2, :, :, :])
max3 = np.nanmax(SkillScores[3, :, :, :])
max4 = np.nanmax(SkillScores[4, :, :, :])

print("        Beq   Et  SS Score")
print("NSE    ", np.round(var1[NSE_max[0]], 4), var2[NSE_max[1]], var3[NSE_max[2]], max0)
print("RMSE   ", np.round(var1[RMSE_min[0]], 4), var2[RMSE_min[1]], var3[RMSE_min[2]], min1)
print("BSS    ", np.round(var1[BSS_max[0]], 4), var2[BSS_max[1]], var3[BSS_max[2]], max2)
print("PC     ", np.round(var1[PC_max[0]], 4), var2[PC_max[1]], var3[PC_max[2]], max3)
print("HSS    ", np.round(var1[HSS_max[0]], 4), var2[HSS_max[1]], var3[HSS_max[2]], max4)


# _____________________________________________
# Plot
label_x = np.round(var2, 3).astype(str)
label_y = np.round(var1, 3).astype(str)
name_x = "Erosiveness (Et)"
name_y = "EQ Beach Slope"

var3ax = int(BSS_max[2])
print("var3ax", var3ax)

Fig = plt.figure(figsize=(11, 9))
Fig.suptitle(name)

ax1 = Fig.add_subplot(221)
cax1 = ax1.matshow(SkillScores[0, :, :, var3ax], cmap='plasma', vmin=0, vmax=1)
ax1.scatter(NSE_max[1], NSE_max[0], marker="*", c="lime")
plt.xlabel(name_x)
plt.ylabel(name_y)
plt.xticks(np.arange(len(label_x)), label_x)
plt.yticks(np.arange(len(label_y)), label_y)
ax1.tick_params(top=False, labeltop=False, bottom=True, labelbottom=True)
cbar = Fig.colorbar(cax1)
# cbar.set_label('Nash-Sutcliffe Model Efficiency (NSE)', rotation=270, labelpad=20)
plt.title('Nash-Sutcliffe Model Efficiency (NSE)')

ax2 = Fig.add_subplot(222)
cax2 = ax2.matshow(SkillScores[1, :, :, var3ax], cmap='plasma')
ax2.scatter(RMSE_min[1], RMSE_min[0], marker="*", c="lime")
plt.xlabel(name_x)
plt.ylabel(name_y)
plt.xticks(np.arange(len(label_x)), label_x)
plt.yticks(np.arange(len(label_y)), label_y)
ax2.tick_params(top=False, labeltop=False, bottom=True, labelbottom=True)
cbar = Fig.colorbar(cax2)
# cbar.set_label('Root-Mean-Square Error (RMSE)', rotation=270, labelpad=20)
plt.title('Root-Mean-Square Error (RMSE)')

ax3 = Fig.add_subplot(223)
cax3 = ax3.matshow(SkillScores[2, :, :, var3ax], cmap='plasma', vmin=0, vmax=1)
ax3.scatter(BSS_max[1], BSS_max[0], marker="*", c="lime")
plt.xlabel(name_x)
plt.ylabel(name_y)
plt.xticks(np.arange(len(label_x)), label_x)
plt.yticks(np.arange(len(label_y)), label_y)
ax3.tick_params(top=False, labeltop=False, bottom=True, labelbottom=True)
cbar = Fig.colorbar(cax3)
# cbar.set_label('Brier Skill Score (BSS)', rotation=270, labelpad=20)
plt.title('Brier Skill Score (BSS)')

ax4 = Fig.add_subplot(224)
cax4 = ax4.matshow(SkillScores[3, :, :, var3ax], cmap='plasma', vmin=0, vmax=1)
ax4.scatter(HSS_max[1], HSS_max[0], marker="*", c="lime")
plt.xlabel(name_x)
plt.ylabel(name_y)
plt.xticks(np.arange(len(label_x)), label_x)
plt.yticks(np.arange(len(label_y)), label_y)
ax4.tick_params(top=False, labeltop=False, bottom=True, labelbottom=True)
cbar = Fig.colorbar(cax4)
# cbar.set_label('Heidke Skill Score (HSS)', rotation=270, labelpad=20)
plt.title('Categorical Percent Correct (PC)')

# ax4 = Fig.add_subplot(224)
# cax4 = ax4.matshow(SkillScores[4, :, :, var3ax], cmap='plasma', vmin=0, vmax=1)
# ax4.scatter(HSS_max[1], HSS_max[0], marker="*", c="lime")
# plt.xlabel(name_x)
# plt.ylabel(name_y)
# plt.xticks(np.arange(len(label_x)), label_x)
# plt.yticks(np.arange(len(label_y)), label_y)
# ax4.tick_params(top=False, labeltop=False, bottom=True, labelbottom=True)
# cbar = Fig.colorbar(cax4)
# # cbar.set_label('Heidke Skill Score (HSS)', rotation=270, labelpad=20)
# plt.title('Heidke Skill Score (HSS)')

plt.tight_layout()

print()
print("Complete.")
plt.show()
