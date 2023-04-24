"""
Script for calibrating MEEB storm parameters with BRUTE FORCE based on best fit to observations.
IRBR 8 Mar 2023
"""

import numpy as np
import matplotlib.pyplot as plt
import routines_meeb as routine
import copy
import time
from tabulate import tabulate


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
        pc = np.nan
        hss = np.nan

    return nse, rmse, bss, pc, hss



def SkillSuite10(topo_pre, obs_topo_final, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, vegf, Rh, Rl, duration, crest, mhw, owmask, slabheight):

    Scores_Beach = np.empty((5, len(v1), len(v2), len(v3), len(v4), len(v5), len(v6), len(v7), len(v8), len(v9), len(v10)))
    Scores_Overwash = np.empty((5, len(v1), len(v2), len(v3), len(v4), len(v5), len(v6), len(v7), len(v8), len(v9), len(v10)))
    longshore, crossshore = topo_pre.shape

    sim_count = 0
    sim_tot = len(v1) * len(v2) * len(v3) * len(v4) * len(v5) * len(v6) * len(v7) * len(v8) * len(v9) * len(v10)

    for b in range(len(v1)):
        for e in range(len(v2)):
            for s in range(len(v3)):
                for r in range(len(v4)):
                    for k in range(len(v5)):
                        for z in range(len(v6)):
                            for u in range(len(v7)):
                                for f in range(len(v8)):
                                    for c in range(len(v9)):
                                        for m in range(len(v10)):

                                            sim_count += 1
                                            print("\r", "Progress: ", sim_count, "/", sim_tot, "simulations", end="")

                                            # _____________________________________________
                                            # Run Beach/Dune Model For This Particular Combintion of Parameter Values

                                            topof = copy.deepcopy(topo_pre)

                                            topof, topo_change_overwash, OWflux, netDischarge, inundated = routine.storm_processes(
                                                topof,
                                                vegf,
                                                Rh,
                                                Rl,
                                                duration,
                                                slabheight_m=slabheight,
                                                threshold_in=0.25,
                                                Rin_i=5,
                                                Rin_r=v4[r],
                                                Cx=v9[c],
                                                AvgSlope=2 / 200,
                                                nn=0.5,
                                                MaxUpSlope=v7[u],
                                                fluxLimit=v8[f],
                                                Qs_min=1,
                                                Kr=v5[k],
                                                Ki=5e-06,
                                                mm=v10[m],
                                                MHW=mhw,
                                                Cbb_i=0.85,
                                                Cbb_r=0.7,
                                                Qs_bb_min=1,
                                                substep_i=6,
                                                substep_r=v6[z],
                                                beach_equilibrium_slope=v1[b],
                                                beach_erosiveness=v2[e],
                                                beach_substeps=v3[s],
                                            )

                                            sim_topo_final = topof * slabheight  # [m]
                                            obs_topo_final_m = obs_topo_final * slabheight  # [m]
                                            topo_pre_m = topo_pre * slabheight  # [m]
                                            topo_change_prestorm = sim_topo_final - topo_pre_m

                                            # _____________________________________________
                                            # Model Skill: Comparisons to Observations

                                            subaerial_mask = sim_topo_final > MHW  # [bool] Map of every cell above water

                                            beach_duneface_mask = np.zeros(sim_topo_final.shape)
                                            for l in range(longshore):
                                                beach_duneface_mask[l, :crest[l]] = True
                                            beach_duneface_mask = np.logical_and(beach_duneface_mask, subaerial_mask)  # [bool] Map of every cell seaward of dune crest

                                            Sim_Obs_OW_Mask = np.logical_or(OW_Mask, inundated) * (~beach_duneface_mask) * subaerial_mask  # [bool] Map of every cell landward of dune crest that was inundated in simulation or observation or both

                                            # Final Elevation Changes
                                            obs_change_m = (obs_topo_final_m - topo_pre_m)  # [m] Observed change
                                            sim_change_m = topo_change_prestorm  # [m] Simulated change

                                            # Beach Mask
                                            obs_change_bmasked = obs_change_m * beach_duneface_mask  # [m]
                                            sim_change_bmasked = sim_change_m * beach_duneface_mask  # [m]
                                            obs_change_mean_bmasked = np.mean(obs_change_m[beach_duneface_mask])  # [m] Average beach change of observations, masked

                                            # Overwash Mask
                                            obs_change_omasked = obs_change_m * owmask * ~beach_duneface_mask  # [m]
                                            sim_change_omasked = sim_change_m * ~beach_duneface_mask  # [m]
                                            obs_change_mean_omasked = np.mean(obs_change_m[Sim_Obs_OW_Mask])  # [m] Average beach change of observations, masked

                                            # ------
                                            # Beach

                                            # Calculate Skill Scores
                                            NSEb, RMSEb, BSSb, PCb, HSSb = model_skill(obs_change_bmasked, sim_change_bmasked, obs_change_mean_bmasked, beach_duneface_mask)

                                            # Store Skill Scores
                                            Scores_Beach[0, b, e, s, r, k, z, u, f, c, m] = NSEb
                                            Scores_Beach[1, b, e, s, r, k, z, u, f, c, m] = RMSEb
                                            Scores_Beach[2, b, e, s, r, k, z, u, f, c, m] = BSSb
                                            Scores_Beach[3, b, e, s, r, k, z, u, f, c, m] = PCb
                                            Scores_Beach[4, b, e, s, r, k, z, u, f, c, m] = HSSb

                                            # --------
                                            # Overwash

                                            # Calculate Skill Scores
                                            NSEo, RMSEo, BSSo, PCo, HSSo = model_skill(obs_change_omasked, sim_change_omasked, obs_change_mean_omasked, Sim_Obs_OW_Mask)

                                            # Store Skill Scores
                                            Scores_Overwash[0, b, e, s, r, k, z, u, f, c, m] = NSEo
                                            Scores_Overwash[1, b, e, s, r, k, z, u, f, c, m] = RMSEo
                                            Scores_Overwash[2, b, e, s, r, k, z, u, f, c, m] = BSSo
                                            Scores_Overwash[3, b, e, s, r, k, z, u, f, c, m] = PCo
                                            Scores_Overwash[4, b, e, s, r, k, z, u, f, c, m] = HSSo

    return Scores_Beach, Scores_Overwash


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

# PARAMETER RANGES

# var1 = [0.02]  # Beq
# var2 = [2]  # Et
# var3 = [20]  # BD Substeps
# var4 = np.arange(start=25, stop=150, step=25)  # Rin_r
# # var5 = np.arange(start=0.0001, stop=0.0007, step=0.0001)  # Kr > KrQ
# var5 = np.arange(start=0.00001, stop=0.00009, step=0.00001)  # Kr > KrQ(S+C)
# var6 = np.arange(start=3, stop=12, step=3)  # OW substeps
# var7 = [1]  # MaxUpSlope
# var8 = [1]  # fluxLimit
# var9 = [15]  # Cx
# var10 = [2]  # mm

var1 = [0.02]  # Beq
var2 = [2]  # Et
var3 = [20]  # BD Substeps
var4 = np.arange(start=50, stop=250, step=50)  # Rin_r
# var5 = np.arange(start=0.0001, stop=0.0007, step=0.0001)  # Kr > KrQ
var5 = np.arange(start=0.000035, stop=0.000085, step=0.00001)  # Kr > KrQ(S+C)
var6 = [3]  # np.arange(start=3, stop=6, step=3)  # OW substeps
var7 = np.arange(start=1, stop=3, step=1)  # MaxUpSlope
var8 = [1]  # fluxLimit
var9 = np.arange(start=10, stop=90, step=20)  # Cx
var10 = [2]  # mm


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

# Find Dune Crest, Beach Slopes
dune_crest = routine.foredune_crest(topo * slabheight_m, veg)
# dune_crest[245: 299] = 171  # 1715-1845  # 2000-2600 TEMP!!!

# Transform water levels to vectors
Rhigh = Rhigh * np.ones(topo_final.shape[0])
Rlow = Rlow * np.ones(topo_final.shape[0])

topo_prestorm = copy.deepcopy(topo)


# _____________________________________________
# Loop Over Parameter Ranges

print(name)
print()

Skil_Scores_Beach, Skill_Scores_Overwash = SkillSuite10(topo_prestorm, topo_final, var1, var2, var3, var4, var5, var6, var7, var8, var9, var10, veg, Rhigh, Rlow, dur, dune_crest, MHW, OW_Mask, slabheight_m)

SimDuration = time.time() - start_time
print()
print("Elapsed Time: ", SimDuration, "sec")


# _____________________________________________
# Identify Best Parameter Values

# 6 Parameters

# ____________
# Beach
NSE_max_b = np.unravel_index(np.nanargmax(Skil_Scores_Beach[0, :, :, :, :, :, :, :, :, :, :]), Skil_Scores_Beach[0, :, :, :, :, :, :, :, :, :, :].shape)
RMSE_min_b = np.unravel_index(np.nanargmin(Skil_Scores_Beach[1, :, :, :, :, :, :, :, :, :, :]), Skil_Scores_Beach[1, :, :, :, :, :, :, :, :, :, :].shape)
BSS_max_b = np.unravel_index(np.nanargmax(Skil_Scores_Beach[2, :, :, :, :, :, :, :, :, :, :]), Skil_Scores_Beach[2, :, :, :, :, :, :, :, :, :, :].shape)
PC_max_b = np.unravel_index(np.nanargmax(Skil_Scores_Beach[3, :, :, :, :, :, :, :, :, :, :]), Skil_Scores_Beach[3, :, :, :, :, :, :, :, :, :, :].shape)
HSS_max_b = np.unravel_index(np.nanargmax(Skil_Scores_Beach[4, :, :, :, :, :, :, :, :, :, :]), Skil_Scores_Beach[4, :, :, :, :, :, :, :, :, :, :].shape)

max0_b = np.nanmax(Skil_Scores_Beach[0, :, :, :, :, :, :, :, :, :, :])
min1_b = np.nanmin(Skil_Scores_Beach[1, :, :, :, :, :, :, :, :, :, :])
max2_b = np.nanmax(Skil_Scores_Beach[2, :, :, :, :, :, :, :, :, :, :])
max3_b = np.nanmax(Skil_Scores_Beach[3, :, :, :, :, :, :, :, :, :, :])
max4_b = np.nanmax(Skil_Scores_Beach[4, :, :, :, :, :, :, :, :, :, :])

print()
print(tabulate({
    "BEACH   ": ["NSE", "RMSE", "BSS", "PC", "HSS"],
    "Beq":  [var1[NSE_max_b[0]], var1[RMSE_min_b[0]], var1[BSS_max_b[0]], var1[PC_max_b[0]], var1[HSS_max_b[0]]],
    "dT":   [var2[NSE_max_b[1]], var2[RMSE_min_b[1]], var2[BSS_max_b[1]], var2[PC_max_b[1]], var2[HSS_max_b[1]]],
    "SSb":  [var3[NSE_max_b[2]], var3[RMSE_min_b[2]], var3[BSS_max_b[2]], var3[PC_max_b[2]], var3[HSS_max_b[2]]],
    "Rin":  [var4[NSE_max_b[3]], var4[RMSE_min_b[3]], var4[BSS_max_b[3]], var4[PC_max_b[3]], var4[HSS_max_b[3]]],
    "Kr":   [var5[NSE_max_b[4]], var5[RMSE_min_b[4]], var5[BSS_max_b[4]], var5[PC_max_b[4]], var5[HSS_max_b[4]]],
    "SSo":  [var6[NSE_max_b[5]], var6[RMSE_min_b[5]], var6[BSS_max_b[5]], var6[PC_max_b[5]], var6[HSS_max_b[5]]],
    "MUS":  [var7[NSE_max_b[6]], var7[RMSE_min_b[6]], var7[BSS_max_b[6]], var7[PC_max_b[6]], var7[HSS_max_b[6]]],
    "FLim": [var8[NSE_max_b[7]], var8[RMSE_min_b[7]], var8[BSS_max_b[7]], var8[PC_max_b[7]], var8[HSS_max_b[7]]],
    "Cx":   [var9[NSE_max_b[8]], var9[RMSE_min_b[8]], var9[BSS_max_b[8]], var9[PC_max_b[8]], var9[HSS_max_b[8]]],
    "mm":   [var10[NSE_max_b[9]], var10[RMSE_min_b[9]], var10[BSS_max_b[9]], var10[PC_max_b[9]], var10[HSS_max_b[9]]],
    "Score": [max0_b, min1_b, max2_b, max3_b, max4_b]
    }, headers="keys", floatfmt=(None, ".3f", ".0f", ".0f", ".0f", ".6f", ".0f", ".0f", ".0f", ".0f", ".0f", ".4f"))
)

# ____________
# Overwash
NSE_max_o = np.unravel_index(np.nanargmax(Skill_Scores_Overwash[0, :, :, :, :, :, :, :, :, :, :]), Skill_Scores_Overwash[0, :, :, :, :, :, :, :, :, :, :].shape)
RMSE_min_o = np.unravel_index(np.nanargmin(Skill_Scores_Overwash[1, :, :, :, :, :, :, :, :, :, :]), Skill_Scores_Overwash[1, :, :, :, :, :, :, :, :, :, :].shape)
BSS_max_o = np.unravel_index(np.nanargmax(Skill_Scores_Overwash[2, :, :, :, :, :, :, :, :, :, :]), Skill_Scores_Overwash[2, :, :, :, :, :, :, :, :, :, :].shape)
PC_max_o = np.unravel_index(np.nanargmax(Skill_Scores_Overwash[3, :, :, :, :, :, :, :, :, :, :]), Skill_Scores_Overwash[3, :, :, :, :, :, :, :, :, :, :].shape)
HSS_max_o = np.unravel_index(np.nanargmax(Skill_Scores_Overwash[4, :, :, :, :, :, :, :, :, :, :]), Skill_Scores_Overwash[4, :, :, :, :, :, :, :, :, :, :].shape)

max0_o = np.nanmax(Skill_Scores_Overwash[0, :, :, :, :, :, :, :, :, :, :])
min1_o = np.nanmin(Skill_Scores_Overwash[1, :, :, :, :, :, :, :, :, :, :])
max2_o = np.nanmax(Skill_Scores_Overwash[2, :, :, :, :, :, :, :, :, :, :])
max3_o = np.nanmax(Skill_Scores_Overwash[3, :, :, :, :, :, :, :, :, :, :])
max4_o = np.nanmax(Skill_Scores_Overwash[4, :, :, :, :, :, :, :, :, :, :])

print()
print(tabulate({
    "OVERWASH": ["NSE", "RMSE", "BSS", "PC", "HSS"],
    "Beq":  [var1[NSE_max_o[0]], var1[RMSE_min_o[0]], var1[BSS_max_o[0]], var1[PC_max_o[0]], var1[HSS_max_o[0]]],
    "dT":   [var2[NSE_max_o[1]], var2[RMSE_min_o[1]], var2[BSS_max_o[1]], var2[PC_max_o[1]], var2[HSS_max_o[1]]],
    "SSb":  [var3[NSE_max_o[2]], var3[RMSE_min_o[2]], var3[BSS_max_o[2]], var3[PC_max_o[2]], var3[HSS_max_o[2]]],
    "Rin":  [var4[NSE_max_o[3]], var4[RMSE_min_o[3]], var4[BSS_max_o[3]], var4[PC_max_o[3]], var4[HSS_max_o[3]]],
    "Kr":   [var5[NSE_max_o[4]], var5[RMSE_min_o[4]], var5[BSS_max_o[4]], var5[PC_max_o[4]], var5[HSS_max_o[4]]],
    "SSo":  [var6[NSE_max_o[5]], var6[RMSE_min_o[5]], var6[BSS_max_o[5]], var6[PC_max_o[5]], var6[HSS_max_o[5]]],
    "MUS":  [var7[NSE_max_o[6]], var7[RMSE_min_o[6]], var7[BSS_max_o[6]], var7[PC_max_o[6]], var7[HSS_max_o[6]]],
    "FLim": [var8[NSE_max_o[7]], var8[RMSE_min_o[7]], var8[BSS_max_o[7]], var8[PC_max_o[7]], var8[HSS_max_o[7]]],
    "Cx":   [var9[NSE_max_o[8]], var9[RMSE_min_o[8]], var9[BSS_max_o[8]], var9[PC_max_o[8]], var9[HSS_max_o[8]]],
    "mm":   [var10[NSE_max_o[9]], var10[RMSE_min_o[9]], var10[BSS_max_o[9]], var10[PC_max_o[9]], var10[HSS_max_o[9]]],
    "Score": [max0_o, min1_o, max2_o, max3_o, max4_o]
    }, headers="keys", floatfmt=(None, ".3f", ".0f", ".0f", ".0f", ".6f", ".0f", ".0f", ".0f", ".0f", ".0f", ".4f"))
)


# _____________________________________________
# Plot Beach
label_x = np.round(var2, 3).astype(str)
label_y = np.round(var1, 3).astype(str)
name_x = "Erosiveness (dT)"
name_y = "EQ Beach Slope"
name_z = "Beach/Dune Substeps"

var3ax = int(BSS_max_b[2])
var4ax = int(BSS_max_b[3])
var5ax = int(BSS_max_b[4])
var6ax = int(BSS_max_b[5])
var7ax = int(BSS_max_b[6])
var8ax = int(BSS_max_b[7])
var9ax = int(BSS_max_b[8])
var10ax = int(BSS_max_b[9])

Fig = plt.figure(figsize=(11, 9))
Fig.suptitle('Beach, ' + name)

ax1 = Fig.add_subplot(221)
cax1 = ax1.matshow(Skil_Scores_Beach[0, :, :, var3ax, var4ax, var5ax, var6ax, var7ax, var8ax, var9ax, var10ax], cmap='plasma', vmin=0, vmax=1)
ax1.scatter(NSE_max_b[1], NSE_max_b[0], marker="*", c="lime")
plt.xlabel(name_x)
plt.ylabel(name_y)
plt.xticks(np.arange(len(label_x)), label_x)
plt.yticks(np.arange(len(label_y)), label_y)
ax1.tick_params(top=False, labeltop=False, bottom=True, labelbottom=True)
cbar = Fig.colorbar(cax1)
# cbar.set_label('Nash-Sutcliffe Model Efficiency (NSE)', rotation=270, labelpad=20)
plt.title('Nash-Sutcliffe Model Efficiency (NSE)')

ax2 = Fig.add_subplot(222)
cax2 = ax2.matshow(Skil_Scores_Beach[1, :, :, var3ax, var4ax, var5ax, var6ax, var7ax, var8ax, var9ax, var10ax], cmap='plasma')
ax2.scatter(RMSE_min_b[1], RMSE_min_b[0], marker="*", c="lime")
plt.xlabel(name_x)
plt.ylabel(name_y)
plt.xticks(np.arange(len(label_x)), label_x)
plt.yticks(np.arange(len(label_y)), label_y)
ax2.tick_params(top=False, labeltop=False, bottom=True, labelbottom=True)
cbar = Fig.colorbar(cax2)
# cbar.set_label('Root-Mean-Square Error (RMSE)', rotation=270, labelpad=20)
plt.title('Root-Mean-Square Error (RMSE)')

ax3 = Fig.add_subplot(223)
cax3 = ax3.matshow(Skil_Scores_Beach[2, :, :, var3ax, var4ax, var5ax, var6ax, var7ax, var8ax, var9ax, var10ax], cmap='plasma', vmin=0, vmax=1)
ax3.scatter(BSS_max_b[1], BSS_max_b[0], marker="*", c="lime")
plt.xlabel(name_x)
plt.ylabel(name_y)
plt.xticks(np.arange(len(label_x)), label_x)
plt.yticks(np.arange(len(label_y)), label_y)
ax3.tick_params(top=False, labeltop=False, bottom=True, labelbottom=True)
cbar = Fig.colorbar(cax3)
# cbar.set_label('Brier Skill Score (BSS)', rotation=270, labelpad=20)
plt.title('Brier Skill Score (BSS)')

ax4 = Fig.add_subplot(224)
cax4 = ax4.matshow(Skil_Scores_Beach[3, :, :, var3ax, var4ax, var5ax, var6ax, var7ax, var8ax, var9ax, var10ax], cmap='plasma', vmin=0, vmax=1)
ax4.scatter(HSS_max_b[1], HSS_max_b[0], marker="*", c="lime")
plt.xlabel(name_x)
plt.ylabel(name_y)
plt.xticks(np.arange(len(label_x)), label_x)
plt.yticks(np.arange(len(label_y)), label_y)
ax4.tick_params(top=False, labeltop=False, bottom=True, labelbottom=True)
cbar = Fig.colorbar(cax4)
# cbar.set_label('Heidke Skill Score (HSS)', rotation=270, labelpad=20)
plt.title('Categorical Percent Correct (PC)')

# ax4 = Fig.add_subplot(224)
# cax4 = ax4.matshow(Skil_Scores_Beach[4, :, :, var3ax, var4ax, var5ax, var6ax, var7ax, var8ax, var9ax, var10ax], cmap='plasma', vmin=0, vmax=1)
# ax4.scatter(HSS_max_b[1], HSS_max_b[0], marker="*", c="lime")
# plt.xlabel(name_x)
# plt.ylabel(name_y)
# plt.xticks(np.arange(len(label_x)), label_x)
# plt.yticks(np.arange(len(label_y)), label_y)
# ax4.tick_params(top=False, labeltop=False, bottom=True, labelbottom=True)
# cbar = Fig.colorbar(cax4)
# # cbar.set_label('Heidke Skill Score (HSS)', rotation=270, labelpad=20)
# plt.title('Heidke Skill Score (HSS)')

plt.tight_layout()


# _____________________________________________
# Plot Overwash
label_x = np.round(var5, 8).astype(str)
label_y = np.round(var4, 3).astype(str)
name_x = "Kr"
name_y = "Rin"
name_z = "Overwash Substeps"

var1ax = int(BSS_max_o[0])
var2ax = int(BSS_max_o[1])
var3ax = int(BSS_max_o[2])
var4ax = int(BSS_max_o[3])
var5ax = int(BSS_max_o[4])
var6ax = int(BSS_max_o[5])
var7ax = int(BSS_max_o[6])
var8ax = int(BSS_max_o[7])
var9ax = int(BSS_max_o[8])
var10ax = int(BSS_max_o[9])

Fig = plt.figure(figsize=(11, 9))
Fig.suptitle('Overwash, ' + name)

ax1 = Fig.add_subplot(221)
cax1 = ax1.matshow(Skill_Scores_Overwash[0, var1ax, var2ax, var3ax, :, :, var6ax, var7ax, var8ax, var9ax, var10ax], cmap='plasma', vmin=0, vmax=1)
ax1.scatter(NSE_max_o[4], NSE_max_o[3], marker="*", c="lime")
plt.xlabel(name_x)
plt.ylabel(name_y)
plt.xticks(np.arange(len(label_x)), label_x)
plt.yticks(np.arange(len(label_y)), label_y)
ax1.tick_params(top=False, labeltop=False, bottom=True, labelbottom=True)
cbar = Fig.colorbar(cax1)
# cbar.set_label('Nash-Sutcliffe Model Efficiency (NSE)', rotation=270, labelpad=20)
plt.title('Nash-Sutcliffe Model Efficiency (NSE)')

ax2 = Fig.add_subplot(222)
cax2 = ax2.matshow(Skill_Scores_Overwash[1, var1ax, var2ax, var3ax, :, :, var6ax, var7ax, var8ax, var9ax, var10ax], cmap='plasma')
ax2.scatter(RMSE_min_o[4], RMSE_min_o[3], marker="*", c="lime")
plt.xlabel(name_x)
plt.ylabel(name_y)
plt.xticks(np.arange(len(label_x)), label_x)
plt.yticks(np.arange(len(label_y)), label_y)
ax2.tick_params(top=False, labeltop=False, bottom=True, labelbottom=True)
cbar = Fig.colorbar(cax2)
# cbar.set_label('Root-Mean-Square Error (RMSE)', rotation=270, labelpad=20)
plt.title('Root-Mean-Square Error (RMSE)')

ax3 = Fig.add_subplot(223)
cax3 = ax3.matshow(Skill_Scores_Overwash[2, var1ax, var2ax, var3ax, :, :, var6ax, var7ax, var8ax, var9ax, var10ax], cmap='plasma', vmin=0, vmax=1)
ax3.scatter(BSS_max_o[4], BSS_max_o[3], marker="*", c="lime")
plt.xlabel(name_x)
plt.ylabel(name_y)
plt.xticks(np.arange(len(label_x)), label_x)
plt.yticks(np.arange(len(label_y)), label_y)
ax3.tick_params(top=False, labeltop=False, bottom=True, labelbottom=True)
cbar = Fig.colorbar(cax3)
# cbar.set_label('Brier Skill Score (BSS)', rotation=270, labelpad=20)
plt.title('Brier Skill Score (BSS)')

ax4 = Fig.add_subplot(224)
cax4 = ax4.matshow(Skill_Scores_Overwash[3, var1ax, var2ax, var3ax, :, :, var6ax, var7ax, var8ax, var9ax, var10ax], cmap='plasma', vmin=0, vmax=1)
ax4.scatter(PC_max_o[4], PC_max_o[3], marker="*", c="lime")
plt.xlabel(name_x)
plt.ylabel(name_y)
plt.xticks(np.arange(len(label_x)), label_x)
plt.yticks(np.arange(len(label_y)), label_y)
ax4.tick_params(top=False, labeltop=False, bottom=True, labelbottom=True)
cbar = Fig.colorbar(cax4)
# cbar.set_label('Heidke Skill Score (HSS)', rotation=270, labelpad=20)
plt.title('Categorical Percent Correct (PC)')

# ax4 = Fig.add_subplot(224)
# cax4 = ax4.matshow(Skill_Scores_Overwash[4, var1ax, var2ax, var3ax, :, :, var6ax, var7ax, var8ax, var9ax, var10ax], cmap='plasma', vmin=0, vmax=1)
# ax4.scatter(HSS_max_o[4], HSS_max_o[3], marker="*", c="lime")
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
