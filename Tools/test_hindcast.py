"""
Script for testing MEEB hindcast simulations.

Runs a hindcast simulation and calculates fitess scores for morphologic and ecologic change between simulated and observed.

IRBR 10 August 2023
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as colors
import routines_meeb as routine
import copy
import time
from tabulate import tabulate

from meeb import MEEB


# __________________________________________________________________________________________________________________________________
# MODEL SKILL

def model_skill(obs, sim, t0, mask):
    """
    Perform suite of model skill assesments and return scores.
    Mask is boolean array with same size as change maps, with cells to be excluded from skill analysis set to FALSE.
    """

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


    return NSE, RMSE, NMAE, MASS, BSS


def model_skill_categorical(obs, sim, catmask):
    """
    Perform categorical skill assesment and return scores.
    Mask is boolean array with same size as change maps, with cells to be excluded from skill analysis set to FALSE.
    """

    threshold = 0.02
    sim_loss = sim < -threshold
    sim_gain = sim > threshold
    sim_no_change = np.logical_and(sim <= threshold, -threshold <= sim)
    obs_loss = obs < -threshold
    obs_gain = obs > threshold
    obs_no_change = np.logical_and(obs <= threshold, -threshold <= obs)

    cat = np.zeros(obs.shape)
    cat[np.logical_and(sim_loss, obs_loss)] = 1  # Hit
    cat[np.logical_and(sim_gain, obs_gain)] = 1  # Hit
    cat[np.logical_and(sim_loss, ~obs_loss)] = 2  # False Alarm
    cat[np.logical_and(sim_gain, ~obs_gain)] = 2  # False Alarm
    cat[np.logical_and(sim_no_change, obs_no_change)] = 3  # Correct Reject
    cat[np.logical_and(sim_no_change, ~obs_no_change)] = 4  # Miss

    hits = np.count_nonzero(cat[catmask] == 1)
    false_alarms = np.count_nonzero(cat[catmask] == 2)
    correct_rejects = np.count_nonzero(cat[catmask] == 3)
    misses = np.count_nonzero(cat[catmask] == 4)
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

    return PC, HSS, cat


# __________________________________________________________________________________________________________________________________
# VARIABLES AND INITIALIZATIONS

# # 2004 - 2009
# start = "Init_NCB-NewDrum-Ocracoke_2004_PostIsabel.npy"
# stop = "Init_NCB-NewDrum-Ocracoke_2009_PreIrene.npy"
# startdate = '20040716'
# hindcast_duration = 5.1

# # 0.92, 20161012
# start = "Init_NCB-NewDrum-Ocracoke_2016_PostMatthew.npy"
# stop = "Init_NCB-NewDrum-Ocracoke_2017_PreFlorence.npy"

# # 2016 - 2018
# start = "Init_NCB-NewDrum-Ocracoke_2016_PostMatthew.npy"
# stop = "Init_NCB-NewDrum-Ocracoke_2018_PostFlorence.npy"
# startdate = '20161012'
# hindcast_duration = 1.96

# 2017 - 2018
start = "Init_NCB-NewDrum-Ocracoke_2017_PreFlorence.npy"
stop = "Init_NCB-NewDrum-Ocracoke_2018_PostFlorence.npy"
startdate = '20170916'
hindcast_duration = 1.06

# # 2004 - 2014
# start = "Init_NCB-NewDrum-Ocracoke_2004_PostIsabel.npy"
# stop = "Init_NCB-NewDrum-Ocracoke_2014_PostSandyNCFMP.npy"
# startdate = '20040716'
# hindcast_duration = 9.72

# # 2004 - 2019
# start = "Init_NCB-NewDrum-Ocracoke_2004_PostIsabel.npy"
# stop = "Init_NCB-NewDrum-Ocracoke_2019_PreDorian.npy"
# startdate = '20040716'
# hindcast_duration = 15.12

# # 6.04, 20110829
# start = "Init_NCB-NewDrum-Ocracoke_2011_PostIrene.npy"
# stop = "Init_NCB-NewDrum-Ocracoke_2017_PreFlorence.npy"

# # 2011 - 2016
# start = "Init_NCB-NewDrum-Ocracoke_2011_PostIrene.npy"
# stop = "Init_NCB-NewDrum-Ocracoke_2016_PostMatthew.npy"
# startdate = '20110829'
# hindcast_duration = 5.12

# # 2012 - 2017
# start = "Init_NCB-NewDrum-Ocracoke_2012_PostSandyUSGS_NoThin.npy"
# stop = "Init_NCB-NewDrum-Ocracoke_2017_PreFlorence.npy"
# startdate = '20121129'
# hindcast_duration = 4.78

# # 2012 - 2018
# start = "Init_NCB-NewDrum-Ocracoke_2012_PostSandyUSGS_MinimalThin.npy"
# stop = "Init_NCB-NewDrum-Ocracoke_2018_PostFlorence.npy"
# startdate = '20121129'
# hindcast_duration = 5.84

# # 1.34, 20121129
# start = "Init_NCB-NewDrum-Ocracoke_2012_PostSandyUSGS_MinimalThin.npy"
# stop = "Init_NCB-NewDrum-Ocracoke_2014_PostSandyNCFMP.npy"

# # 3.44, 20140406
# start = "Init_NCB-NewDrum-Ocracoke_2014_PostSandyNCFMP.npy"
# stop = "Init_NCB-NewDrum-Ocracoke_2017_PreFlorence.npy"

# # 2014 - 2016
# start = "Init_NCB-NewDrum-Ocracoke_2014_PostSandyNCFMP.npy"
# stop = "Init_NCB-NewDrum-Ocracoke_2016_PostMatthew.npy"
# startdate = '20140406'
# hindcast_duration = 2.52

# # 2014 - 2017
# start = "Init_NCB-NewDrum-Ocracoke_2014_PostSandy_NCFMP-Planet.npy"
# stop = "Init_NCB-NewDrum-Ocracoke_2017_PreFlorence.npy"
# startdate = '20140406'
# hindcast_duration = 3.44

# # 2014 - 2018
# start = "Init_NCB-NewDrum-Ocracoke_2014_PostSandyNCFMP.npy"
# stop = "Init_NCB-NewDrum-Ocracoke_2018_PostFlorence.npy"
# startdate = '20140406'
# hindcast_duration = 4.5

# # 2014 - 2019
# start = "Init_NCB-NewDrum-Ocracoke_2014_PostSandyNCFMP.npy"
# stop = "Init_NCB-NewDrum-Ocracoke_2019_PreDorian.npy"
# startdate = '20140406'
# hindcast_duration = 5.4

# # 2009 - 2019
# start = "Init_NCB-NewDrum-Ocracoke_2009_PreIrene.npy"
# stop = "Init_NCB-NewDrum-Ocracoke_2019_PreDorian.npy"
# startdate = '20090824'
# hindcast_duration = 10.02  # Double check

# # 2009 - 2018
# start = "Init_NCB-NewDrum-Ocracoke_2009_PreIrene.npy"
# stop = "Init_NCB-NewDrum-Ocracoke_2018_PostFlorence.npy"
# startdate = '20090824'
# hindcast_duration = 9.12

# _____________________
# Initial Observed Topo
Init = np.load("Input/" + start)
# Final Observed
End = np.load("Input/" + stop)

# Define Alongshore Coordinates of Domain
xmin = 18950  # 575, 2000, 2150, 2000, 3800  # 2650 #6500  #20000 # 5880 # 18950
xmax = 19250  # 825, 2125, 2350, 2600, 4450  # 2850 #6600         # 5980 # 19250

MHW = 0.39  # [m NAVD88]
name = '18950-19250, 2017-2018, 10-52-56-11-55-5-22-18/183-56-2.05-575-6, BermEl 1.78, r=0.004'
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
    simulation_time_yr=hindcast_duration,
    alongshore_domain_boundary_min=xmin,
    alongshore_domain_boundary_max=xmax,
    RSLR=0.004,
    MHW=MHW,
    init_filename=start,
    hindcast=True,
    seeded_random_numbers=True,
    simulation_start_date=startdate,
    storm_timeseries_filename='StormTimeSeries_1980-2020_NCB-CE_Beta0pt039_BermEl1pt78.npy',
    # --- Aeolian --- #
    jumplength=5,
    slabheight=0.02,
    p_dep_sand=0.10,  # Q = hs * L * n * pe/pd
    p_dep_sand_VegMax=0.52,
    p_ero_sand=0.16,
    entrainment_veg_limit=0.11,
    saltation_veg_limit=0.20,
    shadowangle=8,
    repose_bare=20,
    repose_veg=30,
    wind_rose=(0.55, 0.05, 0.22, 0.18),  # (right, down, left, up)
    # --- Storms --- #
    Rin_ru=183,
    Cx=56,
    MaxUpSlope=2.05,
    K_ru=0.0000575,
    substep_ru=6,
    beach_equilibrium_slope=0.02,
    beach_erosiveness=2.73,
    beach_substeps=22,
    # --- Veg --- #
    # sp1_c=1.20,
    # sp2_c=-0.47,
    # sp1_peak=0.307,
    # sp2_peak=0.148,
    # lateral_probability=0.34,
    # pioneer_probability=0.11,
    # Spec1_elev_min=0.60,
    # Spec2_elev_min=0.13,
)

print(meeb.name)

# Loop through time
for time_step in range(int(meeb.iterations)):
    # Print time step to screen
    print("\r", "Time Step: ", (time_step + 1) / meeb.iterations_per_cycle, "years", end="")

    # Run time step
    meeb.update(time_step)

# Print elapsed time of simulation
print()
SimDuration = time.time() - start_time
print()
print("Elapsed Time: ", SimDuration, "sec")

# __________________________________________________________________________________________________________________________________
# ASSESS MODEL SKILL

# Topo change
topo_end_sim = meeb.topo  # [m NAVDD88]
mhw_end_sim = meeb.MHW  # [m NAVD88]
topo_change_sim = topo_end_sim - topo_start  # [m]
topo_change_obs = topo_end_obs - topo_start  # [m]

# Veg change
veg_end_sim = meeb.veg
veg_change_sim = veg_end_sim - veg_start  # [m]
veg_change_obs = veg_end - veg_start  # [m]
veg_present_sim = veg_end_sim > 0.05  # [bool]
veg_present_obs = veg_end > 0.05  # [bool]

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

# Limit interior in analysis by elevation
elev_mask = topo_end_sim > 2.0  # [bool] Mask for every cell above water

# Choose masks
mask = subaerial_mask.copy()
veg_mask = mask.copy()

# # Temp limit interior in analysis to dunes
# mask[:, :835] = False
# mask[:, 950:] = False
mask[:, :1100] = False
mask[:, 1350:] = False

# Optional: Reduce Resolutions
if ResReduc:
    topo_change_obs = routine.reduce_raster_resolution(topo_change_obs, reduc)
    topo_change_sim = routine.reduce_raster_resolution(topo_change_sim, reduc)
    mask = (routine.reduce_raster_resolution(mask, reduc)) == 1
    subaerial_mask = (routine.reduce_raster_resolution(subaerial_mask, reduc)) == 1

    veg_change_obs = routine.reduce_raster_resolution(veg_change_obs, reduc)
    veg_change_sim = routine.reduce_raster_resolution(veg_change_sim, reduc)
    veg_present_obs = routine.reduce_raster_resolution(veg_present_obs, reduc)
    veg_present_sim = routine.reduce_raster_resolution(veg_present_sim, reduc)
    veg_mask = (routine.reduce_raster_resolution(veg_mask, reduc)) == 1

# Model Skill
nse, rmse, nmae, mass, bss = model_skill(topo_change_obs, topo_change_sim, np.zeros(topo_change_obs.shape), mask)  # All cells (excluding masked areas)
nse_dl, rmse_dl, nmae_dl, mass_dl, bss_dl = model_skill(crest_loc_obs.astype('float32'), crest_loc_sim.astype('float32'), crest_loc_obs_start.astype('float32'), np.full(crest_loc_obs.shape, True))  # Foredune location
nse_dh, rmse_dh, nmae_dh, mass_dh, bss_dh = model_skill(crest_height_obs, crest_height_sim, crest_height_obs_start, np.full(crest_height_change_obs.shape, True))  # Foredune elevation

pc_vc, hss_vc, cat_vc = model_skill_categorical(veg_change_obs, veg_change_sim, veg_mask)  # Vegetation skill based on percent cover change
pc_vp, hss_vp, cat_vp = model_skill_categorical(veg_present_obs, veg_present_sim, veg_mask)  # Vegetation skill based on presence or absense

# Combine Skill Scores (Multi-Objective Optimization)
multiobjective_score = np.average([nmae, nmae_dl, nmae_dh], weights=[1, 1, 1])  # This is the skill score used in particle swarms optimization

# Print scores
print()
print(tabulate({
    "Scores": ["All Cells", "Foredune Location", "Foredune Elevation", "Vegetation Change", "Vegetation Presence", "Multi-Objective Score"],
    "NSE": [nse, nse_dl, nse_dh],
    "RMSE": [rmse, rmse_dl, rmse_dh],
    "NMAE": [nmae, nmae_dl, nmae_dh, None, None, multiobjective_score],
    "MASS": [mass, mass_dl, mass_dh],
    "BSS": [bss, bss_dl, bss_dh],
    "PC": [None, None, None, pc_vc, pc_vp],
    "HSS": [None, None, None, hss_vc, hss_vp],
}, headers="keys", floatfmt=(None, ".3f", ".3f", ".3f", ".3f", ".3f", ".3f", ".3f"))
)

# __________________________________________________________________________________________________________________________________
# PLOT RESULTS

xmin = 950#700  # 950
xmax = xmin + 400  # topo_start.shape[1]

# Final Elevation & Vegetation
Fig = plt.figure(figsize=(14, 7.5))
Fig.suptitle(meeb.name, fontsize=13)
topo = meeb.topo[:, xmin: xmax]
topo = np.ma.masked_where(topo <= mhw_end_sim, topo)  # Mask cells below MHW
cmap1 = routine.truncate_colormap(copy.copy(plt.colormaps["terrain"]), 0.5, 0.9)  # Truncate colormap
cmap1.set_bad(color='dodgerblue', alpha=0.5)  # Set cell color below MHW to blue
ax1 = Fig.add_subplot(211)
cax1 = ax1.matshow(topo, cmap=cmap1, vmin=0, vmax=6.0)
cbar = Fig.colorbar(cax1)
cbar.set_label('Elevation [m]', rotation=270, labelpad=20)
ax2 = Fig.add_subplot(212)
veg = meeb.veg[:, xmin: xmax]
veg = np.ma.masked_where(topo <= mhw_end_sim, veg)  # Mask cells below MHW
cmap2 = copy.copy(plt.colormaps["YlGn"])
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
    tco = topo_change_obs[:, xmin_reduc: xmax_reduc] * subaerial_mask[:, xmin_reduc: xmax_reduc]
    tcs = topo_change_sim[:, xmin_reduc: xmax_reduc] * subaerial_mask[:, xmin_reduc: xmax_reduc]
    to = topo_end_obs[:, xmin: xmax]
    ts = topo_end_sim[:, xmin: xmax]
    vco = veg_change_obs[:, xmin_reduc: xmax_reduc] * subaerial_mask[:, xmin_reduc: xmax_reduc]
    vcs = veg_change_sim[:, xmin_reduc: xmax_reduc] * subaerial_mask[:, xmin_reduc: xmax_reduc]
    vo = veg_end[:, xmin: xmax]
    vs = veg_end_sim[:, xmin: xmax]
    cat_vc = cat_vc[:, xmin_reduc: xmax_reduc]
    cat_vp = cat_vp[:, xmin_reduc: xmax_reduc]
else:
    tco = topo_change_obs[:, xmin: xmax] * subaerial_mask[:, xmin: xmax]
    tcs = topo_change_sim[:, xmin: xmax] * subaerial_mask[:, xmin: xmax]
    to = topo_end_obs[:, xmin: xmax] * subaerial_mask[:, xmin: xmax]
    ts = topo_end_sim[:, xmin: xmax] * subaerial_mask[:, xmin: xmax]
    vco = veg_change_obs[:, xmin: xmax] * subaerial_mask[:, xmin: xmax]
    vcs = veg_change_sim[:, xmin: xmax] * subaerial_mask[:, xmin: xmax]
    vo = veg_end[:, xmin: xmax] * subaerial_mask[:, xmin: xmax]
    vs = veg_end_sim[:, xmin: xmax] * subaerial_mask[:, xmin: xmax]
    cat_vc = cat_vc[:, xmin: xmax]
    cat_vp = cat_vp[:, xmin: xmax]

maxx = max(abs(np.min(tco)), abs(np.max(tco)))
maxxx = max(abs(np.min(tcs)), abs(np.max(tcs)))
maxxxx = max(maxx, maxxx)

# Topo
Fig = plt.figure(figsize=(14, 7.5))
Fig.suptitle(meeb.name, fontsize=13)
ax1 = Fig.add_subplot(221)
cax1 = ax1.matshow(to, cmap='terrain', vmin=-1, vmax=6)
plt.title("Observed")

ax2 = Fig.add_subplot(222)
cax2 = ax2.matshow(ts, cmap='terrain', vmin=-1, vmax=6)
plt.title("Simulated")

ax3 = Fig.add_subplot(223)
cax3 = ax3.matshow(tco, cmap='bwr', vmin=-maxxxx, vmax=maxxxx)
if not ResReduc:
    plt.plot(crest_loc_obs - xmin, np.arange(len(dune_crest)), 'black')
    plt.plot(crest_loc_sim - xmin, np.arange(len(dune_crest)), 'green')
    plt.legend(["Observed", "Simulated"])

ax4 = Fig.add_subplot(224)
cax4 = ax4.matshow(tcs, cmap='bwr', vmin=-maxxxx, vmax=maxxxx)
plt.tight_layout()

# Veg
Fig = plt.figure(figsize=(14, 7.5))
Fig.suptitle(meeb.name, fontsize=13)
ax1 = Fig.add_subplot(221)
ax1.matshow(vo, cmap='YlGn', vmin=0, vmax=1)
plt.title("Observed")

ax2 = Fig.add_subplot(222)
cax2 = ax2.matshow(vs, cmap='YlGn', vmin=0, vmax=1)
plt.title("Simulated")
# cbar = Fig.colorbar(cax2)
# cbar.set_label('Vegetation Cover [%]', rotation=270, labelpad=20)

ax3 = Fig.add_subplot(223)
ax3.matshow(vco, cmap='BrBG', vmin=-1, vmax=1)

ax4 = Fig.add_subplot(224)
cax4 = ax4.matshow(vcs, cmap='BrBG', vmin=-1, vmax=1)
# cbar = Fig.colorbar(cax4)
# cbar.set_label('Vegetation Change [%]', rotation=270, labelpad=20)

# Cat
catfig = plt.figure(figsize=(14, 7.5))
cmapcat = colors.ListedColormap(['green', 'yellow', 'gray', 'red'])
bounds = [0.5, 1.5, 2.5, 3.5, 4.5]
norm = colors.BoundaryNorm(bounds, cmapcat.N)
ax1cat = catfig.add_subplot(111)
cax1cat = ax1cat.matshow(cat_vp, cmap=cmapcat, norm=norm)
plt.title('Vegetation Presence/Absence')
cbar1 = plt.colorbar(cax1cat, boundaries=bounds, ticks=[1, 2, 3, 4])
cbar1.set_ticklabels(['Hit', 'False Alarm', 'Correct Reject', 'Miss'])

# Profiles: Observed vs Simulated
Fig = plt.figure(figsize=(14, 7.5))
ax1 = Fig.add_subplot(211)
profile_x = 10
plt.plot(topo_start[profile_x, xmin: xmax], 'k--')
plt.plot(topo_end_obs[profile_x, xmin: xmax], 'k')
plt.plot(topo_end_sim[profile_x, xmin: xmax], 'r')
plt.title("Profile " + str(profile_x))
ax2 = Fig.add_subplot(212)
plt.plot(np.mean(topo_start[:, xmin: xmax], axis=0), 'k--')
plt.plot(np.mean(topo_end_obs[:, xmin: xmax], axis=0), 'k')
plt.plot(np.mean(topo_end_sim[:, xmin: xmax], axis=0), 'r')
plt.legend(['Start', 'Observed', 'Simulated'])
plt.title("Average Profile")


# Animation: Elevation and Vegetation Over Time

def ani_frame(timestep):
    mhw = meeb.RSLR * timestep + MHW

    elev = meeb.topo_TS[:, xmin: xmax, timestep]  # [m]
    elev = np.ma.masked_where(elev <= mhw, elev)  # Mask cells below MHW
    cax1.set_data(elev)
    yrstr = "Year " + str(timestep * meeb.writeyear)
    text1.set_text(yrstr)

    veggie = meeb.veg_TS[:, xmin: xmax, timestep]
    veggie = np.ma.masked_where(elev <= mhw, veggie)  # Mask cells below MHW
    cax2.set_data(veggie)
    text2.set_text(yrstr)

    return cax1, cax2, text1, text2


# Set animation base figure
Fig = plt.figure(figsize=(14, 8))
topo = meeb.topo_TS[:, xmin: xmax, 0] # [m]
topo = np.ma.masked_where(topo <= MHW, topo)  # Mask cells below MHW
cmap1 = routine.truncate_colormap(copy.copy(plt.colormaps["terrain"]), 0.5, 0.9)  # Truncate colormap
cmap1.set_bad(color='dodgerblue', alpha=0.5)  # Set cell color below MHW to blue
ax1 = Fig.add_subplot(211)
cax1 = ax1.matshow(topo, cmap=cmap1, vmin=0, vmax=6.0)
cbar = Fig.colorbar(cax1)
cbar.set_label('Elevation [m]', rotation=270, labelpad=20)
timestr = "Year " + str(0 * meeb.writeyear)
text1 = plt.text(2, meeb.topo.shape[0] - 2, timestr, c='white')

veg = meeb.veg_TS[:, xmin: xmax, 0]
veg = np.ma.masked_where(topo <= MHW, veg)  # Mask cells below MHW
cmap2 = copy.copy(plt.colormaps["YlGn"])
cmap2.set_bad(color='dodgerblue', alpha=0.5)  # Set cell color below MHW to blue
ax2 = Fig.add_subplot(212)
cax2 = ax2.matshow(veg, cmap=cmap2, vmin=0, vmax=1)
cbar = Fig.colorbar(cax2)
cbar.set_label('Vegetation [%]', rotation=270, labelpad=20)
timestr = "Year " + str(0 * meeb.writeyear)
text2 = plt.text(2, meeb.veg.shape[0] - 2, timestr, c='darkblue')
plt.tight_layout()

# Create and save animation
ani = animation.FuncAnimation(Fig, ani_frame, frames=int(meeb.simulation_time_yr / meeb.writeyear) + 1, interval=300, blit=True)
ani.save("Output/SimFrames/meeb_elev.gif", dpi=150, writer="imagemagick")

plt.show()
