"""
Script for testing MEEB hindcast simulations.

Runs a hindcast simulation and calculates fitess scores for morphologic and ecologic change between simulated and observed.

IRBR 4 April 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as colors
import routines_meeb as routine
import copy
import os
from tabulate import tabulate
from tqdm import trange

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

# 2014 - 2017
start = "Init_NCB-NewDrum-Ocracoke_2014_PostSandy_NCFMP-Planet_2m_HighDensity.npy"
stop = "Init_NCB-NewDrum-Ocracoke_2017_PreFlorence_2m.npy"
startdate = '20140406'
hindcast_duration = 3.44

# # 2014 - 2018
# start = "Init_NCB-NewDrum-Ocracoke_2014_PostSandy-NCFMP-Plover.npy"
# stop = "Init_NCB-NewDrum-Ocracoke_2018_PostFlorence-Plover.npy"
# startdate = '20140406'
# hindcast_duration = 4.5

# # 2017 - 2018
# start = "Init_NCB-NewDrum-Ocracoke_2017_PreFlorence.npy"
# stop = "Init_NCB-NewDrum-Ocracoke_2018_PostFlorence-Plover.npy"
# startdate = '20170916'
# hindcast_duration = 1.06


# _____________________
# INPUT

# Define Coordinates of Domain
ymin = 4000  # 22400 8400 4000 20500   # --> 8, 22, 20, 4
ymax = 4500  # 22900 8900 4500 21000
xmin = 600   # 900 700 600 900
xmax = xmin + 800

# Define Cross-Shore Limits for Plotting
plot_xmin = 0
plot_xmax = plot_xmin + 900

# Define Cross-Shore Limits for Skill Score Mask
mask_xmin = 720 - xmin  # 1070  # 930  # 720  # 1070
mask_xmax = 775 - xmin  # 1150  # 990  # 775  # 1120

rslr = 0.006  # [m/yr]
MHW = 0.39  # [m NAVD88]
ResReduc = False  # Option to reduce raster resolution for skill assessment
reduc = 5  # Raster resolution reduction factor
cellsize = 2  # [m]

name = str(ymin) + ' - ' + str(ymax) + ', RSLR=6'

# _____________________
# LOAD INITIAL DOMAINS

# Resize According to Cellsize
ymin = int(ymin / cellsize)  # Alongshore
ymax = int(ymax / cellsize)  # Alongshore
xmin = int(xmin / cellsize)  # Cross-shore
xmax = int(xmax / cellsize)  # Cross-shore
plot_xmin = int(plot_xmin / cellsize)  # Cross-shore plotting
plot_xmax = int(plot_xmax / cellsize)  # Cross-shore plotting
mask_xmin = int(mask_xmin / cellsize)
mask_xmax = int(mask_xmax / cellsize)

# Initial Observed Topo
Init = np.load("Input/" + start)
# Final Observed
End = np.load("Input/" + stop)

# Transform Initial Observed Topo
topo_i = Init[0, ymin: ymax, xmin: xmax]  # [m]
topo_start = copy.deepcopy(topo_i)  # [m] Initialise the topography

# Transform Final Observed Topo
topo_e = End[0, ymin: ymax, xmin: xmax]  # [m]
topo_end_obs = copy.deepcopy(topo_e)  # [m] Initialise the topography

# Set Veg Domain
spec1_i = Init[1, ymin: ymax, xmin: xmax]
spec2_i = Init[2, ymin: ymax, xmin: xmax]
veg_start = spec1_i + spec2_i  # Determine the initial cumulative vegetation effectiveness
veg_start[veg_start > 1] = 1  # Cumulative vegetation effectiveness cannot be negative or larger than one
veg_start[veg_start < 0] = 0

spec1_e = End[1, ymin: ymax, xmin: xmax]
spec2_e = End[2, ymin: ymax, xmin: xmax]
veg_end = spec1_e + spec2_e  # Determine the initial cumulative vegetation effectiveness
veg_end[veg_end > 1] = 1  # Cumulative vegetation effectiveness cannot be negative or larger than one
veg_end[veg_end < 0] = 0


# __________________________________________________________________________________________________________________________________
# RUN MODEL

# Create an instance of the MEEB class
meeb = MEEB(
    name=name,
    simulation_time_yr=hindcast_duration,
    alongshore_domain_boundary_min=ymin,
    alongshore_domain_boundary_max=ymax,
    crossshore_domain_boundary_min=xmin,
    crossshore_domain_boundary_max=xmax,
    cellsize=cellsize,
    RSLR=rslr,
    MHW=MHW,
    init_filename=start,
    hindcast=True,
    seeded_random_numbers=True,
    simulation_start_date=startdate,
    storm_timeseries_filename='StormTimeSeries_1979-2020_NCB-CE_Beta0pt039_BermEl1pt78.npy',
    # --- Aeolian --- #
    saltation_length=2,
    saltation_length_rand_deviation=1,
    slabheight=0.02,
    p_dep_sand=0.09,  # Q = hs * L * n * pe/pd
    p_dep_sand_VegMax=0.17,
    p_ero_sand=0.08,
    entrainment_veg_limit=0.09,
    saltation_veg_limit=0.37,
    repose_threshold=0.37,
    shadowangle=12,
    repose_bare=20,
    repose_veg=30,
    wind_rose=(0.91, 0.04, 0.01, 0.04),  # (right, down, left, up)
    groundwater_depth=0.4,
    # --- Storms --- #
    Rin=245,
    Cs=0.0235,
    MaxUpSlope=1.5,
    marine_flux_limit=1,
    Kow=0.0003615,
    mm=1.05,
    overwash_substeps=25,
    beach_equilibrium_slope=0.021,
    swash_erosive_timescale=1.51,
    beach_substeps=1,
    flow_reduction_max_spec1=0.002,
    flow_reduction_max_spec2=0.02,
    # --- Shoreline --- #
    wave_asymmetry=0.6,
    wave_high_angle_fraction=0.39,
    mean_wave_height=0.98,
    mean_wave_period=6.6,
    alongshore_section_length=25,
    estimate_shoreface_parameters=True,
    shoreline_diffusivity_coefficient=0.07,
    # --- Veg --- #
    sp1_lateral_probability=0.2,
    sp2_lateral_probability=0.2,
    sp1_pioneer_probability=0.05,
    sp2_pioneer_probability=0.03,

    # MY GRASS
    sp1_a=-1.2,
    sp1_b=-0.067,  # Mullins et al. (2019)
    sp1_c=0.5,
    sp1_d=1.2,
    sp1_e=2.1,
    sp1_peak=0.2,
    # MY SHRUB
    sp2_a=-1.0,
    sp2_b=-0.2,  # Day et al. (199?)
    sp2_c=0.0,
    sp2_d=0.2,
    sp2_e=2.1,
    sp2_peak=0.05,
)

print(meeb.name)
print()

# Loop through time
with trange(int(meeb.iterations)) as t:
    for time_step in t:
        # Run time step
        meeb.update(time_step)
        # Update progress bar
        t.set_postfix({'Year': "{:.2f}".format((time_step + 1) / meeb.iterations_per_cycle) + '/' + "{:.2f}".format(meeb.simulation_time_yr)})
        t.update()

print()


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
subaerial_mask = np.logical_and(topo_end_sim > mhw_end_sim, topo_end_obs > mhw_end_sim)  # [bool] Mask for every cell above water

# Beach mask
dune_crest, not_gap = routine.foredune_crest(topo_start, mhw_end_sim, cellsize)
beach_duneface_mask = np.zeros(topo_end_sim.shape)
for l in range(topo_start.shape[0]):
    beach_duneface_mask[l, :dune_crest[l]] = True
beach_duneface_mask = np.logical_and(beach_duneface_mask, subaerial_mask)  # [bool] Map of every cell seaward of dune crest

# Dune crest locations and heights
crest_loc_obs_start, not_gap_obs_start = routine.foredune_crest(topo_start, mhw_end_sim, cellsize)
crest_loc_obs, not_gap_obs = routine.foredune_crest(topo_end_obs, mhw_end_sim, cellsize)
crest_loc_sim, not_gap_sim = routine.foredune_crest(topo_end_sim, mhw_end_sim, cellsize)
crest_loc_change_obs = crest_loc_obs - crest_loc_obs_start
crest_loc_change_sim = crest_loc_sim - crest_loc_obs_start

crest_height_obs_start = topo_start[np.arange(topo_start.shape[0]), crest_loc_obs_start]
crest_height_obs = topo_end_obs[np.arange(topo_end_obs.shape[0]), crest_loc_obs]
crest_height_sim = topo_end_sim[np.arange(topo_end_obs.shape[0]), crest_loc_sim]
crest_height_change_obs = crest_height_obs - crest_height_obs_start
crest_height_change_sim = crest_height_sim - crest_height_obs_start

# Limit interior in analysis by elevation
elev_mask = topo_end_sim > 2.0  # [bool] Mask for every cell above water

# Limit interior in analysis to dunefield (pre-defined cross-shore range)
dunefield_mask = subaerial_mask.copy()
dunefield_mask[:, :mask_xmin] = False
dunefield_mask[:, mask_xmax:] = False

# Limit dune crest analysis to cells with dunes (no dune gaps)
dune_mask = np.logical_and(np.logical_and(not_gap_obs_start, not_gap_obs), not_gap_sim)  # [bool] Dune gap cells set to False

# Choose masks
mask = dunefield_mask.copy()
veg_mask = mask.copy()

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
nse_dh, rmse_dh, nmae_dh, mass_dh, bss_dh = model_skill(crest_height_obs, crest_height_sim, crest_height_obs_start, dune_mask)  # Foredune elevation

pc_vc, hss_vc, cat_vc = model_skill_categorical(veg_change_obs, veg_change_sim, veg_mask)  # Vegetation skill based on percent cover change
pc_vp, hss_vp, cat_vp = model_skill_categorical(veg_present_obs, veg_present_sim, veg_mask)  # Vegetation skill based on presence or absense

# Combine Skill Scores (Multi-Objective Optimization)
multiobjective_score = np.average([bss, bss_dh])

# Print scores
print()
print(tabulate({
    "Scores": ["All Cells", "Foredune Location", "Foredune Elevation", "Vegetation Change", "Vegetation Presence", "Multi-Objective Score"],
    "NSE": [nse, nse_dl, nse_dh],
    "RMSE": [rmse, rmse_dl, rmse_dh],
    "NMAE": [nmae, nmae_dl, nmae_dh],
    "MASS": [mass, mass_dl, mass_dh],
    "BSS": [bss, bss_dl, bss_dh, None, None, multiobjective_score],
    "PC": [None, None, None, pc_vc, pc_vp],
    "HSS": [None, None, None, hss_vc, hss_vp],
}, headers="keys", floatfmt=(None, ".3f", ".3f", ".3f", ".3f", ".3f", ".3f", ".3f"))
)

# __________________________________________________________________________________________________________________________________
# PLOT RESULTS

# -----------------
# Prepare For Plotting

if ResReduc:
    # Reduced Resolutions
    ymin_reduc = int(plot_xmin / reduc)
    ymax_reduc = int(plot_xmax / reduc)
    tco = topo_change_obs[:, ymin_reduc: ymax_reduc] * subaerial_mask[:, ymin_reduc: ymax_reduc]
    tcs = topo_change_sim[:, ymin_reduc: ymax_reduc] * subaerial_mask[:, ymin_reduc: ymax_reduc]
    to = topo_end_obs[:, plot_xmin: plot_xmax]
    ts = topo_end_sim[:, plot_xmin: plot_xmax]
    vco = veg_change_obs[:, ymin_reduc: ymax_reduc] * subaerial_mask[:, ymin_reduc: ymax_reduc]
    vcs = veg_change_sim[:, ymin_reduc: ymax_reduc] * subaerial_mask[:, ymin_reduc: ymax_reduc]
    vo = veg_end[:, plot_xmin: plot_xmax]
    vs = veg_end_sim[:, plot_xmin: plot_xmax]
    cat_vc = cat_vc[:, ymin_reduc: ymax_reduc]
    cat_vp = cat_vp[:, ymin_reduc: ymax_reduc]
else:
    tco = topo_change_obs[:, plot_xmin: plot_xmax] * subaerial_mask[:, plot_xmin: plot_xmax]
    tcs = topo_change_sim[:, plot_xmin: plot_xmax] * subaerial_mask[:, plot_xmin: plot_xmax]
    to = topo_end_obs[:, plot_xmin: plot_xmax] * subaerial_mask[:, plot_xmin: plot_xmax]
    ts = topo_end_sim[:, plot_xmin: plot_xmax] * subaerial_mask[:, plot_xmin: plot_xmax]
    vco = veg_change_obs[:, plot_xmin: plot_xmax] * subaerial_mask[:, plot_xmin: plot_xmax]
    vcs = veg_change_sim[:, plot_xmin: plot_xmax] * subaerial_mask[:, plot_xmin: plot_xmax]
    vo = veg_end[:, plot_xmin: plot_xmax] * subaerial_mask[:, plot_xmin: plot_xmax]
    vs = veg_end_sim[:, plot_xmin: plot_xmax] * subaerial_mask[:, plot_xmin: plot_xmax]
    cat_vc = cat_vc[:, plot_xmin: plot_xmax]
    cat_vp = cat_vp[:, plot_xmin: plot_xmax]

# Set topo change colormap limits
max_change_obs = max(abs(np.min(tco)), abs(np.max(tco)))  # [m]
max_change_sim = max(abs(np.min(tcs)), abs(np.max(tcs)))  # [m]
cmap_lim = 1.5  # max(max_change_obs, max_change_sim)

# Set topo colormap
cmap1 = routine.truncate_colormap(copy.copy(plt.colormaps.get_cmap("terrain")), 0.5, 0.9)  # Truncate colormap
cmap1.set_bad(color='dodgerblue', alpha=0.5)  # Set cell color below MHW to blue

# Combine species
veg_TS = meeb.spec1_TS + meeb.spec2_TS

# -----------------
# Final Elevation & Vegetation
Fig = plt.figure(figsize=(14, 7.5))
Fig.suptitle(meeb.name, fontsize=13)
topo = meeb.topo[:, plot_xmin: plot_xmax]
topo = np.ma.masked_where(topo <= mhw_end_sim, topo)  # Mask cells below MHW
if topo.shape[0] > topo.shape[1]:
    ax1 = Fig.add_subplot(121)
    ax2 = Fig.add_subplot(122)
else:
    ax1 = Fig.add_subplot(211)
    ax2 = Fig.add_subplot(212)
cax1 = ax1.matshow(topo, cmap=cmap1, vmin=0, vmax=6.0)
cbar = Fig.colorbar(cax1)
cbar.set_label('Elevation [m]', rotation=270, labelpad=20)
veg = meeb.veg[:, plot_xmin: plot_xmax]
veg = np.ma.masked_where(topo <= mhw_end_sim, veg)  # Mask cells below MHW
cmap2 = copy.copy(plt.colormaps["YlGn"])
cmap2.set_bad(color='dodgerblue', alpha=0.5)  # Set cell color below MHW to blue
cax2 = ax2.matshow(veg, cmap=cmap2, vmin=0, vmax=1)
cbar = Fig.colorbar(cax2)
cbar.set_label('Vegetation [%]', rotation=270, labelpad=20)
plt.tight_layout()

# -----------------
# Topo Change, Observed vs Simulated
Fig = plt.figure(figsize=(14, 7.5))
Fig.suptitle(meeb.name, fontsize=13)
ax1 = Fig.add_subplot(221)
to = np.ma.masked_where(to < MHW, to)  # Mask cells below MHW
cax1 = ax1.matshow(to, cmap=cmap1, vmin=0, vmax=6)
plt.title("Observed")

ax2 = Fig.add_subplot(222)
ts = np.ma.masked_where(ts < MHW, ts)  # Mask cells below MHW
cax2 = ax2.matshow(ts, cmap=cmap1, vmin=0, vmax=6)
plt.title("Simulated")

ax3 = Fig.add_subplot(223)
cax3 = ax3.matshow(tco, cmap='bwr_r', vmin=-cmap_lim, vmax=cmap_lim)
# if not ResReduc:
#     plt.plot(crest_loc_obs - plot_xmin, np.arange(len(dune_crest)), 'black')
#     plt.plot(crest_loc_sim - plot_xmin, np.arange(len(dune_crest)), 'green')
#     plt.legend(["Observed", "Simulated"])

ax4 = Fig.add_subplot(224)
cax4 = ax4.matshow(tcs, cmap='bwr_r', vmin=-cmap_lim, vmax=cmap_lim)
plt.tight_layout()

# -----------------
# Vegetation Change
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

# -----------------
# Categorical Vegetation Change
catfig = plt.figure(figsize=(14, 7.5))
cmapcat = colors.ListedColormap(['green', 'yellow', 'gray', 'red'])
bounds = [0.5, 1.5, 2.5, 3.5, 4.5]
norm = colors.BoundaryNorm(bounds, cmapcat.N)
ax1cat = catfig.add_subplot(111)
cax1cat = ax1cat.matshow(cat_vp, cmap=cmapcat, norm=norm)
plt.title('Vegetation Presence/Absence')
cbar1 = plt.colorbar(cax1cat, boundaries=bounds, ticks=[1, 2, 3, 4])
cbar1.set_ticklabels(['Hit', 'False Alarm', 'Correct Reject', 'Miss'])

# -----------------
# Profiles: Observed vs Simulated
Fig = plt.figure(figsize=(14, 7.5))
ax1 = Fig.add_subplot(211)
profile_x = int(10 / cellsize)
plt.plot(topo_start[profile_x, plot_xmin: plot_xmax], 'k--')
plt.plot(topo_end_obs[profile_x, plot_xmin: plot_xmax], 'k')
plt.plot(topo_end_sim[profile_x, plot_xmin: plot_xmax], 'r')
plt.title("Profile " + str(profile_x))
ax2 = Fig.add_subplot(212)
plt.plot(np.mean(topo_start[:, plot_xmin: plot_xmax], axis=0), 'k--')
plt.plot(np.mean(topo_end_obs[:, plot_xmin: plot_xmax], axis=0), 'k')
plt.plot(np.mean(topo_end_sim[:, plot_xmin: plot_xmax], axis=0), 'r')
plt.legend(['Start', 'Observed', 'Simulated'])
plt.title("Average Profile")

# -----------------
# Shoreline Position Over Time
step = 1  # [yr] Plotting interval
plt.figure(figsize=(14, 7.5))
plt.xlabel('Cross-shore Position [m]')
plt.ylabel('Alongshore Position [m]')
for t in range(0, meeb.x_s_TS.shape[0], int(step * meeb.storm_iterations_per_year)):
    plt.plot(meeb.x_s_TS[t, :] * cellsize, np.arange(len(dune_crest)))
ax = plt.gca()
ax.invert_yaxis()


# -----------------
# Animation: Elevation and Vegetation Over Time
def ani_frame(timestep):
    mhw = meeb.RSLR * (timestep * meeb.save_frequency) + MHW

    elev = meeb.topo_TS[:, plot_xmin: plot_xmax, timestep]  # [m]
    elev = np.ma.masked_where(elev <= mhw, elev)  # Mask cells below MHW
    cax1.set_data(elev)
    yrstr = "Year " + str(timestep * meeb.save_frequency)
    text1.set_text(yrstr)

    veggie = veg_TS[:, plot_xmin: plot_xmax, timestep]
    veggie = np.ma.masked_where(elev <= mhw, veggie)  # Mask cells below MHW
    cax2.set_data(veggie)
    text2.set_text(yrstr)

    return cax1, cax2, text1, text2


# Set animation base figure
Fig = plt.figure(figsize=(14, 8))
topo = meeb.topo_TS[:, plot_xmin: plot_xmax, 0]  # [m]
topo = np.ma.masked_where(topo <= MHW, topo)  # Mask cells below MHW
cmap1 = routine.truncate_colormap(copy.copy(plt.colormaps["terrain"]), 0.5, 0.9)  # Truncate colormap
cmap1.set_bad(color='dodgerblue', alpha=0.5)  # Set cell color below MHW to blue
if topo.shape[0] > topo.shape[1]:
    ax1 = Fig.add_subplot(121)
    ax2 = Fig.add_subplot(122)
else:
    ax1 = Fig.add_subplot(211)
    ax2 = Fig.add_subplot(212)
cax1 = ax1.matshow(topo, cmap=cmap1, vmin=0, vmax=6.0)
cbar = Fig.colorbar(cax1)
cbar.set_label('Elevation [m]', rotation=270, labelpad=20)
timestr = "Year " + str(0 * meeb.save_frequency)
text1 = plt.text(2, meeb.topo.shape[0] - 2, timestr, c='white')

veg = veg_TS[:, plot_xmin: plot_xmax, 0]
veg = np.ma.masked_where(topo <= MHW, veg)  # Mask cells below MHW
cmap2 = copy.copy(plt.colormaps["YlGn"])
cmap2.set_bad(color='dodgerblue', alpha=0.5)  # Set cell color below MHW to blue
cax2 = ax2.matshow(veg, cmap=cmap2, vmin=0, vmax=1)
cbar = Fig.colorbar(cax2)
cbar.set_label('Vegetation [%]', rotation=270, labelpad=20)
timestr = "Year " + str(0 * meeb.save_frequency)
text2 = plt.text(2, meeb.veg.shape[0] - 2, timestr, c='darkblue')
plt.tight_layout()

# Create and save animation
ani = animation.FuncAnimation(Fig, ani_frame, frames=int(meeb.simulation_time_yr / meeb.save_frequency) + 1, interval=300, blit=True)
c = 1
while os.path.exists("Output/Animation/meeb_elev_" + str(c) + ".gif"):
    c += 1
ani.save("Output/Animation/meeb_elev_" + str(c) + ".gif", dpi=150, writer="imagemagick")

plt.show()
