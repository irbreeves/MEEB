"""
Script for running MEEB simulations.

IRBR 29 July 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import copy
import gc
from tqdm import trange

import routines_meeb as routine
from meeb import MEEB


# __________________________________________________________________________________________________________________________________
# VARIABLES AND INITIALIZATIONS

# 2018
start = "Init_NCB-2200-34200_2018_PostFlorence_2m.npy"
startdate = '20181015'

# _____________________

sim_duration = 10
MHW = 0.39  # [m NAVD88]
cellsize = 2  # [m]
name = '18000-18500, 2018-2028, RSLR=9.6'  # Name of simulation
animate = False

# _____________________
# Define Coordinates of Model Domain
ymin = 18000  # Alongshore
ymax = 18500  # Alongshore
xmin = 700  # Cross-shore
xmax = 1500  # Cross-shore
plot_xmin = 0  # Cross-shore plotting
plot_xmax = 1500  # Cross-shore plotting

# Resize according to cellsize
ymin = int(ymin / cellsize)  # Alongshore
ymax = int(ymax / cellsize)  # Alongshore
xmin = int(xmin / cellsize)  # Cross-shore
xmax = int(xmax / cellsize)  # Cross-shore
plot_xmin = int(plot_xmin / cellsize)  # Cross-shore plotting
plot_xmax = int(plot_xmax / cellsize)  # Cross-shore plotting

# Load Initial Domains
Init = np.load("Input/" + start)
topo_start = Init[0, ymin: ymax, xmin: xmax].copy()
spec1_start = Init[1, ymin: ymax, xmin: xmax].copy()
spec2_start = Init[2, ymin: ymax, xmin: xmax].copy()

del Init
gc.collect()


# __________________________________________________________________________________________________________________________________
# RUN MODEL

# Create an instance of the MEEB class
meeb = MEEB(
    name=name,
    simulation_time_yr=sim_duration,
    alongshore_domain_boundary_min=ymin,
    alongshore_domain_boundary_max=ymax,
    crossshore_domain_boundary_min=xmin,
    crossshore_domain_boundary_max=xmax,
    cellsize=cellsize,
    RSLR=0.0096,
    MHW=MHW,
    init_filename=start,
    hindcast=False,
    shift_mean_storm_intensity_start=1.485,
    shift_mean_storm_intensity_end=4.199,
    storm_twl_duration_correlation=28.31,
    seeded_random_numbers=True,
    simulation_start_date=startdate,
    storm_timeseries_filename='StormTimeSeries_1979-2020_NCB-CE_Beta0pt039_BermEl1pt78.npy',  # For hindcasts
    storm_list_filename='SyntheticStorms_NCB-CE_10k_1979-2020_Beta0pt039_BermEl1pt78.npy',  # For forecasts
    init_by_file=False,
    init_elev_array=topo_start,
    init_spec1_array=spec1_start,
    init_spec2_array=spec2_start,
    save_frequency=0.1,
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
    H_flow_reduction_max=0.002,
    W_flow_reduction_max=0.02,
    # --- Shoreline --- #
    wave_asymmetry=0.6,
    wave_high_angle_fraction=0.39,
    mean_wave_height=0.98,
    mean_wave_period=6.6,
    alongshore_section_length=25,
    estimate_shoreface_parameters=True,
    shoreline_diffusivity_coefficient=0.07,
    # --- Veg --- #
)

print(meeb.name, end='\n' * 2)

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
# ASSESS MODEL RESULTS

# Topo change
topo_start_sim = meeb.topo_TS[:, :, 0].astype(np.float32)  # [m NAVDD88]
topo_end_sim = meeb.topo_TS[:, :, -1].astype(np.float32)  # [m NAVDD88]
mhw_end_sim = meeb.MHW  # [m NAVD88]
topo_change_sim = topo_end_sim - topo_start_sim  # [m]

# Veg change
veg_TS = meeb.veg_fraction_TS[:, :, 2, :] + meeb.veg_fraction_TS[:, :, 4, :] + meeb.veg_fraction_TS[:, :, 6, :] + meeb.veg_fraction_TS[:, :, 7, :]
veg_start_sim = veg_TS[:, :, 0]
veg_end_sim = veg_TS[:, :, -1]
veg_change_sim = veg_end_sim - veg_start_sim  # [m]
veg_present_sim = veg_end_sim > 0.05  # [bool]
woody_TS = meeb.veg_fraction_TS[:, :, 6, :] + meeb.veg_fraction_TS[:, :, 7, :]
woody_start_sim = woody_TS[:, :, 0]
woody_end_sim = woody_TS[:, :, -1]
woody_change_sim = woody_end_sim - woody_start_sim  # [m]
woody_present_sim = woody_end_sim > 0.05  # [bool]

# Subaerial mask
subaerial_mask = topo_end_sim > mhw_end_sim  # [bool] Mask for every cell above water

# Dune crest height
dune_crest, not_gap = routine.foredune_crest(topo_start_sim, mhw_end_sim, cellsize)
dune_crest_end, not_gap = routine.foredune_crest(topo_end_sim, mhw_end_sim, cellsize)

# __________________________________________________________________________________________________________________________________
# PLOT RESULTS

# -----------------
# Prepare For Plotting
tcs = topo_change_sim[:, plot_xmin: plot_xmax] * subaerial_mask[:, plot_xmin: plot_xmax]  # Masked sim topo change
ts = topo_end_sim[:, plot_xmin: plot_xmax] * subaerial_mask[:, plot_xmin: plot_xmax]  # Masked sim topo
vcs = veg_change_sim[:, plot_xmin: plot_xmax] * subaerial_mask[:, plot_xmin: plot_xmax]  # Masked sim veg change
vs = veg_end_sim[:, plot_xmin: plot_xmax] * subaerial_mask[:, plot_xmin: plot_xmax]  # Masked sim veg

wcs = woody_change_sim[:, plot_xmin: plot_xmax] * subaerial_mask[:, plot_xmin: plot_xmax]  # Masked sim veg change
ws = woody_end_sim[:, plot_xmin: plot_xmax] * subaerial_mask[:, plot_xmin: plot_xmax]  # Masked sim veg

cmap1 = routine.truncate_colormap(copy.copy(plt.colormaps["terrain"]), 0.5, 0.9)  # Truncate colormap
cmap1.set_bad(color='dodgerblue', alpha=0.5)  # Set cell color below MHW to blue
cmap_lim = max(abs(np.min(tcs)), abs(np.max(tcs)))

cmap2 = copy.copy(plt.colormaps["YlGn"])
cmap2.set_bad(color='dodgerblue', alpha=0.5)  # Set cell color below MHW to blue

cmap3 = copy.copy(plt.colormaps["BrBG"])
cmap3.set_bad(color='dodgerblue', alpha=0.5)  # Set cell color below MHW to blue

# -----------------
# Final Elevation & Vegetation
Fig = plt.figure(figsize=(14, 7.5))
Fig.suptitle(meeb.name, fontsize=13)
topo = meeb.topo[:, plot_xmin: plot_xmax]
topo = np.ma.masked_where(topo <= mhw_end_sim, topo)  # Mask cells below MHW
ax1 = Fig.add_subplot(211)
cax1 = ax1.matshow(topo, cmap=cmap1, vmin=0, vmax=6.0)
cbar = Fig.colorbar(cax1)
cbar.set_label('Elevation [m]', rotation=270, labelpad=20)
ax2 = Fig.add_subplot(212)
veg = meeb.veg_fraction_TS[:, plot_xmin: plot_xmax, 2, -1] + meeb.veg_fraction_TS[:, plot_xmin: plot_xmax, 4, -1] + meeb.veg_fraction_TS[:, plot_xmin: plot_xmax, 6, -1] + meeb.veg_fraction_TS[:, plot_xmin: plot_xmax, 7, -1]
veg = np.ma.masked_where(topo <= mhw_end_sim, veg)  # Mask cells below MHW
cax2 = ax2.matshow(veg, cmap=cmap2, vmin=0, vmax=1)
cbar = Fig.colorbar(cax2)
cbar.set_label('Vegetation [%]', rotation=270, labelpad=20)
plt.tight_layout()

# -----------------
# Topo Change
Fig = plt.figure(figsize=(14, 7.5))
Fig.suptitle(meeb.name, fontsize=13)
ax1 = Fig.add_subplot(211)
ax1.matshow(topo, cmap=cmap1, vmin=0, vmax=6.0)
# ax1.plot(dune_crest_end - plot_xmin, np.arange(len(dune_crest)), c='black', alpha=0.6)
ax2 = Fig.add_subplot(212)
ax2.matshow(tcs, cmap='bwr_r', vmin=-cmap_lim, vmax=cmap_lim)
plt.tight_layout()

# -----------------
# Vegetation Change
Fig = plt.figure(figsize=(14, 7.5))
Fig.suptitle(meeb.name, fontsize=13)
ax1 = Fig.add_subplot(211)
ax1.matshow(veg, cmap=cmap2, vmin=0, vmax=1)
plt.title("Simulated")
# cbar = Fig.colorbar(cax2)
# cbar.set_label('Vegetation Cover [%]', rotation=270, labelpad=20)
vcs = np.ma.masked_where(topo <= mhw_end_sim, vcs)  # Mask cells below MHW
ax2 = Fig.add_subplot(212)
ax2.matshow(vcs, cmap=cmap3, vmin=-1, vmax=1)

# Woody Vegetation Change
Fig = plt.figure(figsize=(14, 7.5))
Fig.suptitle(meeb.name, fontsize=13)
ax1 = Fig.add_subplot(211)
ws = np.ma.masked_where(topo <= mhw_end_sim, ws)  # Mask cells below MHW
ax1.matshow(ws, cmap=cmap2, vmin=0, vmax=1)
plt.title("Simulated - Woody")
# cbar = Fig.colorbar(cax2)
# cbar.set_label('Vegetation Cover [%]', rotation=270, labelpad=20)
wcs = np.ma.masked_where(topo <= mhw_end_sim, wcs)  # Mask cells below MHW
ax2 = Fig.add_subplot(212)
ax2.matshow(wcs, cmap=cmap3, vmin=-1, vmax=1)

# -----------------
# # Profiles
# Fig = plt.figure(figsize=(14, 7.5))
# ax1 = Fig.add_subplot(211)
# profile_x = int(140 / cellsize)
# plt.plot(topo_start_sim[profile_x, plot_xmin: plot_xmax], 'k--')
# plt.plot(topo_end_sim[profile_x, plot_xmin: plot_xmax], 'r')
# plt.title("Profile " + str(profile_x))
# ax2 = Fig.add_subplot(212)
# plt.plot(np.mean(topo_start_sim[:, plot_xmin: plot_xmax], axis=0), 'k--')
# plt.plot(np.mean(topo_end_sim[:, plot_xmin: plot_xmax], axis=0), 'r')
# plt.legend(['Start', 'Simulated'])
# plt.title("Average Profile")

# profx = int(140 / cellsize)
# proffig2 = plt.figure(figsize=(11, 7.5))
# for t in range(0, int(meeb.simulation_time_yr / meeb.save_frequency), 2):
#     prof = meeb.topo_TS[profx, :, t]
#     plt.plot(prof)
# prof = meeb.topo_TS[profx, :, -1]
# crest_loc_elev = prof[dune_crest_end[profx]]
# plt.scatter(dune_crest_end[profx], crest_loc_elev)
# plt.title(name + ", x =" + str(profx))

# -----------------
# Shoreline Position Over Time
Fig = plt.figure()
plt.tight_layout()
ax_1 = Fig.add_subplot(211)
plt.ylabel('Meters Cross-Shore')

color = plt.cm.viridis(np.arange(meeb.x_s_TS.shape[0]))

for it in range(meeb.x_s_TS.shape[0]):
    shoreline_it = meeb.x_s_TS[it, :] * cellsize  # Find relative ocean shoreline positions and convert y-axis to meters
    shoreline_it = np.repeat(shoreline_it, cellsize)  # Convert x-axis to meters
    if it == 0:
        ax_1.plot(shoreline_it, c=color[it], label='Start')
    if it == meeb.x_s_TS.shape[0] - 1:
        ax_1.plot(shoreline_it, c=color[it], label='End')
    else:
        ax_1.plot(shoreline_it, c=color[it], label='_')
plt.legend()

# Short and long-term shoreline change
ax_2 = Fig.add_subplot(212)
plt.xlabel('Meters Alongshore')
plt.ylabel('Shoreline Change Rate [m/yr]')
ax_2.plot(np.arange(int(meeb.x_s_TS.shape[1] * cellsize)), np.zeros([int(meeb.x_s_TS.shape[1] * cellsize)]), 'k--', alpha=0.3, label='_Zero Line')
long_term_shoreline_change_rate = (meeb.x_s_TS[-1, :] - meeb.x_s_TS[0, :]) / (meeb.x_s_TS.shape[0] / meeb.storm_iterations_per_year) * cellsize  # [m/yr]
long_term_shoreline_change_rate = np.repeat(long_term_shoreline_change_rate, cellsize)
ax_2.plot(long_term_shoreline_change_rate, 'darkred', label='Long-term Shoreline Change (Full Simulation Duration)')
if sim_duration >= 10:
    short_term_shoreline_change_rate = (meeb.x_s_TS[int(10 * meeb.storm_iterations_per_year), :] - meeb.x_s_TS[0, :]) / (meeb.x_s_TS.shape[0] / meeb.storm_iterations_per_year) * cellsize  # First decade
    short_term_shoreline_change_rate = np.repeat(short_term_shoreline_change_rate, cellsize)
    ax_2.plot(short_term_shoreline_change_rate, 'cornflowerblue', label='Short-term Shoreline Change (First Decade)')
plt.legend()

# -----------------
# # Storm Sequence
# Fig = plt.figure(figsize=(14, 7.5))
# storms = meeb.StormRecord
# twl_it = ((storms[:, 0] - 1) * meeb.iterations_per_cycle) + storms[:, 1]
# plt.scatter(twl_it, storms[:, 2])
# plt.xlabel("Simulation Iteration")
# plt.ylabel("TWL (m NAVD88)")


# -----------------
tx = np.linspace(0, sim_duration, meeb.veg_fraction_TS.shape[3])
plt.figure(figsize=(9.55, 7))
ls = 128
cs = 158
plt.plot(tx, meeb.veg_fraction_TS[ls, cs, 0, :], c='black')
plt.plot(tx, meeb.veg_fraction_TS[ls, cs, 1, :], c='turquoise')
plt.plot(tx, meeb.veg_fraction_TS[ls, cs, 2, :], c='green')
plt.plot(tx, meeb.veg_fraction_TS[ls, cs, 3, :], c='deeppink')
plt.plot(tx, meeb.veg_fraction_TS[ls, cs, 4, :], c='purple')
plt.plot(tx, meeb.veg_fraction_TS[ls, cs, 5, :], c='gold')
plt.plot(tx, meeb.veg_fraction_TS[ls, cs, 6, :], c='red')
plt.plot(tx, meeb.veg_fraction_TS[ls, cs, 7, :], c='brown')
plt.legend(['Bare', 'Grass_Seedling_1', 'Grass_1', 'Grass_Seedling_2', 'Grass_2', 'Shrub_Seedling', 'Shrub', 'Shrub_Dead'])
plt.ylabel('Fraction of Carrying Capacity')
plt.xlabel('Years')
plt.title("(" + str(ls) + ", " + str(cs) + ")")

plt.figure(figsize=(9.55, 7))
ls = 53
cs = 85
plt.plot(tx, meeb.veg_fraction_TS[ls, cs, 0, :], c='black')
plt.plot(tx, meeb.veg_fraction_TS[ls, cs, 1, :], c='turquoise')
plt.plot(tx, meeb.veg_fraction_TS[ls, cs, 2, :], c='green')
plt.plot(tx, meeb.veg_fraction_TS[ls, cs, 3, :], c='deeppink')
plt.plot(tx, meeb.veg_fraction_TS[ls, cs, 4, :], c='purple')
plt.plot(tx, meeb.veg_fraction_TS[ls, cs, 5, :], c='gold')
plt.plot(tx, meeb.veg_fraction_TS[ls, cs, 6, :], c='red')
plt.plot(tx, meeb.veg_fraction_TS[ls, cs, 7, :], c='brown')
plt.legend(['Bare', 'Grass_Seedling_1', 'Grass_1', 'Grass_Seedling_2', 'Grass_2', 'Shrub_Seedling', 'Shrub', 'Shrub_Dead'])
plt.ylabel('Fraction of Carrying Capacity')
plt.xlabel('Years')
plt.title("(" + str(ls) + ", " + str(cs) + ")")


# -----------------
# Animation: Elevation and Vegetation Over Time
def ani_frame(timestep):
    mhw = meeb.RSLR * (timestep * meeb.save_frequency) + MHW

    elev = meeb.topo_TS[:, plot_xmin: plot_xmax, timestep]  # [m]
    elev = np.ma.masked_where(elev <= mhw, elev)  # Mask cells below MHW
    cax1.set_data(elev)
    yrstr = "Year " + str(round(timestep * meeb.save_frequency, 2))
    text1.set_text(yrstr)

    veggie = veg_TS[:, plot_xmin: plot_xmax, timestep]
    veggie = np.ma.masked_where(elev <= mhw, veggie)  # Mask cells below MHW
    cax2.set_data(veggie)
    text2.set_text(yrstr)

    return cax1, cax2, text1, text2


if animate:
    # Set animation base figure
    Fig = plt.figure(figsize=(14, 8))
    topo = meeb.topo_TS[:, plot_xmin: plot_xmax, 0]  # [m]
    topo = np.ma.masked_where(topo <= MHW, topo)  # Mask cells below MHW
    cmap1 = routine.truncate_colormap(copy.copy(plt.colormaps["terrain"]), 0.5, 0.9)  # Truncate colormap
    cmap1.set_bad(color='dodgerblue', alpha=0.5)  # Set cell color below MHW to blue
    if topo.shape[0] > topo.shape[1]:
        ax1 = Fig.add_subplot(121)
    else:
        ax1 = Fig.add_subplot(211)
    cax1 = ax1.matshow(topo, cmap=cmap1, vmin=0, vmax=6.0)
    # cax1 = ax1.matshow(topo, cmap='terrain', vmin=-2, vmax=6.0)
    cbar = Fig.colorbar(cax1)
    cbar.set_label('Elevation [m]', rotation=270, labelpad=20)
    timestr = "Year " + str(round(0 * meeb.save_frequency, 2))
    text1 = plt.text(2, meeb.topo.shape[0] - 2, timestr, c='white')

    veg = veg_TS[:, plot_xmin: plot_xmax, 0]
    veg = np.ma.masked_where(topo <= MHW, veg)  # Mask cells below MHW
    cmap2 = copy.copy(plt.colormaps["YlGn"])
    cmap2.set_bad(color='dodgerblue', alpha=0.5)  # Set cell color below MHW to blue
    if topo.shape[0] > topo.shape[1]:
        ax2 = Fig.add_subplot(122)
    else:
        ax2 = Fig.add_subplot(212)
    cax2 = ax2.matshow(veg, cmap=cmap2, vmin=0, vmax=1)
    cbar = Fig.colorbar(cax2)
    cbar.set_label('Vegetation [%]', rotation=270, labelpad=20)
    timestr = "Year " + str(round(0 * meeb.save_frequency, 2))
    text2 = plt.text(2, meeb.topo.shape[0] - 2, timestr, c='white')
    plt.tight_layout()

    # Create and save animation
    ani = animation.FuncAnimation(Fig, ani_frame, frames=int(meeb.simulation_time_yr / meeb.save_frequency) + 1, interval=300, blit=True)
    c = 1
    while os.path.exists("Output/Animation/meeb_elev_" + str(c) + ".gif"):
        c += 1
    ani.save("Output/Animation/meeb_elev_" + str(c) + ".gif", dpi=150, writer="imagemagick")


plt.show()
