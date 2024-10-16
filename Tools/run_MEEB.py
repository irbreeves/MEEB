"""
Script for running MEEB simulations.

IRBR 16 October 2024
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import copy
from tqdm import trange

import routines_meeb as routine
from meeb import MEEB


# __________________________________________________________________________________________________________________________________
# VARIABLES AND INITIALIZATIONS

# # 2014
# start = "Init_NCB-NewDrum-Ocracoke_2014_PostSandy-NCFMP-Plover.npy"
# startdate = '20140406'

# # 2017
# start = "Init_NCB-NewDrum-Ocracoke_2017_PreFlorence.npy"
# startdate = '20170916'

# # 2018
# start = "Init_NCB-NewDrum-Ocracoke_2018_PostFlorence-Plover_2m.npy"
# startdate = '20181007'

# 2018
start = "Init_NCB-NewDrum-Ocracoke_2018_PostFlorence_18400-23400.npy"
startdate = '20181007'

# _____________________

sim_duration = 12
MHW = 0.39  # [m NAVD88]
cellsize = 1  # [m]
name = '250-750, 2018-2030, RSLR=0.0124'  # Name of simulation

# _____________________
# Define Coordinates of Model Domain
ymin = 250  # Alongshore
ymax = 750  # Alongshore
xmin = 0  # Cross-shore
xmax = 850  # Cross-shore
plot_xmin = 0  # Cross-shore plotting
plot_xmax = 800  # Cross-shore plotting

# Resize according to cellsize
ymin = int(ymin / cellsize)  # Alongshore
ymax = int(ymax / cellsize)  # Alongshore
xmin = int(xmin / cellsize)  # Cross-shore
xmax = int(xmax / cellsize)  # Cross-shore
plot_xmin = int(plot_xmin / cellsize)  # Cross-shore plotting
plot_xmax = int(plot_xmax / cellsize)  # Cross-shore plotting

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
    RSLR=0.0124,
    MHW=MHW,
    init_filename=start,
    hindcast=False,
    seeded_random_numbers=True,
    simulation_start_date=startdate,
    storm_timeseries_filename='StormTimeSeries_1979-2020_NCB-CE_Beta0pt039_BermEl1pt78.npy',  # For hindcasts
    storm_list_filename='SyntheticStorms_NCB-CE_10k_1979-2020_Beta0pt039_BermEl1pt78.npy',  # For forecasts
    # --- Aeolian --- #
    saltation_length=5,
    saltation_length_rand_deviation=2,
    slabheight=0.02,
    p_dep_sand=0.22,  # Q = hs * L * n * pe/pd
    p_dep_sand_VegMax=0.54,
    p_ero_sand=0.10,
    entrainment_veg_limit=0.10,
    saltation_veg_limit=0.35,
    shadowangle=12,
    repose_bare=20,
    repose_veg=30,
    wind_rose=(0.81, 0.04, 0.06, 0.09),  # (right, down, left, up)
    # --- Storms --- #
    Rin=249,
    Cs=0.0283,
    MaxUpSlope=1.5,
    marine_flux_limit=1,
    Kow=0.0001684,
    mm=1.04,
    overwash_substeps=50,
    beach_equilibrium_slope=0.022,
    swash_erosive_timescale=1.48,
    beach_substeps=25,
    flow_reduction_max_spec1=0.02,
    flow_reduction_max_spec2=0.05,
    # --- Shoreline --- #
    wave_asymmetry=0.6,
    wave_high_angle_fraction=0.39,
    mean_wave_height=0.98,
    mean_wave_period=6.6,
    alongshore_section_length=30,
    estimate_shoreface_parameters=True,
    # --- Veg --- #
    sp1_lateral_probability=0.2,
    sp2_lateral_probability=0.2,
    sp1_pioneer_probability=0.05,
    sp2_pioneer_probability=0.05,
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
topo_start_sim = meeb.topo_TS[:, :, 0]  # [m NAVDD88]
topo_end_sim = meeb.topo_TS[:, :, -1]  # [m NAVDD88]
mhw_end_sim = meeb.MHW  # [m NAVD88]
topo_change_sim = topo_end_sim - topo_start_sim  # [m]

# Veg change
veg_start_sim = meeb.veg_TS[:, :, 0]
veg_end_sim = meeb.veg_TS[:, :, -1]
veg_change_sim = veg_end_sim - veg_start_sim  # [m]
veg_present_sim = veg_end_sim > 0.05  # [bool]

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
veg = meeb.veg[:, plot_xmin: plot_xmax]
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
ax1.plot(dune_crest_end - plot_xmin, np.arange(len(dune_crest)), c='black', alpha=0.6)
ax2 = Fig.add_subplot(212)
ax2.matshow(tcs, cmap='bwr', vmin=-cmap_lim, vmax=cmap_lim)
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

# -----------------
# Profiles
Fig = plt.figure(figsize=(14, 7.5))
ax1 = Fig.add_subplot(211)
profile_x = int(140 / cellsize)
plt.plot(topo_start_sim[profile_x, plot_xmin: plot_xmax], 'k--')
plt.plot(topo_end_sim[profile_x, plot_xmin: plot_xmax], 'r')
plt.title("Profile " + str(profile_x))
ax2 = Fig.add_subplot(212)
plt.plot(np.mean(topo_start_sim[:, plot_xmin: plot_xmax], axis=0), 'k--')
plt.plot(np.mean(topo_end_sim[:, plot_xmin: plot_xmax], axis=0), 'r')
plt.legend(['Start', 'Simulated'])
plt.title("Average Profile")

profx = int(140 / cellsize)
proffig2 = plt.figure(figsize=(11, 7.5))
for t in range(0, int(meeb.simulation_time_yr / meeb.save_frequency), 2):
    prof = meeb.topo_TS[profx, :, t]
    plt.plot(prof)
prof = meeb.topo_TS[profx, :, -1]
crest_loc_elev = prof[dune_crest_end[profx]]
plt.scatter(dune_crest_end[profx], crest_loc_elev)
plt.title(name + ", x =" + str(profx))

# -----------------
# Shoreline Position Over Time
step = 1  # [yr] Plotting interval
plt.figure(figsize=(14, 7.5))
plt.xlabel('Cross-shore Position [m]')
plt.ylabel('Alongshore Position [m]')
plt.title(name)
for t in range(0, meeb.x_s_TS.shape[0], int(step * meeb.storm_iterations_per_year)):
    plt.plot(meeb.x_s_TS[t, :] * cellsize, np.arange(len(dune_crest)))
ax = plt.gca()
ax.invert_yaxis()

# -----------------
# Storm Sequence
Fig = plt.figure(figsize=(14, 7.5))
storms = meeb.StormRecord
twl_it = ((storms[:, 0] - 1) * meeb.iterations_per_cycle) + storms[:, 1]
plt.scatter(twl_it, storms[:, 2])
plt.xlabel("Simulation Iteration")
plt.ylabel("TWL (m NAVD88)")


# -----------------
# Animation: Elevation and Vegetation Over Time
def ani_frame(timestep):
    mhw = meeb.RSLR * (timestep * meeb.save_frequency) + MHW

    elev = meeb.topo_TS[:, plot_xmin: plot_xmax, timestep]  # [m]
    elev = np.ma.masked_where(elev <= mhw, elev)  # Mask cells below MHW
    cax1.set_data(elev)
    yrstr = "Year " + str(timestep * meeb.save_frequency)
    text1.set_text(yrstr)

    veggie = meeb.veg_TS[:, plot_xmin: plot_xmax, timestep]
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
else:
    ax1 = Fig.add_subplot(211)
cax1 = ax1.matshow(topo, cmap=cmap1, vmin=0, vmax=6.0)
# cax1 = ax1.matshow(topo, cmap='terrain', vmin=-2, vmax=6.0)
cbar = Fig.colorbar(cax1)
cbar.set_label('Elevation [m]', rotation=270, labelpad=20)
timestr = "Year " + str(0 * meeb.save_frequency)
text1 = plt.text(2, meeb.topo.shape[0] - 2, timestr, c='white')

veg = meeb.veg_TS[:, plot_xmin: plot_xmax, 0]
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
timestr = "Year " + str(0 * meeb.save_frequency)
text2 = plt.text(2, meeb.veg.shape[0] - 2, timestr, c='white')
plt.tight_layout()

# Create and save animation
ani = animation.FuncAnimation(Fig, ani_frame, frames=int(meeb.simulation_time_yr / meeb.save_frequency) + 1, interval=300, blit=True)
c = 1
while os.path.exists("Output/Animation/meeb_elev_" + str(c) + ".gif"):
    c += 1
ani.save("Output/Animation/meeb_elev_" + str(c) + ".gif", dpi=150, writer="imagemagick")


plt.show()
