"""
Script for running MEEB simulations.

IRBR 11 March 2024
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

# 2018
start = "Init_NCB-NewDrum-Ocracoke_2018_PostFlorence-Plover.npy"
startdate = '20181007'

# _____________________

sim_duration = 32
MHW = 0.39  # [m NAVD88]
name = '19000-19500, 2018-2050'  # Name of simulation

# _____________________
# Define Coordinates of Model Domain
ymin = 19000  # Alongshore
ymax = 19500  # Alongshore
xmin = 900  # Cross-shore
xmax = xmin + 900  # Cross-shore
plot_xmin = 0  # Cross-shore plotting
plot_xmax = plot_xmin + 600  # Cross-shore plotting


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
    RSLR=0.006,
    MHW=MHW,
    init_filename=start,
    hindcast=False,
    seeded_random_numbers=True,
    simulation_start_date=startdate,
    storm_timeseries_filename='StormTimeSeries_1979-2020_NCB-CE_Beta0pt039_BermEl1pt78.npy',  # For hindcasts
    storm_list_filename='SyntheticStorms_NCB-CE_10k_1979-2020_Beta0pt039_BermEl1pt78.npy',  # For forecasts
    # --- Aeolian --- #
    jumplength=5,
    slabheight=0.02,
    p_dep_sand=0.36,  # Q = hs * L * n * pe/pd
    p_dep_sand_VegMax=0.60,
    p_ero_sand=0.13,
    entrainment_veg_limit=0.37,
    saltation_veg_limit=0.37,
    shadowangle=5,
    repose_bare=20,
    repose_veg=30,
    wind_rose=(0.83, 0.02, 0.12, 0.03),  # (right, down, left, up)
    # --- Storms --- #
    Rin=213,
    Cx=36,
    MaxUpSlope=1.57,
    Kow=0.0000501,
    mm=1.02,
    beach_equilibrium_slope=0.027,
    swash_transport_coefficient=0.001,
    wave_period_storm=9.4,
    flow_reduction_max_spec1=0.02,
    flow_reduction_max_spec2=0.05,
    # --- Shoreline --- #
    wave_asymetry=0.6,
    wave_high_angle_fraction=0.39,
    mean_wave_height=0.98,
    mean_wave_period=6.6,
    alongshore_section_length=25,
    estimate_shoreface_parameters=True,
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
dune_crest, not_gap = routine.foredune_crest(topo_start_sim, mhw_end_sim)


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
profile_x = 10
plt.plot(topo_start_sim[profile_x, plot_xmin: plot_xmax], 'k--')
plt.plot(topo_end_sim[profile_x, plot_xmin: plot_xmax], 'r')
plt.title("Profile " + str(profile_x))
ax2 = Fig.add_subplot(212)
plt.plot(np.mean(topo_start_sim[:, plot_xmin: plot_xmax], axis=0), 'k--')
plt.plot(np.mean(topo_end_sim[:, plot_xmin: plot_xmax], axis=0), 'r')
plt.legend(['Start', 'Simulated'])
plt.title("Average Profile")

# -----------------
# Shoreline Position Over Time
step = 1  # [yr] Plotting interval
plt.figure(figsize=(14, 7.5))
plt.xlabel('Cross-shore Position [m]')
plt.ylabel('Alongshore Position [m]')
plt.title(name)
for t in range(0, meeb.x_s_TS.shape[0], int(step * meeb.storm_iterations_per_year)):
    plt.plot(meeb.x_s_TS[t, :], np.arange(len(dune_crest)))
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
    mhw = meeb.RSLR * timestep + MHW

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
