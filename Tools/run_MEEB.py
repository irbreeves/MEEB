"""
Script for running MEEB simulations.

IRBR 26 October 2023
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import copy
import time

import routines_meeb
import routines_meeb as routine
from meeb import MEEB


# __________________________________________________________________________________________________________________________________
# VARIABLES AND INITIALIZATIONS

# # 2004
# start = "Init_NCB-NewDrum-Ocracoke_2004_PostIsabel.npy"
# startdate = '20040716'

# # 2009
# start = "Init_NCB-NewDrum-Ocracoke_2009_PreIrene.npy"
# startdate = '20090824'

# # 2011
# start = "Init_NCB-NewDrum-Ocracoke_2011_PostIrene.npy"
# startdate = '20110829'

# # 2012
# start = "Init_NCB-NewDrum-Ocracoke_2012_PostSandyUSGS_MinimalThin.npy"
# startdate = '20121129'

# # 2014
# start = "Init_NCB-NewDrum-Ocracoke_2014_PostSandy-NCFMP-Plover.npy"
# startdate = '20140406'

# # 2016
# start = "Init_NCB-NewDrum-Ocracoke_2016_PostMatthew.npy"
# startdate = '20161012'

# # 2017
# start = "Init_NCB-NewDrum-Ocracoke_2017_PreFlorence.npy"
# startdate = '20170916'

# 2018
start = "Init_NCB-NewDrum-Ocracoke_2018_PostFlorence-Plover.npy"
startdate = '20181007'


sim_duration = 8


# _____________________
# Initial Conditions

# Define Alongshore Coordinates of Domain
ymin = 6500  # Alongshore
ymax = 6700  # Alongshore
xmin = 700  # Cross-shore   900
xmax = 1400  # Cross-shore  1500
MHW = 0.39  # [m NAVD88]

name = '6500-6700, 8 yr, RSLR8'

# Load Initial Domains
Init = np.load("Input/" + start)
# Initial Topo
topo_start = Init[0, ymin: ymax, :]

# Set Veg Domain
spec1_i = Init[1, ymin: ymax, :]
spec2_i = Init[2, ymin: ymax, :]
veg_start = spec1_i + spec2_i  # Determine the initial cumulative vegetation effectiveness


# __________________________________________________________________________________________________________________________________
# RUN MODEL

start_time = time.time()  # Record time at start of simulation

# Create an instance of the MEEB class
meeb = MEEB(
    name=name,
    simulation_time_yr=sim_duration,
    alongshore_domain_boundary_min=ymin,
    alongshore_domain_boundary_max=ymax,
    RSLR=0.008,
    MHW=MHW,
    init_filename=start,
    hindcast=False,
    seeded_random_numbers=True,
    simulation_start_date=startdate,
    storm_timeseries_filename='StormTimeSeries_1979-2020_NCB-CE_Beta0pt039_BermEl1pt78.npy',
    storm_list_filename='SyntheticStorms_NCB-CE_10k_1979-2020_Beta0pt039_BermEl1pt78.npy',
    # --- Aeolian --- #
    jumplength=5,
    slabheight=0.02,
    p_dep_sand=0.10,  # Q = hs * L * n * pe/pd
    p_dep_sand_VegMax=0.22,
    p_ero_sand=0.16,
    entrainment_veg_limit=0.11,
    saltation_veg_limit=0.20,
    shadowangle=8,
    repose_bare=20,
    repose_veg=30,
    wind_rose=(0.55, 0.05, 0.22, 0.18),  # (right, down, left, up)
    # --- Storms --- #
    Rin_ru=246,
    Cx=27,
    MaxUpSlope=0.63,
    K_ru=0.0000622,
    substep_ru=7,
    beach_equilibrium_slope=0.039,
    swash_transport_coefficient=1e-3,
    beach_substeps=10,
    flow_reduction_max_spec1=0.17,
    flow_reduction_max_spec2=0.44,
    # --- Shoreline --- #
    wave_asymetry=0.6,
    wave_high_angle_fraction=0.39,
    mean_wave_height=0.98,
    mean_wave_period=6.6,
    alongshore_section_length=25,
    estimate_shoreface_parameters=True,
    # --- Veg --- #
    # sp1_c=1.20,
    # sp2_c=-0.47,
    # sp1_peak=0.307,
    # sp2_peak=0.148,
    # lateral_probability=0.34,
    # pioneer_probability=0.11,
    # Spec1_elev_min=0.60,
    # Spec2_elev_min=0.13,
    effective_veg_sigma=3,
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
dune_crest = routine.foredune_crest(topo_start_sim, mhw_end_sim)


# __________________________________________________________________________________________________________________________________
# PLOT RESULTS

# -----------------
# Prepare For Plotting
tcs = topo_change_sim[:, xmin: xmax] * subaerial_mask[:, xmin: xmax]
ts = topo_end_sim[:, xmin: xmax] * subaerial_mask[:, xmin: xmax]
vcs = veg_change_sim[:, xmin: xmax] * subaerial_mask[:, xmin: xmax]
vs = veg_end_sim[:, xmin: xmax] * subaerial_mask[:, xmin: xmax]

maxxxx = max(abs(np.min(tcs)), abs(np.max(tcs)))

MEEB_elevation = routines_meeb.get_MEEB_colormap()

# -----------------
# Final Elevation & Vegetation
Fig = plt.figure(figsize=(14, 7.5))
Fig.suptitle(meeb.name, fontsize=13)
topo = meeb.topo[:, xmin: xmax]
topo = np.ma.masked_where(topo <= mhw_end_sim, topo)  # Mask cells below MHW
cmap1 = routine.truncate_colormap(copy.copy(plt.colormaps["terrain"]), 0.5, 0.9)  # Truncate colormap
cmap1.set_bad(color='dodgerblue', alpha=0.5)  # Set cell color below MHW to blue
ax1 = Fig.add_subplot(211)
cax1 = ax1.matshow(topo, cmap=cmap1, vmin=0, vmax=6.0)
# cax1 = ax1.matshow(topo, cmap='terrain', vmin=-2, vmax=6.0)
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

# # -----------------
# # Topo Change
# Fig = plt.figure(figsize=(14, 7.5))
# Fig.suptitle(meeb.name, fontsize=13)
#
# ax1 = Fig.add_subplot(211)
# ax1.matshow(ts, cmap='terrain', vmin=-1, vmax=6)
#
# ax2 = Fig.add_subplot(212)
# ax2.matshow(tcs, cmap='bwr', vmin=-maxxxx, vmax=maxxxx)
# plt.tight_layout()

# # -----------------
# # Vegetation Change
# Fig = plt.figure(figsize=(14, 7.5))
# Fig.suptitle(meeb.name, fontsize=13)
# ax1 = Fig.add_subplot(211)
# ax1.matshow(vs, cmap='YlGn', vmin=0, vmax=1)
# plt.title("Simulated")
# # cbar = Fig.colorbar(cax2)
# # cbar.set_label('Vegetation Cover [%]', rotation=270, labelpad=20)
# ax2 = Fig.add_subplot(212)
# ax2.matshow(vcs, cmap='BrBG', vmin=-1, vmax=1)

# # -----------------
# # Profiles
# Fig = plt.figure(figsize=(14, 7.5))
# ax1 = Fig.add_subplot(211)
# profile_x = 10
# plt.plot(topo_start_sim[profile_x, xmin: xmax], 'k--')
# plt.plot(topo_end_sim[profile_x, xmin: xmax], 'r')
# plt.title("Profile " + str(profile_x))
# ax2 = Fig.add_subplot(212)
# plt.plot(np.mean(topo_start_sim[:, xmin: xmax], axis=0), 'k--')
# plt.plot(np.mean(topo_end_sim[:, xmin: xmax], axis=0), 'r')
# plt.legend(['Start', 'Simulated'])
# plt.title("Average Profile")

# -----------------
# Shoreline Position Over Time
step = 1  # [yr] Plotting interval
plt.figure(figsize=(14, 7.5))
plt.xlabel('Cross-shore Position [m]')
plt.ylabel('Alongshore Position [m]')
plt.title(name)
for t in range(0, meeb.x_s_TS.shape[0], int(step * meeb.storm_iterations_per_year)):
    plt.plot(meeb.x_s_TS[t, :], np.arange(len(dune_crest)), 'k')
    # plt.plot(meeb.x_t_TS[t, :], np.arange(len(dune_crest)), 'r')
ax = plt.gca()
ax.invert_yaxis()


# -----------------
# Animation: Elevation and Vegetation Over Time
def ani_frame(timestep):
    mhw = meeb.RSLR * timestep + MHW

    elev = meeb.topo_TS[:, xmin: xmax, timestep]  # [m]
    elev = np.ma.masked_where(elev <= mhw, elev)  # Mask cells below MHW
    cax1.set_data(elev)
    yrstr = "Year " + str(timestep * meeb.save_frequency)
    text1.set_text(yrstr)

    veggie = meeb.veg_TS[:, xmin: xmax, timestep]
    veggie = np.ma.masked_where(elev <= mhw, veggie)  # Mask cells below MHW
    cax2.set_data(veggie)
    text2.set_text(yrstr)

    return cax1, cax2, text1, text2


# Set animation base figure
Fig = plt.figure(figsize=(14, 8))
topo = meeb.topo_TS[:, xmin: xmax, 0]  # [m]
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

veg = meeb.veg_TS[:, xmin: xmax, 0]
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
