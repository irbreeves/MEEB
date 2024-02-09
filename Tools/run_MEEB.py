"""
Script for running MEEB simulations.

IRBR 8 February 2024
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


sim_duration = 10


# _____________________
# Initial Conditions

# Define Alongshore Coordinates of Domain
ymin = 21000  # Alongshore
ymax = 21500  # Alongshore
xmin = 900  # Cross-shore   1000
xmax = xmin + 600  # Cross-shore  1500
MHW = 0.39  # [m NAVD88]

name = '21000-21500, 2018-2028, RSLR=4'

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

# Create an instance of the MEEB class
meeb = MEEB(
    name=name,
    simulation_time_yr=sim_duration,
    alongshore_domain_boundary_min=ymin,
    alongshore_domain_boundary_max=ymax,
    RSLR=0.004,
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
    p_dep_sand=0.42,  # Q = hs * L * n * pe/pd
    p_dep_sand_VegMax=0.67,
    p_ero_sand=0.15,
    entrainment_veg_limit=0.07,
    saltation_veg_limit=0.3,
    shadowangle=5,
    repose_bare=20,
    repose_veg=30,
    wind_rose=(0.81, 0.06, 0.11, 0.02),  # (right, down, left, up)
    # --- Storms --- #
    Rin_ru=138,
    Cx=68,
    MaxUpSlope=1,
    K_ru=0.0000227,
    mm=1.04,
    substep_ru=4,
    beach_equilibrium_slope=0.024,
    swash_transport_coefficient=0.00083,
    wave_period_storm=9.4,
    beach_substeps=20,
    flow_reduction_max_spec1=0.2,
    flow_reduction_max_spec2=0.3,
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

# -----------------
# Profiles
Fig = plt.figure(figsize=(14, 7.5))
ax1 = Fig.add_subplot(211)
profile_x = 10
plt.plot(topo_start_sim[profile_x, xmin: xmax], 'k--')
plt.plot(topo_end_sim[profile_x, xmin: xmax], 'r')
plt.title("Profile " + str(profile_x))
ax2 = Fig.add_subplot(212)
plt.plot(np.mean(topo_start_sim[:, xmin: xmax], axis=0), 'k--')
plt.plot(np.mean(topo_end_sim[:, xmin: xmax], axis=0), 'r')
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
