"""
Script for plotting saved output data from MEEB v2.0 comparison simulations.

IRBR 10 September 2026
"""

import numpy as np
import matplotlib.pyplot as plt
import copy
import time
import gc
from netCDF4 import Dataset

import routines_meeb as routine

# __________________________________________________________________________________________________________________________________
# LOAD DATA

# Specify Filenames and Location
data_loc = 'Output/SimData/'
data_filetype_NetCDF = True  # True for NetCDF (.nc) data files, False for NumPy (.npy) data files
elevation_filename = '0-vs-4_Mixed_Km13-28_22Apr26_Elevation.nc'
vegetation_filename = '0-vs-4_Mixed_Km13-28_22Apr26_Vegetation.nc'

name1 = '+0C Mixed'
name2 = '+4C Mixed'
cellsize = 2  # [m]
MHW = 0.39  # [m NAVD88]
save_frequency = 0.1
RSLR = 0.0096  # [m/yr]
sim_duration = 32  # [yr]
H1_a_relative_effectiveness = 0.75  # AMBR
H2_a_relative_effectiveness = 0.5  # UNPA
W_a_relative_effectiveness = 1  # MOCE
W_d_relative_effectiveness = 0.85  # MOCE

# Sim Specifications
plot_start = 60  # Iteration to start plotting from
plot_xmin = 0  # [m] Cross-shore
plot_xmax = 1475  # [m] Cross-shore
plot_ymin = 0  # [m] Alongshore
plot_ymax = 15000  # [m] Alongshore
plot_maps = True
animate = False

plot_xmin = int(plot_xmin / cellsize)  # Cross-shore plotting
plot_xmax = int(plot_xmax / cellsize)  # Cross-shore plotting
plot_ymin = int(plot_ymin / cellsize)  # Alongshore plotting
plot_ymax = int(plot_ymax / cellsize)  # Alongshore plotting

start_time = time.time()  # Record time at start of simulation


class SimulationElevVeg:
    def __init__(self, elev, veg, name):
        self.topo_TS = elev
        self.veg_fraction_TS = veg
        self.name = name


# Load Simulation Data
if data_filetype_NetCDF:
    elev_all = Dataset("Output/SimData/" + elevation_filename, 'r').variables['elevation'][:].data
    veg_all = Dataset("Output/SimData/" + vegetation_filename, 'r').variables['fractional_cover'][:].data
    meeb1 = SimulationElevVeg(elev_all[0, :, :, :], veg_all[0, :, :, :, :], name1)
    meeb2 = SimulationElevVeg(elev_all[1, :, :, :], veg_all[1, :, :, :, :], name2)
else:  # Load numpy arrays
    elev_all = np.load("Output/SimData/" + elevation_filename)
    veg_all = np.load("Output/SimData/" + vegetation_filename)
    meeb1 = SimulationElevVeg(elev_all[0, :, :, :], veg_all[0, :, :, :, :], name1)
    meeb2 = SimulationElevVeg(elev_all[1, :, :, :], veg_all[1, :, :, :, :], name2)

del elev_all, veg_all
gc.collect()

# __________________________________________________________________________________________________________________________________
# ASSESS MODEL RESULTS

# Topo change
topo_start = meeb1.topo_TS[plot_ymin:plot_ymax, :, plot_start].astype(np.float32)  # [m NAVDD88]
mhw_end = MHW + sim_duration * RSLR  # [m NAVD88]

topo_end_1 = meeb1.topo_TS[plot_ymin:plot_ymax, :, -1].astype(np.float32)  # [m NAVDD88]
topo_change_1 = topo_end_1 - topo_start  # [m]

topo_end_2 = meeb2.topo_TS[plot_ymin:plot_ymax, :, -1].astype(np.float32)  # [m NAVDD88]
topo_change_2 = topo_end_2 - topo_start  # [m]

# Veg change
timesteps_in_year = int(1 / save_frequency)
veg_TS_1 = meeb1.veg_fraction_TS[plot_ymin:plot_ymax, :, 2, :] + meeb1.veg_fraction_TS[plot_ymin:plot_ymax, :, 4, :] + meeb1.veg_fraction_TS[plot_ymin:plot_ymax, :, 6, :] + meeb1.veg_fraction_TS[plot_ymin:plot_ymax, :, 7, :]
veg_start = np.mean(veg_TS_1[:, :, plot_start: plot_start + timesteps_in_year], axis=2)

veg_end_1 = np.mean(veg_TS_1[:, :, -timesteps_in_year:-1], axis=2)
veg_change_1 = veg_end_1 - veg_start  # [m]

veg_TS_2 = meeb2.veg_fraction_TS[plot_ymin:plot_ymax, :, 2, :] + meeb2.veg_fraction_TS[plot_ymin:plot_ymax, :, 4, :] + meeb2.veg_fraction_TS[plot_ymin:plot_ymax, :, 6, :] + meeb2.veg_fraction_TS[plot_ymin:plot_ymax, :, 7, :]
veg_end_2 = np.mean(veg_TS_2[:, :, -timesteps_in_year:-1], axis=2)
veg_change_2 = veg_end_2 - veg_start  # [m]

effective_veg_fraction_end_1 = (np.mean(meeb1.veg_fraction_TS[plot_ymin:plot_ymax, :, 2, -timesteps_in_year:-1], axis=2) * H1_a_relative_effectiveness
                                + np.mean(meeb1.veg_fraction_TS[plot_ymin:plot_ymax, :, 4, -timesteps_in_year:-1], axis=2) * H2_a_relative_effectiveness
                                + np.mean(meeb1.veg_fraction_TS[plot_ymin:plot_ymax, :, 6, -timesteps_in_year:-1], axis=2) * W_a_relative_effectiveness
                                + np.mean(meeb1.veg_fraction_TS[plot_ymin:plot_ymax, :, 7, -timesteps_in_year:-1], axis=2) * W_d_relative_effectiveness)

effective_veg_fraction_end_2 = (np.mean(meeb2.veg_fraction_TS[plot_ymin:plot_ymax, :, 2, -timesteps_in_year:-1], axis=2) * H1_a_relative_effectiveness
                                + np.mean(meeb2.veg_fraction_TS[plot_ymin:plot_ymax, :, 4, -timesteps_in_year:-1], axis=2) * H2_a_relative_effectiveness
                                + np.mean(meeb2.veg_fraction_TS[plot_ymin:plot_ymax, :, 6, -timesteps_in_year:-1], axis=2) * W_a_relative_effectiveness
                                + np.mean(meeb2.veg_fraction_TS[plot_ymin:plot_ymax, :, 7, -timesteps_in_year:-1], axis=2) * W_d_relative_effectiveness)

# Woody veg change
woody_TS_1 = meeb1.veg_fraction_TS[plot_ymin:plot_ymax, :, 6, :] + meeb1.veg_fraction_TS[plot_ymin:plot_ymax, :, 7, :]
woody_start = woody_TS_1[:, :, plot_start]

woody_end_1 = woody_TS_1[:, :, -1]
woody_change_1 = woody_end_1 - woody_start  # [m]

woody_TS_2 = meeb2.veg_fraction_TS[plot_ymin:plot_ymax, :, 6, :] + meeb2.veg_fraction_TS[plot_ymin:plot_ymax, :, 7, :]
woody_end_2 = woody_TS_2[:, :, -1]
woody_change_2 = woody_end_2 - woody_start  # [m]

# Dune crest location
dune_crest_start, not_gap_start = routine.foredune_crest(topo_start, mhw_end, cellsize)
dune_crest_end_1, not_gap_end_1 = routine.foredune_crest(topo_end_1, mhw_end, cellsize)
dune_crest_end_2, not_gap_end_2 = routine.foredune_crest(topo_end_2, mhw_end, cellsize)

# Dune crest elevation
dune_elev_start = topo_start[np.arange(topo_start.shape[0]), dune_crest_start]
dune_elev_end_1 = topo_end_1[np.arange(topo_end_1.shape[0]), dune_crest_end_1]
dune_elev_end_2 = topo_end_2[np.arange(topo_end_2.shape[0]), dune_crest_end_2]

# Dune toe location
dune_toe_start = routine.foredune_toe(topo_start, dune_crest_start, MHW, not_gap_start, cellsize)
dune_toe_end_1 = routine.foredune_toe(topo_end_1, dune_crest_end_1, mhw_end, not_gap_end_1, cellsize)
dune_toe_end_2 = routine.foredune_toe(topo_end_2, dune_crest_end_2, mhw_end, not_gap_end_2, cellsize)

# Dune toe elevation
dune_toe_elev_start = topo_start[np.arange(topo_start.shape[0]), dune_toe_start]
dune_toe_elev_end_1 = topo_end_1[np.arange(topo_end_1.shape[0]), dune_toe_end_1]
dune_toe_elev_end_2 = topo_end_2[np.arange(topo_end_2.shape[0]), dune_toe_end_2]

# Dune heel location
dune_heel_start = routine.foredune_heel(topo_start, dune_crest_start, not_gap_start, dune_toe_start, mhw_end, cellsize)
dune_heel_end_1 = routine.foredune_heel(topo_end_1, dune_crest_end_1, not_gap_end_1, dune_toe_end_1, mhw_end, cellsize)
dune_heel_end_2 = routine.foredune_heel(topo_end_2, dune_crest_end_2, not_gap_end_2, dune_toe_end_2, mhw_end, cellsize)

# Dune Width
dune_width_start = ((dune_heel_start - dune_toe_start) * cellsize).astype(float)
dune_width_end_1 = ((dune_heel_end_1 - dune_toe_end_1) * cellsize).astype(float)
dune_width_end_2 = ((dune_heel_end_2 - dune_toe_end_2) * cellsize).astype(float)
dune_width_end_1[not_gap_end_1 == 0] = np.nan  # Don't include areas where there is no dune
dune_width_end_2[not_gap_end_2 == 0] = np.nan  # Don't include areas where there is no dune

# Dune half-width
dune_half_width_start = ((dune_crest_start - dune_toe_start) * cellsize).astype(float)
dune_half_width_end_1 = ((dune_crest_end_1 - dune_toe_end_1) * cellsize).astype(float)
dune_half_width_end_2 = ((dune_crest_end_2 - dune_toe_end_2) * cellsize).astype(float)
dune_half_width_end_1[not_gap_end_1 == 0] = np.nan  # Don't include areas where there is no dune
dune_half_width_end_2[not_gap_end_2 == 0] = np.nan  # Don't include areas where there is no dune

# Shoreline location
shoreline_loc_start = routine.ocean_shoreline(topo_start, MHW)  # Note: this method finds slightly different ocean shoreline than in the run script
shoreline_loc_end_1 = routine.ocean_shoreline(topo_end_1, mhw_end)
shoreline_loc_end_2 = routine.ocean_shoreline(topo_end_2, mhw_end)

# Shoreline elevation
shoreline_elev_start = topo_start[np.arange(topo_start.shape[0]), shoreline_loc_start]
shoreline_elev_end_1 = topo_end_1[np.arange(topo_end_1.shape[0]), shoreline_loc_end_1]
shoreline_elev_end_2 = topo_end_2[np.arange(topo_end_2.shape[0]), shoreline_loc_end_2]

# Beach width
beach_width_start = (dune_toe_start - shoreline_loc_start) * cellsize
beach_width_end_1 = (dune_toe_end_1 - shoreline_loc_end_1) * cellsize
beach_width_end_2 = (dune_toe_end_2 - shoreline_loc_end_2) * cellsize

# Beach slope
beach_slope_start = (dune_toe_elev_start - shoreline_elev_start) / beach_width_start
beach_slope_end_1 = (dune_toe_elev_end_1 - shoreline_elev_end_1) / beach_width_end_1
beach_slope_end_2 = (dune_toe_elev_end_2 - shoreline_elev_end_2) / beach_width_end_2

# Dune stoss slope
dune_slope_start = (dune_elev_start - dune_toe_elev_start) / dune_half_width_start
dune_slope_end_1 = (dune_elev_end_1 - dune_toe_elev_end_1) / dune_half_width_end_1
dune_slope_end_2 = (dune_elev_end_2 - dune_toe_elev_end_2) / dune_half_width_end_2
dune_slope_end_1[not_gap_end_1 == 0] = np.nan  # Don't include areas where there is no dune
dune_slope_end_2[not_gap_end_2 == 0] = np.nan  # Don't include areas where there is no dune

# Final Fractional Cover by Species (Average Cover of Last Year)
H1_cover_end_1 = np.mean(meeb1.veg_fraction_TS[plot_ymin:plot_ymax, :, 2, -timesteps_in_year:-1], axis=2)
H2_cover_end_1 = np.mean(meeb1.veg_fraction_TS[plot_ymin:plot_ymax, :, 4, -timesteps_in_year:-1], axis=2)
W_cover_end_1 = np.mean(meeb1.veg_fraction_TS[plot_ymin:plot_ymax, :, 6, -timesteps_in_year:-1], axis=2) + np.mean(meeb1.veg_fraction_TS[plot_ymin:plot_ymax, :, 7, -timesteps_in_year:-1], axis=2)
H1_cover_end_2 = np.mean(meeb2.veg_fraction_TS[plot_ymin:plot_ymax, :, 2, -timesteps_in_year:-1], axis=2)
H2_cover_end_2 = np.mean(meeb2.veg_fraction_TS[plot_ymin:plot_ymax, :, 4, -timesteps_in_year:-1], axis=2)
W_cover_end_2 = np.mean(meeb2.veg_fraction_TS[plot_ymin:plot_ymax, :, 6, -timesteps_in_year:-1], axis=2) + np.mean(meeb2.veg_fraction_TS[plot_ymin:plot_ymax, :, 7, -timesteps_in_year:-1], axis=2)

H1_avg_end_1 = np.average(H1_cover_end_1[(H1_cover_end_1 + H2_cover_end_1) > 0.01])
H2_avg_end_1 = np.average(H2_cover_end_1[(H1_cover_end_1 + H2_cover_end_1) > 0.01])
W_avg_end_1 = np.average(W_cover_end_1[W_cover_end_1 > 0.01])

H1_avg_end_2 = np.average(H1_cover_end_2[(H1_cover_end_2 + H2_cover_end_2) > 0.01])
H2_avg_end_2 = np.average(H2_cover_end_2[(H1_cover_end_2 + H2_cover_end_2) > 0.01])
W_avg_end_2 = np.average(W_cover_end_2[W_cover_end_2 > 0.01])

# Species Presence
H1_cell_presence_start_1 = np.sum(meeb1.veg_fraction_TS[plot_ymin:plot_ymax, :, 2, plot_start] > 0)
H2_cell_presence_start_1 = np.sum(meeb1.veg_fraction_TS[plot_ymin:plot_ymax, :, 4, plot_start] > 0)
W_cell_presence_start_1 = np.sum(meeb1.veg_fraction_TS[plot_ymin:plot_ymax, :, 6, plot_start] > 0) + np.sum(meeb1.veg_fraction_TS[plot_ymin:plot_ymax, :, 7, 0] > 0)

H1_cell_presence_start_2 = np.sum(meeb2.veg_fraction_TS[plot_ymin:plot_ymax, :, 2, plot_start] > 0)
H2_cell_presence_start_2 = np.sum(meeb2.veg_fraction_TS[plot_ymin:plot_ymax, :, 4, plot_start] > 0)
W_cell_presence_start_2 = np.sum(meeb2.veg_fraction_TS[plot_ymin:plot_ymax, :, 6, plot_start] > 0) + np.sum(meeb2.veg_fraction_TS[plot_ymin:plot_ymax, :, 7, 0] > 0)

H1_cell_presence_end_1 = np.sum(meeb1.veg_fraction_TS[plot_ymin:plot_ymax, :, 2, -1] > 0)
H2_cell_presence_end_1 = np.sum(meeb1.veg_fraction_TS[plot_ymin:plot_ymax, :, 4, -1] > 0)
W_cell_presence_end_1 = np.sum(meeb1.veg_fraction_TS[plot_ymin:plot_ymax, :, 6, -1] > 0) + np.sum(meeb1.veg_fraction_TS[plot_ymin:plot_ymax, :, 7, -1] > 0)

H1_cell_presence_end_2 = np.sum(meeb2.veg_fraction_TS[plot_ymin:plot_ymax, :, 2, -1] > 0)
H2_cell_presence_end_2 = np.sum(meeb2.veg_fraction_TS[plot_ymin:plot_ymax, :, 4, -1] > 0)
W_cell_presence_end_2 = np.sum(meeb2.veg_fraction_TS[plot_ymin:plot_ymax, :, 6, -1] > 0) + np.sum(meeb2.veg_fraction_TS[plot_ymin:plot_ymax, :, 7, -1] > 0)

# __________________________________________________________________________________________________________________________________
# PLOT RESULTS

# -----------------
# Prepare Colormaps

cmap1 = routine.truncate_colormap(copy.copy(plt.colormaps["terrain"]), 0.5, 0.9)  # Truncate colormap
cmap1.set_bad(color='dodgerblue', alpha=0.5)  # Set cell color below MHW to blue

cmap2 = copy.copy(plt.colormaps["YlGn"])
cmap2.set_bad(color='dodgerblue', alpha=0.5)  # Set cell color below MHW to blue

cmap3 = copy.copy(plt.colormaps["BrBG"])
cmap3.set_bad(color='dodgerblue', alpha=0.5)  # Set cell color below MHW to blue

if plot_maps:
    # -----------------
    # Final Elevation
    Fig1 = plt.figure(figsize=(10, 10))
    Fig1.suptitle('Final Elevation', fontsize=13)
    topo_1 = np.ma.masked_where(topo_end_1[:, plot_xmin: plot_xmax] <= mhw_end, topo_end_1[:, plot_xmin: plot_xmax])  # Mask cells below MHW
    ax1 = Fig1.add_subplot(121)
    cax1 = ax1.matshow(topo_1, cmap=cmap1, vmin=0, vmax=6.0)
    ax1.plot(dune_crest_end_1, np.arange(dune_crest_end_1.shape[0]), 'red', linewidth=1)
    cbar_1 = Fig1.colorbar(cax1)
    cbar_1.set_label('Elevation [m NAVD88]', rotation=270, labelpad=20)
    plt.title(meeb1.name, fontsize=11)

    ax2 = Fig1.add_subplot(122)
    topo_2 = np.ma.masked_where(topo_end_2[:, plot_xmin: plot_xmax] <= mhw_end, topo_end_2[:, plot_xmin: plot_xmax])  # Mask cells below MHW
    cax2 = ax2.matshow(topo_2, cmap=cmap1, vmin=0, vmax=6.0)
    ax2.plot(dune_crest_end_2, np.arange(dune_crest_end_1.shape[0]), 'red', linewidth=1)
    cbar_2 = Fig1.colorbar(cax2)
    cbar_2.set_label('Elevation [m NAVD88]', rotation=270, labelpad=20)
    plt.title(meeb2.name, fontsize=11)

    plt.tight_layout()

    # -----------------
    # Final Vegetation
    Fig = plt.figure(figsize=(10, 10))
    Fig.suptitle('Final Vegetation', fontsize=13)
    veg_1 = np.ma.masked_where(topo_end_1[:, plot_xmin: plot_xmax] <= mhw_end, veg_end_1[:, plot_xmin: plot_xmax])  # Mask cells below MHW
    ax1 = Fig.add_subplot(121)
    cax1 = ax1.matshow(veg_1, cmap=cmap2, vmin=0, vmax=1.0)
    ax1.plot(dune_crest_end_1, np.arange(dune_crest_end_1.shape[0]), 'black', linewidth=1)
    cbar_1 = Fig.colorbar(cax1)
    cbar_1.set_label('Fractional Cover', rotation=270, labelpad=20)
    plt.title(meeb1.name, fontsize=11)

    ax2 = Fig.add_subplot(122)
    veg_2 = np.ma.masked_where(topo_end_2[:, plot_xmin: plot_xmax] <= mhw_end, veg_end_2[:, plot_xmin: plot_xmax])  # Mask cells below MHW
    cax2 = ax2.matshow(veg_2, cmap=cmap2, vmin=0, vmax=1.0)
    ax2.plot(dune_crest_end_2, np.arange(dune_crest_end_2.shape[0]), 'black', linewidth=1)
    cbar_2 = Fig.colorbar(cax2)
    cbar_2.set_label('Fractional Cover', rotation=270, labelpad=20)
    plt.title(meeb2.name, fontsize=11)

    plt.tight_layout()

    # -----------------
    # Final Grass Vegetation
    grass1_1 = np.mean(meeb1.veg_fraction_TS[plot_ymin:plot_ymax, :, 2, -timesteps_in_year:-1], axis=2)
    grass2_1 = np.mean(meeb1.veg_fraction_TS[plot_ymin:plot_ymax, :, 4, -timesteps_in_year:-1], axis=2)
    grass1_2 = np.mean(meeb2.veg_fraction_TS[plot_ymin:plot_ymax, :, 2, -timesteps_in_year:-1], axis=2)
    grass2_2 = np.mean(meeb2.veg_fraction_TS[plot_ymin:plot_ymax, :, 4, -timesteps_in_year:-1], axis=2)

    Fig = plt.figure(figsize=(10, 10))
    Fig.suptitle('Final Grass Vegetation', fontsize=13)
    g1_1 = np.ma.masked_where(topo_end_1[:, plot_xmin: plot_xmax] <= mhw_end, grass1_1[:, plot_xmin: plot_xmax])  # Mask cells below MHW
    ax1 = Fig.add_subplot(221)
    ax1.matshow(g1_1, cmap=cmap2, vmin=0, vmax=1.0)
    plt.title(meeb1.name + ', Grass 1', fontsize=11)

    g2_1 = np.ma.masked_where(topo_end_1[:, plot_xmin: plot_xmax] <= mhw_end, grass2_1[:, plot_xmin: plot_xmax])  # Mask cells below MHW
    ax2 = Fig.add_subplot(222)
    ax2.matshow(g2_1, cmap=cmap2, vmin=0, vmax=1.0)
    plt.title(meeb1.name + ', Grass 2', fontsize=11)

    g1_2 = np.ma.masked_where(topo_end_1[:, plot_xmin: plot_xmax] <= mhw_end, grass1_2[:, plot_xmin: plot_xmax])  # Mask cells below MHW
    ax3 = Fig.add_subplot(223)
    ax3.matshow(g1_2, cmap=cmap2, vmin=0, vmax=1.0)
    plt.title(meeb2.name + ', Grass 1', fontsize=11)

    g2_2 = np.ma.masked_where(topo_end_1[:, plot_xmin: plot_xmax] <= mhw_end, grass2_2[:, plot_xmin: plot_xmax])  # Mask cells below MHW
    ax4 = Fig.add_subplot(224)
    ax4.matshow(g2_2, cmap=cmap2, vmin=0, vmax=1.0)
    plt.title(meeb2.name + ', Grass 2', fontsize=11)

    plt.tight_layout()

    # -----------------
    # Final Woody Vegetation
    Fig = plt.figure(figsize=(10, 10))
    Fig.suptitle('Final Woody Vegetation', fontsize=13)
    woody_1 = np.ma.masked_where(topo_end_1[:, plot_xmin: plot_xmax] <= mhw_end, woody_end_1[:, plot_xmin: plot_xmax])  # Mask cells below MHW
    ax1 = Fig.add_subplot(121)
    cax1 = ax1.matshow(woody_1, cmap=cmap2, vmin=0, vmax=1.0)
    cbar_1 = Fig.colorbar(cax1)
    cbar_1.set_label('Fractional Cover', rotation=270, labelpad=20)
    plt.title(meeb1.name, fontsize=11)

    ax2 = Fig.add_subplot(122)
    woody_2 = np.ma.masked_where(topo_end_2[:, plot_xmin: plot_xmax] <= mhw_end, woody_end_2[:, plot_xmin: plot_xmax])  # Mask cells below MHW
    cax2 = ax2.matshow(woody_2, cmap=cmap2, vmin=0, vmax=1.0)
    cbar_2 = Fig.colorbar(cax2)
    cbar_2.set_label('Fractional Cover', rotation=270, labelpad=20)
    plt.title(meeb2.name, fontsize=11)

    plt.tight_layout()

    # -----------------
    # Final Effective Veg Fraction
    Fig = plt.figure(figsize=(10, 10))
    Fig.suptitle('Final Vegetation Effectiveness', fontsize=13)
    eff_veg_1 = np.ma.masked_where(topo_end_1[:, plot_xmin: plot_xmax] <= mhw_end, effective_veg_fraction_end_1[:, plot_xmin: plot_xmax])  # Mask cells below MHW
    ax1 = Fig.add_subplot(131)
    cax1 = ax1.matshow(eff_veg_1, cmap=cmap2, vmin=0, vmax=1.0)
    plt.plot(dune_crest_end_1, np.arange(dune_crest_end_1.shape[0]), 'black', linewidth=1)
    cbar_1 = Fig.colorbar(cax1)
    cbar_1.set_label('Effective Fractional Cover', rotation=270, labelpad=20)
    plt.title(meeb1.name, fontsize=11)

    ax2 = Fig.add_subplot(132)
    eff_veg_2 = np.ma.masked_where(topo_end_2[:, plot_xmin: plot_xmax] <= mhw_end, effective_veg_fraction_end_2[:, plot_xmin: plot_xmax])  # Mask cells below MHW
    cax2 = ax2.matshow(eff_veg_2, cmap=cmap2, vmin=0, vmax=1.0)
    plt.plot(dune_crest_end_2, np.arange(dune_crest_end_2.shape[0]), 'black', linewidth=1)
    cbar_2 = Fig.colorbar(cax2)
    cbar_2.set_label('Effective Fractional Cover', rotation=270, labelpad=20)
    plt.title(meeb2.name, fontsize=11)

    ax3 = Fig.add_subplot(133)
    eff_veg_diff = eff_veg_2 - eff_veg_1
    cax3 = ax3.matshow(eff_veg_diff, cmap='bwr_r', vmin=-1.0, vmax=1.0)
    plt.plot(dune_crest_end_2, np.arange(dune_crest_end_2.shape[0]), 'black', linewidth=1)
    cbar_3 = Fig.colorbar(cax3)
    cbar_3.set_label('Difference in Effective Cover: Sim 2 - Sim 1', rotation=270, labelpad=20)
    plt.title(meeb2.name, fontsize=11)

    plt.tight_layout()

    # -----------------
    # Elevation Change and Difference
    Fig = plt.figure(figsize=(10, 10))
    Fig.suptitle('Elevation Change and Difference', fontsize=13)
    ax1 = Fig.add_subplot(131)
    cax1 = ax1.matshow(topo_change_1[:, plot_xmin: plot_xmax], cmap='bwr_r', vmin=-1.5, vmax=1.5)
    plt.plot(dune_crest_end_1, np.arange(dune_crest_end_1.shape[0]), 'black', linewidth=1)
    cbar_1 = Fig.colorbar(cax1)
    cbar_1.set_label('Elevation Change [m]', rotation=270, labelpad=20)
    plt.title(meeb1.name, fontsize=11)

    ax2 = Fig.add_subplot(132)
    cax2 = ax2.matshow(topo_change_2[:, plot_xmin: plot_xmax], cmap='bwr_r', vmin=-1.5, vmax=1.5)
    plt.plot(dune_crest_end_2, np.arange(dune_crest_end_2.shape[0]), 'black', linewidth=1)
    cbar_2 = Fig.colorbar(cax2)
    cbar_2.set_label('Elevation Change [m]', rotation=270, labelpad=20)
    plt.title(meeb2.name, fontsize=11)

    ax3 = Fig.add_subplot(133)
    cax3 = ax3.matshow(topo_end_2[:, plot_xmin: plot_xmax] - topo_end_1[:, plot_xmin: plot_xmax], cmap='bwr_r', vmin=-1.5, vmax=1.5)
    plt.plot(dune_crest_end_1, np.arange(dune_crest_end_1.shape[0]), 'black', linewidth=1)
    plt.plot(dune_crest_end_2, np.arange(dune_crest_end_2.shape[0]), '--k', linewidth=1)
    cbar_3 = Fig.colorbar(cax3)
    cbar_3.set_label('Elevation Difference: Sim 2 - Sim 1 [m]', rotation=270, labelpad=20)
    plt.title('Elevation Difference: Sim 1 vs Sim 2', fontsize=11)
    plt.tight_layout()

    # -----------------
    # Vegetation Change and Difference
    Fig = plt.figure(figsize=(10, 10))
    Fig.suptitle('Vegetation Change and Difference', fontsize=13)
    ax1 = Fig.add_subplot(131)
    cax1 = ax1.matshow(veg_change_1[:, plot_xmin: plot_xmax], cmap='BrBG', vmin=-1, vmax=1)
    plt.plot(dune_crest_end_1, np.arange(dune_crest_end_1.shape[0]), 'red', linewidth=1)
    cbar_1 = Fig.colorbar(cax1)
    cbar_1.set_label('Vegetation Change [fraction]', rotation=270, labelpad=20)
    plt.title(meeb1.name, fontsize=11)

    ax2 = Fig.add_subplot(132)
    cax2 = ax2.matshow(veg_change_2[:, plot_xmin: plot_xmax], cmap='BrBG', vmin=-1, vmax=1)
    plt.plot(dune_crest_end_2, np.arange(dune_crest_end_1.shape[0]), 'red', linewidth=1)
    cbar_2 = Fig.colorbar(cax2)
    cbar_2.set_label('Vegetation Change [fraction]', rotation=270, labelpad=20)
    plt.title(meeb2.name, fontsize=11)

    ax3 = Fig.add_subplot(133)
    cax3 = ax3.matshow(veg_end_2[:, plot_xmin: plot_xmax] - veg_end_1[:, plot_xmin: plot_xmax], cmap='bwr_r', vmin=-1, vmax=1)
    cbar_3 = Fig.colorbar(cax3)
    cbar_3.set_label('Vegetation Difference: Sim 2 - Sim 1 [fraction]', rotation=270, labelpad=20)
    plt.title('Vegetation Difference: Sim 1 vs Sim 2', fontsize=11)
    plt.tight_layout()

    # -----------------
    # Final Effective Veg Fraction
    Fig = plt.figure(figsize=(10, 10))
    Fig.suptitle('Final Vegetation Effectiveness', fontsize=13)
    eff_veg_1 = np.ma.masked_where(topo_end_1[:, plot_xmin: plot_xmax] <= mhw_end, effective_veg_fraction_end_1[:, plot_xmin: plot_xmax])  # Mask cells below MHW
    ax1 = Fig.add_subplot(131)
    cax1 = ax1.matshow(eff_veg_1, cmap=cmap2, vmin=0, vmax=1.0)
    ax1.plot(dune_crest_end_1, np.arange(dune_crest_end_1.shape[0]), 'black', linewidth=1)
    cbar_1 = Fig.colorbar(cax1)
    cbar_1.set_label('Effective Fractional Cover', rotation=270, labelpad=20)
    plt.title(meeb1.name, fontsize=11)

    ax2 = Fig.add_subplot(132)
    eff_veg_2 = np.ma.masked_where(topo_end_2[:, plot_xmin: plot_xmax] <= mhw_end, effective_veg_fraction_end_2[:, plot_xmin: plot_xmax])  # Mask cells below MHW
    cax2 = ax2.matshow(eff_veg_2, cmap=cmap2, vmin=0, vmax=1.0)
    ax2.plot(dune_crest_end_2, np.arange(dune_crest_end_2.shape[0]), 'black', linewidth=1)
    cbar_2 = Fig.colorbar(cax2)
    cbar_2.set_label('Effective Fractional Cover', rotation=270, labelpad=20)
    plt.title(meeb2.name, fontsize=11)

    ax3 = Fig.add_subplot(133)
    eff_veg_diff = eff_veg_2 - eff_veg_1
    cax3 = ax3.matshow(eff_veg_diff, cmap='bwr_r', vmin=-1.0, vmax=1.0)
    ax3.plot(dune_crest_end_2, np.arange(dune_crest_end_2.shape[0]), 'black', linewidth=1)
    cbar_3 = Fig.colorbar(cax3)
    cbar_3.set_label('Difference in Effective Cover: Sim 2 - Sim 1', rotation=270, labelpad=20)
    plt.title(meeb2.name, fontsize=11)

    plt.tight_layout()

# -----------------
# Dune Crest Elevation
Fig = plt.figure(figsize=(14, 10))
Fig.suptitle('Dune Crest Elevation', fontsize=13)
ax1 = Fig.add_subplot(2, 1, 1)
x = np.arange(0, meeb1.topo_TS[plot_ymin:plot_ymax, :, :].shape[0] * cellsize, 2)
ax1.plot(x, dune_elev_start, '--k')
ax1.plot(x, dune_elev_end_1, 'teal')
ax1.plot(x, dune_elev_end_2, 'red')
plt.legend(['Initial', 'End: ' + str(name1), 'End: ' + str(name2)])
plt.xlabel('Distance Alongshore [m]')
plt.ylabel('Dune Crest Elevation [m NAVD88]')
ax2 = Fig.add_subplot(2, 2, 3)
# plt.boxplot([dune_elev_end_1, dune_elev_end_2], labels=[name1, name2])
# plt.ylabel('Dune Crest Elevation [m NAVD88]')
vplot = ax2.violinplot([dune_elev_end_1, dune_elev_end_2], showmedians=True)
for i, pc in enumerate(vplot['bodies']):
    pc.set_facecolor(['teal', 'red'][i])
labels = [name1, name2]
ax2.set_xticks(np.arange(1, len(labels) + 1), labels=labels)
ax2.set_xlim(0.25, len(labels) + 0.75)
plt.ylabel('Dune Crest Elevation [m NAVD88]')
ax3 = Fig.add_subplot(2, 2, 4)
ax3.hist(dune_elev_end_1, alpha=0.5, bins=32, density=True, color='teal', edgecolor='black', linewidth=0.5, range=(min(np.nanmin(dune_elev_end_1), np.nanmin(dune_elev_end_2)), max(np.nanmax(dune_elev_end_1), np.nanmax(dune_elev_end_2))))
ax3.hist(dune_elev_end_2, alpha=0.5, bins=32, density=True, color='red', edgecolor='black', linewidth=0.5, range=(min(np.nanmin(dune_elev_end_1), np.nanmin(dune_elev_end_2)), max(np.nanmax(dune_elev_end_1), np.nanmax(dune_elev_end_2))))
plt.ylabel('Probability Density')
plt.xlabel('Dune Crest Elevation [m NAVD88]')
plt.legend(['End: ' + str(name1), 'End: ' + str(name2)])

# -----------------
# Dune Width
Fig = plt.figure(figsize=(14, 10))
Fig.suptitle('Dune Width', fontsize=13)
ax1 = Fig.add_subplot(2, 1, 1)
x = np.arange(0, meeb1.topo_TS[plot_ymin:plot_ymax, :, :].shape[0] * cellsize, 2)
ax1.plot(x, dune_width_start, '--k')
ax1.plot(x, dune_width_end_1, 'teal')
ax1.plot(x, dune_width_end_2, 'red')
plt.legend(['Initial', 'End: ' + str(name1), 'End: ' + str(name2)])
plt.xlabel('Distance Alongshore [m]')
plt.ylabel('Dune Width [m]')
ax2 = Fig.add_subplot(2, 2, 3)
vplot = ax2.violinplot([dune_width_end_1[~np.isnan(dune_width_end_1)], dune_width_end_2[~np.isnan(dune_width_end_2)]], showmedians=True)
for i, pc in enumerate(vplot['bodies']):
    pc.set_facecolor(['teal', 'red'][i])
labels = [name1, name2]
ax2.set_xticks(np.arange(1, len(labels) + 1), labels=labels)
ax2.set_xlim(0.25, len(labels) + 0.75)
plt.ylabel('Dune Width [m]')
ax3 = Fig.add_subplot(2, 2, 4)
ax3.hist(dune_width_end_1, alpha=0.5, bins=32, density=True, color='teal', edgecolor='black', linewidth=0.5, range=(min(np.nanmin(dune_width_end_1), np.nanmin(dune_width_end_2)), max(np.nanmax(dune_width_end_1), np.nanmax(dune_width_end_2))))
ax3.hist(dune_width_end_2, alpha=0.5, bins=32, density=True, color='red', edgecolor='black', linewidth=0.5, range=(min(np.nanmin(dune_width_end_1), np.nanmin(dune_width_end_2)), max(np.nanmax(dune_width_end_1), np.nanmax(dune_width_end_2))))
plt.ylabel('Probability Density')
plt.xlabel('Dune Width [m]')
plt.legend(['End: ' + str(name1), 'End: ' + str(name2)])

# -----------------
# Dune Half-Width and Stoss Slope
Fig = plt.figure(figsize=(14, 10))
Fig.suptitle('Dune Half-Width', fontsize=13)
ax1 = Fig.add_subplot(2, 1, 1)
x = np.arange(0, meeb1.topo_TS[plot_ymin:plot_ymax, :, :].shape[0] * cellsize, 2)
ax1.plot(x, dune_half_width_start, '--k')
ax1.plot(x, dune_half_width_end_1, 'teal')
ax1.plot(x, dune_half_width_end_2, 'red')
plt.legend(['Initial', 'End: ' + str(name1), 'End: ' + str(name2)])
plt.xlabel('Distance Alongshore [m]')
plt.ylabel('Dune Half-Width [m]')
ax2 = Fig.add_subplot(2, 2, 3)
ax2.hist(dune_half_width_end_1, alpha=0.5, bins=32, density=True, color='teal', edgecolor='black', linewidth=0.5, range=(min(np.nanmin(dune_half_width_end_1), np.nanmin(dune_half_width_end_2)), max(np.nanmax(dune_half_width_end_1), np.nanmax(dune_half_width_end_2))))
ax2.hist(dune_half_width_end_2, alpha=0.5, bins=32, density=True, color='red', edgecolor='black', linewidth=0.5, range=(min(np.nanmin(dune_half_width_end_1), np.nanmin(dune_half_width_end_2)), max(np.nanmax(dune_half_width_end_1), np.nanmax(dune_half_width_end_2))))
plt.ylabel('Probability Density')
plt.xlabel('Dune Half-Width [m]')
plt.legend(['End: ' + str(name1), 'End: ' + str(name2)])
ax3 = Fig.add_subplot(2, 2, 4)
DS_1 = dune_slope_end_1.copy()
DS_1[not_gap_end_1 == 0] = np.nan
DS_1 = [i for i in DS_1 if ~np.isnan(i)]
DS_2 = dune_slope_end_2.copy()
DS_2[not_gap_end_2 == 0] = np.nan
DS_2 = [i for i in DS_2 if ~np.isnan(i)]
ax3.hist(DS_1, alpha=0.5, bins=32, density=True, color='teal', edgecolor='black', linewidth=0.5, range=(min(np.nanmin(DS_1), np.nanmin(DS_2)), max(np.nanmax(DS_1), np.nanmax(DS_2))))
ax3.hist(DS_2, alpha=0.5, bins=32, density=True, color='red', edgecolor='black', linewidth=0.5, range=(min(np.nanmin(DS_1), np.nanmin(DS_2)), max(np.nanmax(DS_1), np.nanmax(DS_2))))
plt.ylabel('Probability Density')
plt.xlabel('Dune Stoss Slope')
plt.legend(['End: ' + str(name1), 'End: ' + str(name2)])

# -----------------
# Beach Width & Slope
Fig = plt.figure(figsize=(14, 10))
Fig.suptitle('Beach Width', fontsize=13)
ax1 = Fig.add_subplot(2, 1, 1)
x = np.arange(0, meeb1.topo_TS[plot_ymin:plot_ymax, :, :].shape[0] * cellsize, 2)
ax1.plot(x, beach_width_start, '--k')
ax1.plot(x, beach_width_end_1, 'teal')
ax1.plot(x, beach_width_end_2, 'red')
plt.legend(['Initial', 'End: ' + str(name1), 'End: ' + str(name2)])
plt.xlabel('Distance Alongshore [m]')
plt.ylabel('Beach Width [m]')
ax2 = Fig.add_subplot(2, 2, 3)
ax2.hist(beach_width_end_1, alpha=0.5, bins=32, density=True, color='teal', edgecolor='black', linewidth=0.5, range=(min(np.min(beach_width_end_1), np.min(beach_width_end_2)), max(np.max(beach_width_end_1), np.max(beach_width_end_2))))
ax2.hist(beach_width_end_2, alpha=0.5, bins=32, density=True, color='red', edgecolor='black', linewidth=0.5, range=(min(np.min(beach_width_end_1), np.min(beach_width_end_2)), max(np.max(beach_width_end_1), np.max(beach_width_end_2))))
plt.ylabel('Probability Density')
plt.xlabel('Beach Width [m]')
plt.legend(['End: ' + str(name1), 'End: ' + str(name2)])
ax3 = Fig.add_subplot(2, 2, 4)
ax3.boxplot([beach_slope_end_1, beach_slope_end_2], labels=[name1, name2])
plt.ylabel('Beach Slope')

# -----------------
# Species Abundance

# Sim 1 vs Sim 2 Average Fractional Cover
x = np.arange(2)
width = 0.2
Fig = plt.figure(figsize=(14, 10))
ax1 = Fig.add_subplot(211)
ax1.bar(x - 0.2, [H1_avg_end_1, H1_avg_end_2], width, color=['indigo', 'indigo'], alpha=0.5)
ax1.bar(x, [H2_avg_end_1, H2_avg_end_2], width, color=['goldenrod', 'goldenrod'], alpha=0.5)
ax1.bar(x + 0.2, [H1_avg_end_1 + H2_avg_end_1, H1_avg_end_2 + H2_avg_end_2], width, color=['black', 'black'], alpha=0.5)
plt.xticks([0, 1], [name1, name2])
plt.ylabel('Average Fractional Cover')
plt.legend(['AMBR', 'UNPA', 'Combined'])
ax2 = Fig.add_subplot(212)
vplot = ax2.violinplot([H1_cover_end_1[(H1_cover_end_1 + H2_cover_end_1) > 0.01],
                        H2_cover_end_1[(H1_cover_end_1 + H2_cover_end_1) > 0.01],
                        H1_cover_end_2[(H1_cover_end_2 + H2_cover_end_2) > 0.01],
                        H2_cover_end_2[(H1_cover_end_2 + H2_cover_end_2) > 0.01]],
                       showmedians=True)
for i, pc in enumerate(vplot['bodies']):
    pc.set_facecolor(['indigo', 'goldenrod', 'indigo', 'goldenrod'][i])
labels = ['G1 ' + str(name1), 'G2 ' + str(name1), 'G1 ' + str(name2), 'G2 ' + str(name2)]
ax2.set_xticks(np.arange(1, len(labels) + 1), labels=labels)
ax2.set_xlim(0.25, len(labels) + 0.75)
plt.ylabel('Average Fractional Cover')

# Species Abundance Over Time
H1_avg_cover_TS_1 = []
H2_avg_cover_TS_1 = []
Hboth_avg_cover_TS_1 = []
H1_avg_cover_TS_2 = []
H2_avg_cover_TS_2 = []
Hboth_avg_cover_TS_2 = []
for ts in range(plot_start, meeb1.veg_fraction_TS.shape[3]):
    H1_avg_cover_TS_1.append(np.mean(meeb1.veg_fraction_TS[plot_ymin:plot_ymax, :, 2, ts][np.logical_and(topo_end_1 > mhw_end, (H1_cover_end_1 + H2_cover_end_1) > 0.01)]))
    H2_avg_cover_TS_1.append(np.mean(meeb1.veg_fraction_TS[plot_ymin:plot_ymax, :, 4, ts][np.logical_and(topo_end_1 > mhw_end, (H1_cover_end_1 + H2_cover_end_1) > 0.01)]))
    Hboth_avg_cover_TS_1.append(np.mean(meeb1.veg_fraction_TS[plot_ymin:plot_ymax, :, 2, ts][np.logical_and(topo_end_1 > mhw_end, (H1_cover_end_1 + H2_cover_end_1) > 0.01)]) + np.mean(meeb1.veg_fraction_TS[plot_ymin:plot_ymax, :, 4, ts][np.logical_and(topo_end_1 > mhw_end, (H1_cover_end_1 + H2_cover_end_1) > 0.01)]))
    H1_avg_cover_TS_2.append(np.mean(meeb2.veg_fraction_TS[plot_ymin:plot_ymax, :, 2, ts][np.logical_and(topo_end_2 > mhw_end, (H1_cover_end_2 + H2_cover_end_2) > 0.01)]))
    H2_avg_cover_TS_2.append(np.mean(meeb2.veg_fraction_TS[plot_ymin:plot_ymax, :, 4, ts][np.logical_and(topo_end_2 > mhw_end, (H1_cover_end_2 + H2_cover_end_2) > 0.01)]))
    Hboth_avg_cover_TS_2.append(np.mean(meeb2.veg_fraction_TS[plot_ymin:plot_ymax, :, 2, ts][np.logical_and(topo_end_2 > mhw_end, (H1_cover_end_2 + H2_cover_end_2) > 0.01)]) + np.mean(meeb2.veg_fraction_TS[plot_ymin:plot_ymax, :, 4, ts][np.logical_and(topo_end_2 > mhw_end, (H1_cover_end_2 + H2_cover_end_2) > 0.01)]))

Fig = plt.figure(figsize=(14, 10))
Fig.add_subplot(2, 1, 1)
x = np.arange(0, len(H1_avg_cover_TS_1) * save_frequency, save_frequency)
plt.plot(x, H1_avg_cover_TS_1, 'indigo')
plt.plot(x, H2_avg_cover_TS_1, 'dodgerblue')
plt.plot(x, H1_avg_cover_TS_2, 'red')
plt.plot(x, H2_avg_cover_TS_2, 'goldenrod')
plt.legend(['AMBR: ' + meeb1.name, 'UNPA: ' + meeb1.name, 'AMBR: ' + meeb2.name, 'UNPA: ' + meeb2.name])
plt.ylabel('Average Fractional Cover')
plt.xlabel('Model Year')
Fig.add_subplot(2, 1, 2)
plt.plot(x, Hboth_avg_cover_TS_1, 'teal')
plt.plot(x, Hboth_avg_cover_TS_2, 'red')
plt.legend(['Grass Combined: ' + meeb1.name, 'Grass Combined: ' + meeb2.name])
plt.ylabel('Average Fractional Cover')
plt.xlabel('Model Year')

# -----------------
# Effective Veg Cover at Dune Crest
Fig = plt.figure(figsize=(14, 10))
ax1 = Fig.add_subplot(2, 1, 1)
x = np.arange(0, meeb1.topo_TS[plot_ymin:plot_ymax, :, :].shape[0] * cellsize, 2)
ax1.plot(x, veg_end_1[np.arange(topo_end_1.shape[0]), dune_crest_end_1], 'teal')
ax1.plot(x, veg_end_2[np.arange(topo_end_2.shape[0]), dune_crest_end_2], 'red')
ax1.plot(x, effective_veg_fraction_end_1[np.arange(topo_end_1.shape[0]), dune_crest_end_1], 'purple')
ax1.plot(x, effective_veg_fraction_end_2[np.arange(topo_end_2.shape[0]), dune_crest_end_2], 'orange')
plt.legend(['Fractional Cover: ' + str(name1), 'Fractional Cover: ' + str(name2), 'Effective Fractional Cover: ' + str(name1), 'Effective Fractional Cover: ' + str(name2)])
plt.xlabel('Distance Alongshore [m]')
plt.ylabel('Vegetation Cover at Dune Crest')
plt.title('Vegetation Cover at Dune Crest')
ax2 = Fig.add_subplot(2, 1, 2)
vplot = ax2.violinplot([veg_end_1[np.arange(topo_end_1.shape[0]), dune_crest_end_1],
                        veg_end_2[np.arange(topo_end_2.shape[0]), dune_crest_end_2],
                        effective_veg_fraction_end_1[np.arange(topo_end_1.shape[0]), dune_crest_end_1],
                        effective_veg_fraction_end_2[np.arange(topo_end_2.shape[0]), dune_crest_end_2]],
                       showmedians=True)
for i, pc in enumerate(vplot['bodies']):
    pc.set_facecolor(['teal', 'red', 'purple', 'orange'][i])
labels = [str(name1), str(name2), 'Effective ' + str(name1), 'Effective ' + str(name2)]
ax2.set_xticks(np.arange(1, len(labels) + 1), labels=labels)
ax2.set_xlim(0.25, len(labels) + 0.75)
plt.ylabel('Vegetation Cover at Dune Crest')

# -----------------
# Effective Veg Cover Between Shoreline and Dune Crest
x_s_end_1 = routine.ocean_shoreline(topo_end_1, mhw_end)
x_s_end_2 = routine.ocean_shoreline(topo_end_2, mhw_end)
eff_veg_shoreline_to_crest_1 = [np.mean(effective_veg_fraction_end_1[ll, x_s_end_1[ll]: dune_crest_end_1[ll]]) for ll in np.arange(topo_end_1.shape[0])]
eff_veg_shoreline_to_crest_2 = [np.mean(effective_veg_fraction_end_2[ll, x_s_end_2[ll]: dune_crest_end_2[ll]]) for ll in np.arange(topo_end_2.shape[0])]

Fig = plt.figure(figsize=(14, 10))
ax1 = Fig.add_subplot(2, 1, 1)
ax1.plot(eff_veg_shoreline_to_crest_1, 'purple')
ax1.plot(eff_veg_shoreline_to_crest_2, 'orange')
plt.legend([str(name1), str(name2)])
plt.xlabel('Distance Alongshore [m]')
plt.ylabel('Avg Eff Veg Cover: Shoreline to Dune Crest')
ax2 = Fig.add_subplot(2, 1, 2)
vplot = ax2.violinplot([effective_veg_fraction_end_1[np.arange(topo_end_1.shape[0]), dune_crest_end_1],
                        effective_veg_fraction_end_2[np.arange(topo_end_2.shape[0]), dune_crest_end_2]],
                       showmedians=True)
for i, pc in enumerate(vplot['bodies']):
    pc.set_facecolor(['teal', 'red'][i])
labels = [str(name1), str(name2)]
ax2.set_xticks(np.arange(1, len(labels) + 1), labels=labels)
ax2.set_xlim(0.25, len(labels) + 0.75)
plt.ylabel('Avg Eff Veg Cover: Shoreline to Dune Crest')

# -----------------
# Crest-aligned Average Profiles of Topo and Veg
avg_crest_loc_1 = int(np.mean(dune_crest_end_1))
avg_crest_loc_2 = int(np.mean(dune_crest_end_2))
crest_shift_1 = avg_crest_loc_1 - dune_crest_end_1
crest_shift_2 = avg_crest_loc_2 - dune_crest_end_2 - (avg_crest_loc_2 - avg_crest_loc_1)

rows, cols = topo_end_1.shape
col_indices = np.arange(cols)
new_indices_1 = (col_indices - crest_shift_1[:, None]) % cols
topo_centered_1 = topo_end_1[np.arange(rows)[:, None], new_indices_1]
eff_veg_centered_1 = effective_veg_fraction_end_1[np.arange(rows)[:, None], new_indices_1]
H1_cover_centered_1 = H1_cover_end_1[np.arange(rows)[:, None], new_indices_1]
H2_cover_centered_1 = H2_cover_end_1[np.arange(rows)[:, None], new_indices_1]
W_cover_centered_1 = W_cover_end_1[np.arange(rows)[:, None], new_indices_1]

new_indices_2 = (col_indices - crest_shift_2[:, None]) % cols
topo_centered_2 = topo_end_2[np.arange(rows)[:, None], new_indices_2]
eff_veg_centered_2 = effective_veg_fraction_end_2[np.arange(rows)[:, None], new_indices_2]
H1_cover_centered_2 = H1_cover_end_2[np.arange(rows)[:, None], new_indices_2]
H2_cover_centered_2 = H2_cover_end_2[np.arange(rows)[:, None], new_indices_2]
W_cover_centered_2 = W_cover_end_2[np.arange(rows)[:, None], new_indices_2]

mean_topo_profile_1 = np.mean(topo_centered_1, axis=0)
mean_topo_profile_2 = np.mean(topo_centered_2, axis=0)
mean_eff_veg_profile_1 = np.mean(eff_veg_centered_1, axis=0)
mean_eff_veg_profile_2 = np.mean(eff_veg_centered_2, axis=0)
mean_H1_cover_end_1 = np.mean(H1_cover_centered_1, axis=0)
mean_H1_cover_end_2 = np.mean(H1_cover_centered_2, axis=0)
mean_H2_cover_end_1 = np.mean(H2_cover_centered_1, axis=0)
mean_H2_cover_end_2 = np.mean(H2_cover_centered_2, axis=0)

mean_W_cover_end_1 = np.mean(W_cover_centered_1, axis=0)
mean_W_cover_end_2 = np.mean(W_cover_centered_2, axis=0)

fig, ax1 = plt.subplots()
xmin = np.min([np.where(mean_topo_profile_1 >= mhw_end)[0][0], np.where(mean_topo_profile_2 >= mhw_end)[0][0]])
xmax = int(np.max([np.mean(dune_heel_end_1), np.mean(dune_heel_end_2)])) + (20 * cellsize)
x = (np.arange(xmax - xmin) - (avg_crest_loc_1 - xmin)) * cellsize
ymin = np.min([np.min(mean_topo_profile_1[xmin: xmax]), np.min(mean_topo_profile_2[xmin: xmax])])
ax1.set_xlabel('Distance From Dune Crest [m]')
ax1.set_ylabel('Elevation [m NAVD88]')
# ax1.plot(x, mean_topo_profile_1[xmin: xmax], 'k')
ax1.fill_between(x, mean_topo_profile_1[xmin: xmax], ymin, color='teal', alpha=0.5)
# ax1.plot(x, mean_topo_profile_2[xmin: xmax], '--k')
ax1.fill_between(x, mean_topo_profile_2[xmin: xmax], ymin, color='red', alpha=0.5)
ax1.tick_params(axis='y', labelcolor='k')
ax1.legend([name1, name2])
ax2 = ax1.twinx()
# ax2.set_yxlabel('Effective Fractional Cover')
ax2.plot(x, mean_eff_veg_profile_1[xmin: xmax], 'black')
ax2.plot(x, mean_eff_veg_profile_2[xmin: xmax], 'black', ls='--')
ax2.plot(x, mean_H1_cover_end_1[xmin: xmax], 'indigo')
ax2.plot(x, mean_H1_cover_end_2[xmin: xmax], 'indigo', ls='--')
ax2.plot(x, mean_H2_cover_end_1[xmin: xmax], 'goldenrod')
ax2.plot(x, mean_H2_cover_end_2[xmin: xmax], 'goldenrod', ls='--')
ax2.plot(x, mean_W_cover_end_1[xmin: xmax], 'brown')
ax2.plot(x, mean_W_cover_end_2[xmin: xmax], 'brown', ls='--')
ax2.tick_params(axis='y', labelcolor='g')
ax2.legend([name1 + ': Combined Effective', name2 + ': Combined Effective', name1 + ': AMBR', name2 + ': AMBR', name1 + ': UNPA', name2 + ': UNPA', name1 + ': MOCE', name2 + ': MOCE'])
# ax2.legend([name1 + ': AMBR', name2 + ': AMBR', name1 + ': UNPA', name2 + ': UNPA', name1 + ': MOCE', name2 + ': MOCE'])
fig.tight_layout()
plt.show()

# -----------------
# Shoreline-aligned Average Profiles of Topo and Veg
avg_xs_loc_1 = int(np.mean(shoreline_loc_end_1))
avg_xs_loc_2 = int(np.mean(shoreline_loc_end_2))
xs_shift_1 = avg_xs_loc_1 - shoreline_loc_end_1
xs_shift_2 = avg_xs_loc_2 - shoreline_loc_end_2 - (avg_xs_loc_2 - avg_xs_loc_1)

rows, cols = topo_end_1.shape
col_indices = np.arange(cols)
new_indices_1 = (col_indices - xs_shift_1[:, None]) % cols
topo_centered_1 = topo_end_1[np.arange(rows)[:, None], new_indices_1]
eff_veg_centered_1 = effective_veg_fraction_end_1[np.arange(rows)[:, None], new_indices_1]
H1_cover_centered_1 = H1_cover_end_1[np.arange(rows)[:, None], new_indices_1]
H2_cover_centered_1 = H2_cover_end_1[np.arange(rows)[:, None], new_indices_1]
W_cover_centered_1 = W_cover_end_1[np.arange(rows)[:, None], new_indices_1]

new_indices_2 = (col_indices - xs_shift_2[:, None]) % cols
topo_centered_2 = topo_end_2[np.arange(rows)[:, None], new_indices_2]
eff_veg_centered_2 = effective_veg_fraction_end_2[np.arange(rows)[:, None], new_indices_2]
H1_cover_centered_2 = H1_cover_end_2[np.arange(rows)[:, None], new_indices_2]
H2_cover_centered_2 = H2_cover_end_2[np.arange(rows)[:, None], new_indices_2]
W_cover_centered_2 = W_cover_end_2[np.arange(rows)[:, None], new_indices_2]

mean_topo_profile_1 = np.mean(topo_centered_1, axis=0)
mean_topo_profile_2 = np.mean(topo_centered_2, axis=0)
mean_eff_veg_profile_1 = np.mean(eff_veg_centered_1, axis=0)
mean_eff_veg_profile_2 = np.mean(eff_veg_centered_2, axis=0)
mean_H1_cover_end_1 = np.mean(H1_cover_centered_1, axis=0)
mean_H1_cover_end_2 = np.mean(H1_cover_centered_2, axis=0)
mean_H2_cover_end_1 = np.mean(H2_cover_centered_1, axis=0)
mean_H2_cover_end_2 = np.mean(H2_cover_centered_2, axis=0)

mean_W_cover_end_1 = np.mean(W_cover_centered_1, axis=0)
mean_W_cover_end_2 = np.mean(W_cover_centered_2, axis=0)

fig, ax1 = plt.subplots()
xmin = np.min([np.where(mean_topo_profile_1 >= mhw_end)[0][0], np.where(mean_topo_profile_2 >= mhw_end)[0][0]])
xmax = int(np.max([np.mean(dune_heel_end_1), np.mean(dune_heel_end_2)])) + (20 * cellsize)
x = (np.arange(xmax - xmin) - (avg_xs_loc_1 - xmin)) * cellsize
ymin = np.min([np.min(mean_topo_profile_1[xmin: xmax]), np.min(mean_topo_profile_2[xmin: xmax])])
ax1.set_xlabel('Distance From Shoreline [m]')
ax1.set_ylabel('Elevation [m NAVD88]')
# ax1.plot(x, mean_topo_profile_1[xmin: xmax], 'k')
ax1.fill_between(x, mean_topo_profile_1[xmin: xmax], ymin, color='teal', alpha=0.5)
# ax1.plot(x, mean_topo_profile_2[xmin: xmax], '--k')
ax1.fill_between(x, mean_topo_profile_2[xmin: xmax], ymin, color='red', alpha=0.5)
ax1.tick_params(axis='y', labelcolor='k')
ax1.legend([name1, name2])
ax2 = ax1.twinx()
# ax2.set_yxlabel('Effective Fractional Cover')
ax2.plot(x, mean_eff_veg_profile_1[xmin: xmax], 'black')
ax2.plot(x, mean_eff_veg_profile_2[xmin: xmax], 'black', ls='--')
ax2.plot(x, mean_H1_cover_end_1[xmin: xmax], 'indigo')
ax2.plot(x, mean_H1_cover_end_2[xmin: xmax], 'indigo', ls='--')
ax2.plot(x, mean_H2_cover_end_1[xmin: xmax], 'goldenrod')
ax2.plot(x, mean_H2_cover_end_2[xmin: xmax], 'goldenrod', ls='--')
ax2.plot(x, mean_W_cover_end_1[xmin: xmax], 'brown')
ax2.plot(x, mean_W_cover_end_2[xmin: xmax], 'brown', ls='--')
ax2.tick_params(axis='y', labelcolor='g')
ax2.legend([name1 + ': Combined Effective', name2 + ': Combined Effective', name1 + ': AMBR', name2 + ': AMBR', name1 + ': UNPA', name2 + ': UNPA', name1 + ': MOCE', name2 + ': MOCE'])
# ax2.legend([name1 + ': AMBR', name2 + ': AMBR', name1 + ': UNPA', name2 + ': UNPA', name1 + ': MOCE', name2 + ': MOCE'])
fig.tight_layout()
plt.show()

plt.show()

# Print elapsed time
print()
duration = time.time() - start_time
print()
print("Elapsed Time: ", duration, "sec")
