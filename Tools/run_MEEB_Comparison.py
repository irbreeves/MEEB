"""
Script for running and comparing 2 distinct MEEB v2.0 simulations.

IRBR 2 September 2026
"""

import numpy as np
import matplotlib.pyplot as plt
import copy
import gc
from tqdm import trange, tqdm
from joblib import Parallel, delayed

import routines_meeb as routine
from meeb import MEEB


# __________________________________________________________________________________________________________________________________
# VARIABLES AND INITIALIZATIONS

# Init File
start = "Init_NCB-2200-34200_2018_PostFlorence_2m.npy"
startdate = '20181015'

# Simulation Specifications
sim_duration = 32
cellsize = 2  # [m]
name = '+0 vs +4C, 2018-2050, Km 13-28, RSLR=9.6, HWEI=+4.2%, UNPA Monoculture, New Calib, 24Apr26'
name1 = '+0C UNPA'
name2 = '+4C UNPA'
shift_mean_atmospheric_temperature = [0, 4]  # [deg C]
H1_a_proportion = [0, 0]  # Initial proportional of H1 adult fractional cover relative to H2

# Output Data Filenames
outloc = 'Output/SimData/'
save_name = '0-vs-4_UNPA_Km13-28_24Apr26'

# Define Coordinates of Model Domain
ymin = 13000  # Alongshore
ymax = 28000  # Alongshore
xmin = 625  # Cross-shore
xmax = 1475  # Cross-shore
plot_xmin = 0  # Cross-shore plotting
plot_xmax = 1500  # Cross-shore plotting
plot_ymin = 0  # Alongshore plotting
plot_ymax = 15000  # Alongshore plotting

parallel = False
plot = False
plot_maps = False
animate = False
save_data = True
plot_start = 60  # Iteration to start plotting from


# _____________________
# Resize according to cellsize
ymin = int(ymin / cellsize)  # Alongshore
ymax = int(ymax / cellsize)  # Alongshore
xmin = int(xmin / cellsize)  # Cross-shore
xmax = int(xmax / cellsize)  # Cross-shore
plot_xmin = int(plot_xmin / cellsize)  # Cross-shore plotting
plot_xmax = int(plot_xmax / cellsize)  # Cross-shore plotting
plot_ymin = int(plot_ymin / cellsize)  # Alongshore plotting
plot_ymax = int(plot_ymax / cellsize)  # Alongshore plotting

# Load Initial Domains
Init = np.load("Input/" + start)
topo_init = Init[0, ymin: ymax, xmin: xmax].copy()
spec1_init = Init[1, ymin: ymax, xmin: xmax].copy()
spec2_init = Init[2, ymin: ymax, xmin: xmax].copy()

print(name)
print()

del Init
gc.collect()


# __________________________________________________________________________________________________________________________________
# RUN MODEL

def run_sim(name_x, H1_a_prop, atm_temp):

    # _______________________________
    # Create instance of the MEEB class
    meeb = MEEB(
        name=name_x,
        simulation_time_yr=sim_duration,
        alongshore_domain_boundary_min=ymin,
        alongshore_domain_boundary_max=ymax,
        crossshore_domain_boundary_min=xmin,
        crossshore_domain_boundary_max=xmax,
        cellsize=cellsize,
        RSLR=0.0096,
        MHW=0.39,  # [m NAVD88]
        init_filename=start,
        hindcast=False,
        shift_mean_storm_intensity_start=1.485,
        shift_mean_storm_intensity_end=4.199,
        storm_twl_duration_correlation=28.31,
        seeded_random_numbers=True,
        simulation_start_date=startdate,
        storm_list_filename='SyntheticStorms_NCB-CE_10k_1979-2020_Beta0pt039_BermEl1pt78.npy',  # For forecasts
        init_by_file=False,
        init_elev_array=topo_init,
        init_spec1_array=spec1_init,
        init_spec2_array=spec2_init,
        save_frequency=0.1,
        # --- Aeolian --- #
        saltation_length=2,
        saltation_length_rand_deviation=1,
        slabheight=0.02,
        p_dep_sand=0.05,
        p_dep_sand_VegMax=0.47,
        p_ero_sand=0.08,
        entrainment_veg_limit=0.4,
        saltation_veg_limit=0.3,
        repose_threshold=0.3,
        shadowangle=9,
        repose_bare=20,
        repose_veg=30,
        wind_rose=(0.80, 0.03, 0.13, 0.04),
        groundwater_depth=0.4,
        # --- Storms --- #
        Rin=250,
        Cs=0.0311,
        MaxUpSlope=1.5,
        marine_flux_limit=1,
        Kow=0.0003701,
        Kl=0.38,
        mm=1.01,
        overwash_substeps=25,
        beach_equilibrium_slope=0.017,
        swash_erosive_timescale=1.23,
        beach_substeps=1,
        # --- Other --- #
        shift_mean_atmospheric_temperature=atm_temp,
        H1_a_proportion=H1_a_prop,
    )

    # Loop through time
    for time_step in range(int(meeb.iterations)):
        # Run time step
        meeb.update(time_step)

    return meeb


if parallel:
    # Run in parallel
    with routine.tqdm_joblib(tqdm(desc="Comparison", total=2)) as progress_bar:
        meeb1, meeb2 = Parallel(n_jobs=2)(delayed(run_sim)(
            [name1, name2][i],
            H1_a_proportion[i],
            shift_mean_atmospheric_temperature[i],
        ) for i in range(2))

# _______________________________
else:  # Not parallel

    meeb1 = MEEB(
        name=name1,
        simulation_time_yr=sim_duration,
        alongshore_domain_boundary_min=ymin,
        alongshore_domain_boundary_max=ymax,
        crossshore_domain_boundary_min=xmin,
        crossshore_domain_boundary_max=xmax,
        cellsize=cellsize,
        RSLR=0.0096,
        MHW=0.39,  # [m NAVD88]
        init_filename=start,
        hindcast=False,
        shift_mean_storm_intensity_start=1.485,
        shift_mean_storm_intensity_end=4.199,
        storm_twl_duration_correlation=28.31,
        seeded_random_numbers=True,
        simulation_start_date=startdate,
        storm_list_filename='SyntheticStorms_NCB-CE_10k_1979-2020_Beta0pt039_BermEl1pt78.npy',  # For forecasts
        init_by_file=False,
        init_elev_array=topo_init,
        init_spec1_array=spec1_init,
        init_spec2_array=spec2_init,
        save_frequency=0.1,
        # --- Aeolian --- #
        saltation_length=2,
        saltation_length_rand_deviation=1,
        slabheight=0.02,
        p_dep_sand=0.05,
        p_dep_sand_VegMax=0.47,
        p_ero_sand=0.08,
        entrainment_veg_limit=0.4,
        saltation_veg_limit=0.3,
        repose_threshold=0.3,
        shadowangle=9,
        repose_bare=20,
        repose_veg=30,
        wind_rose=(0.80, 0.03, 0.13, 0.04),
        groundwater_depth=0.4,
        # --- Storms --- #
        Rin=250,
        Cs=0.0311,
        MaxUpSlope=1.5,
        marine_flux_limit=1,
        Kow=0.0003701,
        Kl=0.38,
        mm=1.01,
        overwash_substeps=25,
        beach_equilibrium_slope=0.017,
        swash_erosive_timescale=1.23,
        beach_substeps=1,
        # --- Other --- #
        shift_mean_atmospheric_temperature=shift_mean_atmospheric_temperature[0],
        H1_a_proportion=H1_a_proportion[0],
    )

    meeb2 = MEEB(
        name=name2,
        simulation_time_yr=sim_duration,
        alongshore_domain_boundary_min=ymin,
        alongshore_domain_boundary_max=ymax,
        crossshore_domain_boundary_min=xmin,
        crossshore_domain_boundary_max=xmax,
        cellsize=cellsize,
        RSLR=0.0096,
        MHW=0.39,  # [m NAVD88]
        init_filename=start,
        hindcast=False,
        shift_mean_storm_intensity_start=1.485,
        shift_mean_storm_intensity_end=4.199,
        storm_twl_duration_correlation=28.31,
        seeded_random_numbers=True,
        simulation_start_date=startdate,
        storm_list_filename='SyntheticStorms_NCB-CE_10k_1979-2020_Beta0pt039_BermEl1pt78.npy',  # For forecasts
        init_by_file=False,
        init_elev_array=topo_init,
        init_spec1_array=spec1_init,
        init_spec2_array=spec2_init,
        save_frequency=0.1,
        # --- Aeolian --- #
        saltation_length=2,
        saltation_length_rand_deviation=1,
        slabheight=0.02,
        p_dep_sand=0.05,
        p_dep_sand_VegMax=0.47,
        p_ero_sand=0.08,
        entrainment_veg_limit=0.4,
        saltation_veg_limit=0.3,
        repose_threshold=0.3,
        shadowangle=9,
        repose_bare=20,
        repose_veg=30,
        wind_rose=(0.80, 0.03, 0.13, 0.04),
        groundwater_depth=0.4,
        # --- Storms --- #
        Rin=250,
        Cs=0.0311,
        MaxUpSlope=1.5,
        marine_flux_limit=1,
        Kow=0.0003701,
        Kl=0.38,
        mm=1.01,
        overwash_substeps=25,
        beach_equilibrium_slope=0.017,
        swash_erosive_timescale=1.23,
        beach_substeps=1,
        # --- Other --- #
        shift_mean_atmospheric_temperature=shift_mean_atmospheric_temperature[1],
        H1_a_proportion=H1_a_proportion[1],
    )

    # Run not in parallel (uses less memory)
    with trange(int(meeb1.iterations)) as t:
        for time_step in t:
            # Run time step
            meeb1.update(time_step)
            meeb2.update(time_step)
            # Update progress bar
            t.set_postfix({'Year': "{:.2f}".format((time_step + 1) / meeb1.iterations_per_cycle) + '/' + "{:.2f}".format(meeb1.simulation_time_yr)})
            t.update()

# Save data
if save_data:
    # Combine Elevation and Vegetation Data
    elev_all = np.zeros((2, *meeb1.topo_TS.shape), dtype=np.float32)
    veg_all = np.zeros((2, *meeb1.veg_fraction_TS.shape), dtype=np.float32)

    elev_all[0, :, :, :] = meeb1.topo_TS
    veg_all[0, :, :, :, ] = meeb1.veg_fraction_TS

    elev_all[1, :, :, :] = meeb2.topo_TS
    veg_all[1, :, :, :, ] = meeb2.veg_fraction_TS

    # Specify path
    save_name_elev = save_name + "_Elevation.npy"
    save_loc_elev = "Output/SimData/" + save_name_elev
    save_name_veg = save_name + "_Vegetation.npy"
    save_loc_veg = "Output/SimData/" + save_name_veg

    # Save data
    np.save(save_loc_elev, elev_all)
    np.save(save_loc_veg, veg_all)

print()

if plot:

    # __________________________________________________________________________________________________________________________________
    # ASSESS MODEL RESULTS

    # Topo change
    topo_start = meeb1.topo_TS[:, :, plot_start].astype(np.float32)  # [m NAVDD88]
    mhw_end = meeb1.MHW  # [m NAVD88]

    topo_end_1 = meeb1.topo_TS[:, :, -1].astype(np.float32)  # [m NAVDD88]
    topo_change_1 = topo_end_1 - topo_start  # [m]

    topo_end_2 = meeb2.topo_TS[:, :, -1].astype(np.float32)  # [m NAVDD88]
    topo_change_2 = topo_end_2 - topo_start  # [m]

    # Veg change
    timesteps_in_year = int(1 / meeb1.save_frequency)
    veg_TS_1 = meeb1.veg_fraction_TS[:, :, 2, :] + meeb1.veg_fraction_TS[:, :, 4, :] + meeb1.veg_fraction_TS[:, :, 6, :] + meeb1.veg_fraction_TS[:, :, 7, :]
    veg_start = np.mean(veg_TS_1[:, :, plot_start: plot_start + timesteps_in_year], axis=2)

    veg_end_1 = np.mean(veg_TS_1[:, :, -timesteps_in_year:-1], axis=2)
    veg_change_1 = veg_end_1 - veg_start  # [m]

    veg_TS_2 = meeb2.veg_fraction_TS[:, :, 2, :] + meeb2.veg_fraction_TS[:, :, 4, :] + meeb2.veg_fraction_TS[:, :, 6, :] + meeb2.veg_fraction_TS[:, :, 7, :]
    veg_end_2 = np.mean(veg_TS_2[:, :, -timesteps_in_year:-1], axis=2)
    veg_change_2 = veg_end_2 - veg_start  # [m]

    effective_veg_fraction_end_1 = (np.mean(meeb1.veg_fraction_TS[:, :, 2, -timesteps_in_year:-1], axis=2) * meeb1._H1_a_relative_effectiveness
                                    + np.mean(meeb1.veg_fraction_TS[:, :, 4, -timesteps_in_year:-1], axis=2) * meeb1._H2_a_relative_effectiveness
                                    + np.mean(meeb1.veg_fraction_TS[:, :, 6, -timesteps_in_year:-1], axis=2) * meeb1._W_a_relative_effectiveness
                                    + np.mean(meeb1.veg_fraction_TS[:, :, 7, -timesteps_in_year:-1], axis=2) * meeb1._W_d_relative_effectiveness)

    effective_veg_fraction_end_2 = (np.mean(meeb2.veg_fraction_TS[:, :, 2, -timesteps_in_year:-1], axis=2) * meeb2._H1_a_relative_effectiveness
                                    + np.mean(meeb2.veg_fraction_TS[:, :, 4, -timesteps_in_year:-1], axis=2) * meeb2._H2_a_relative_effectiveness
                                    + np.mean(meeb2.veg_fraction_TS[:, :, 6, -timesteps_in_year:-1], axis=2) * meeb2._W_a_relative_effectiveness
                                    + np.mean(meeb2.veg_fraction_TS[:, :, 7, -timesteps_in_year:-1], axis=2) * meeb2._W_d_relative_effectiveness)

    # Woody veg change
    woody_TS_1 = meeb1.veg_fraction_TS[:, :, 6, :] + meeb1.veg_fraction_TS[:, :, 7, :]
    woody_start = woody_TS_1[:, :, plot_start]

    woody_end_1 = woody_TS_1[:, :, -1]
    woody_change_1 = woody_end_1 - woody_start  # [m]

    woody_TS_2 = meeb2.veg_fraction_TS[:, :, 6, :] + meeb2.veg_fraction_TS[:, :, 7, :]
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
    dune_toe_start = routine.foredune_toe(topo_start, dune_crest_start, meeb1.MHW_init, not_gap_start, cellsize)
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
    shoreline_loc_start = meeb1.x_s_TS[plot_start, :].astype(int)
    shoreline_loc_end_1 = meeb1.x_s_TS[-1, :].astype(int)
    shoreline_loc_end_2 = meeb2.x_s_TS[-1, :].astype(int)

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
    H1_cover_end_1 = np.mean(meeb1.veg_fraction_TS[:, :, 2, -timesteps_in_year:-1], axis=2)
    H2_cover_end_1 = np.mean(meeb1.veg_fraction_TS[:, :, 4, -timesteps_in_year:-1], axis=2)
    W_cover_end_1 = np.mean(meeb1.veg_fraction_TS[:, :, 6, -timesteps_in_year:-1], axis=2) + np.mean(meeb1.veg_fraction_TS[:, :, 7, -timesteps_in_year:-1], axis=2)
    H1_cover_end_2 = np.mean(meeb2.veg_fraction_TS[:, :, 2, -timesteps_in_year:-1], axis=2)
    H2_cover_end_2 = np.mean(meeb2.veg_fraction_TS[:, :, 4, -timesteps_in_year:-1], axis=2)
    W_cover_end_2 = np.mean(meeb2.veg_fraction_TS[:, :, 6, -timesteps_in_year:-1], axis=2) + np.mean(meeb2.veg_fraction_TS[:, :, 7, -timesteps_in_year:-1], axis=2)

    H1_avg_end_1 = np.average(H1_cover_end_1[(H1_cover_end_1 + H2_cover_end_1) > 0.01])
    H2_avg_end_1 = np.average(H2_cover_end_1[(H1_cover_end_1 + H2_cover_end_1) > 0.01])
    W_avg_end_1 = np.average(W_cover_end_1[W_cover_end_1 > 0.01])

    H1_avg_end_2 = np.average(H1_cover_end_2[(H1_cover_end_2 + H2_cover_end_2) > 0.01])
    H2_avg_end_2 = np.average(H2_cover_end_2[(H1_cover_end_2 + H2_cover_end_2) > 0.01])
    W_avg_end_2 = np.average(W_cover_end_2[W_cover_end_2 > 0.01])

    # Species Presence
    H1_cell_presence_start_1 = np.sum(meeb1.veg_fraction_TS[:, :, 2, plot_start] > 0)
    H2_cell_presence_start_1 = np.sum(meeb1.veg_fraction_TS[:, :, 4, plot_start] > 0)
    W_cell_presence_start_1 = np.sum(meeb1.veg_fraction_TS[:, :, 6, plot_start] > 0) + np.sum(meeb1.veg_fraction_TS[:, :, 7, 0] > 0)

    H1_cell_presence_start_2 = np.sum(meeb2.veg_fraction_TS[:, :, 2, plot_start] > 0)
    H2_cell_presence_start_2 = np.sum(meeb2.veg_fraction_TS[:, :, 4, plot_start] > 0)
    W_cell_presence_start_2 = np.sum(meeb2.veg_fraction_TS[:, :, 6, plot_start] > 0) + np.sum(meeb2.veg_fraction_TS[:, :, 7, 0] > 0)

    H1_cell_presence_end_1 = np.sum(meeb1.veg_fraction_TS[:, :, 2, -1] > 0)
    H2_cell_presence_end_1 = np.sum(meeb1.veg_fraction_TS[:, :, 4, -1] > 0)
    W_cell_presence_end_1 = np.sum(meeb1.veg_fraction_TS[:, :, 6, -1] > 0) + np.sum(meeb1.veg_fraction_TS[:, :, 7, -1] > 0)

    H1_cell_presence_end_2 = np.sum(meeb2.veg_fraction_TS[:, :, 2, -1] > 0)
    H2_cell_presence_end_2 = np.sum(meeb2.veg_fraction_TS[:, :, 4, -1] > 0)
    W_cell_presence_end_2 = np.sum(meeb2.veg_fraction_TS[:, :, 6, -1] > 0) + np.sum(meeb2.veg_fraction_TS[:, :, 7, -1] > 0)

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
    # Overwash Flux
    Fig = plt.figure(figsize=(14, 10))
    Fig.suptitle('Overwash Flux', fontsize=13)
    Fig.add_subplot(2, 1, 1)
    x = np.arange(0, meeb1.topo_TS[plot_ymin:plot_ymax, :, :].shape[0] * cellsize, 2)
    plt.plot(x, meeb1.OWflux_cumul[plot_ymin:plot_ymax], 'teal')
    plt.plot(x, meeb2.OWflux_cumul[plot_ymin:plot_ymax], 'red')
    plt.legend(['End: ' + str(name1), 'End: ' + str(name2)])
    plt.xlabel('Distance Alongshore [m]')
    plt.ylabel('Cumulative Overwash Flux [m^3]')
    Fig.add_subplot(2, 2, 3)
    plt.boxplot([meeb1.OWflux_cumul[plot_ymin:plot_ymax], meeb2.OWflux_cumul[plot_ymin:plot_ymax]], labels=[name1, name2])
    plt.ylabel('Cumulative Overwash Flux [m^3]')
    Fig.add_subplot(2, 2, 4)
    OW1_nozero = [i for i in meeb1.OWflux_cumul[plot_ymin:plot_ymax] if i > 0]  # Filter out all locations where cumul overwash flux was 0
    OW2_nozero = [i for i in meeb2.OWflux_cumul[plot_ymin:plot_ymax] if i > 0]  # Filter out all locations where cumul overwash flux was 0
    plt.boxplot([OW1_nozero, OW2_nozero], labels=[name1, name2])
    plt.ylabel('Cumulative Overwash Flux, where > 0 [m^3]')

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
        H1_avg_cover_TS_1.append(np.mean(meeb1.veg_fraction_TS[plot_ymin:plot_ymax, :, 2, ts][np.logical_and(topo_end_1 > meeb1.MHW, (H1_cover_end_1 + H2_cover_end_1) > 0.01)]))
        H2_avg_cover_TS_1.append(np.mean(meeb1.veg_fraction_TS[plot_ymin:plot_ymax, :, 4, ts][np.logical_and(topo_end_1 > meeb1.MHW, (H1_cover_end_1 + H2_cover_end_1) > 0.01)]))
        Hboth_avg_cover_TS_1.append(np.mean(meeb1.veg_fraction_TS[plot_ymin:plot_ymax, :, 2, ts][np.logical_and(topo_end_1 > meeb1.MHW, (H1_cover_end_1 + H2_cover_end_1) > 0.01)]) + np.mean(meeb1.veg_fraction_TS[plot_ymin:plot_ymax, :, 4, ts][np.logical_and(topo_end_1 > meeb1.MHW, (H1_cover_end_1 + H2_cover_end_1) > 0.01)]))
        H1_avg_cover_TS_2.append(np.mean(meeb2.veg_fraction_TS[plot_ymin:plot_ymax, :, 2, ts][np.logical_and(topo_end_2 > meeb2.MHW, (H1_cover_end_2 + H2_cover_end_2) > 0.01)]))
        H2_avg_cover_TS_2.append(np.mean(meeb2.veg_fraction_TS[plot_ymin:plot_ymax, :, 4, ts][np.logical_and(topo_end_2 > meeb2.MHW, (H1_cover_end_2 + H2_cover_end_2) > 0.01)]))
        Hboth_avg_cover_TS_2.append(np.mean(meeb2.veg_fraction_TS[plot_ymin:plot_ymax, :, 2, ts][np.logical_and(topo_end_2 > meeb2.MHW, (H1_cover_end_2 + H2_cover_end_2) > 0.01)]) + np.mean(meeb2.veg_fraction_TS[plot_ymin:plot_ymax, :, 4, ts][np.logical_and(topo_end_2 > meeb2.MHW, (H1_cover_end_2 + H2_cover_end_2) > 0.01)]))

    Fig = plt.figure(figsize=(14, 10))
    Fig.add_subplot(2, 1, 1)
    x = np.arange(0, len(H1_avg_cover_TS_1) * meeb1.save_frequency, meeb1.save_frequency)
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
    x_s_end_1 = routine.ocean_shoreline(topo_end_1, meeb1.MHW)
    x_s_end_2 = routine.ocean_shoreline(topo_end_2, meeb2.MHW)
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

    # -----------------
    # Average Aeolian Flux
    average_aeolian_flux_1 = meeb1.cumulative_max_aeolian_flux[plot_ymin:plot_ymax, :] / meeb1.iterations
    average_aeolian_flux_2 = meeb2.cumulative_max_aeolian_flux[plot_ymin:plot_ymax, :] / meeb2.iterations
    Fig = plt.figure(figsize=(14, 10))
    ax1 = Fig.add_subplot(131)
    cax1 = ax1.matshow(average_aeolian_flux_1)
    plt.xlabel('Meters Cross-shore')
    plt.ylabel('Meters Alongshore')
    plt.title(name1)
    cbar = Fig.colorbar(cax1)
    cbar.set_label('Average Maximum Aeolian Flux [m^3/y]', rotation=270, labelpad=20)
    ax2 = Fig.add_subplot(132)
    cax2 = ax2.matshow(average_aeolian_flux_2)
    plt.xlabel('Meters Cross-shore')
    plt.ylabel('Meters Alongshore')
    plt.title(name2)
    cbar = Fig.colorbar(cax2)
    cbar.set_label('Average Maximum Aeolian Flux [m^3/y]', rotation=270, labelpad=20)
    ax3 = Fig.add_subplot(133)
    diff = average_aeolian_flux_2 - average_aeolian_flux_1
    diff_lim = max(np.abs(np.min(diff)), np.max(diff))
    cax3 = ax3.matshow(diff, cmap='bwr_r', vmin=-diff_lim, vmax=diff_lim)
    plt.plot(dune_crest_end_1, np.arange(dune_crest_end_1.shape[0]), 'black', linewidth=1)
    plt.plot(dune_crest_end_2, np.arange(dune_crest_end_2.shape[0]), '--k', linewidth=1)
    plt.xlabel('Meters Cross-shore')
    plt.ylabel('Meters Alongshore')
    cbar = Fig.colorbar(cax3)
    cbar.set_label('Difference (Sim2 - Sim1) [m^3/y]', rotation=270, labelpad=20)

    plt.show()
