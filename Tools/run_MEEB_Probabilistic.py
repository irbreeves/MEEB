"""
Probabilistic framework for running MEEB simulations. Generates probabilistic projections of future change.

IRBR 16 October 2024
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import gc
import scipy
from tqdm import tqdm
from matplotlib import colors
from joblib import Parallel, delayed
sys.path.append(os.getcwd())

import routines_meeb as routine
from meeb import MEEB


# __________________________________________________________________________________________________________________________________
# FUNCTIONS


def run_individual_sim(rslr, shift_mean_storm_intensity):
    """Runs uniqe individual MEEB simulation."""

    # Create an instance of the MEEB class
    meeb = MEEB(
        name=name,
        simulation_time_yr=sim_duration,
        alongshore_domain_boundary_min=ymin,
        alongshore_domain_boundary_max=ymax,
        crossshore_domain_boundary_min=xmin,
        crossshore_domain_boundary_max=xmax,
        cellsize=cellsize,
        RSLR=rslr,
        shift_mean_storm_intensity=shift_mean_storm_intensity,
        MHW=MHW_init,
        init_filename=start,
        hindcast=False,
        seeded_random_numbers=False,
        simulation_start_date=startdate,
        storm_timeseries_filename='StormTimeSeries_1979-2020_NCB-CE_Beta0pt039_BermEl1pt78.npy',
        storm_list_filename='SyntheticStorms_NCB-CE_10k_1979-2020_Beta0pt039_BermEl1pt78.npy',
        save_frequency=save_frequency,
        init_by_file=False,
        init_elev_array=topo_start,
        init_spec1_array=spec1_start,
        init_spec2_array=spec2_start,
        # --- Aeolian --- #
        saltation_length=2,
        saltation_length_rand_deviation=1,
        p_dep_sand=0.22,
        p_dep_sand_VegMax=0.54,
        p_ero_sand=0.10,
        entrainment_veg_limit=0.10,
        saltation_veg_limit=0.35,
        shadowangle=12,
        repose_bare=20,
        repose_veg=30,
        wind_rose=(0.81, 0.04, 0.06, 0.09),  # (right, down, left, up)
        groundwater_depth=0.4,
        # --- Storms --- #
        Rin=249,
        Cs=0.0283,
        MaxUpSlope=1.5,
        marine_flux_limit=1,
        Kow=0.0001684,
        mm=1.04,
        overwash_substeps=25,
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
        alongshore_section_length=25,
        estimate_shoreface_parameters=True,
        # --- Veg --- #
        sp1_lateral_probability=0.2,
        sp2_lateral_probability=0.2,
        sp1_pioneer_probability=0.05,
        sp2_pioneer_probability=0.05,
    )

    # Loop through time
    for time_step in range(int(meeb.iterations)):
        # Run time step
        meeb.update(time_step)

    # Topo change
    topo_start_sim = meeb.topo_TS[:, :, 0]  # [m NAVDD88]
    topo_end_sim = meeb.topo_TS[:, :, -1]  # [m NAVDD88]

    # Subaerial mask
    subaerial_mask = topo_end_sim > MHW_init  # [bool] Mask for every cell above initial MHW; Note: this should be changed if modeling sea-level fall

    topo_change_sim_TS = np.zeros(meeb.topo_TS.shape, dtype=np.float32)
    for ts in range(meeb.topo_TS.shape[2]):
        topo_change_ts = (meeb.topo_TS[:, :, ts] - topo_start_sim) * subaerial_mask  # Disregard change that is not subaerial
        topo_change_sim_TS[:, :, ts] = topo_change_ts

    # Create classified map
    elevation_classification = classify_topo_change(meeb.topo_TS.shape[2], topo_change_sim_TS)
    inundation_classification = classify_overwash_frequency(meeb.topo_TS.shape[2], meeb.storm_inundation_TS, meeb.topo_TS, meeb.MHW_init, meeb.RSLR)
    habitat_state_classification = classify_ecogeomorphic_habitat_state(meeb.topo_TS.shape[2], meeb.topo_TS, meeb.veg_TS, meeb.MHW_init, meeb.RSLR, vegetated_threshold=0.25)

    classes = [elevation_classification, habitat_state_classification, inundation_classification]

    del elevation_classification, inundation_classification, habitat_state_classification, meeb, topo_change_sim_TS
    gc.collect()

    return classes


def classify_topo_change(TS, topo_change_sim_TS):
    """Classify according to range of elevation change."""

    topo_change_bin = np.zeros([elev_num_classes, num_saves, longshore, crossshore], dtype=np.float32)

    for b in range(len(elev_class_edges) - 1):
        lower = elev_class_edges[b]
        upper = elev_class_edges[b + 1]

        for ts in range(TS):
            bin_change = np.logical_and(lower < topo_change_sim_TS[:, :, ts], topo_change_sim_TS[:, :, ts] <= upper).astype(int)
            topo_change_bin[b, ts, :, :] += bin_change

    return topo_change_bin


def classify_ecogeomorphic_state(TS, topo_TS, veg_TS, mhw_init, rslr, vegetated_threshold):
    """Classify by ecogeomorphic state using topography and vegetation."""

    state_classes = np.zeros([state_num_classes, num_saves, longshore, crossshore], dtype=np.float32)  # Initialize

    # Run Categorization
    # Loop through saved timesteps
    for ts in range(TS):

        # Find MHW for this time step
        MHW = mhw_init + rslr * ts * save_frequency

        # Smooth topography to remove small-scale variability
        topo = scipy.ndimage.gaussian_filter(topo_TS[:, :, ts], 5, mode='reflect')

        # Find dune crest, toe, and heel lines
        dune_crestline, not_gap = routine.foredune_crest(topo, MHW, cellsize)
        dune_toeline = routine.foredune_toe(topo, dune_crestline, MHW, not_gap, cellsize)
        dune_heelline = routine.foredune_heel(topo, dune_crestline, not_gap, cellsize, threshold=0.6)

        # Make boolean maps of locations landward and seaward of dune lines
        seaward_of_dunetoe = np.zeros(topo.shape, dtype=bool)
        for ls in range(longshore):
            seaward_of_dunetoe[ls, :dune_toeline[ls]] = True

        landward_of_dunetoe = np.zeros(topo.shape, dtype=bool)
        for ls in range(longshore):
            landward_of_dunetoe[ls, dune_toeline[ls]:] = True

        seaward_of_duneheel = np.zeros(topo.shape, dtype=bool)
        for ls in range(longshore):
            seaward_of_duneheel[ls, :dune_heelline[ls]] = True

        landward_of_duneheel = np.zeros(topo.shape, dtype=bool)
        for ls in range(longshore):
            landward_of_duneheel[ls, dune_heelline[ls]:] = True

        seaward_of_dunecrest = np.zeros(topo.shape, dtype=bool)
        for ls in range(longshore):
            seaward_of_dunecrest[ls, :dune_crestline[ls]] = True

        landward_of_dunecrest = np.zeros(topo.shape, dtype=bool)
        for ls in range(longshore):
            landward_of_dunecrest[ls, dune_crestline[ls]:] = True

        # Make boolean maps of locations fronted by dune gaps
        fronting_dune_gap_simple = np.ones(topo.shape, dtype=bool)  # [bool] Areas directly orthogonal to dune gaps
        for ls in range(longshore):
            if not_gap[ls]:
                fronting_dune_gap_simple[ls, :] = False

        fronting_dune_gap = np.zeros(topo.shape, dtype=bool)  # Areas with 90 deg spread of dune gaps
        for ls in range(longshore):
            if not not_gap[ls]:
                crest_loc = dune_crestline[ls]
                temp = np.ones([crossshore - crest_loc, longshore], dtype=np.float32)
                right_diag = np.tril(temp, k=ls)
                left_diag = np.fliplr(np.tril(temp, k=len(dune_crestline) - ls - 1))
                spread = np.rot90(right_diag * left_diag, 1)
                fronting_dune_gap[:, crest_loc:] = np.flipud(np.logical_or(fronting_dune_gap[:, crest_loc:], spread))
                trim = max(0, right_diag.shape[0] - crest_loc)
                fronting_dune_gap[:, max(0, crest_loc - right_diag.shape[0]): crest_loc] = np.fliplr(fronting_dune_gap[:, crest_loc:])[:, trim:]

        # Find dune crest heights
        dune_crestheight = np.zeros(longshore, dtype=np.float32)
        for ls in range(longshore):
            dune_crestheight[ls] = topo[ls, dune_crestline[ls]]  # [m NAVD88]

        # Make boolean maps of locations fronted by low dunes (quite similar to fronting_dune_gaps)
        fronting_low_dune = np.zeros(topo.shape, dtype=bool)
        for ls in range(longshore):
            if dune_crestheight[ls] - MHW < 2.7:
                crest_loc = dune_crestline[ls]
                temp = np.ones([crossshore - crest_loc, longshore], dtype=np.float32)
                right_diag = np.tril(temp, k=ls)
                left_diag = np.fliplr(np.tril(temp, k=len(dune_crestline) - ls - 1))
                spread = np.rot90(right_diag * left_diag, 1)
                fronting_low_dune[:, crest_loc:] = np.flipud(np.logical_or(fronting_low_dune[:, crest_loc:], spread))

        # Smooth vegetation to remove small-scale variability
        veg = scipy.ndimage.gaussian_filter(veg_TS[:, :, ts], 5, mode='constant')
        unvegetated = veg < vegetated_threshold  # [bool] Map of unvegetated areas

        # Categorize Cells of Model Domain

        # Subaqueous: below MHW
        subaqueous_class_TS = topo < MHW
        state_classes[0, ts, :, :] += subaqueous_class_TS

        # Dune: between toe and heel, and not fronting dune gap
        dune_class_TS = landward_of_dunetoe * seaward_of_duneheel * ~fronting_dune_gap_simple * ~subaqueous_class_TS
        state_classes[2, ts, :, :] += dune_class_TS

        # Beach: seaward of dune crest, and not dune or subaqueous
        beach_class_TS = seaward_of_dunecrest * ~dune_class_TS * ~subaqueous_class_TS
        state_classes[1, ts, :, :] += beach_class_TS

        # Washover: landward of dune crest, unvegetated, and fronting dune gap
        washover_class_TS = landward_of_dunecrest * unvegetated * fronting_dune_gap * ~dune_class_TS * ~beach_class_TS * ~subaqueous_class_TS
        state_classes[3, ts, :, :] += washover_class_TS

        # Interior: all other cells landward of dune crest
        interior_class_TS = landward_of_dunecrest * ~washover_class_TS * ~dune_class_TS * ~beach_class_TS * ~subaqueous_class_TS
        state_classes[4, ts, :, :] += interior_class_TS

    del topo, seaward_of_dunecrest, seaward_of_duneheel, seaward_of_dunetoe, landward_of_dunecrest, landward_of_duneheel, landward_of_dunetoe, subaqueous_class_TS, dune_class_TS, beach_class_TS, washover_class_TS, interior_class_TS
    gc.collect()

    return state_classes


def classify_ecogeomorphic_habitat_state(TS, topo_TS, veg_TS, mhw_init, rslr, vegetated_threshold):
    """Classify by ecogeomorphic state using topography and vegetatio, with focus on bird and turtle habitat."""

    habitat_state_classes = np.zeros([habitat_state_num_classes, num_saves, longshore, crossshore], dtype=np.float32)  # Initialize

    # Run Categorization
    # Loop through saved timesteps
    for ts in range(TS):

        # Find MHW for this time step
        MHW = mhw_init + rslr * ts * save_frequency

        # Smooth topography to remove small-scale variability
        topo = scipy.ndimage.gaussian_filter(topo_TS[:, :, ts], 5, mode='reflect')

        # Find dune crest, toe, and heel lines
        dune_crestline, not_gap = routine.foredune_crest(topo, MHW, cellsize)
        dune_toeline = routine.foredune_toe(topo, dune_crestline, MHW, not_gap, cellsize)
        dune_heelline = routine.foredune_heel(topo, dune_crestline, not_gap, cellsize, threshold=0.6)

        # Beach slope
        x_s = routine.ocean_shoreline(topo, MHW)
        toe_elev = topo[np.arange(longshore), dune_toeline]
        beach_width = (dune_toeline - x_s) * cellsize
        for ls in range(longshore):
            if beach_width[ls] <= cellsize:
                if ls == 0:
                    beach_width[ls] = cellsize * 2
                else:
                    beach_width[ls] = beach_width[ls - 1]
        beach_slopes = (toe_elev - MHW) / beach_width

        # Make boolean maps of locations landward and seaward of dune lines
        seaward_of_dunetoe = np.zeros(topo.shape, dtype=bool)
        for ls in range(longshore):
            seaward_of_dunetoe[ls, :dune_toeline[ls]] = True

        landward_of_dunetoe = np.zeros(topo.shape, dtype=bool)
        for ls in range(longshore):
            landward_of_dunetoe[ls, dune_toeline[ls]:] = True

        seaward_of_duneheel = np.zeros(topo.shape, dtype=bool)
        for ls in range(longshore):
            seaward_of_duneheel[ls, :dune_heelline[ls]] = True

        landward_of_duneheel = np.zeros(topo.shape, dtype=bool)
        for ls in range(longshore):
            landward_of_duneheel[ls, dune_heelline[ls]:] = True

        seaward_of_dunecrest = np.zeros(topo.shape, dtype=bool)
        for ls in range(longshore):
            seaward_of_dunecrest[ls, :dune_crestline[ls]] = True

        landward_of_dunecrest = np.zeros(topo.shape, dtype=bool)
        for ls in range(longshore):
            landward_of_dunecrest[ls, dune_crestline[ls]:] = True

        # Make boolean maps of locations fronted by dune gaps
        fronting_dune_gap_simple = np.ones(topo.shape, dtype=bool)  # [bool] Areas directly orthogonal to dune gaps
        for ls in range(longshore):
            if not_gap[ls]:
                fronting_dune_gap_simple[ls, :] = False

        fronting_dune_gap = np.zeros(topo.shape, dtype=bool)  # Areas with 90 deg spread of dune gaps
        for ls in range(longshore):
            if not not_gap[ls]:
                crest_loc = dune_crestline[ls]
                temp = np.ones([crossshore - crest_loc, longshore], dtype=np.float32)
                right_diag = np.tril(temp, k=ls)
                left_diag = np.fliplr(np.tril(temp, k=len(dune_crestline) - ls - 1))
                spread = np.rot90(right_diag * left_diag, 1)
                fronting_dune_gap[:, crest_loc:] = np.flipud(np.logical_or(fronting_dune_gap[:, crest_loc:], spread))
                trim = max(0, right_diag.shape[0] - crest_loc)
                fronting_dune_gap[:, max(0, crest_loc - right_diag.shape[0]): crest_loc] = np.fliplr(fronting_dune_gap[:, crest_loc:])[:, trim:]

        # Find dune crest heights
        dune_crestheight = np.zeros(longshore, dtype=np.float32)
        for ls in range(longshore):
            dune_crestheight[ls] = topo[ls, dune_crestline[ls]]  # [m NAVD88]

        # Make boolean maps of locations fronted by low dunes (quite similar to fronting_dune_gaps)
        fronting_low_dune = np.zeros(topo.shape, dtype=bool)
        for ls in range(longshore):
            if dune_crestheight[ls] - MHW < 2.7:
                crest_loc = dune_crestline[ls]
                temp = np.ones([crossshore - crest_loc, longshore], dtype=np.float32)
                right_diag = np.tril(temp, k=ls)
                left_diag = np.fliplr(np.tril(temp, k=len(dune_crestline) - ls - 1))
                spread = np.rot90(right_diag * left_diag, 1)
                fronting_low_dune[:, crest_loc:] = np.flipud(np.logical_or(fronting_low_dune[:, crest_loc:], spread))

        # Make boolean maps of areas alongshore with beach slope greater than threshold (i.e., turtle habitat)
        steep_beach_threshold = 0.02
        steep_beach_slope = np.rot90(np.array([beach_slopes > steep_beach_threshold] * crossshore), -1)

        # Smooth vegetation to remove small-scale variability
        veg = scipy.ndimage.gaussian_filter(veg_TS[:, :, ts], 5, mode='constant')
        unvegetated = veg < vegetated_threshold  # [bool] Map of unvegetated areas

        # Categorize Cells of Model Domain

        # Subaqueous: below MHW
        subaqueous_class_TS = topo < MHW
        habitat_state_classes[0, ts, :, :] += subaqueous_class_TS

        # Dune: between toe and heel, and not fronting dune gap
        dune_class_TS = landward_of_dunetoe * seaward_of_duneheel * ~fronting_dune_gap_simple * ~subaqueous_class_TS
        habitat_state_classes[3, ts, :, :] += dune_class_TS

        # Beach-Steep: seaward of dune crest, not dune or subaqueous, beach slope > threshold
        beach_class_steep_TS = seaward_of_dunecrest * ~dune_class_TS * ~subaqueous_class_TS * steep_beach_slope
        habitat_state_classes[1, ts, :, :] += beach_class_steep_TS

        # Beach-Shallow: seaward of dune crest, and not dune or subaqueous, , beach slope >= threshold
        beach_class_shallow_TS = seaward_of_dunecrest * ~dune_class_TS * ~subaqueous_class_TS * ~steep_beach_slope
        habitat_state_classes[2, ts, :, :] += beach_class_shallow_TS

        # Washover: landward of dune crest, unvegetated, and fronting dune gap
        washover_class_TS = landward_of_dunecrest * unvegetated * fronting_dune_gap * ~dune_class_TS * ~beach_class_steep_TS * ~beach_class_shallow_TS * ~subaqueous_class_TS
        habitat_state_classes[4, ts, :, :] += washover_class_TS

        # Interior: all other cells landward of dune crest
        interior_class_TS = landward_of_dunecrest * ~washover_class_TS * ~dune_class_TS * ~beach_class_steep_TS * ~beach_class_shallow_TS * ~subaqueous_class_TS
        habitat_state_classes[5, ts, :, :] += interior_class_TS

    return habitat_state_classes


def classify_inundation(TS, inundation_TS, topo, mhw_init, rslr):
    """Classify according to inundation from storms (active) or RSLR (passive), cumulative through time."""

    inundation = np.zeros([num_saves, longshore, crossshore], dtype=np.float32)

    inundation_TS = inundation_TS > 0  # Convert to bool

    for ts in range(TS):
        MHW = mhw_init + rslr * ts * save_frequency
        if ts == 0:
            # [bool] Find inundation for this timestep, rslr and storms
            rslr_inun = topo[:, :, ts] < MHW
            storm_inun = inundation_TS[:, :, ts]
            inundation[ts, :, :] = np.logical_or(rslr_inun, storm_inun)
        else:
            # [bool] Find inundation for this timestep, rslr and storms
            rslr_inun = topo[:, :, ts] < MHW
            storm_inun = inundation_TS[:, :, ts]
            inundation_next = np.logical_or(rslr_inun, storm_inun)
            # Find inundation, cumulative over time
            inundation_prev = inundation[ts - 1, :, :]
            inundation[ts, :, :] = np.logical_or(inundation_next, inundation_prev)

    return inundation


def classify_overwash_frequency(TS, inundation_TS, topo, mhw_init, rslr):
    """Classify according to number of times inundated from storm overwash."""

    overwash = np.zeros([num_saves, longshore, crossshore], dtype=np.float32)

    for ts in range(TS):
        MHW = mhw_init + rslr * ts * save_frequency
        storm_inun = inundation_TS[:, :, ts]
        storm_inun[topo[:, :, ts] < MHW] = 0
        if ts == 0:
            overwash[ts, :, :] += storm_inun
        else:
            overwash[ts, :, :] += storm_inun + overwash[ts - 1, :, :]  # Cumulative

    return overwash


def intrinsic_probability():
    """Runs a batch of duplicate simulations, for a range of scenarios for external forcing, to find the classification probability from stochastic processes intrinsic to the system, particularly storm
    occurence & intensity, aeolian dynamics, and vegetation dynamics."""

    # Create array of simulations of all parameter combinations and duplicates
    sims = np.zeros([2, len(ExSE_A_bins) * len(ExSE_B_bins)])
    col = 0
    for a in range(len(ExSE_A_bins)):
        for b in range(len(ExSE_B_bins)):
            sims[0, col] = a
            sims[1, col] = b
            col += 1

    sims = np.repeat(sims, duplicates, axis=1)
    sims = sims.astype(int)
    num_sims = np.arange(sims.shape[1])

    # Run through simulations in parallel
    with routine.tqdm_joblib(tqdm(desc="Probabilistic Simulation Batch", total=len(num_sims))) as progress_bar:
        class_duplicates = Parallel(n_jobs=core_num)(delayed(run_individual_sim)(ExSE_A_bins[sims[0, i]], ExSE_B_bins[sims[1, i]]) for i in num_sims)

    # Unpack resulting data
    elev_intrinsic_prob = np.zeros([len(ExSE_A_bins), len(ExSE_A_bins), elev_num_classes, num_saves, longshore, crossshore], dtype=np.float32)  # Initialize
    habitat_state_intrinsic_prob = np.zeros([len(ExSE_A_bins), len(ExSE_A_bins), habitat_state_num_classes, num_saves, longshore, crossshore], dtype=np.float32)  # Initialize
    inundation_intrinsic_prob = np.zeros([len(ExSE_A_bins), len(ExSE_A_bins), num_saves, longshore, crossshore], dtype=np.float32)  # Initialize

    for ts in range(num_saves):
        for b in range(elev_num_classes):
            for n in range(len(num_sims)):
                exse_a = sims[0, n]
                exse_b = sims[1, n]
                elev_intrinsic_prob[exse_a, exse_b, b, ts, :, :] += class_duplicates[n][0][b, ts, :, :]
                # state_intrinsic_prob[exse_a, exse_b, b, ts, :, :] += class_duplicates[n][1][b, ts, :, :]
        for b in range(habitat_state_num_classes):
            for n in range(len(num_sims)):
                exse_a = sims[0, n]
                exse_b = sims[1, n]
                habitat_state_intrinsic_prob[exse_a, exse_b, b, ts, :, :] += class_duplicates[n][1][b, ts, :, :]
        for n in range(len(num_sims)):
            exse_a = sims[0, n]
            exse_b = sims[1, n]
            inundation_intrinsic_prob[exse_a, exse_b, ts, :, :] += class_duplicates[n][2][ts, :, :]

    del class_duplicates
    gc.collect()

    # Find average of duplicates
    elev_intrinsic_prob /= duplicates
    habitat_state_intrinsic_prob /= duplicates
    inundation_intrinsic_prob /= duplicates

    return elev_intrinsic_prob, inundation_intrinsic_prob, habitat_state_intrinsic_prob


def joint_probability():
    """Finds the joint external-intrinsic probabilistic classification. Runs a range of probabilistic scenarios to find the classification probability from
    stochastic processes external to the system (e.g., RSLR, atmospheric temperature) and and duplicates of each scenario to find the classification probability
    from stochastic processes intrinsic to the system (i.e., the inherent randomness of natural phenomena).
    """

    # Find intrinsic probability
    elev_intrinsic_prob, inundation_intrinsic_prob, habitat_state_intrinsic_prob = intrinsic_probability()

    # Create storage array for joint probability
    elev_joint_prob = np.zeros([elev_num_classes, num_saves, longshore, crossshore], dtype=np.float32)
    habitat_state_joint_prob = np.zeros([habitat_state_num_classes, num_saves, longshore, crossshore], dtype=np.float32)
    inundation_joint_prob = np.zeros([num_saves, longshore, crossshore], dtype=np.float32)

    # Apply external probability to get joint probability
    for a in range(len(ExSE_A_bins)):
        for b in range(len(ExSE_B_bins)):
            elev_external_prob = elev_intrinsic_prob[a, b, :, :, :, :] * ExSE_A_prob[a] * ExSE_B_prob[b]  # To add more external drivers: add nested for loop and multiply here, e.g. * temp_prob[t]
            # state_external_prob = state_intrinsic_prob[a, b, :, :, :, :] * ExSE_A_prob[a] * ExSE_B_prob[b]
            habitat_state_external_prob = habitat_state_intrinsic_prob[a, b, :, :, :, :] * ExSE_A_prob[a] * ExSE_B_prob[b]
            inundation_external_prob = inundation_intrinsic_prob[a, b, :, :, :] * ExSE_A_prob[a] * ExSE_B_prob[b]
            elev_joint_prob += elev_external_prob
            # state_joint_prob += state_external_prob
            habitat_state_joint_prob += habitat_state_external_prob
            inundation_joint_prob += inundation_external_prob

    return elev_joint_prob, inundation_joint_prob, habitat_state_joint_prob


def plot_cell_prob_bar(class_probabilities, class_labels, classification_label, it, l, c):
    """For a particular cell, makes bar plot of the probabilities of each class.

    Parameters
    ----------
    class_probabilities : ndarray
        Probabilities of each class over space and time.
    class_labels : list
        List of class names.
    classification_label : str
        String of classification scheme for axis label.
    it : int
        Iteration to draw probabilities from.
    l : int
        Longshore (y) coordinate of cell.
    c : int
        Cross-shore (x) coordinate of cell.
    """

    probs = class_probabilities[:, it, l, c]

    plt.figure(figsize=(8, 8))
    plt.bar(np.arange(len(probs)), probs)
    x_locs = np.arange(len(probs))
    plt.xticks(x_locs, class_labels)
    ax = plt.gca()
    ax.set_ylim([0, 1])
    plt.xlabel(classification_label)
    plt.ylabel('Probability')
    plt.title('Loc: (' + str(c) + ', ' + str(l) + '), Iteration: ' + str(it))


def plot_most_probable_class(class_probabilities, class_cmap, class_labels, it, orientation='vertical'):
    """Plots the most probable class across the domain at a particular time step, with separate panel indicating confidence in most likely class prediction.
    Note: this returns the first max occurance, i.e. if multiple bins are tied for the maximum probability of occuring, the first one will be plotted as the most likely.

    Parameters
    ----------
    class_probabilities : ndarray
        Probabilities of each class over space and time.
    class_cmap
        Discrete colormap for plotting classes.
    class_labels : list
        List of class names.
    it : int
        Iteration to draw probabilities from.
    orientation : str
        ['vertical' or 'horizontal'] Orientation to plot domain: vertical will plot ocean along left edge of domain, 'horizontal' along bottom.
    """

    num_classes = class_probabilities.shape[0]
    mmax_idx = np.argmax(class_probabilities[:, it, :, plot_xmin: plot_xmax], axis=0)  # Bin of most probable outcome
    confidence = np.max(class_probabilities[:, it, :, plot_xmin: plot_xmax], axis=0)  # Confidence, i.e. probability of most probable outcome
    min_confidence = 1 / num_classes

    if orientation == 'vertical':
        Fig = plt.figure(figsize=(8, 10))
        ax1 = Fig.add_subplot(121)
        ax2 = Fig.add_subplot(122)
    elif orientation == 'horizontal':
        mmax_idx = np.rot90(mmax_idx, k=1)
        confidence = np.rot90(confidence, k=1)
        Fig = plt.figure(figsize=(14, 10))
        ax1 = Fig.add_subplot(211)
        ax2 = Fig.add_subplot(212)
    else:
        raise ValueError("plot_most_probable_class: orientation invalid, must use 'vertical' or 'horizontal'")

    im_ratio = mmax_idx.shape[0] / mmax_idx.shape[1]
    cax1 = ax1.matshow(mmax_idx, cmap=class_cmap, vmin=0, vmax=num_classes - 1)
    tic = np.linspace(start=((num_classes - 1) / num_classes) / 2, stop=num_classes - 1 - ((num_classes - 1) / num_classes) / 2, num=num_classes)
    mcbar = Fig.colorbar(cax1, fraction=0.046 * im_ratio, ticks=tic)
    mcbar.ax.set_yticklabels(class_labels)
    plt.xlabel('Alongshore Distance [m]')
    plt.ylabel('Cross-Shore Distance [m]')

    cax2 = ax2.matshow(confidence, cmap=cmap_conf, vmin=min_confidence, vmax=1)
    Fig.colorbar(cax2, fraction=0.046 * im_ratio)
    plt.xlabel('Alongshore Distance [m]')
    plt.ylabel('Cross-Shore Distance [m]')

    plt.tight_layout()


def plot_class_probability(class_probabilities, it, class_label, orientation='vertical'):
    """Plots the probability of a class (e.g., inundation) across the domain at a particular time step.

    Parameters
    ----------
    class_probabilities : ndarray
        Probabilities of a class over space and time.
    it : int
        Iteration to draw probabilities from.
    class_label : str
        Name/description of class for labeling colorbar.
    orientation : str
        ['vertical' or 'horizontal'] Orientation to plot domain: vertical will plot ocean along left edge of domain, 'horizontal' along bottom.
    """

    inun_prob = class_probabilities[it, :, :]

    if orientation == 'vertical':
        Fig = plt.figure(figsize=(8, 10))
        ax1 = Fig.add_subplot(111)
    elif orientation == 'horizontal':
        inun_prob = np.rot90(inun_prob, k=1)
        Fig = plt.figure(figsize=(14, 10))
        ax1 = Fig.add_subplot(111)
    else:
        raise ValueError("plot_most_probable_class: orientation invalid, must use 'vertical' or 'horizontal'")

    im_ratio = inun_prob.shape[0] / inun_prob.shape[1]
    cax1 = ax1.matshow(inun_prob, cmap=cmap_class_prob, vmin=0, vmax=1)
    cb_label = 'Probability of ' + class_label
    Fig.colorbar(cax1, label=cb_label, fraction=0.046 * im_ratio)
    plt.xlabel('Meters Alongshore')
    plt.ylabel('Meters Cross-Shore')

    plt.tight_layout()


def plot_class_frequency(class_probabilities, it, class_label, orientation='vertical'):
    """Plots the frequency of a class (e.g., overwash inundation) across the domain at a particular time step.

    Parameters
    ----------
    class_probabilities : ndarray
        Probabilities of a class over space and time.
    it : int
        Iteration to draw probabilities from.
    class_label : str
        Name/description of class for labeling colorbar.
    orientation : str
        ['vertical' or 'horizontal'] Orientation to plot domain: vertical will plot ocean along left edge of domain, 'horizontal' along bottom.
    """

    inun_prob = class_probabilities[it, :, :]

    if orientation == 'vertical':
        Fig = plt.figure(figsize=(8, 10))
        ax1 = Fig.add_subplot(111)
    elif orientation == 'horizontal':
        inun_prob = np.rot90(inun_prob, k=1)
        Fig = plt.figure(figsize=(14, 10))
        ax1 = Fig.add_subplot(111)
    else:
        raise ValueError("plot_most_probable_class: orientation invalid, must use 'vertical' or 'horizontal'")

    cmap_class_freq = plt.get_cmap('inferno', int(np.max(inun_prob)))

    im_ratio = inun_prob.shape[0] / inun_prob.shape[1]
    cax1 = ax1.matshow(inun_prob, cmap=cmap_class_freq, norm=colors.LogNorm())  # Log colorbar
    cb_label = 'Number of ' + class_label
    Fig.colorbar(cax1, label=cb_label, fraction=0.046 * im_ratio)
    plt.xlabel('Meters Alongshore')
    plt.ylabel('Meters Cross-Shore')

    plt.tight_layout()


def plot_class_area_change_over_time(class_probabilities, class_labels):

    num_classes = class_probabilities.shape[0]

    plt.figure()
    xx = np.arange(0, num_saves) * save_frequency

    for n in range(num_classes):
        class_change_TS = np.zeros([num_saves])  # Initialize
        class_0 = np.sum(class_probabilities[n, 0, :, plot_xmin: plot_xmax])
        for ts in range(1, num_saves):
            class_change_TS[ts] = (np.sum(class_probabilities[n, ts, :, plot_xmin: plot_xmax]) - class_0)

        plt.plot(xx, class_change_TS)

    plt.legend(class_labels)
    plt.ylabel('Change in Area')
    plt.xlabel('Forecast Year')


def plot_transitions_area_matrix(class_probabilities, class_labels, norm='class'):

    num_classes = class_probabilities.shape[0]
    transition_matrix = np.zeros([num_classes, num_classes])

    start_class = np.argmax(class_probabilities[:, 0, :, :], axis=0)  # Bin of most probable outcome
    end_class = np.argmax(class_probabilities[:, -1, :, :], axis=0)  # Bin of most probable outcome

    if norm == 'total':  # Area normalized by total change in area of all classes
        for class_from in range(num_classes):
            for class_to in range(num_classes):
                if class_from == class_to:
                    transition_matrix[class_from, class_to] = 0
                else:
                    transition_matrix[class_from, class_to] = np.sum(np.logical_and(start_class == class_from, end_class == class_to))
        sum_all_transition = np.sum(transition_matrix)
        transition_matrix = transition_matrix / sum_all_transition
        cbar_label = 'Proportion of Net Change in Area From State Transitions'
    elif norm == 'class':  # Area normalized based on initial area of from class
        for class_from in range(num_classes):
            for class_to in range(num_classes):
                if class_from == class_to:
                    transition_matrix[class_from, class_to] = 0
                else:
                    transition_matrix[class_from, class_to] = np.sum(np.logical_and(start_class == class_from, end_class == class_to)) / np.sum(start_class == class_from)
        cbar_label = 'Proportional Net Change in Area of From Class'
    else:
        raise ValueError("Invalid entry in norm field: must use 'class' or 'total'")

    fig, ax = plt.subplots()
    cax = ax.matshow(transition_matrix, cmap='binary')
    tic_locs = np.arange(len(class_labels))
    plt.xticks(tic_locs, class_labels)
    plt.yticks(tic_locs, class_labels)
    plt.ylabel('From Class')
    plt.xlabel('To Class')
    plt.title('Ecogeomorphic State Transitions')
    fig.colorbar(cax, label=cbar_label)


def plot_most_likely_transition_maps(class_probabilities):

    most_likely_ts = np.argmax(class_probabilities[:, -1, :, :], axis=0)  # Bin of most probable outcome
    prev_most_likely = np.argmax(class_probabilities[:, 0, :, :], axis=0)  # Bin of most probable outcome

    Fig = plt.figure(figsize=(14, 7.5))
    Fig.suptitle('Most Likely Transitions', fontsize=13)

    # Subaqueous to..
    subaqueous_to = np.zeros([longshore, crossshore])
    subaqueous_to[np.logical_and(most_likely_ts == 0, prev_most_likely == 0)] = 1
    subaqueous_to[np.logical_and(most_likely_ts == 1, prev_most_likely == 0)] = 2
    subaqueous_to[np.logical_and(most_likely_ts == 2, prev_most_likely == 0)] = 3
    subaqueous_to[np.logical_and(most_likely_ts == 3, prev_most_likely == 0)] = 4
    subaqueous_to[np.logical_and(most_likely_ts == 4, prev_most_likely == 0)] = 5
    subaqueous_to[np.logical_and(most_likely_ts == 5, prev_most_likely == 0)] = 6

    s_to_ticks = ['', 'No Change', 'Beach-Steep Beach', 'Beach-Shallow', 'Dune', 'Washover', 'Interior']
    cmap1 = colors.ListedColormap(['white', 'black', 'gold', 'tan', 'saddlebrown', 'red', 'green'])
    ax_1 = Fig.add_subplot(231)
    cax_1 = ax_1.matshow(subaqueous_to[:, plot_xmin: plot_xmax], cmap=cmap1, vmin=0, vmax=len(s_to_ticks) - 1)
    tic = np.linspace(start=((len(s_to_ticks) - 1) / len(s_to_ticks)) / 2, stop=len(s_to_ticks) - 1 - ((len(s_to_ticks) - 1) / len(s_to_ticks)) / 2, num=len(s_to_ticks))
    mcbar = Fig.colorbar(cax_1, ticks=tic)
    mcbar.ax.set_yticklabels(s_to_ticks)
    plt.title('From Subaqueous to...')

    # Beach-Steep to..
    beach_to = np.zeros([longshore, crossshore])
    beach_to[np.logical_and(most_likely_ts == 1, prev_most_likely == 1)] = 1
    beach_to[np.logical_and(most_likely_ts == 0, prev_most_likely == 1)] = 2
    beach_to[np.logical_and(most_likely_ts == 2, prev_most_likely == 1)] = 3
    beach_to[np.logical_and(most_likely_ts == 3, prev_most_likely == 1)] = 4
    beach_to[np.logical_and(most_likely_ts == 4, prev_most_likely == 1)] = 5
    beach_to[np.logical_and(most_likely_ts == 5, prev_most_likely == 1)] = 6

    b_to_ticks = ['', 'No Change', 'Subaqueous', 'Beach-Shallow', 'Dune', 'Washover', 'Interior']
    cmap2 = colors.ListedColormap(['white', 'black', 'blue', 'tan', 'saddlebrown', 'red', 'green'])
    ax_2 = Fig.add_subplot(232)
    cax_2 = ax_2.matshow(beach_to[:, plot_xmin: plot_xmax], cmap=cmap2, vmin=0, vmax=len(b_to_ticks) - 1)
    tic = np.linspace(start=((len(b_to_ticks) - 1) / len(b_to_ticks)) / 2, stop=len(b_to_ticks) - 1 - ((len(b_to_ticks) - 1) / len(b_to_ticks)) / 2, num=len(b_to_ticks))
    mcbar = Fig.colorbar(cax_2, ticks=tic)
    mcbar.ax.set_yticklabels(b_to_ticks)
    plt.title('From Beach-Steep to...')

    # Beach-Shallow to..
    dune_to = np.zeros([longshore, crossshore])
    dune_to[np.logical_and(most_likely_ts == 2, prev_most_likely == 2)] = 1
    dune_to[np.logical_and(most_likely_ts == 0, prev_most_likely == 2)] = 2
    dune_to[np.logical_and(most_likely_ts == 1, prev_most_likely == 2)] = 3
    dune_to[np.logical_and(most_likely_ts == 3, prev_most_likely == 2)] = 4
    dune_to[np.logical_and(most_likely_ts == 4, prev_most_likely == 2)] = 5
    dune_to[np.logical_and(most_likely_ts == 5, prev_most_likely == 2)] = 6

    d_to_ticks = ['', 'No Change', 'Subaqueous', 'Beach-Steep', 'Dune', 'Washover', 'Interior']
    cmap3 = colors.ListedColormap(['white', 'black', 'blue', 'gold', 'saddlebrown', 'red', 'green'])
    ax_3 = Fig.add_subplot(233)
    cax_3 = ax_3.matshow(dune_to[:, plot_xmin: plot_xmax], cmap=cmap3, vmin=0, vmax=len(d_to_ticks) - 1)
    tic = np.linspace(start=((len(d_to_ticks) - 1) / len(d_to_ticks)) / 2, stop=len(d_to_ticks) - 1 - ((len(d_to_ticks) - 1) / len(d_to_ticks)) / 2, num=len(d_to_ticks))
    mcbar = Fig.colorbar(cax_3, ticks=tic)
    mcbar.ax.set_yticklabels(d_to_ticks)
    plt.title('From Beach-Shallow to...')

    # Dune to..
    washover_to = np.zeros([longshore, crossshore])
    washover_to[np.logical_and(most_likely_ts == 3, prev_most_likely == 3)] = 1
    washover_to[np.logical_and(most_likely_ts == 0, prev_most_likely == 3)] = 2
    washover_to[np.logical_and(most_likely_ts == 1, prev_most_likely == 3)] = 3
    washover_to[np.logical_and(most_likely_ts == 2, prev_most_likely == 3)] = 4
    washover_to[np.logical_and(most_likely_ts == 4, prev_most_likely == 3)] = 5
    washover_to[np.logical_and(most_likely_ts == 5, prev_most_likely == 3)] = 6

    w_to_ticks = ['', 'No Change', 'Subaqueous', 'Beach-Steep', 'Beach-Shallow', 'Washover', 'Interior']
    cmap4 = colors.ListedColormap(['white', 'black', 'blue', 'gold', 'tan', 'red', 'green'])
    ax_4 = Fig.add_subplot(234)
    cax_4 = ax_4.matshow(washover_to[:, plot_xmin: plot_xmax], cmap=cmap4, vmin=0, vmax=len(w_to_ticks) - 1)
    tic = np.linspace(start=((len(w_to_ticks) - 1) / len(w_to_ticks)) / 2, stop=len(w_to_ticks) - 1 - ((len(w_to_ticks) - 1) / len(w_to_ticks)) / 2, num=len(w_to_ticks))
    mcbar = Fig.colorbar(cax_4, ticks=tic)
    mcbar.ax.set_yticklabels(w_to_ticks)
    plt.title('From Dune to...')

    # Washover to..
    washover_to = np.zeros([longshore, crossshore])
    washover_to[np.logical_and(most_likely_ts == 4, prev_most_likely == 4)] = 1
    washover_to[np.logical_and(most_likely_ts == 0, prev_most_likely == 4)] = 2
    washover_to[np.logical_and(most_likely_ts == 1, prev_most_likely == 4)] = 3
    washover_to[np.logical_and(most_likely_ts == 2, prev_most_likely == 4)] = 4
    washover_to[np.logical_and(most_likely_ts == 3, prev_most_likely == 4)] = 5
    washover_to[np.logical_and(most_likely_ts == 5, prev_most_likely == 4)] = 6

    w_to_ticks = ['', 'No Change', 'Subaqueous', 'Beach-Steep', 'Beach-Shallow', 'Dune', 'Interior']
    cmap4 = colors.ListedColormap(['white', 'black', 'blue', 'gold', 'tan', 'saddlebrown', 'green'])
    ax_4 = Fig.add_subplot(235)
    cax_4 = ax_4.matshow(washover_to[:, plot_xmin: plot_xmax], cmap=cmap4, vmin=0, vmax=len(w_to_ticks) - 1)
    tic = np.linspace(start=((len(w_to_ticks) - 1) / len(w_to_ticks)) / 2, stop=len(w_to_ticks) - 1 - ((len(w_to_ticks) - 1) / len(w_to_ticks)) / 2, num=len(w_to_ticks))
    mcbar = Fig.colorbar(cax_4, ticks=tic)
    mcbar.ax.set_yticklabels(w_to_ticks)
    plt.title('From Washover to...')

    # Interior to..
    interior_to = np.zeros([longshore, crossshore])
    interior_to[np.logical_and(most_likely_ts == 5, prev_most_likely == 5)] = 1
    interior_to[np.logical_and(most_likely_ts == 0, prev_most_likely == 5)] = 2
    interior_to[np.logical_and(most_likely_ts == 1, prev_most_likely == 5)] = 3
    interior_to[np.logical_and(most_likely_ts == 2, prev_most_likely == 5)] = 4
    interior_to[np.logical_and(most_likely_ts == 3, prev_most_likely == 5)] = 5
    interior_to[np.logical_and(most_likely_ts == 4, prev_most_likely == 5)] = 6

    i_to_ticks = ['', 'No Change', 'Subaqueous', 'Beach-Steep', 'Beach-Shallow', 'Dune', 'Washover']
    cmap5 = colors.ListedColormap(['white', 'black', 'blue', 'gold', 'tan', 'saddlebrown', 'red'])
    ax_5 = Fig.add_subplot(236)
    cax_5 = ax_5.matshow(interior_to[:, plot_xmin: plot_xmax], cmap=cmap5, vmin=0, vmax=len(i_to_ticks) - 1)
    tic = np.linspace(start=((len(i_to_ticks) - 1) / len(i_to_ticks)) / 2, stop=len(i_to_ticks) - 1 - ((len(i_to_ticks) - 1) / len(i_to_ticks)) / 2, num=len(i_to_ticks))
    mcbar = Fig.colorbar(cax_5, ticks=tic)
    mcbar.ax.set_yticklabels(i_to_ticks)
    plt.title('From Interior to...')


def plot_class_maps(class_probabilities, class_labels, it):
    """Plots probability of occurance across the domain for each class at a particular timestep.

    Parameters
    ----------
    class_probabilities : ndarray
        Probabilities of each class over space and time.
    class_labels : list
        List of class names.
    it : int
        Iteration to draw probabilities from.
    """

    num_classes = class_probabilities.shape[0]
    plot_columns = 3
    plot_rows = int(np.ceil(num_classes / plot_columns))
    bFig = plt.figure(figsize=(14, 7.5))
    for n in range(num_classes):

        bax = bFig.add_subplot(plot_rows, plot_columns, n + 1)
        bax.matshow(class_probabilities[n, it, :, plot_xmin: plot_xmax], vmin=0, vmax=1)
        plt.title(class_labels[n])

    bFig.suptitle(name, fontsize=13)
    # cbar = Fig.colorbar(bcax5)
    plt.tight_layout()


def ani_frame_bins(timestep, class_probabilities, cax1, cax2, cax3, cax4, cax5, cax6, text1, text2, text3, text4, text5, text6):

    prob1 = class_probabilities[0, timestep, :, plot_xmin: plot_xmax]
    cax1.set_data(prob1)
    yrstr = "Year " + str(timestep * save_frequency)
    text1.set_text(yrstr)

    prob2 = class_probabilities[1, timestep, :, plot_xmin: plot_xmax]
    cax2.set_data(prob2)
    text2.set_text(yrstr)

    prob3 = class_probabilities[2, timestep, :, plot_xmin: plot_xmax]
    cax3.set_data(prob3)
    text3.set_text(yrstr)

    prob4 = class_probabilities[3, timestep, :, plot_xmin: plot_xmax]
    cax4.set_data(prob4)
    text4.set_text(yrstr)

    prob5 = class_probabilities[4, timestep, :, plot_xmin: plot_xmax]
    cax5.set_data(prob5)
    text5.set_text(yrstr)

    if class_probabilities.shape[0] > 5:
        prob6 = class_probabilities[5, timestep, :, plot_xmin: plot_xmax]
        cax6.set_data(prob6)
        text6.set_text(yrstr)

    return cax1, cax2, cax3, cax4, cax5, cax6, text1, text2, text3, text4, text5, text6


def ani_frame_most_probable_outcome(timestep, class_probabilities, cax1, cax2, text1, text2, orientation):

    Max_idx = np.argmax(class_probabilities[:, timestep, :, plot_xmin: plot_xmax], axis=0)
    Conf = np.max(class_probabilities[:, timestep, :, plot_xmin: plot_xmax], axis=0)

    if orientation == 'horizontal':
        Max_idx = np.rot90(Max_idx, k=1)
        Conf = np.rot90(Conf, k=1)

    cax1.set_data(Max_idx)
    cax2.set_data(Conf)
    yrstr = "Year " + str(timestep * save_frequency)
    text1.set_text(yrstr)
    text2.set_text(yrstr)

    return cax1, cax2, text1, text2


def ani_frame_class_probability(timestep, class_probabilities, cax1, text1, orientation):

    inun_prob = class_probabilities[timestep, :, :]

    if orientation == 'horizontal':
        inun_prob = np.rot90(inun_prob, k=1)

    cax1.set_data(inun_prob)
    yrstr = "Year " + str(timestep * save_frequency)
    text1.set_text(yrstr)

    return cax1, text1


def bins_animation(class_probabilities, class_labels):
    # Set animation base figure
    Fig = plt.figure(figsize=(8, 10.5))
    plt.tight_layout()
    Fig.suptitle(name, fontsize=13)
    ax1 = Fig.add_subplot(231)
    cax1 = ax1.matshow(class_probabilities[0, 0, :, plot_xmin: plot_xmax], vmin=0, vmax=1)
    plt.title(class_labels[0])
    timestr = "Year " + str(0)
    text1 = plt.text(2, longshore - 2, timestr, c='white')

    ax2 = Fig.add_subplot(232)
    cax2 = ax2.matshow(class_probabilities[1, 0, :, plot_xmin: plot_xmax], vmin=0, vmax=1)
    plt.title(class_labels[1])
    text2 = plt.text(2, longshore - 2, timestr, c='white')

    ax3 = Fig.add_subplot(233)
    cax3 = ax3.matshow(class_probabilities[2, 0, :, plot_xmin: plot_xmax], vmin=0, vmax=1)
    plt.title(class_labels[2])
    text3 = plt.text(2, longshore - 2, timestr, c='white')

    ax4 = Fig.add_subplot(234)
    cax4 = ax4.matshow(class_probabilities[3, 0, :, plot_xmin: plot_xmax], vmin=0, vmax=1)
    plt.title(class_labels[3])
    text4 = plt.text(2, longshore - 2, timestr, c='white')

    ax5 = Fig.add_subplot(235)
    cax5 = ax5.matshow(class_probabilities[4, 0, :, plot_xmin: plot_xmax], vmin=0, vmax=1)
    plt.title(class_labels[4])
    text5 = plt.text(2, longshore - 2, timestr, c='white')
    # cbar = Fig.colorbar(cax5)

    if class_probabilities.shape[0] > 5:
        ax6 = Fig.add_subplot(236)
        cax6 = ax6.matshow(class_probabilities[5, 0, :, plot_xmin: plot_xmax], vmin=0, vmax=1)
        plt.title(class_labels[5])
        text6 = plt.text(2, longshore - 2, timestr, c='white')
        # cbar = Fig.colorbar(cax5)
    else:
        cax6 = cax5
        text6 = text5

    # Create and save animation
    ani1 = animation.FuncAnimation(Fig, ani_frame_bins, frames=num_saves, fargs=(class_probabilities, cax1, cax2, cax3, cax4, cax5, cax6, text1, text2, text3, text4, text5, text6,), interval=300, blit=True)
    c = 1
    while os.path.exists("Output/Animation/meeb_prob_bins_" + str(c) + ".gif"):
        c += 1
    ani1.save("Output/Animation/meeb_prob_bins_" + str(c) + ".gif", dpi=150, writer="imagemagick")


def most_likely_animation(class_probabilities, class_cmap, class_labels, orientation='vertical'):

    num_classes = class_probabilities.shape[0]

    # Set animation base figure
    max_idx = np.argmax(class_probabilities[:, 0, :, plot_xmin: plot_xmax], axis=0)  # Bin of most probable outcome
    conf = np.max(class_probabilities[:, 0, :, plot_xmin: plot_xmax], axis=0)  # Confidence, i.e. probability of most probable outcome
    min_conf = 1 / num_classes

    if orientation == 'vertical':
        Fig = plt.figure(figsize=(8, 10.5))
        plt.tight_layout()
        ax1 = Fig.add_subplot(121)
        ax2 = Fig.add_subplot(122)
    elif orientation == 'horizontal':
        max_idx = np.rot90(max_idx, k=1)
        conf = np.rot90(conf, k=1)
        Fig = plt.figure(figsize=(16, 6))
        plt.tight_layout()
        ax1 = Fig.add_subplot(211)
        ax2 = Fig.add_subplot(212)
    else:
        raise ValueError("plot_most_probable_class: orientation invalid, must use 'vertical' or 'horizontal'")

    im_ratio = max_idx.shape[0] / max_idx.shape[1]
    cax1 = ax1.matshow(max_idx, cmap=class_cmap, vmin=0, vmax=num_classes - 1)
    tic = np.linspace(start=((num_classes - 1) / num_classes) / 2, stop=num_classes - 1 - ((num_classes - 1) / num_classes) / 2, num=num_classes)
    mcbar = Fig.colorbar(cax1, fraction=0.046 * im_ratio, ticks=tic)
    mcbar.ax.set_yticklabels(class_labels)
    timestr = "Year " + str(0)
    text1 = plt.text(2, longshore - 2, timestr, c='black')

    cax2 = ax2.matshow(conf, cmap=cmap_conf, vmin=min_conf, vmax=1)
    Fig.colorbar(cax2, fraction=0.046 * im_ratio)
    timestr = "Year " + str(0)
    text2 = plt.text(2, longshore - 2, timestr, c='white')

    # Create and save animation
    ani3 = animation.FuncAnimation(Fig, ani_frame_most_probable_outcome, frames=num_saves, fargs=(class_probabilities, cax1, cax2, text1, text2, orientation), interval=300, blit=True)
    c = 1
    while os.path.exists("Output/Animation/meeb_most_likely_" + str(c) + ".gif"):
        c += 1
    ani3.save("Output/Animation/meeb_most_likely_" + str(c) + ".gif", dpi=150, writer="imagemagick")


def class_probability_animation(class_probabilities, orientation='vertical'):

    inun_prob = class_probabilities[0, :, :]
    timestr = "Year " + str(0)

    if orientation == 'vertical':
        Fig = plt.figure(figsize=(8, 10.5))
        plt.tight_layout()
        ax1 = Fig.add_subplot(111)
        text1 = plt.text(2, longshore - 2, timestr, c='black')
    elif orientation == 'horizontal':
        inun_prob = np.rot90(inun_prob, k=1)
        Fig = plt.figure(figsize=(16, 6))
        plt.tight_layout()
        ax1 = Fig.add_subplot(111)
        text1 = plt.text(2, crossshore - 2, timestr, c='black')
    else:
        raise ValueError("class_probability_animation: orientation invalid, must use 'vertical' or 'horizontal'")

    im_ratio = inun_prob.shape[0] / inun_prob.shape[1]
    cax1 = ax1.matshow(inun_prob, cmap=cmap_class_prob, vmin=0, vmax=1)
    Fig.colorbar(cax1, fraction=0.046 * im_ratio)
    plt.xlabel('Meters Alongshore')
    plt.ylabel('Meters Cross-Shore')

    # Create and save animation
    ani4 = animation.FuncAnimation(Fig, ani_frame_class_probability, frames=num_saves, fargs=(class_probabilities, cax1, text1, orientation), interval=300, blit=True)
    c = 1
    while os.path.exists("Output/Animation/meeb_class_probability_" + str(c) + ".gif"):
        c += 1
    ani4.save("Output/Animation/meeb_class_probability_" + str(c) + ".gif", dpi=150, writer="imagemagick")


def class_frequency_animation(class_probabilities, orientation='vertical'):

    vmax = np.max(class_probabilities)
    vmin = np.min(class_probabilities[class_probabilities > 0])

    inun_prob = class_probabilities[0, :, :]
    timestr = "Year " + str(0)

    if orientation == 'vertical':
        Fig = plt.figure(figsize=(8, 10.5))
        plt.tight_layout()
        ax1 = Fig.add_subplot(111)
        text1 = plt.text(2, longshore - 2, timestr, c='black')
    elif orientation == 'horizontal':
        inun_prob = np.rot90(inun_prob, k=1)
        Fig = plt.figure(figsize=(16, 6))
        plt.tight_layout()
        ax1 = Fig.add_subplot(111)
        text1 = plt.text(2, crossshore - 2, timestr, c='black')
    else:
        raise ValueError("class_frequency_animation: orientation invalid, must use 'vertical' or 'horizontal'")

    cmap_class_freq = plt.get_cmap('inferno', int(vmax))

    im_ratio = inun_prob.shape[0] / inun_prob.shape[1]
    cax1 = ax1.matshow(inun_prob, cmap=cmap_class_freq, norm=colors.LogNorm(vmin=vmin, vmax=vmax))
    Fig.colorbar(cax1, label='Number of Occurences', fraction=0.046 * im_ratio)
    plt.xlabel('Meters Alongshore')
    plt.ylabel('Meters Cross-Shore')

    # Create and save animation
    ani5 = animation.FuncAnimation(Fig, ani_frame_class_probability, frames=num_saves, fargs=(class_probabilities, cax1, text1, orientation), interval=300, blit=True)
    c = 1
    while os.path.exists("Output/Animation/meeb_class_frequency_" + str(c) + ".gif"):
        c += 1
    ani5.save("Output/Animation/meeb_class_frequency_" + str(c) + ".gif", dpi=150, writer="imagemagick")


# __________________________________________________________________________________________________________________________________
# VARIABLES AND INITIALIZATIONS

# # 2014
# start = "Init_NCB-NewDrum-Ocracoke_2014_PostSandy-NCFMP-Plover.npy"
# startdate = '20140406'

# 2018
start = "Init_NCB-NewDrum-Ocracoke_2018_PostFlorence-Plover_2m.npy"
startdate = '20181007'

# _____________________
# EXTERNAL STOCHASTIC ELEMENTS (ExSE)

# RSLR
ExSE_A_bins = [0.0068, 0.0096, 0.0124]  # [m/yr] Bins of future RSLR rates up to 2050
ExSE_A_prob = [0.26, 0.55, 0.19]  # Probability of future RSLR bins (must sum to 1.0)

# Mean Storm Intensity
ExSE_B_bins = [0]  # [0.005, 0.135, 0.266]  # [%/yr] Bins of yearly percent shift in mean storm intensity up to 2050
ExSE_B_prob = [1]  # [0.296, 0.526, 0.178]  # Probability of future storm intensity bins (must sum to 1.0)

# _____________________
# CLASSIFICATION SCHEME SPECIFICATIONS
elev_classification_label = 'Elevation Change [m]'  # Axes labels on figures
elev_class_edges = [-np.inf, -0.5, -0.1, 0.1, 0.5, np.inf]  # [m] Elevation change
elev_class_labels = ['< -0.5', '-0.5 - -0.1', '-0.1 - 0.1', '0.1 - 0.5', '> 0.5']
elev_class_cmap = colors.ListedColormap(['#ca0020', '#f4a582', '#f7f7f7', '#92c5de', '#0571b0'])

state_classification_label = 'Ecogeomorphic State'  # Axes labels on figures
state_class_edges = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]  # [m] State change
state_class_labels = ['Subaqueous', 'Beach', 'Dune', 'Washover', 'Interior']
state_class_cmap = colors.ListedColormap(['blue', 'gold', 'saddlebrown', 'red', 'green'])

habitat_state_classification_label = 'Habitat-Ecogeomorphic State'  # Axes labels on figures
habitat_state_class_edges = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5]  # [m] State change
habitat_state_class_labels = ['Subaqueous', 'Beach-Steep', 'Beach-Shallow', 'Dune', 'Washover', 'Interior']
habitat_state_class_cmap = colors.ListedColormap(['blue', 'gold', 'tan', 'saddlebrown', 'red', 'green'])

# Class Probability
cmap_class_prob = plt.get_cmap('cividis', 5)

# Confidence
cmap_conf = plt.get_cmap('BuPu', 4)  # 4 discrete colors

# _____________________
# INITIAL PARAMETERS

sim_duration = 32  # [yr] Note: For probabilistic projections, use a duration that is divisible by the save_frequency
save_frequency = 1  # [yr] Time step for probability calculations

duplicates = 25  # To account for intrinsic stochasticity (e.g., storms, aeolian)

# Number of cores to use in the parallelization
core_num = int(os.environ['SLURM_CPUS_PER_TASK'])  # --> Use this if running on HPC
# core_num = 12  # --> Use this if running on local machine

# Define Horizontal and Vertical References of Domain
ymin = 13000  # [m] Alongshore coordinate
ymax = 21000  # [m] Alongshore coordinate
xmin = 900  # [m] Cross-shore coordinate
xmax = 1700  # [m] Cross-shore coordinate
plot_xmin = 0  # [m] Cross-shore coordinate (for plotting), relative to trimmed domain
plot_xmax = 800  # [m] Cross-shore coordinate (for plotting), relative to trimmed domain
MHW_init = 0.39  # [m NAVD88] Initial mean high water
cellsize = 2  # [m]

name = '16Sep24, 13000-21000, 2018-2050, n=25'  # Name of simulation suite

plot = False  # [bool]
animate = False  # [bool]
save_data = True  # [bool]
savename = '16Sep24_13000-21000'

# _____________________
# INITIAL CONDITIONS

# Resize according to cellsize
ymin = int(ymin / cellsize)  # Alongshore
ymax = int(ymax / cellsize)  # Alongshore
xmin = int(xmin / cellsize)  # Cross-shore
xmax = int(xmax / cellsize)  # Cross-shore
plot_xmin = int(plot_xmin / cellsize)  # Cross-shore plotting
plot_xmax = int(plot_xmax / cellsize)  # Cross-shore plotting

# Load Initial Domains
Init = np.load("Input/" + start)
topo_start = Init[0, ymin: ymax, xmin: xmax]
spec1_start = Init[1, ymin: ymax, xmin: xmax]
spec2_start = Init[2, ymin: ymax, xmin: xmax]
longshore, crossshore = topo_start.shape

elev_num_classes = len(elev_class_edges) - 1
state_num_classes = len(state_class_edges) - 1
habitat_state_num_classes = len(habitat_state_class_edges) - 1
num_saves = int(np.floor(sim_duration/save_frequency)) + 1

del Init
gc.collect()

# __________________________________________________________________________________________________________________________________
# RUN MODEL

print()
print(name)
print()

start_time = time.time()  # Record time at start of simulation

# Determine classification probabilities cross space and time for joint intrinsic-external stochastic elements
elev_class_probabilities, inundation_class_probabilities, habitat_state_class_probabilities = joint_probability()

# Print elapsed time of simulation
print()
SimDuration = time.time() - start_time
print()
print("Elapsed Time: ", SimDuration, "sec")


# __________________________________________________________________________________________________________________________________
# PLOT RESULTS

if plot:
    plot_class_maps(elev_class_probabilities, elev_class_labels, it=-1)
    plot_class_maps(habitat_state_class_probabilities, habitat_state_class_labels, it=-1)
    plot_most_probable_class(elev_class_probabilities, elev_class_cmap, elev_class_labels, it=-1, orientation='horizontal')
    plot_most_probable_class(habitat_state_class_probabilities, habitat_state_class_cmap, habitat_state_class_labels, it=-1, orientation='horizontal')
    plot_class_frequency(inundation_class_probabilities, it=-1, class_label='Overwash Events', orientation='horizontal')
    plot_class_area_change_over_time(habitat_state_class_probabilities, habitat_state_class_labels)
    plot_most_likely_transition_maps(habitat_state_class_probabilities)
    plot_transitions_area_matrix(habitat_state_class_probabilities, habitat_state_class_labels)
if animate:
    bins_animation(elev_class_probabilities, elev_class_labels)
    bins_animation(habitat_state_class_probabilities, habitat_state_class_labels)
    most_likely_animation(elev_class_probabilities, elev_class_cmap, elev_class_labels, orientation='horizontal')
    most_likely_animation(habitat_state_class_probabilities, habitat_state_class_cmap, habitat_state_class_labels, orientation='horizontal')
    class_frequency_animation(inundation_class_probabilities, orientation='horizontal')
plt.show()


# __________________________________________________________________________________________________________________________________
# SAVE DATA

if save_data:
    # Elevation
    elev_name = "ElevClassProbabilities_" + savename
    elev_outloc = "Output/SimData/" + elev_name
    np.save(elev_outloc, elev_class_probabilities)
    # State
    state_name = "HabitatStateClassProbabilities_" + savename
    state_outloc = "Output/SimData/" + state_name
    np.save(state_outloc, habitat_state_class_probabilities)
    # Inundation
    inun_name = "InundationClassProbabilities_" + savename
    inun_outloc = "Output/SimData/" + inun_name
    np.save(inun_outloc, inundation_class_probabilities)

