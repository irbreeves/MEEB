"""
Probabilistic framework for running MEEB simulations using MPI distributed memory parallelism on super computers. Generates probabilistic projections
of future change.

IRBR 11 March 2025
"""

import os
import sys
import numpy as np
import time
import gc
import scipy
from tqdm import tqdm
from mpi4py import MPI

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
        shift_mean_storm_intensity_start=shift_mean_storm_intensity[0],
        shift_mean_storm_intensity_end=shift_mean_storm_intensity[1],
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
        Rin=232,
        Cs=0.0235,
        MaxUpSlope=1.5,
        marine_flux_limit=1,
        Kow=0.0003615,
        mm=1.05,
        overwash_substeps=25,
        beach_equilibrium_slope=0.021,
        swash_erosive_timescale=1.51,
        beach_substeps=1,
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
        sp2_pioneer_probability=0.03,
        # MY GRASS
        sp1_a=-1.2,
        sp1_b=-0.2,  # Mullins et al. (2019)
        sp1_c=0.5,
        sp1_d=1.2,
        sp1_e=2.1,
        sp1_peak=0.2,
        # MY SHRUB
        sp2_a=-1.0,
        sp2_b=-0.2,  # Conn and Day (1993)
        sp2_c=0.0,
        sp2_d=0.2,
        sp2_e=2.1,
        sp2_peak=0.05,
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
    habitat_state_classification = classify_ecogeomorphic_habitat_state(meeb.topo_TS.shape[2], meeb.topo_TS, meeb.spec1_TS, meeb.spec2_TS, meeb.MHW_init, meeb.RSLR, vegetated_threshold=0.37)

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


def classify_ecogeomorphic_habitat_state(TS, topo_TS, spec1_TS, spec2_TS, mhw_init, rslr, vegetated_threshold, beach_slope_threshold=0.019):
    """Classify by ecogeomorphic state using topography and vegetatio, with focus on bird and turtle habitat."""

    habitat_state_classes = np.zeros([habitat_state_num_classes, num_saves, longshore, crossshore], dtype=np.float32)  # Initialize

    # Run Categorization
    # Loop through saved timesteps
    for ts in range(TS):

        # Find MHW for this time step
        MHW = mhw_init + rslr * ts * save_frequency

        # Smooth topography to remove small-scale variability
        topo = scipy.ndimage.gaussian_filter(topo_TS[:, :, ts], 5, mode='reflect').astype(np.float32)

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

        # Find dune crest heights
        dune_crestheight = np.zeros(longshore, dtype=np.float32)
        for ls in range(longshore):
            dune_crestheight[ls] = topo[ls, dune_crestline[ls]]  # [m NAVD88]

        # Make boolean maps of areas alongshore with beach slope greater than threshold (i.e., turtle habitat)
        steep_beach_slope = np.rot90(np.array([beach_slopes > beach_slope_threshold] * crossshore), -1)

        # Smooth vegetation to remove small-scale variability
        veg = scipy.ndimage.gaussian_filter((spec1_TS[:, :, ts] + spec2_TS[:, :, ts]).astype(np.float32), 5, mode='constant').astype(np.float16)
        unvegetated = veg < vegetated_threshold  # [bool] Map of unvegetated areas

        # Categorize Cells of Model Domain

        # Subaqueous: below MHW
        subaqueous_class_TS = topo < MHW
        habitat_state_classes[0, ts, :, :] += subaqueous_class_TS

        # Dune: between toe and heel, and not fronting dune gap
        dune_class_TS = landward_of_dunetoe * seaward_of_duneheel * ~fronting_dune_gap_simple * ~subaqueous_class_TS
        habitat_state_classes[3, ts, :, :] += dune_class_TS

        # Beach-Shallow: seaward of dune crest, and not dune or subaqueous, , beach slope >= threshold
        beach_class_shallow_TS = seaward_of_dunecrest * ~dune_class_TS * ~subaqueous_class_TS * ~steep_beach_slope
        habitat_state_classes[1, ts, :, :] += beach_class_shallow_TS

        # Beach-Steep: seaward of dune crest, not dune or subaqueous, beach slope > threshold
        beach_class_steep_TS = seaward_of_dunecrest * ~dune_class_TS * ~subaqueous_class_TS * steep_beach_slope
        habitat_state_classes[2, ts, :, :] += beach_class_steep_TS

        # Unvegetated Interior: landward of dune crest and unvegetated
        unveg_interior_class_TS = landward_of_dunecrest * unvegetated * ~dune_class_TS * ~beach_class_steep_TS * ~beach_class_shallow_TS * ~subaqueous_class_TS
        habitat_state_classes[4, ts, :, :] += unveg_interior_class_TS

        # Vegetated Interior: all other cells landward of dune crest
        interior_class_TS = landward_of_dunecrest * ~unveg_interior_class_TS * ~dune_class_TS * ~beach_class_steep_TS * ~beach_class_shallow_TS * ~subaqueous_class_TS
        habitat_state_classes[5, ts, :, :] += interior_class_TS

    return habitat_state_classes


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


def run_parallel_sims(sims):
    """Run through simulations in parallel using MPI parallelism"""

    comm = MPI.COMM_WORLD
    rank = comm.rank  # Number of local processor
    size = comm.size

    total_sims = sims.shape[1]
    local_sim_num = int(total_sims / size)  # Number of sims to complete on this processor
    local_results = []

    if rank == 0:
        print()
        print(name)
        print("Total simulations:", total_sims)
        print("Processors assigned:", size)
        print()

    if total_sims % size != 0:
        raise ValueError('The number of simulations must be divisible by number of processors.')

    # Local parameter values
    for i in tqdm(range(local_sim_num), desc=f"Processor {rank}", position=rank):

        rslr = ExSE_A_bins[sims[0, rank * local_sim_num + i]]
        shift_mean_storm_intensity = ExSE_B_bins[sims[1, rank * local_sim_num + i]]

        # Run simulation with local parameter values
        local_results.append(run_individual_sim(rslr, shift_mean_storm_intensity))

    # Gather results from all processes
    all_results = comm.gather(local_results, root=0)

    if rank == 0:
        return all_results


def probabilistic_simulation():
    """Runs a batch of duplicate simulations, for a range of scenarios for external forcing, to find the classification probability from stochastic processes intrinsic to the system, particularly storm
    occurence & intensity, aeolian dynamics, and vegetation dynamics."""

    # Create array of simulations of all parameter combinations and duplicates
    sims = np.zeros([2, len(ExSE_A_bins) * len(ExSE_B_bins)])
    col = 0
    for a in range(len(ExSE_A_bins)):
        for b in reversed(range(len(ExSE_B_bins))):
            sims[0, col] = a
            sims[1, col] = b
            col += 1

    sims = np.repeat(sims, duplicates, axis=1)
    sims = sims.astype(int)
    num_sims = np.arange(sims.shape[1])

    # ----------------------------------------------
    # INTRINSIC PROBABILITY
    """Runs a batch of duplicate simulations, for a range of scenarios for external forcing, to find the classification probability from stochastic processes intrinsic to the system, particularly storm
    occurence & intensity, aeolian dynamics, and vegetation dynamics."""
    # ----------------------------------------------

    # Run through simulations in parallel using MPI parallelism
    class_duplicates = run_parallel_sims(sims)

    if class_duplicates is not None:  # Gathered results from master rank

        class_duplicates = [item for t in class_duplicates for item in t]

        # Unpack resulting data
        elev_intrinsic_prob = np.zeros([len(ExSE_A_bins), len(ExSE_B_bins), elev_num_classes, num_saves, longshore, crossshore], dtype=np.float32)  # Initialize
        habitat_state_intrinsic_prob = np.zeros([len(ExSE_A_bins), len(ExSE_B_bins), habitat_state_num_classes, num_saves, longshore, crossshore], dtype=np.float32)  # Initialize
        inundation_intrinsic_prob = np.zeros([len(ExSE_A_bins), len(ExSE_B_bins), num_saves, longshore, crossshore], dtype=np.float32)  # Initialize

        for ts in range(num_saves):
            for b in range(elev_num_classes):
                for n in range(len(num_sims)):
                    exse_a = sims[0, n]
                    exse_b = sims[1, n]
                    elev_intrinsic_prob[exse_a, exse_b, b, ts, :, :] += class_duplicates[n][0][b, ts, :, :]
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

        # ----------------------------------------------
        # JOINT PROBABILITY
        """Finds the joint external-intrinsic probabilistic classification. Runs a range of probabilistic scenarios to find the classification probability from
        stochastic processes external to the system (e.g., RSLR, atmospheric temperature) and and duplicates of each scenario to find the classification probability
        from stochastic processes intrinsic to the system (i.e., the inherent randomness of natural phenomena)."""
        # ----------------------------------------------

        # Create storage array for joint probability
        elev_joint_prob = np.zeros([elev_num_classes, num_saves, longshore, crossshore], dtype=np.float32)
        habitat_state_joint_prob = np.zeros([habitat_state_num_classes, num_saves, longshore, crossshore], dtype=np.float32)
        inundation_joint_prob = np.zeros([num_saves, longshore, crossshore], dtype=np.float32)

        # Apply external probability to get joint probability
        for a in range(len(ExSE_A_bins)):
            for b in range(len(ExSE_B_bins)):
                elev_external_prob = elev_intrinsic_prob[a, b, :, :, :, :] * ExSE_A_prob[a] * ExSE_B_prob[b]  # To add more external drivers: add nested for loop and multiply here, e.g. * temp_prob[t]
                habitat_state_external_prob = habitat_state_intrinsic_prob[a, b, :, :, :, :] * ExSE_A_prob[a] * ExSE_B_prob[b]
                inundation_external_prob = inundation_intrinsic_prob[a, b, :, :, :] * ExSE_A_prob[a] * ExSE_B_prob[b]
                elev_joint_prob += elev_external_prob
                habitat_state_joint_prob += habitat_state_external_prob
                inundation_joint_prob += inundation_external_prob

        # Print Elapsed Time of Probabilistic Simulation
        SimDuration_total = time.time() - start_time_total
        print()
        print("Total Elapsed Time: ", SimDuration_total, "sec")
        print()

        # ----------------------------------------------
        # SAVE DATA
        # ----------------------------------------------
        if save_data:
            # Elevation
            elev_name = "ElevClassProbabilities_" + savename
            elev_outloc = "Output/SimData/" + elev_name
            np.save(elev_outloc, elev_joint_prob)
            # State
            state_name = "HabitatStateClassProbabilities_" + savename
            state_outloc = "Output/SimData/" + state_name
            np.save(state_outloc, habitat_state_joint_prob)
            # Inundation
            inun_name = "InundationClassProbabilities_" + savename
            inun_outloc = "Output/SimData/" + inun_name
            np.save(inun_outloc, inundation_joint_prob)


# __________________________________________________________________________________________________________________________________

if __name__ == '__main__':

    # _____________________
    # EXTERNAL STOCHASTIC ELEMENTS (ExSE)

    # RSLR
    ExSE_A_bins = [0.0068, 0.0096, 0.0124]  # [m/yr] Bins of future RSLR rates up to 2050
    ExSE_A_prob = [0.26, 0.55, 0.19]  # Probability of future RSLR bins (must sum to 1.0)

    ExSE_B_bins = [(1.485, 4.199)]  # [(0.059, 0.167), (1.485, 4.199), (2.910, 8.231)]  # [%] Bins of percent shift in mean storm intensity at simulation start ([0]) and end ([1])
    ExSE_B_prob = [1]  # [0.297, 0.525, 0.178]  # Probability of future storm intensity bins (must sum to 1.0)

    # _____________________
    # CLASSIFICATION SCHEME SPECIFICATIONS

    elev_class_edges = [-np.inf, -0.5, -0.1, 0.1, 0.5, np.inf]  # [m] Elevation change

    state_class_edges = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]  # [m] State change

    habitat_state_class_edges = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5]  # [m] State change

    # _____________________
    # INITIAL PARAMETERS

    duplicates = 4

    start = "Init_NCB-2200-34200_2018_USACE_PostFlorence_2m.npy"
    startdate = '20181015'

    sim_duration = 20  # [yr] Note: For probabilistic projections, use a duration that is divisible by the save_frequency
    save_frequency = 1  # [yr] Time step for probability calculations

    # Define Horizontal and Vertical References of Domain
    ymin = 21000  # [m] Alongshore coordinate
    ymax = 21400  # [m] Alongshore coordinate
    xmin = 700  # [m] Cross-shore coordinate
    xmax = 1500  # [m] Cross-shore coordinate
    MHW_init = 0.39  # [m NAVD88] Initial mean high water
    cellsize = 2  # [m]

    name = '21Jan25, 21000-21400, 2018-2038, MPI Test 1'  # Name of simulation suite

    save_data = True  # [bool]
    savename = '21Jan25_MPI-Test-1'

    # _____________________
    # INITIAL CONDITIONS

    # Resize according to cellsize
    ymin = int(ymin / cellsize)  # Alongshore
    ymax = int(ymax / cellsize)  # Alongshore
    xmin = int(xmin / cellsize)  # Cross-shore
    xmax = int(xmax / cellsize)  # Cross-shore

    # Load Initial Domains
    Init = np.load("Input/" + start)
    topo_start = Init[0, ymin: ymax, xmin: xmax].copy()
    spec1_start = Init[1, ymin: ymax, xmin: xmax].copy()
    spec2_start = Init[2, ymin: ymax, xmin: xmax].copy()
    longshore, crossshore = topo_start.shape

    elev_num_classes = len(elev_class_edges) - 1
    state_num_classes = len(state_class_edges) - 1
    habitat_state_num_classes = len(habitat_state_class_edges) - 1
    num_saves = int(np.floor(sim_duration/save_frequency)) + 1

    del Init
    gc.collect()

    # __________________________________________________________________________________________________________________________________
    # RUN MODEL

    start_time_total = time.time()  # Record time at start of simulation

    # Determine classification probabilities cross space and time for joint intrinsic-external stochastic elements
    probabilistic_simulation()
