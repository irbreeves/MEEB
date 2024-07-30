"""
Probabilistic framework for running MEEB simulations. Generates probabilistic projections of future change.

IRBR 30 July 2024
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import os
import scipy
from tqdm import tqdm
from matplotlib import colors
from joblib import Parallel, delayed
import routines_meeb as routine

from meeb import MEEB


# __________________________________________________________________________________________________________________________________
# FUNCTIONS


def run_individual_sim(rslr):
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
        MHW=MHW_init,
        init_filename=start,
        hindcast=False,
        seeded_random_numbers=False,
        simulation_start_date=startdate,
        storm_timeseries_filename='StormTimeSeries_1979-2020_NCB-CE_Beta0pt039_BermEl1pt78.npy',
        storm_list_filename='SyntheticStorms_NCB-CE_10k_1979-2020_Beta0pt039_BermEl1pt78.npy',
        save_frequency=save_frequency,
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
        wave_asymetry=0.6,
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

    topo_change_sim_TS = np.zeros(meeb.topo_TS.shape)
    for ts in range(meeb.topo_TS.shape[2]):
        topo_change_ts = (meeb.topo_TS[:, :, ts] - topo_start_sim) * subaerial_mask  # Disregard change that is not subaerial
        topo_change_sim_TS[:, :, ts] = topo_change_ts

    # Create classified map
    elevation_classification = classify_topo_change(meeb.topo_TS.shape[2], topo_change_sim_TS)
    state_classification = classify_ecogeomorphic_state(meeb.topo_TS.shape[2], meeb.topo_TS, meeb.veg_TS, meeb.MHW_init, meeb.RSLR, vegetated_threshold=0.25)  # vegetated_threshold was 0.12

    classes = [elevation_classification, state_classification]

    return classes


def classify_topo_change(TS, topo_change_sim_TS):
    """Classify according to range of elevation change."""

    topo_change_bin = np.zeros([elev_num_classes, num_saves, longshore, crossshore])

    for b in range(len(elev_class_edges) - 1):
        lower = elev_class_edges[b]
        upper = elev_class_edges[b + 1]

        for ts in range(TS):
            bin_change = np.logical_and(lower < topo_change_sim_TS[:, :, ts], topo_change_sim_TS[:, :, ts] <= upper).astype(int)
            topo_change_bin[b, ts, :, :] += bin_change

    return topo_change_bin


def classify_ecogeomorphic_state(TS, topo_TS, veg_TS, mhw_init, rslr, vegetated_threshold):
    """Classify by ecogeomorphic state using topography and vegetation."""

    state_classes = np.zeros([state_num_classes, num_saves, longshore, crossshore])  # Initialize

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
                temp = np.ones([crossshore - crest_loc, longshore])
                right_diag = np.tril(temp, k=ls)
                left_diag = np.fliplr(np.tril(temp, k=len(dune_crestline) - ls - 1))
                spread = np.rot90(right_diag * left_diag, 1)
                fronting_dune_gap[:, crest_loc:] = np.flipud(np.logical_or(fronting_dune_gap[:, crest_loc:], spread))
                trim = max(0, right_diag.shape[0] - crest_loc)
                fronting_dune_gap[:, max(0, crest_loc - right_diag.shape[0]): crest_loc] = np.fliplr(fronting_dune_gap[:, crest_loc:])[:, trim:]

        # Find dune crest heights
        dune_crestheight = np.zeros(longshore)
        for ls in range(longshore):
            dune_crestheight[ls] = topo[ls, dune_crestline[ls]]  # [m NAVD88]

        # Make boolean maps of locations fronted by low dunes (quite similar to fronting_dune_gaps)
        fronting_low_dune = np.zeros(topo.shape, dtype=bool)
        for ls in range(longshore):
            if dune_crestheight[ls] - MHW < 2.7:
                crest_loc = dune_crestline[ls]
                temp = np.ones([crossshore - crest_loc, longshore])
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

    return state_classes


def intrinsic_probability(sims):
    """Runs a batch of duplicate simulations, for a range of scenarios for external forcing, to find the classification probability from stochastic processes intrinsic to the system, particularly storm
    occurence & intensity, aeolian dynamics, and vegetation dynamics."""

    # Create array of simulations of all parameter combinations and duplicates
    sims = np.repeat(np.arange(len(RSLR_bin)), duplicates)

    # Run through simulations in parallel
    with routine.tqdm_joblib(tqdm(desc="Probabilistic Simulation Batch", total=len(sims))) as progress_bar:
        class_duplicates = Parallel(n_jobs=core_num)(delayed(run_individual_sim)(RSLR_bin[i]) for i in sims)

    # Organize resulting data
    elev_class_bins = np.zeros([len(RSLR_bin), elev_num_classes, num_saves, longshore, crossshore])  # Initialize
    state_class_bins = np.zeros([len(RSLR_bin), state_num_classes, num_saves, longshore, crossshore])  # Initialize

    for ts in range(num_saves):
        for b in range(elev_num_classes):
            for n in range(len(sims)):
                rslr = sims[n]
                elev_class_bins[rslr, b, ts, :, :] += class_duplicates[n][0][b, ts, :, :]
                state_class_bins[rslr, b, ts, :, :] += class_duplicates[n][1][b, ts, :, :]

    elev_internal_prob = elev_class_bins / duplicates
    state_internal_prob = state_class_bins / duplicates

    return elev_internal_prob, state_internal_prob


def joint_probability():
    """Finds the joint external-intrinsic probabilistic classification. Runs a range of probabilistic scenarios to find the classification probability from
    stochastic processes external to the system (e.g., RSLR, atmospheric temperature) and and duplicates of each scenario to find the classification probability
    from stochastic processes intrinsic to the system (i.e., the inherent randomness of natural phenomena).
    """

    # Create array of simulations of all parameter combinations and duplicates
    sims = np.repeat(np.arange(len(RSLR_bin)), duplicates)

    # Find intrinsic probability
    elev_internal_prob, state_internal_prob = intrinsic_probability(sims)

    # Create storage array for joint probability
    elev_joint_prob = np.zeros([elev_num_classes, num_saves, longshore, crossshore])
    state_joint_prob = np.zeros([state_num_classes, num_saves, longshore, crossshore])

    # Apply external probability to get joint probability
    for r in range(len(RSLR_bin)):
        elev_external_prob = elev_internal_prob[r, :, :, :, :] * RSLR_prob[r]  # To add more external drivers: add nested for loop and multiply here, e.g. * temp_prob[t]
        state_external_prob = state_internal_prob[r, :, :, :, :] * RSLR_prob[r]
        elev_joint_prob += elev_external_prob
        state_joint_prob += state_external_prob

    return elev_joint_prob, state_joint_prob


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


def plot_most_probable_class(class_probabilities, class_cmap, class_labels, it):
    """Plots the most probable class across the domain at a particular time step. Note: this returns the first max occurance,
    i.e. if multiple bins are tied for the maximum probability of occuring, the first one will be plotted as the most likely.

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
    """

    num_classes = class_probabilities.shape[0]
    mmax_idx = np.argmax(class_probabilities[:, it, :, plot_xmin: plot_xmax], axis=0)  # Bin of most probable outcome
    confidence = 1 - np.max(class_probabilities[:, it, :, plot_xmin: plot_xmax], axis=0)  # Confidence, i.e. probability of most probable outcome

    conf_cmap = colors.ListedColormap(['white'])

    fig, ax = plt.subplots()
    cax = ax.matshow(mmax_idx, cmap=class_cmap, vmin=0, vmax=num_classes - 1)
    cax2 = ax.matshow(np.ones(mmax_idx.shape), cmap=conf_cmap, vmin=0, vmax=1, alpha=confidence)
    tic = np.linspace(start=((num_classes - 1) / num_classes) / 2, stop=num_classes - 1 - ((num_classes - 1) / num_classes) / 2, num=num_classes)
    mcbar = fig.colorbar(cax, ticks=tic)
    mcbar.ax.set_yticklabels(class_labels)
    plt.xlabel('Alongshore Distance [m]')
    plt.ylabel('Cross-Shore Distance [m]')
    plt.title('Iteration: ' + str(it))


def plot_most_probable_class_2(class_probabilities, class_cmap, class_labels, it):
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
    """

    num_classes = class_probabilities.shape[0]
    mmax_idx = np.argmax(class_probabilities[:, it, :, plot_xmin: plot_xmax], axis=0)  # Bin of most probable outcome
    confidence = np.max(class_probabilities[:, it, :, plot_xmin: plot_xmax], axis=0)  # Confidence, i.e. probability of most probable outcome
    min_confidence = 1 / num_classes

    Fig = plt.figure(figsize=(10, 11))
    ax1 = Fig.add_subplot(211)
    cax1 = ax1.matshow(mmax_idx, cmap=class_cmap, vmin=0, vmax=num_classes - 1)
    tic = np.linspace(start=((num_classes - 1) / num_classes) / 2, stop=num_classes - 1 - ((num_classes - 1) / num_classes) / 2, num=num_classes)
    mcbar = Fig.colorbar(cax1, ticks=tic)
    mcbar.ax.set_yticklabels(class_labels)
    plt.xlabel('Alongshore Distance [m]')
    plt.ylabel('Cross-Shore Distance [m]')

    ax2 = Fig.add_subplot(212)
    cax2 = ax2.matshow(confidence, cmap=cmap_conf, vmin=min_confidence, vmax=1)
    Fig.colorbar(cax2)
    plt.xlabel('Alongshore Distance [m]')
    plt.ylabel('Cross-Shore Distance [m]')

    plt.tight_layout()


def plot_class_area_change_over_time(class_probabilities):

    num_classes = class_probabilities.shape[0]
    class_change_TS = np.zeros([num_classes, num_saves])  # Initialize

    subaqueous_0 = np.sum(class_probabilities[0, 0, :, plot_xmin: plot_xmax])
    beach_0 = np.sum(class_probabilities[1, 0, :, plot_xmin: plot_xmax])
    dune_0 = np.sum(class_probabilities[2, 0, :, plot_xmin: plot_xmax])
    washover_0 = np.sum(class_probabilities[3, 0, :, plot_xmin: plot_xmax])
    interior_0 = np.sum(class_probabilities[4, 0, :, plot_xmin: plot_xmax])

    for ts in range(1, num_saves):
        delta_s = (np.sum(class_probabilities[0, ts, :, plot_xmin: plot_xmax]) - subaqueous_0)
        delta_b = (np.sum(class_probabilities[1, ts, :, plot_xmin: plot_xmax]) - beach_0)
        delta_d = (np.sum(class_probabilities[2, ts, :, plot_xmin: plot_xmax]) - dune_0)
        delta_w = (np.sum(class_probabilities[3, ts, :, plot_xmin: plot_xmax]) - washover_0)
        delta_i = (np.sum(class_probabilities[4, ts, :, plot_xmin: plot_xmax]) - interior_0)

        class_change_TS[0, ts] = delta_s
        class_change_TS[1, ts] = delta_b
        class_change_TS[2, ts] = delta_d
        class_change_TS[3, ts] = delta_w
        class_change_TS[4, ts] = delta_i

    plt.figure()
    xx = np.arange(0, num_saves/2, save_frequency)
    for n in range(num_classes):
        plt.plot(xx, class_change_TS[n, :])
    plt.legend(['Subaqueous', 'Beach', 'Dune', 'Washover', 'Interior'])
    plt.ylabel('Change in Area')
    plt.xlabel('Forecast Year')


def plot_transitions_area_matrix(class_probabilities, class_labels):

    num_classes = class_probabilities.shape[0]
    transition_matrix = np.zeros([num_classes, num_classes])

    start_class = np.argmax(class_probabilities[:, 0, :, :], axis=0)  # Bin of most probable outcome
    end_class = np.argmax(class_probabilities[:, -1, :, :], axis=0)  # Bin of most probable outcome

    for class_from in range(num_classes):
        for class_to in range(num_classes):
            if class_from == class_to:
                transition_matrix[class_from, class_to] = 0
            else:
                transition_matrix[class_from, class_to] = np.sum(np.logical_and(start_class == class_from, end_class == class_to))

    sum_all_transition = np.sum(transition_matrix)

    transition_matrix = transition_matrix / sum_all_transition

    fig, ax = plt.subplots()
    cax = ax.matshow(transition_matrix, cmap='YlOrBr', vmin=0, vmax=1)
    tic_locs = np.arange(len(class_labels))
    plt.xticks(tic_locs, class_labels)
    plt.yticks(tic_locs, class_labels)
    plt.ylabel('From Class')
    plt.xlabel('To Class')
    fig.colorbar(cax, label='Net Change in Area From Ecogeomorphic State Transition (m^2)')


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

    s_to_ticks = ['', 'No Change', 'Beach', 'Dune', 'Washover', 'Interior']
    cmap1 = colors.ListedColormap(['white', 'black', 'gold', 'saddlebrown', 'red', 'green'])
    ax_1 = Fig.add_subplot(231)
    cax_1 = ax_1.matshow(subaqueous_to[:, plot_xmin: plot_xmax], cmap=cmap1, vmin=0, vmax=len(s_to_ticks) - 1)
    tic = np.linspace(start=((len(s_to_ticks) - 1) / len(s_to_ticks)) / 2, stop=len(s_to_ticks) - 1 - ((len(s_to_ticks) - 1) / len(s_to_ticks)) / 2, num=len(s_to_ticks))
    mcbar = Fig.colorbar(cax_1, ticks=tic)
    mcbar.ax.set_yticklabels(s_to_ticks)
    plt.title('From Subaqueous to...')

    # Beach to..
    beach_to = np.zeros([longshore, crossshore])
    beach_to[np.logical_and(most_likely_ts == 1, prev_most_likely == 1)] = 1
    beach_to[np.logical_and(most_likely_ts == 0, prev_most_likely == 1)] = 2
    beach_to[np.logical_and(most_likely_ts == 2, prev_most_likely == 1)] = 3
    beach_to[np.logical_and(most_likely_ts == 3, prev_most_likely == 1)] = 4
    beach_to[np.logical_and(most_likely_ts == 4, prev_most_likely == 1)] = 5

    b_to_ticks = ['', 'No Change', 'Subaqueous', 'Dune', 'Washover', 'Interior']
    cmap2 = colors.ListedColormap(['white', 'black', 'blue', 'saddlebrown', 'red', 'green'])
    ax_2 = Fig.add_subplot(232)
    cax_2 = ax_2.matshow(beach_to[:, plot_xmin: plot_xmax], cmap=cmap2, vmin=0, vmax=len(b_to_ticks) - 1)
    tic = np.linspace(start=((len(b_to_ticks) - 1) / len(b_to_ticks)) / 2, stop=len(b_to_ticks) - 1 - ((len(b_to_ticks) - 1) / len(b_to_ticks)) / 2, num=len(b_to_ticks))
    mcbar = Fig.colorbar(cax_2, ticks=tic)
    mcbar.ax.set_yticklabels(b_to_ticks)
    plt.title('From Beach to...')

    # Dune to..
    dune_to = np.zeros([longshore, crossshore])
    dune_to[np.logical_and(most_likely_ts == 2, prev_most_likely == 2)] = 1
    dune_to[np.logical_and(most_likely_ts == 0, prev_most_likely == 2)] = 2
    dune_to[np.logical_and(most_likely_ts == 1, prev_most_likely == 2)] = 3
    dune_to[np.logical_and(most_likely_ts == 3, prev_most_likely == 2)] = 4
    dune_to[np.logical_and(most_likely_ts == 4, prev_most_likely == 2)] = 5

    d_to_ticks = ['', 'No Change', 'Subaqueous', 'Beach', 'Washover', 'Interior']
    cmap3 = colors.ListedColormap(['white', 'black', 'blue', 'gold', 'red', 'green'])
    ax_3 = Fig.add_subplot(233)
    cax_3 = ax_3.matshow(dune_to[:, plot_xmin: plot_xmax], cmap=cmap3, vmin=0, vmax=len(d_to_ticks) - 1)
    tic = np.linspace(start=((len(d_to_ticks) - 1) / len(d_to_ticks)) / 2, stop=len(d_to_ticks) - 1 - ((len(d_to_ticks) - 1) / len(d_to_ticks)) / 2, num=len(d_to_ticks))
    mcbar = Fig.colorbar(cax_3, ticks=tic)
    mcbar.ax.set_yticklabels(d_to_ticks)
    plt.title('From Dune to...')

    # Washover to..
    washover_to = np.zeros([longshore, crossshore])
    washover_to[np.logical_and(most_likely_ts == 3, prev_most_likely == 3)] = 1
    washover_to[np.logical_and(most_likely_ts == 0, prev_most_likely == 3)] = 2
    washover_to[np.logical_and(most_likely_ts == 1, prev_most_likely == 3)] = 3
    washover_to[np.logical_and(most_likely_ts == 2, prev_most_likely == 3)] = 4
    washover_to[np.logical_and(most_likely_ts == 4, prev_most_likely == 3)] = 5

    w_to_ticks = ['', 'No Change', 'Subaqueous', 'Beach', 'Dune', 'Interior']
    cmap4 = colors.ListedColormap(['white', 'black', 'blue', 'gold', 'saddlebrown', 'green'])
    ax_4 = Fig.add_subplot(234)
    cax_4 = ax_4.matshow(washover_to[:, plot_xmin: plot_xmax], cmap=cmap4, vmin=0, vmax=len(w_to_ticks) - 1)
    tic = np.linspace(start=((len(w_to_ticks) - 1) / len(w_to_ticks)) / 2, stop=len(w_to_ticks) - 1 - ((len(w_to_ticks) - 1) / len(w_to_ticks)) / 2, num=len(w_to_ticks))
    mcbar = Fig.colorbar(cax_4, ticks=tic)
    mcbar.ax.set_yticklabels(w_to_ticks)
    plt.title('From Washover to...')

    # Interior to..
    interior_to = np.zeros([longshore, crossshore])
    interior_to[np.logical_and(most_likely_ts == 4, prev_most_likely == 4)] = 1
    interior_to[np.logical_and(most_likely_ts == 0, prev_most_likely == 4)] = 2
    interior_to[np.logical_and(most_likely_ts == 1, prev_most_likely == 4)] = 3
    interior_to[np.logical_and(most_likely_ts == 2, prev_most_likely == 4)] = 4
    interior_to[np.logical_and(most_likely_ts == 3, prev_most_likely == 4)] = 5

    i_to_ticks = ['', 'No Change', 'Subaqueous', 'Beach', 'Dune', 'Washover']
    cmap5 = colors.ListedColormap(['white', 'black', 'blue', 'gold', 'saddlebrown', 'red'])
    ax_5 = Fig.add_subplot(235)
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

    bFig = plt.figure(figsize=(14, 7.5))
    bFig.suptitle(name, fontsize=13)
    bax1 = bFig.add_subplot(231)
    bcax1 = bax1.matshow(class_probabilities[0, it, :, plot_xmin: plot_xmax], vmin=0, vmax=1)
    plt.title(class_labels[0])

    bax2 = bFig.add_subplot(232)
    bcax2 = bax2.matshow(class_probabilities[1, it, :, plot_xmin: plot_xmax], vmin=0, vmax=1)
    plt.title(class_labels[1])

    bax3 = bFig.add_subplot(233)
    bcax3 = bax3.matshow(class_probabilities[2, it, :, plot_xmin: plot_xmax], vmin=0, vmax=1)
    plt.title(class_labels[2])

    bax4 = bFig.add_subplot(234)
    bcax4 = bax4.matshow(class_probabilities[3, it, :, plot_xmin: plot_xmax], vmin=0, vmax=1)
    plt.title(class_labels[3])

    bax5 = bFig.add_subplot(235)
    bcax5 = bax5.matshow(class_probabilities[4, it, :, plot_xmin: plot_xmax], vmin=0, vmax=1)
    plt.title(class_labels[4])
    # cbar = Fig.colorbar(bcax5)
    plt.tight_layout()


def ani_frame_bins(timestep, class_probabilities, cax1, cax2, cax3, cax4, cax5, text1, text2, text3, text4, text5):

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

    return cax1, cax2, cax3, cax4, cax5, text1, text2, text3, text4, text5


def ani_frame_most_probable_outcome(timestep, class_probabilities, cax1, cax2, text1):

    Max_idx = np.argmax(class_probabilities[:, timestep, :, plot_xmin: plot_xmax], axis=0)
    Conf = 1 - np.max(class_probabilities[:, timestep, :, plot_xmin: plot_xmax], axis=0)
    cax1.set_data(Max_idx)
    cax2.set_alpha(Conf)
    yrstr = "Year " + str(timestep * save_frequency)
    text1.set_text(yrstr)

    return cax1, cax2, text1


def ani_frame_most_probable_outcome_2(timestep, class_probabilities, cax1, cax2, text1, text2):

    Max_idx = np.argmax(class_probabilities[:, timestep, :, plot_xmin: plot_xmax], axis=0)
    Conf = np.max(class_probabilities[:, timestep, :, plot_xmin: plot_xmax], axis=0)
    cax1.set_data(Max_idx)
    cax2.set_data(Conf)
    yrstr = "Year " + str(timestep * save_frequency)
    text1.set_text(yrstr)
    text2.set_text(yrstr)

    return cax1, cax2, text1, text2


def bins_animation(class_probabilities, class_labels):
    # Set animation base figure
    Fig = plt.figure(figsize=(14, 7.5))
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
    plt.tight_layout()

    # Create and save animation
    ani1 = animation.FuncAnimation(Fig, ani_frame_bins, frames=num_saves, fargs=(class_probabilities, cax1, cax2, cax3, cax4, cax5, text1, text2, text3, text4, text5,), interval=300, blit=True)
    c = 1
    while os.path.exists("Output/Animation/meeb_prob_bins_" + str(c) + ".gif"):
        c += 1
    ani1.save("Output/Animation/meeb_prob_bins_" + str(c) + ".gif", dpi=150, writer="imagemagick")


def most_likely_animation(class_probabilities, class_cmap, class_labels):

    num_classes = class_probabilities.shape[0]

    # Set animation base figure
    Fig = plt.figure(figsize=(14, 7.5))
    ax1 = Fig.add_subplot(111)
    conf_cmap = colors.ListedColormap(['white'])
    max_idx = np.argmax(class_probabilities[:, 0, :, plot_xmin: plot_xmax], axis=0)
    conf = 1 - np.max(class_probabilities[:, 0, :, plot_xmin: plot_xmax], axis=0)
    cax1 = ax1.matshow(max_idx, cmap=class_cmap, vmin=0, vmax=num_classes - 1)
    cax2 = ax1.matshow(np.ones(conf.shape), cmap=conf_cmap, vmin=0, vmax=1, alpha=conf)
    ticks = np.linspace(start=((num_classes - 1) / num_classes) / 2, stop=num_classes - 1 - ((num_classes - 1) / num_classes) / 2, num=num_classes)
    cbar = Fig.colorbar(cax1, ticks=ticks)
    cbar.ax.set_yticklabels(class_labels)
    plt.title(name)
    timestr = "Year " + str(0)
    text1 = plt.text(2, longshore - 2, timestr, c='white')

    # Create and save animation
    ani2 = animation.FuncAnimation(Fig, ani_frame_most_probable_outcome, frames=num_saves, fargs=(class_probabilities, cax1, cax2, text1), interval=300, blit=True)
    c = 1
    while os.path.exists("Output/Animation/meeb_most_likely_" + str(c) + ".gif"):
        c += 1
    ani2.save("Output/Animation/meeb_most_likely_" + str(c) + ".gif", dpi=150, writer="imagemagick")


def most_likely_animation_2(class_probabilities, class_cmap, class_labels):

    num_classes = class_probabilities.shape[0]

    # Set animation base figure
    max_idx = np.argmax(class_probabilities[:, 0, :, plot_xmin: plot_xmax], axis=0)  # Bin of most probable outcome
    conf = np.max(class_probabilities[:, 0, :, plot_xmin: plot_xmax], axis=0)  # Confidence, i.e. probability of most probable outcome
    min_conf = 1 / num_classes

    Fig = plt.figure(figsize=(10, 11))
    ax1 = Fig.add_subplot(211)
    cax1 = ax1.matshow(max_idx, cmap=class_cmap, vmin=0, vmax=num_classes - 1)
    tic = np.linspace(start=((num_classes - 1) / num_classes) / 2, stop=num_classes - 1 - ((num_classes - 1) / num_classes) / 2, num=num_classes)
    mcbar = Fig.colorbar(cax1, ticks=tic)
    mcbar.ax.set_yticklabels(class_labels)
    timestr = "Year " + str(0)
    text1 = plt.text(2, longshore - 2, timestr, c='black')

    ax2 = Fig.add_subplot(212)
    cax2 = ax2.matshow(conf, cmap=cmap_conf, vmin=min_conf, vmax=1)
    Fig.colorbar(cax2)
    timestr = "Year " + str(0)
    text2 = plt.text(2, longshore - 2, timestr, c='white')

    plt.tight_layout()

    # Create and save animation
    ani3 = animation.FuncAnimation(Fig, ani_frame_most_probable_outcome_2, frames=num_saves, fargs=(class_probabilities, cax1, cax2, text1, text2), interval=300, blit=True)
    c = 1
    while os.path.exists("Output/Animation/meeb_most_likely_" + str(c) + ".gif"):
        c += 1
    ani3.save("Output/Animation/meeb_most_likely_" + str(c) + ".gif", dpi=150, writer="imagemagick")


# __________________________________________________________________________________________________________________________________
# VARIABLES AND INITIALIZATIONS

# # 2014
# start = "Init_NCB-NewDrum-Ocracoke_2014_PostSandy-NCFMP-Plover.npy"
# startdate = '20140406'

# 2018
start = "Init_NCB-NewDrum-Ocracoke_2018_PostFlorence-Plover_2m.npy"
startdate = '20181007'


# _____________________
# PROBABILISTIC PROJECTIONS
RSLR_bin = [0.0068, 0.0096, 0.0124]  # [m/yr] Bins of future RSLR rates
RSLR_prob = [0.26, 0.55, 0.19]  # Probability of future RSLR bins (must sum to 1.0)

# _____________________
# CLASSIFICATION SCHEME SPECIFICATIONS
elev_classification_label = 'Elevation Change [m]'  # Axes labels on figures
elev_class_edges = [-np.inf, -0.5, -0.1, 0.1, 0.5, np.inf]  # [m] Elevation change
elev_class_labels = ['< -0.5', '-0.5 - -0.1', '-0.1 - 0.1', '0.1 - 0.5', '> 0.5']
elev_class_cmap = colors.ListedColormap(['red', 'gold', 'black', 'aquamarine', 'mediumblue'])
elev_class_cmap_2 = colors.ListedColormap(['#ca0020', '#f4a582', '#f7f7f7', '#92c5de', '#0571b0'])

state_classification_label = 'Ecogeomorphic State'  # Axes labels on figures
state_class_edges = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]  # [m] State change
state_class_labels = ['Subaqueous', 'Beach', 'Dune', 'Washover', 'Interior']
state_class_cmap = colors.ListedColormap(['blue', 'gold', 'saddlebrown', 'red', 'green'])

# Confidence
cmap_conf = plt.get_cmap('BuPu', 4)  # 4 discrete colors

# _____________________
# INITIAL PARAMETERS

sim_duration = 32  # [yr] Note: For probabilistic projections, use a duration that is divisible by the save_frequency
save_frequency = 0.5  # [yr] Time step for probability calculations

duplicates = 10  # To account for internal stochasticity (e.g., storms, aeolian)
core_num = 8  # min(duplicates, 20)  # Number of cores to use in the parallelization (IR PC: 24)

# Define Horizontal and Vertical References of Domain
ymin = 19000  # [m] Alongshore coordinate
ymax = 19500  # [m] Alongshore coordinate
xmin = 900  # [m] Cross-shore coordinate
xmax = 1600  # [m] Cross-shore coordinate
plot_xmin = 0  # [m] Cross-shore coordinate (for plotting), relative to trimmed domain
plot_xmax = plot_xmin + 600  # [m] Cross-shore coordinate (for plotting), relative to trimmed domain
MHW_init = 0.39  # [m NAVD88] Initial mean high water
cellsize = 2  # [m]

name = '19000-19500, 2018-2050, 10 duplicates, Elevation and Ecogeomorphic State, RSLR Rate(6.8, 9.6, 12.4) Prob(0.26, 0.55, 0.19), 30Jul24'  # Name of simulation suite

plot = True  # [bool]
animate = False  # [bool]
save_data = False  # [bool]
savename = '19000-19500'

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

longshore, crossshore = topo_start.shape

elev_num_classes = len(elev_class_edges) - 1
state_num_classes = len(state_class_edges) - 1
num_saves = int(np.floor(sim_duration/save_frequency)) + 1

del Init

# __________________________________________________________________________________________________________________________________
# RUN MODEL

print()
print(name)
print()

start_time = time.time()  # Record time at start of simulation

# Determine classification probabilities cross space and time for joint intrinsic-external stochastic elements
elev_class_probabilities, state_class_probabilities = joint_probability()

# Print elapsed time of simulation
print()
SimDuration = time.time() - start_time
print()
print("Elapsed Time: ", SimDuration, "sec")


# __________________________________________________________________________________________________________________________________
# PLOT RESULTS

if plot:
    plot_class_maps(elev_class_probabilities, elev_class_labels, it=-1)
    plot_class_maps(state_class_probabilities, state_class_labels, it=-1)
    plot_most_probable_class_2(elev_class_probabilities, elev_class_cmap_2, elev_class_labels, it=-1)
    plot_most_probable_class_2(state_class_probabilities, state_class_cmap, state_class_labels, it=-1)
    plot_class_area_change_over_time(state_class_probabilities)
    plot_most_likely_transition_maps(state_class_probabilities)
    plot_transitions_area_matrix(state_class_probabilities, state_class_labels)
if animate:
    bins_animation(elev_class_probabilities, elev_class_labels)
    bins_animation(state_class_probabilities, state_class_labels)
    most_likely_animation_2(elev_class_probabilities, elev_class_cmap_2, elev_class_labels)
    most_likely_animation_2(state_class_probabilities, state_class_cmap, state_class_labels)
plt.show()


# __________________________________________________________________________________________________________________________________
# SAVE DATA

if save_data:
    # Elevation
    elev_name = "ElevClassProbabilities_" + savename
    elev_outloc = "Output/SimData/" + elev_name
    np.save(elev_outloc, elev_class_probabilities)
    # State
    state_name = "StateClassProbabilities_" + savename
    state_outloc = "Output/SimData/" + state_name
    np.save(state_outloc, state_class_probabilities)

