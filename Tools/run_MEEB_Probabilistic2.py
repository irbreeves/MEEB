"""
Probabilistic framework for running MEEB simulations. Generates probabilistic projections of future change.

IRBR 12 February 2024
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
        groundwater_depth=0.4,
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

    # Loop through time
    for time_step in range(int(meeb.iterations)):
        # Run time step
        meeb.update(time_step)

    # Topo change
    topo_start_sim = meeb.topo_TS[:, :, 0]  # [m NAVDD88]
    topo_end_sim = meeb.topo_TS[:, :, -1]  # [m NAVDD88]
    mhw_end_sim = meeb.MHW  # [m NAVD88]

    # Veg change
    veg_start_sim = meeb.veg_TS[:, :, 0]
    veg_end_sim = meeb.veg_TS[:, :, -1]

    # Subaerial mask
    subaerial_mask = topo_end_sim > MHW_init  # [bool] Mask for every cell above initial MHW; Note: this should be changed if modeling sea-level fall

    topo_change_sim_TS = np.zeros(meeb.topo_TS.shape)
    for ts in range(meeb.topo_TS.shape[2]):
        topo_change_ts = (meeb.topo_TS[:, :, ts] - topo_start_sim) * subaerial_mask  # Disregard change that is not subaerial
        topo_change_sim_TS[:, :, ts] = topo_change_ts

    # Create classified map
    # elevation_classification = classify_topo_change(meeb.topo_TS.shape[2], topo_change_sim_TS)
    state_classification = classify_ecogeomorphic_state(meeb.topo_TS.shape[2], meeb.topo_TS, meeb.veg_TS, meeb.MHW_init, meeb.RSLR, vegetated_threshold=0.12)

    return state_classification


def classify_topo_change(TS, topo_change_sim_TS):
    """Classify according to range of elevation change."""

    topo_change_bin = np.zeros([num_classes, num_saves, longshore, crossshore])

    for b in range(len(class_edges) - 1):
        lower = class_edges[b]
        upper = class_edges[b + 1]

        for ts in range(TS):
            bin_change = np.logical_and(lower < topo_change_sim_TS[:, :, ts], topo_change_sim_TS[:, :, ts] <= upper).astype(int)
            topo_change_bin[b, ts, :, :] += bin_change

    return topo_change_bin


def classify_ecogeomorphic_state(TS, topo_TS, veg_TS, mhw_init, rslr, vegetated_threshold):
    """Classify by ecogeomorphic state using topography and vegetation."""

    state_classes = np.zeros([num_classes, num_saves, longshore, crossshore])  # Initialize

    # Run Categorization
    # Loop through saved timesteps
    for ts in range(TS):

        # Find MHW for this time step
        MHW = mhw_init + rslr * ts * save_frequency

        # Smooth topography to remove small-scale variability
        topo = scipy.ndimage.gaussian_filter(topo_TS[:, :, ts], 5, mode='constant')

        # Find dune crest, toe, and heel lines
        dune_crestline, not_gap = routine.foredune_crest(topo, MHW)
        dune_toeline = routine.foredune_toe(topo, dune_crestline, MHW, not_gap)
        dune_heelline = routine.foredune_heel(topo, dune_crestline, not_gap, threshold=0.6)

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
                fronting_dune_gap[:, crest_loc - right_diag.shape[0]: crest_loc] = np.fliplr(fronting_dune_gap[:, crest_loc:])

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


def internal_probability(rslr):
    """Runs duplicate simulations to find the classification probability from stochastic processes intrinsic to the system, particularly storm
    occurence & intensity, aeolian dynamics, and vegetation dynamics."""

    # Create storage array
    class_bins = np.zeros([num_classes, num_saves, longshore, crossshore])

    with routine.tqdm_joblib(tqdm(desc="RSLR[" + str(rslr) + "]", total=duplicates)) as progress_bar:
        class_duplicates = Parallel(n_jobs=core_num)(delayed(run_individual_sim)(rslr) for i in range(duplicates))

    for ts in range(num_saves):
        for b in range(num_classes):
            for n in range(duplicates):
                class_bins[b, ts, :, :] += class_duplicates[n][b, ts, :, :]

    internal_prob = class_bins / duplicates

    return internal_prob


def joint_probability():
    """Runs a range of probabilistic scenarios to find the classification probability from stochastic processes external to the system, e.g. RSLR, atmospheric temperature."""

    # Create storage array
    joint_prob = np.zeros([num_classes, num_saves, longshore, crossshore])

    for r in range(len(RSLR_prob)):
        internal_prob = internal_probability(RSLR_bin[r])
        external_prob = internal_prob * RSLR_prob[r]  # To add more external drivers: add nested for loop and multiply here, e.g. * temp_prob[t]
        joint_prob += external_prob

    return joint_prob


def plot_cell_prob_bar(it, l, c):
    """For a particular cell, makes bar plot of the probabilities of each class.

    Parameters
    ----------
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


def plot_most_probable_class(it):
    """Plots the most probable class across the domain at a particular time step. Note: this returns the first max occurance,
    i.e. if multiple bins are tied for the maximum probability of occuring, the first one will be plotted as the most likely.

    Parameters
    ----------
    it : int
        Iteration to draw probabilities from.
    """

    mmax_idx = np.argmax(class_probabilities[:, it, :, xmin: xmax], axis=0)  # Bin of most probable outcome
    confidence = 1 - np.max(class_probabilities[:, it, :, xmin: xmax], axis=0)  # Confidence, i.e. probability of most probable outcome

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


def plot_class_maps(it):
    """Plots probability of occurance across the domain for each class at a particular timestep.

    Parameters
    ----------
    it : int
        Iteration to draw probabilities from.
    """

    bFig = plt.figure(figsize=(14, 7.5))
    bFig.suptitle(name, fontsize=13)
    bax1 = bFig.add_subplot(231)
    bcax1 = bax1.matshow(class_probabilities[0, it, :, xmin: xmax], vmin=0, vmax=1)
    plt.title(class_labels[0])

    bax2 = bFig.add_subplot(232)
    bcax2 = bax2.matshow(class_probabilities[1, it, :, xmin: xmax], vmin=0, vmax=1)
    plt.title(class_labels[1])

    bax3 = bFig.add_subplot(233)
    bcax3 = bax3.matshow(class_probabilities[2, it, :, xmin: xmax], vmin=0, vmax=1)
    plt.title(class_labels[2])

    bax4 = bFig.add_subplot(234)
    bcax4 = bax4.matshow(class_probabilities[3, it, :, xmin: xmax], vmin=0, vmax=1)
    plt.title(class_labels[3])

    bax5 = bFig.add_subplot(235)
    bcax5 = bax5.matshow(class_probabilities[4, it, :, xmin: xmax], vmin=0, vmax=1)
    plt.title(class_labels[4])
    # cbar = Fig.colorbar(bcax5)
    plt.tight_layout()


def ani_frame_bins(timestep):

    prob1 = class_probabilities[0, timestep, :, xmin: xmax]
    cax1.set_data(prob1)
    yrstr = "Year " + str(timestep * save_frequency)
    text1.set_text(yrstr)

    prob2 = class_probabilities[1, timestep, :, xmin: xmax]
    cax2.set_data(prob2)
    text2.set_text(yrstr)

    prob3 = class_probabilities[2, timestep, :, xmin: xmax]
    cax3.set_data(prob3)
    text3.set_text(yrstr)

    prob4 = class_probabilities[3, timestep, :, xmin: xmax]
    cax4.set_data(prob4)
    text4.set_text(yrstr)

    prob5 = class_probabilities[4, timestep, :, xmin: xmax]
    cax5.set_data(prob5)
    text5.set_text(yrstr)

    return cax1, cax2, cax3, cax4, cax5, text1, text2, text3, text4, text5


def ani_frame_most_probable_outcome(timestep):

    Max_idx = np.argmax(class_probabilities[:, timestep, :, xmin: xmax], axis=0)
    Conf = 1 - np.max(class_probabilities[:, timestep, :, xmin: xmax], axis=0)
    cax1.set_data(Max_idx)
    cax2.set_alpha(Conf)
    yrstr = "Year " + str(timestep * save_frequency)
    text1.set_text(yrstr)

    return cax1, cax2, text1


# __________________________________________________________________________________________________________________________________
# VARIABLES AND INITIALIZATIONS

# # 2014
# start = "Init_NCB-NewDrum-Ocracoke_2014_PostSandy-NCFMP-Plover.npy"
# startdate = '20140406'

# 2018
start = "Init_NCB-NewDrum-Ocracoke_2018_PostFlorence-Plover.npy"
startdate = '20181007'


# _____________________
# PROBABILISTIC PROJECTIONS
RSLR_bin = [0.0068, 0.0096, 0.0124]  # [m/yr] Bins of future RSLR rates
RSLR_prob = [0.26, 0.55, 0.19]  # Probability of future RSLR bins (must sum to 1.0)

# _____________________
# CLASSIFICATION SCHEME SPECIFICATIONS
# classification_label = 'Elevation Change [m]'  # Axes labels on figures
# class_edges = [-np.inf, -0.5, -0.1, 0.1, 0.5, np.inf]  # [m] Elevation change
# class_labels = ['< -0.5', '-0.5 - -0.1', '-0.1 - 0.1', '0.1 - 0.5', '> 0.5']
# class_cmap = colors.ListedColormap(['red', 'gold', 'black', 'aquamarine', 'mediumblue'])

classification_label = 'Ecogeomorphic State'  # Axes labels on figures
class_edges = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]  # [m] Elevation change
class_labels = ['Subaqueous', 'Beach', 'Dune', 'Washover', 'Interior']
class_cmap = colors.ListedColormap(['blue', 'gold', 'saddlebrown', 'red', 'green'])

# _____________________
# INITIAL PARAMETERS

sim_duration = 32  # [yr] Note: For probabilistic projections, use a duration that is divisible by the save_frequency
save_frequency = 0.5  # [yr] Time step for probability calculations

duplicates = 24  # To account for internal stochasticity (e.g., storms, aeolian)
core_num = min(duplicates, 12)  # Number of cores to use in the parallelization (IR PC: 24)

# Define Horizontal and Vertical References of Domain
ymin = 19000  # [m] Alongshore coordinate
ymax = 19500  # [m] Alongshore coordinate
xmin = 900  # [m] Cross-shore coordinate (for plotting)
xmax = xmin + 600  # [m] Cross-shore coordinate (for plotting)
MHW_init = 0.39  # [m NAVD88] Initial mean high water

name = '19000-19500, 24 duplicates, 2018-2050, RSLR Rate(6.8, 9.6, 12.4) Prob(0.26, 0.55, 0.19)'  # Name of simulation suite

# 21000


# _____________________
# INITIAL CONDITIONS

# Load Initial Domains
Init = np.load("Input/" + start)
topo_start = Init[0, ymin: ymax, :]

longshore, crossshore = topo_start.shape

num_classes = len(class_edges) - 1
num_saves = int(np.floor(sim_duration/save_frequency)) + 1


# __________________________________________________________________________________________________________________________________
# RUN MODEL

print()
print(name)
print()

start_time = time.time()  # Record time at start of simulation

# Determine classification probabilities cross space and time for joint intrinsic-external stochastic elements
class_probabilities = joint_probability()

# Print elapsed time of simulation
print()
SimDuration = time.time() - start_time
print()
print("Elapsed Time: ", SimDuration, "sec")


# __________________________________________________________________________________________________________________________________
# PLOT RESULTS

plot_class_maps(it=-1)
plot_most_probable_class(it=-1)

# -----------------
# Animation 1

# Set animation base figure
Fig = plt.figure(figsize=(14, 7.5))
Fig.suptitle(name, fontsize=13)
ax1 = Fig.add_subplot(231)
cax1 = ax1.matshow(class_probabilities[0, 0, :, xmin: xmax], vmin=0, vmax=1)
plt.title(class_labels[0])
timestr = "Year " + str(0)
text1 = plt.text(2, longshore - 2, timestr, c='white')

ax2 = Fig.add_subplot(232)
cax2 = ax2.matshow(class_probabilities[1, 0, :, xmin: xmax], vmin=0, vmax=1)
plt.title(class_labels[1])
text2 = plt.text(2, longshore - 2, timestr, c='white')

ax3 = Fig.add_subplot(233)
cax3 = ax3.matshow(class_probabilities[2, 0, :, xmin: xmax], vmin=0, vmax=1)
plt.title(class_labels[2])
text3 = plt.text(2, longshore - 2, timestr, c='white')

ax4 = Fig.add_subplot(234)
cax4 = ax4.matshow(class_probabilities[3, 0, :, xmin: xmax], vmin=0, vmax=1)
plt.title(class_labels[3])
text4 = plt.text(2, longshore - 2, timestr, c='white')

ax5 = Fig.add_subplot(235)
cax5 = ax5.matshow(class_probabilities[4, 0, :, xmin: xmax], vmin=0, vmax=1)
plt.title(class_labels[4])
text5 = plt.text(2, longshore - 2, timestr, c='white')
# cbar = Fig.colorbar(cax5)
plt.tight_layout()

# Create and save animation
ani1 = animation.FuncAnimation(Fig, ani_frame_bins, frames=num_saves, interval=300, blit=True)
c = 1
while os.path.exists("Output/Animation/meeb_prob_bins_" + str(c) + ".gif"):
    c += 1
ani1.save("Output/Animation/meeb_prob_bins_" + str(c) + ".gif", dpi=150, writer="imagemagick")


# -----------------
# Animation 2

# Set animation base figure
Fig = plt.figure(figsize=(14, 7.5))
ax1 = Fig.add_subplot(111)
conf_cmap = colors.ListedColormap(['white'])
max_idx = np.argmax(class_probabilities[:, 0, :, xmin: xmax], axis=0)
conf = 1 - np.max(class_probabilities[:, 0, :, xmin: xmax], axis=0)
cax1 = ax1.matshow(max_idx, cmap=class_cmap, vmin=0, vmax=num_classes - 1)
cax2 = ax1.matshow(np.ones(conf.shape), cmap=conf_cmap, vmin=0, vmax=1, alpha=conf)
ticks = np.linspace(start=((num_classes - 1) / num_classes) / 2, stop=num_classes - 1 - ((num_classes - 1) / num_classes) / 2, num=num_classes)
cbar = Fig.colorbar(cax1, ticks=ticks)
cbar.ax.set_yticklabels(class_labels)
plt.title(name)
timestr = "Year " + str(0)
text1 = plt.text(2, longshore - 2, timestr, c='white')

# Create and save animation
ani2 = animation.FuncAnimation(Fig, ani_frame_most_probable_outcome, frames=num_saves, interval=300, blit=True)
c = 1
while os.path.exists("Output/Animation/meeb_most_likely_" + str(c) + ".gif"):
    c += 1
ani2.save("Output/Animation/meeb_most_likely_" + str(c) + ".gif", dpi=150, writer="imagemagick")

plt.show()
