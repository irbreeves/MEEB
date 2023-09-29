"""
Probabilistic framework for running MEEB simulations. Generates probabilistic projections of future change.

IRBR 21 September 2023
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import os
from matplotlib import colors
from multiprocessing import Pool
from joblib import Parallel, delayed
from meeb import MEEB


# __________________________________________________________________________________________________________________________________
# FUNCTIONS


def run_individual_sim():
    """Runs uniqe individual MEEB simulation."""

    global sim_count

    # Create an instance of the MEEB class
    meeb = MEEB(
        name=name,
        simulation_time_yr=sim_duration,
        alongshore_domain_boundary_min=ymin,
        alongshore_domain_boundary_max=ymax,
        RSLR=0.001,
        MHW=MHW,
        init_filename=start,
        hindcast=False,
        seeded_random_numbers=False,
        simulation_start_date=startdate,
        storm_timeseries_filename='StormTimeSeries_1980-2020_NCB-CE_Beta0pt039_BermEl1pt78.npy',
        storm_list_filename='NCB_SimStorms.npy',
        save_frequency=save_frequency,
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
        beach_equilibrium_slope=0.012,
        beach_erosiveness=1.84,
        beach_substeps=51,
        flow_reduction_max_spec1=0.17,
        flow_reduction_max_spec2=0.44,
        # --- Shoreline --- #
        wave_asymetry=0.6,
        wave_high_angle_fraction=0.39,
        mean_wave_height=0.98,
        mean_wave_period=6.6,
        alongshore_section_length=50,
        estimate_shoreface_parameters=True,
        # --- Veg --- #
    )

    sim_count = sim_count + 1
    print("\r", "Running simulation: ", sim_count, " / ", num_sim, end="")  # TODO: Not working

    # Loop through time
    for time_step in range(int(meeb.iterations)):
        # print("\r", "Time Step: ", (time_step + 1) / meeb.iterations_per_cycle, "years", end="")

        # Run time step
        meeb.update(time_step)

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

    topo_change_sim_TS = np.zeros(meeb.topo_TS.shape)
    for ts in range(meeb.topo_TS.shape[2]):
        topo_change_ts = (meeb.topo_TS[:, :, ts] - topo_start_sim) * subaerial_mask  # Disregard change that is not subaerial
        topo_change_sim_TS[:, :, ts] = topo_change_ts

    # Disregard change that is not subaerial
    topo_change_sim *= subaerial_mask

    # Bin by elevation change
    topo_change_bin = np.zeros([num_bins, num_saves, longshore, crossshore])

    for b in range(len(bin_edges) - 1):
        lower = bin_edges[b]
        upper = bin_edges[b + 1]

        for ts in range(meeb.topo_TS.shape[2]):
            bin_change = np.logical_and(lower < topo_change_sim_TS[:, :, ts], topo_change_sim_TS[:, :, ts] <= upper).astype(int)
            topo_change_bin[b, ts, :, :] += bin_change

    return topo_change_bin


def interal_probabilistic_change():

    # Create storage array
    topo_change_bins = np.zeros([num_bins, num_saves, longshore, crossshore])

    topo_change_bins_duplicates = Parallel(n_jobs=core_num)(delayed(run_individual_sim)() for i in range(duplicates))

    # TODO: try Pool multiprocessing as alternative?

    for ts in range(num_saves):
        for b in range(num_bins):
            for n in range(duplicates):
                topo_change_bins[b, ts, :, :] += topo_change_bins_duplicates[n][b, ts, :, :]

    prob_change = topo_change_bins / duplicates

    return prob_change


def plot_cell_prob_bar(it, l, c):
    """For a particular cell, makes bar plot of the probabilities of each elevation change bin.

    Parameters
    ----------
    it : int
        Iteration to draw probabilities from.
    l : int
        Longshore (y) coordinate of cell.
    c : int
        Cross-shore (x) coordinate of cell.
    """

    probs = prob_change_internal[:, it, l, c]

    plt.figure(figsize=(8, 8))
    plt.bar(np.arange(len(probs)), probs)
    x_locs = np.arange(len(probs))
    plt.xticks(x_locs, bin_labels)
    ax = plt.gca()
    ax.set_ylim([0, 1])
    plt.xlabel('Elevation Change [m]')
    plt.ylabel('Probability')
    plt.title('Loc: (' + str(c) + ', ' + str(l) + '), Iteration: ' + str(it))


def plot_most_probable_outcome(it):
    """Plots the most probable outcome across the domain at a particular time step. Note: this returns the first max occurance,
    i.e. if multiple bins are tied for the maximum probability of occuring, the first one will be plotted as the most likely.

    Parameters
    ----------
    it : int
        Iteration to draw probabilities from.
    """

    mmax_idx = np.argmax(prob_change_internal[:, it, :, xmin: xmax], axis=0)  # Bin of most probably outcome
    conf = 1 - np.max(prob_change_internal[:, it, :, xmin: xmax], axis=0)  # Confidence, i.e. probability of most probable outcome

    cmap1 = colors.ListedColormap(['yellow', 'red', 'black', 'blue', 'green'])
    cmap2 = colors.ListedColormap(['white'])

    fig, ax = plt.subplots()
    cax = ax.matshow(mmax_idx, cmap=cmap1, vmin=0, vmax=num_bins-1)
    cax2 = ax.matshow(np.ones(mmax_idx.shape), cmap=cmap2, vmin=0, vmax=1, alpha=conf)
    ticks = np.linspace(start=((num_bins - 1) / num_bins) / 2, stop=num_bins - 1 - ((num_bins - 1) / num_bins) / 2, num=num_bins)
    mcbar = fig.colorbar(cax, ticks=ticks)
    mcbar.ax.set_yticklabels(bin_labels)
    plt.xlabel('Alongshore Distance [m]')
    plt.ylabel('Cross-Shore Distance [m]')
    plt.title('Iteration: ' + str(it))


def plot_bin_maps(it):
    """Plots probability of occurance across the domain for each scenario bin at a particular timestep.

    Parameters
    ----------
    it : int
        Iteration to draw probabilities from.
    """

    bFig = plt.figure(figsize=(14, 7.5))
    bFig.suptitle(name, fontsize=13)
    bax1 = bFig.add_subplot(231)
    bcax1 = bax1.matshow(prob_change_internal[0, it, :, xmin: xmax], vmin=0, vmax=1)
    plt.title(bin_labels[0])

    bax2 = bFig.add_subplot(232)
    bcax2 = bax2.matshow(prob_change_internal[1, it, :, xmin: xmax], vmin=0, vmax=1)
    plt.title(bin_labels[1])

    bax3 = bFig.add_subplot(233)
    bcax3 = bax3.matshow(prob_change_internal[2, it, :, xmin: xmax], vmin=0, vmax=1)
    plt.title(bin_labels[2])

    bax4 = bFig.add_subplot(234)
    bcax4 = bax4.matshow(prob_change_internal[3, it, :, xmin: xmax], vmin=0, vmax=1)
    plt.title(bin_labels[3])

    bax5 = bFig.add_subplot(235)
    bcax5 = bax5.matshow(prob_change_internal[4, it, :, xmin: xmax], vmin=0, vmax=1)
    plt.title(bin_labels[4])
    # cbar = Fig.colorbar(bcax5)
    plt.tight_layout()


def ani_frame_bins(timestep):

    prob1 = prob_change_internal[0, timestep, :, xmin: xmax]
    cax1.set_data(prob1)
    yrstr = "Year " + str(timestep * save_frequency)
    text1.set_text(yrstr)

    prob2 = prob_change_internal[1, timestep, :, xmin: xmax]
    cax2.set_data(prob2)
    text2.set_text(yrstr)

    prob3 = prob_change_internal[2, timestep, :, xmin: xmax]
    cax3.set_data(prob3)
    text3.set_text(yrstr)

    prob4 = prob_change_internal[3, timestep, :, xmin: xmax]
    cax4.set_data(prob4)
    text4.set_text(yrstr)

    prob5 = prob_change_internal[4, timestep, :, xmin: xmax]
    cax5.set_data(prob5)
    text5.set_text(yrstr)

    return cax1, cax2, cax3, cax4, cax5, text1, text2, text3, text4, text5


def ani_frame_most_probable_outcome(timestep):

    Max_idx = np.argmax(prob_change_internal[:, timestep, :, xmin: xmax], axis=0)
    Conf = 1 - np.max(prob_change_internal[:, timestep, :, xmin: xmax], axis=0)
    cax1.set_data(Max_idx)
    cax2.set_data(Conf)
    yrstr = "Year " + str(timestep * save_frequency)
    text1.set_text(yrstr)

    return cax1, cax2, text1  # TODO: Not updating the confidence layer


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


# _____________________
# PROBABILISTIC PROJECTIONS
RSLR_bin = [1, 3, 5, 7]  # Bins of future RSLR rates
RSLR_prob = [0.1, 0.5, 0.3, 0.1]  # Probability of future RSLR bins (sum to 1.0)

bin_edges = [-np.inf, -0.5, -0.1, 0.1, 0.5, np.inf]  # [m] Elevation change
bin_labels = ['< -0.5', '-0.5 - -0.1', '-0.1 - 0.1', '0.1 - 0.5', '> 0.5']


# _____________________
# INITIAL PARAMETERS

sim_duration = 0.1  # [yr]
save_frequency = 0.5  # [yr]

duplicates = 4  # To account for internal stochasticity (e.g., storms, aeolian)
core_num = 2  # Number of cores to use in the parallelization

# Define Horizontal and Vertical References of Domain
ymin = 18950  # [m] Alongshore coordinate
ymax = 19250  # [m] Alongshore coordinate
xmin = 1000  # [m] Cross-shore coordinate (for plotting)
xmax = 1600  # [m] Cross-shore coordinate (for plotting)
MHW = 0.39  # [m NAVD88] Initial mean high water

name = '2018-2018, 4 duplicates'  # Name of simulation suite


# _____________________
# INITIAL CONDITIONS

# Load Initial Domains
Init = np.load("Input/" + start)
topo_start = Init[0, ymin: ymax, :]

longshore, crossshore = topo_start.shape

num_bins = len(bin_edges) - 1
num_saves = int(np.floor(sim_duration/save_frequency)) + 1


# __________________________________________________________________________________________________________________________________
# RUN MODEL

print()
print(name)

sim_count = 0  # Initialize sim counter
num_sim = duplicates  #* len(RSLR_bin)  # Total number of simulations
start_time = time.time()  # Record time at start of simulation

# Determine probabilistic change from internal stochasticity
prob_change_internal = interal_probabilistic_change()

# Print elapsed time of simulation
print()
SimDuration = time.time() - start_time
print()
print("Elapsed Time: ", SimDuration, "sec")


# __________________________________________________________________________________________________________________________________
# PLOT RESULTS

plot_bin_maps(it=-1)
plot_most_probable_outcome(it=-1)
plot_cell_prob_bar(it=-1, l=182, c=1311)


# -----------------
# Animation 1

# Set animation base figure
Fig = plt.figure(figsize=(14, 7.5))
Fig.suptitle(name, fontsize=13)
ax1 = Fig.add_subplot(231)
cax1 = ax1.matshow(prob_change_internal[0, 0, :, xmin: xmax], vmin=0, vmax=1)
plt.title(bin_labels[0])
timestr = "Year " + str(0)
text1 = plt.text(2, longshore - 2, timestr, c='white')

ax2 = Fig.add_subplot(232)
cax2 = ax2.matshow(prob_change_internal[1, 0, :, xmin: xmax], vmin=0, vmax=1)
plt.title(bin_labels[1])
text2 = plt.text(2, longshore - 2, timestr, c='white')

ax3 = Fig.add_subplot(233)
cax3 = ax3.matshow(prob_change_internal[2, 0, :, xmin: xmax], vmin=0, vmax=1)
plt.title(bin_labels[2])
text3 = plt.text(2, longshore - 2, timestr, c='white')

ax4 = Fig.add_subplot(234)
cax4 = ax4.matshow(prob_change_internal[3, 0, :, xmin: xmax], vmin=0, vmax=1)
plt.title(bin_labels[3])
text4 = plt.text(2, longshore - 2, timestr, c='white')

ax5 = Fig.add_subplot(235)
cax5 = ax5.matshow(prob_change_internal[4, 0, :, xmin: xmax], vmin=0, vmax=1)
plt.title(bin_labels[4])
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
cmap1 = colors.ListedColormap(['yellow', 'red', 'black', 'blue', 'green'])
cmap2 = colors.ListedColormap(['white'])
max_idx = np.argmax(prob_change_internal[:, 0, :, xmin: xmax], axis=0)
conf = 1 - np.max(prob_change_internal[:, 0, :, xmin: xmax], axis=0)
cax1 = ax1.matshow(max_idx, cmap=cmap1, vmin=0, vmax=num_bins-1)
cax2 = ax1.matshow(np.ones(conf.shape), cmap=cmap2, vmin=0, vmax=1, alpha=conf)
ticks = np.linspace(start=((num_bins - 1) / num_bins) / 2, stop=num_bins - 1 - ((num_bins - 1) / num_bins) / 2, num=num_bins)
cbar = Fig.colorbar(cax1, ticks=ticks)
cbar.ax.set_yticklabels(bin_labels)
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
