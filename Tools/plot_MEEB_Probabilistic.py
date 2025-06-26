"""
Script for plotting output from datafiles of probabilistic MEEB simulation.

IRBR 19 May 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import os
from matplotlib import colors


# __________________________________________________________________________________________________________________________________
# PLOTTING FUNCTIONS


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


def plot_class_frequency(class_probabilities, it, start_step, class_label, orientation='vertical'):
    """Plots the frequency of a class (e.g., overwash inundation) across the domain at a particular time step.

    Parameters
    ----------
    class_probabilities : ndarray
        Probabilities of a class over space and time.
    it : int
        Iteration to draw probabilities from.
    start_step : int
        Iteration to begin count from.
    class_label : str
        Name/description of class for labeling colorbar.
    orientation : str
        ['vertical' or 'horizontal'] Orientation to plot domain: vertical will plot ocean along left edge of domain, 'horizontal' along bottom.
    """

    inun_prob = class_probabilities[it, :, :] - class_probabilities[start_step, :, :]

    if orientation == 'vertical':
        Fig = plt.figure(figsize=(8, 10))
        ax1 = Fig.add_subplot(111)
    elif orientation == 'horizontal':
        inun_prob = np.rot90(inun_prob, k=1)
        Fig = plt.figure(figsize=(14, 10))
        ax1 = Fig.add_subplot(111)
    else:
        raise ValueError("plot_most_probable_class: orientation invalid, must use 'vertical' or 'horizontal'")

    inun_prob[inun_prob < 0.01] = 0
    cmap_class_freq = plt.get_cmap('inferno', int(np.max(inun_prob)))

    im_ratio = inun_prob.shape[0] / inun_prob.shape[1]
    cax1 = ax1.matshow(inun_prob, cmap=cmap_class_freq, norm=colors.LogNorm())
    cb_label = 'Number of ' + class_label
    Fig.colorbar(cax1, label=cb_label, fraction=0.046 * im_ratio)
    plt.xlabel('Meters Alongshore')
    plt.ylabel('Meters Cross-Shore')

    plt.tight_layout()


def plot_most_likely_ocean_shoreline(class_probabilities, start_step):

    shorelines = np.zeros([num_saves, int(class_probabilities.shape[2] * cellsize)])
    Fig = plt.figure()
    plt.tight_layout()

    # Shorelines over time
    ax_1 = Fig.add_subplot(211)
    plt.ylabel('Meters Cross-Shore')

    color = iter(plt.cm.viridis(np.linspace(0, 1, num_saves)))

    for it in range(start_step, num_saves):
        most_likely_state_it = np.argmax(class_probabilities[:, it, :, plot_xmin: plot_xmax], axis=0)  # Bin of most probable outcome at t=it
        shoreline_it = np.argmax(most_likely_state_it > 0, axis=1) * cellsize  # Find relative ocean shoreline positions and convert y-axis to meters
        shoreline_it = np.repeat(shoreline_it, cellsize)  # Convert x-axis to meters
        shorelines[it, :] = shoreline_it
        if it == start_step:
            ax_1.plot(shoreline_it, c=next(color), label='Start')
        if it == num_saves - 1:
            ax_1.plot(shoreline_it, c=next(color), label='End')
        else:
            ax_1.plot(shoreline_it, c=next(color), label='_')
    plt.legend()

    # Short and long-term shoreline change
    ax_2 = Fig.add_subplot(212)
    plt.xlabel('Meters Alongshore')
    plt.ylabel('Shoreline Change Rate [m/yr]')
    dur = shorelines[start_step:, :].shape[0] * save_frequency  # [yr]
    long_term_shoreline_change_rate = (shorelines[-1, :] - shorelines[start_step, :]) / dur
    short_term_shoreline_change_rate = (shorelines[start_step + int(10 / save_frequency), :] - shorelines[start_step, :]) / 10  # First decade
    ax_2.plot(np.arange(shorelines.shape[1]), np.zeros([shorelines.shape[1]]), 'k--', alpha=0.3, label='_Zero Line')
    ax_2.plot(short_term_shoreline_change_rate, 'cornflowerblue', label='Short-term Shoreline Change (First Decade)')
    ax_2.plot(long_term_shoreline_change_rate, 'darkred', label='Long-term Shoreline Change (Full Simulation Duration)')
    plt.legend()


def plot_most_likely_backbarrier_shoreline(class_probabilities, start_step):

    shorelines = np.zeros([num_saves, int(class_probabilities.shape[2] * cellsize)])
    Fig = plt.figure()
    plt.tight_layout()

    # Shorelines over time
    ax_1 = Fig.add_subplot(211)
    plt.ylabel('Meters Cross-Shore')

    color = iter(plt.cm.viridis(np.linspace(0, 1, num_saves)))

    for it in range(start_step, num_saves):
        most_likely_state_it = np.argmax(class_probabilities[:, it, :, plot_xmin: plot_xmax], axis=0)  # Bin of most probable outcome at t=it
        # shoreline_it = (most_likely_state_it.shape[1] - np.argmax(np.fliplr(most_likely_state_it) > 0, axis=1) - 1) * cellsize  # Find relative ocean shoreline positions and convert y-axis to meters

        # ----
        water = most_likely_state_it == 0
        shoreline_it = np.zeros(most_likely_state_it.shape[0], dtype=np.int32)

        # Finds the first continous section of N subaqeuous cells landward of the ocean shoreline, and takes the first cell of that section as the back-barrier shoreline
        N = 25  # [cells] Threshold number of cells for subaqueous section; assumes any subaqeuous cells < N is interior pond
        for ls in range(most_likely_state_it.shape[0]):
            x_s = np.argwhere(water[ls, :] < 1)[0][0]
            bb_water = np.argwhere(water[ls, x_s:] > 0)
            x_bb = most_likely_state_it.shape[1] - 1
            if len(bb_water) > 0:
                for q in range(len(bb_water)):
                    x_bb_temp = bb_water[q][0] + x_s - 1

                    if x_bb_temp + N <= most_likely_state_it.shape[1]:
                        if np.all(water[ls, x_bb_temp: x_bb_temp + N] > 0):
                            x_bb = x_bb_temp
                            break
                    else:
                        if np.all(water[ls, x_bb_temp: most_likely_state_it.shape[1]] > 0):
                            x_bb = x_bb_temp
                            break

            shoreline_it[ls] = np.int32(x_bb * cellsize)
        # ----

        shoreline_it = np.repeat(shoreline_it, cellsize)  # Convert x-axis to meters
        shorelines[it, :] = shoreline_it
        if it == start_step:
            ax_1.plot(shoreline_it, c=next(color), label='Start')
        if it == num_saves - 1:
            ax_1.plot(shoreline_it, c=next(color), label='End')
        else:
            ax_1.plot(shoreline_it, c=next(color), label='_')
    plt.legend()

    # Short and long-term shoreline change
    ax_2 = Fig.add_subplot(212)
    plt.xlabel('Meters Alongshore')
    plt.ylabel('Backbarrier Shoreline Change Rate [m/yr]')
    dur = shorelines[start_step:, :].shape[0] * save_frequency  # [yr]
    long_term_shoreline_change_rate = (shorelines[-1, :] - shorelines[start_step, :]) / dur
    short_term_shoreline_change_rate = (shorelines[start_step + int(10 / save_frequency), :] - shorelines[start_step, :]) / 10  # First decade
    ax_2.plot(np.arange(shorelines.shape[1]), np.zeros([shorelines.shape[1]]), 'k--', alpha=0.3, label='_Zero Line')
    ax_2.plot(short_term_shoreline_change_rate, 'cornflowerblue', label='Short-term Shoreline Change (First Decade)')
    ax_2.plot(long_term_shoreline_change_rate, 'darkred', label='Long-term Shoreline Change (Full Simulation Duration)')
    plt.legend()


def plot_most_likely_barrier_width(class_probabilities, start_step):

    widths = np.zeros([num_saves, int(class_probabilities.shape[2] * cellsize)])
    Fig = plt.figure()
    plt.tight_layout()

    # Width over time
    ax_1 = Fig.add_subplot(211)
    plt.ylabel('Barrier Width [m]')

    color = iter(plt.cm.viridis(np.linspace(0, 1, num_saves)))

    for it in range(start_step, num_saves):
        most_likely_state_it = np.argmax(class_probabilities[:, it, :, plot_xmin: plot_xmax], axis=0)  # Bin of most probable outcome at t=it

        # Back-barrier Shoreline
        water = most_likely_state_it == 0
        bbshoreline_it = np.zeros(most_likely_state_it.shape[0], dtype=np.int32)
        N = 25  # [cells] Threshold number of cells for subaqueous section; assumes any subaqeuous cells < N is interior pond
        for ls in range(most_likely_state_it.shape[0]):
            x_s = np.argwhere(water[ls, :] < 1)[0][0]
            bb_water = np.argwhere(water[ls, x_s:] > 0)
            x_bb = most_likely_state_it.shape[1] - 1
            if len(bb_water) > 0:
                for q in range(len(bb_water)):
                    x_bb_temp = bb_water[q][0] + x_s - 1

                    if x_bb_temp + N <= most_likely_state_it.shape[1]:
                        if np.all(water[ls, x_bb_temp: x_bb_temp + N] > 0):
                            x_bb = x_bb_temp
                            break
                    else:
                        if np.all(water[ls, x_bb_temp: most_likely_state_it.shape[1]] > 0):
                            x_bb = x_bb_temp
                            break

            bbshoreline_it[ls] = np.int32(x_bb * cellsize)

        # Ocean Shoreline
        oshoreline_it = np.argmax(most_likely_state_it > 0, axis=1) * cellsize  # Find relative ocean shoreline positions and convert y-axis to meters

        # Barrier Width
        oshoreline_it = np.repeat(oshoreline_it, cellsize)  # Convert x-axis to meters
        bbshoreline_it = np.repeat(bbshoreline_it, cellsize)  # Convert x-axis to meters
        barrier_width = bbshoreline_it - oshoreline_it
        widths[it, :] = barrier_width

        if it == start_step:
            ax_1.plot(barrier_width, c=next(color), label='Start')
        if it == num_saves - 1:
            ax_1.plot(barrier_width, c=next(color), label='End')
        else:
            ax_1.plot(barrier_width, c=next(color), label='_')
    plt.legend()

    # Short and long-term shoreline change
    ax_2 = Fig.add_subplot(212)
    plt.xlabel('Meters Alongshore')
    plt.ylabel('Barrier Width Change [m]')
    long_term_width_change = (widths[-1, :] - widths[start_step, :])
    short_term_width_change = (widths[start_step + int(10 / save_frequency), :] - widths[start_step, :])  # First decade
    ax_2.plot(np.arange(widths.shape[1]), np.zeros([widths.shape[1]]), 'k--', alpha=0.3, label='_Zero Line')
    ax_2.plot(short_term_width_change, 'cornflowerblue', label='Short-term Barrier Width Change (First Decade)')
    ax_2.plot(long_term_width_change, 'darkred', label='Long-term Barrier Width Change (Full Simulation Duration)')
    plt.legend()


def plot_beach_and_barrier_width_change_box(class_probabilities, start_step):

    barrier_widths = np.zeros([num_saves, int(class_probabilities.shape[2])])
    beach_widths = np.zeros([num_saves, int(class_probabilities.shape[2])])

    for it in range(start_step, num_saves):
        most_likely_state_it = np.argmax(class_probabilities[:, it, :, plot_xmin: plot_xmax], axis=0)  # Bin of most probable outcome at t=it

        # Back-barrier Shoreline
        water = most_likely_state_it == 0
        bbshoreline_it = np.zeros(most_likely_state_it.shape[0], dtype=np.int32)
        N = 25  # [cells] Threshold number of cells for subaqueous section; assumes any subaqeuous cells < N is interior pond
        for ls in range(most_likely_state_it.shape[0]):
            x_s = np.argwhere(water[ls, :] < 1)[0][0]
            bb_water = np.argwhere(water[ls, x_s:] > 0)
            x_bb = most_likely_state_it.shape[1] - 1
            if len(bb_water) > 0:
                for q in range(len(bb_water)):
                    x_bb_temp = bb_water[q][0] + x_s - 1

                    if x_bb_temp + N <= most_likely_state_it.shape[1]:
                        if np.all(water[ls, x_bb_temp: x_bb_temp + N] > 0):
                            x_bb = x_bb_temp
                            break
                    else:
                        if np.all(water[ls, x_bb_temp: most_likely_state_it.shape[1]] > 0):
                            x_bb = x_bb_temp
                            break

            bbshoreline_it[ls] = np.int32(x_bb * cellsize)

            # Beach Width
            beach_widths[it, ls] = np.sum(np.logical_or(most_likely_state_it[ls, :] == 1, most_likely_state_it[ls, :] == 2)) * cellsize

        # Ocean Shoreline
        oshoreline_it = np.argmax(most_likely_state_it > 0, axis=1) * cellsize  # Find relative ocean shoreline positions and convert y-axis to meters

        # Barrier Width
        barrier_widths[it, :] = bbshoreline_it - oshoreline_it

    # Change - Start to End
    change_in_barrier_width = barrier_widths[-1, :] - barrier_widths[start_step, :]
    change_in_beach_width = beach_widths[-1, :] - beach_widths[start_step, :]

    Fig = plt.figure()
    plt.tight_layout()
    plt.boxplot([change_in_barrier_width, change_in_beach_width], labels=['Change in Barrier Width [m]', 'Change in Beach Width[m]'])


def plot_dune_alongshore_extent(class_probabilities, start_step):

    most_likely_state_start = np.argmax(class_probabilities[:, start_step, :, plot_xmin: plot_xmax], axis=0)  # Bin of most probable outcome at t=start
    most_likely_state_end = np.argmax(class_probabilities[:, -1, :, plot_xmin: plot_xmax], axis=0)  # Bin of most probable outcome at t=end

    dune_start = np.zeros([1, most_likely_state_start.shape[0]])
    dune_end = np.zeros([1, most_likely_state_start.shape[0]])

    dune_width_start = np.zeros(most_likely_state_start.shape[0])
    dune_width_end = np.zeros(most_likely_state_start.shape[0])

    for ls in range(most_likely_state_end.shape[0]):

        dune_width_start[ls] = np.sum(most_likely_state_start[ls, :] == 3)
        if dune_width_start[ls] > 0:
            dune_start[:, ls] = 1

        dune_width_end[ls] = np.sum(most_likely_state_end[ls, :] == 3)
        if dune_width_end[ls] > 0:
            dune_end[:, ls] = 1

    dune_width_change = (dune_width_end - dune_width_start) * cellsize
    dune_change = dune_end - dune_start

    Fig = plt.figure()
    ax1 = Fig.add_subplot(211)
    ax1.plot(dune_width_change)
    plt.xlabel('Alongshore Extent')
    plt.ylabel('Change in Dune Cross-shore Width (m)')

    ax2 = Fig.add_subplot(212)
    # ax2.plot(dune_change)
    ax2.matshow(np.repeat(dune_change, 500, axis=0), cmap='bwr_r')
    plt.xlabel('Alongshore Extent')
    plt.ylabel('Change in Dune Presence (+Gain, -Loss)')

    plt.tight_layout()

    print('Change in Dune Area:', np.sum(dune_width_change))
    print('Change in Dune Area of New Dune Locations', np.sum(dune_width_change[dune_change[0, :] > 0]))
    print('Change in Alongshore Dune Coverage:', np.sum(dune_change))


def plot_overwash_intensity_over_time(inun_class_probabilities, sta_class_probabilities, dy):
    """"""

    dy = int(dy / cellsize)
    n_dy = int(longshore / dy)
    ow_count = np.zeros([num_saves, n_dy])

    for it in range(num_saves):
        ow_domain = inun_class_probabilities[it, :, :]

        state_num = np.argmax(sta_class_probabilities[:, it, :, :], axis=0)  # Bin of most probable outcome
        interior = np.logical_or(state_num == 4, state_num == 5)
        ow_domain[~interior] = 0
        ow_domain = np.rot90(ow_domain, k=1)

        for nn in range(n_dy):
            ow_count[it, nn] = np.sum(ow_domain[:, nn * dy: nn * dy + dy])

    total_overwash_time = np.sum(ow_count.copy(), axis=1)
    total_overwash_alongshore = np.sum(ow_count.copy(), axis=0)
    ow_count = np.repeat(ow_count, int(n_dy / 4 / num_saves), axis=0)

    cmap_ow_intensity = plt.get_cmap('binary', int(np.max(ow_count)))
    fig, ax = plt.subplots()
    im_ratio = ow_count.shape[0] / ow_count.shape[1]
    cax = ax.matshow(ow_count, cmap=cmap_ow_intensity)

    t_loc = np.arange(0, num_saves, 2) * int(n_dy / 4 / num_saves) + int(n_dy / 4 / num_saves / 2)
    t_lab = np.arange(0, num_saves, 2) * save_frequency
    plt.yticks(t_loc, t_lab)

    fig.colorbar(cax, fraction=0.046 * im_ratio, label='Overwash Intensity')
    plt.xlabel('Meters Alongshore')
    plt.ylabel('Forecast Year')
    plt.tight_layout()

    # Total overwash through time (whole domain)
    plt.figure()
    xx = np.arange(0, num_saves) * save_frequency
    plt.plot(xx, total_overwash_time)
    plt.xlabel('Forecast Year')
    plt.ylabel('Overwash Intensity (Whole Domain)')

    # Cumulative overwash alongshore
    plt.figure()
    plt.plot(total_overwash_alongshore)
    plt.xlabel('Alongshore')
    plt.ylabel('Overwash Intensity')


def plot_class_area_change_over_time(class_probabilities, class_labels, start_step=1):

    num_classes = class_probabilities.shape[0]
    most_likely_class = np.argmax(class_probabilities[:, :, :, :], axis=0)

    plt.figure()
    xx = np.arange(start_step, num_saves) * save_frequency

    for c in range(num_classes):
        class_change_TS = np.zeros([num_saves - start_step])  # Initialize
        class_0 = np.sum(most_likely_class[start_step, :, plot_xmin: plot_xmax] == c)
        for ts in range(start_step, num_saves):
            class_change_TS[ts - start_step] = (np.sum(most_likely_class[ts, :, plot_xmin: plot_xmax] == c) - class_0) * cellsize ** 2 / 1e6  # [km2]

        plt.plot(xx, class_change_TS, c=state_class_colors[c])

    plt.legend(class_labels)
    plt.ylabel('Change in Area [km2]')
    plt.xlabel('Forecast Year')


def plot_probabilistic_class_area_change_over_time(class_probabilities, class_labels, norm='total', start_step=1):

    num_classes = class_probabilities.shape[0]

    plt.figure()
    xx = np.arange(start_step, num_saves) * save_frequency
    y_lab = 'Change in Area [km2]'

    for c in range(num_classes):
        class_change_TS = np.zeros([num_saves - start_step])  # Initialize
        class_0 = np.sum(class_probabilities[c, start_step, :, :])

        if norm == 'total':
            for ts in range(start_step, num_saves):
                class_change_TS[ts - start_step] = (np.sum(class_probabilities[c, ts, :, :]) - class_0) * cellsize ** 2 / 1e6  # [km2]
        elif norm == 'class':
            for ts in range(start_step, num_saves):
                class_change_TS[ts - start_step] = (np.sum(class_probabilities[c, ts, :, :]) - class_0) / class_0
                y_lab = 'Area Proportional to Initial'
        else:
            raise ValueError("Invalid entry in norm field: must use 'class' or 'total'")
        plt.plot(xx, class_change_TS, c=state_class_colors[c])

    plt.legend(class_labels)
    plt.ylabel(y_lab)
    plt.xlabel('Forecast Year')


def plot_class_area_loss_gain_over_time(class_probabilities, class_labels, start_step=1):

    num_classes = class_probabilities.shape[0]
    most_likely_class = np.argmax(class_probabilities[:, :, :, :], axis=0)

    plt.figure()
    xx = np.arange(start_step, num_saves) * save_frequency

    for c in range(num_classes):
        class_loss_TS = np.zeros([num_saves - start_step])  # Initialize
        class_gain_TS = np.zeros([num_saves - start_step])
        for ts in range(max(1, start_step), num_saves):
            class_loss_TS[ts - start_step] = np.sum(np.logical_and(most_likely_class[ts, :, plot_xmin: plot_xmax] != c,  most_likely_class[ts - 1, :, plot_xmin: plot_xmax] == c)) * cellsize ** 2 / 1e6  # [km2]
            class_gain_TS[ts - start_step] = np.sum(np.logical_and(most_likely_class[ts, :, plot_xmin: plot_xmax] == c, most_likely_class[ts - 1, :, plot_xmin: plot_xmax] != c)) * cellsize ** 2 / 1e6  # [km2]

        plt.plot(xx, class_loss_TS, c=state_class_colors[c], linestyle='--', label=(class_labels[c] + ' Loss'))
        plt.plot(xx, class_gain_TS, c=state_class_colors[c], label=(class_labels[c] + ' Gain'))

    plt.legend()
    plt.ylabel('Area Gained and Lost [km2]')
    plt.xlabel('Forecast Year')

    plt.figure()
    for c in range(num_classes):
        class_loss_TS = np.zeros([num_saves - start_step])  # Initialize
        class_gain_TS = np.zeros([num_saves - start_step])
        initial = np.sum(most_likely_class[start_step, :, plot_xmin: plot_xmax] == c) * cellsize ** 2 / 1e6  # [km2]
        for ts in range(max(1, start_step), num_saves):
            class_loss_TS[ts - start_step] = np.sum(np.logical_and(most_likely_class[ts, :, plot_xmin: plot_xmax] != c,  most_likely_class[ts - 1, :, plot_xmin: plot_xmax] == c)) * cellsize ** 2 / 1e6  # [km2]
            class_gain_TS[ts - start_step] = np.sum(np.logical_and(most_likely_class[ts, :, plot_xmin: plot_xmax] == c, most_likely_class[ts - 1, :, plot_xmin: plot_xmax] != c)) * cellsize ** 2 / 1e6  # [km2]

        class_gain_TS /= initial
        class_loss_TS /= initial

        plt.plot(xx, class_loss_TS, c=state_class_colors[c], linestyle='--', label=(class_labels[c] + ' Loss'))
        plt.plot(xx, class_gain_TS, c=state_class_colors[c], label=(class_labels[c] + ' Gain'))

    plt.legend()
    plt.ylabel('Area Gained and Lost, Proportional to Initial 2024')
    plt.xlabel('Forecast Year')

    plt.figure()
    for c in range(num_classes):
        class_loss_TS = np.zeros([num_saves - start_step])  # Initialize
        class_gain_TS = np.zeros([num_saves - start_step])
        for ts in range(max(1, start_step), num_saves):
            class_loss_TS[ts - start_step] = np.sum(np.logical_and(most_likely_class[ts, :, plot_xmin: plot_xmax] != c,  most_likely_class[ts - 1, :, plot_xmin: plot_xmax] == c)) * cellsize ** 2 / 1e6  # [km2]
            class_gain_TS[ts - start_step] = np.sum(np.logical_and(most_likely_class[ts, :, plot_xmin: plot_xmax] == c, most_likely_class[ts - 1, :, plot_xmin: plot_xmax] != c)) * cellsize ** 2 / 1e6  # [km2]

        absolute_changed = class_gain_TS + class_loss_TS

        plt.plot(xx, absolute_changed, c=state_class_colors[c], label=(class_labels[c]))

    plt.legend()
    plt.ylabel('Absolute Area Changed (Gained + Lost) [km2]')
    plt.xlabel('Forecast Year')

    plt.figure()
    for c in range(num_classes):
        class_loss_TS = np.zeros([num_saves - start_step])  # Initialize
        class_gain_TS = np.zeros([num_saves - start_step])
        initial = np.sum(most_likely_class[start_step, :, plot_xmin: plot_xmax] == c) * cellsize ** 2 / 1e6  # [km2]
        for ts in range(max(1, start_step), num_saves):
            class_loss_TS[ts - start_step] = np.sum(np.logical_and(most_likely_class[ts, :, plot_xmin: plot_xmax] != c,  most_likely_class[ts - 1, :, plot_xmin: plot_xmax] == c)) * cellsize ** 2 / 1e6  # [km2]
            class_gain_TS[ts - start_step] = np.sum(np.logical_and(most_likely_class[ts, :, plot_xmin: plot_xmax] == c, most_likely_class[ts - 1, :, plot_xmin: plot_xmax] != c)) * cellsize ** 2 / 1e6  # [km2]

        class_gain_TS /= initial
        class_loss_TS /= initial

        absolute_changed = class_gain_TS + class_loss_TS

        plt.plot(xx, absolute_changed, c=state_class_colors[c], label=(class_labels[c]))

    plt.legend()
    plt.ylabel('Absolute Area Changed (Gained + Lost), Proportional to Initial 2024')
    plt.xlabel('Forecast Year')


def plot_class_area_loss_gain_over_time_subplots(class_probabilities, class_labels, start_step=1):

    cnum = class_probabilities.shape[0]
    cols = 3
    rows = int(math.ceil(cnum / cols))
    most_likely_class = np.argmax(class_probabilities[:, :, :, :], axis=0)

    fig, ax = plt.subplots(rows, cols)
    ax = ax.flatten()
    xx = np.arange(start_step, num_saves) * save_frequency

    ymax = 0  # Initialize

    for c in range(cnum):
        class_loss_TS = np.zeros([num_saves - start_step])  # Initialize
        class_gain_TS = np.zeros([num_saves - start_step])
        initial = np.sum(most_likely_class[start_step, :, plot_xmin: plot_xmax] == c) * cellsize ** 2 / 1e6  # [km2]
        for ts in range(max(1, start_step), num_saves):
            class_loss_TS[ts - start_step] = np.sum(np.logical_and(most_likely_class[ts, :, plot_xmin: plot_xmax] != c,  most_likely_class[ts - 1, :, plot_xmin: plot_xmax] == c)) * cellsize ** 2 / 1e6  # [km2]
            class_gain_TS[ts - start_step] = np.sum(np.logical_and(most_likely_class[ts, :, plot_xmin: plot_xmax] == c, most_likely_class[ts - 1, :, plot_xmin: plot_xmax] != c)) * cellsize ** 2 / 1e6  # [km2]

        class_gain_TS /= initial
        class_loss_TS /= initial

        ymax = max(ymax, max(np.max(class_gain_TS), np.max(class_loss_TS)))

        ax[c].plot(xx, class_loss_TS, c=state_class_colors[c], linestyle='--', label='Loss')
        ax[c].plot(xx, class_gain_TS, c=state_class_colors[c], label=' Gain')
        ax[c].set_title(class_labels[c])
        ax[c].set_ylabel('Change in Area, Proportional to 2024')
        ax[c].set_xlabel('Year')
        ax[c].legend()

    plt.setp(ax, ylim=(0, ymax + ymax * 0.05))
    plt.tight_layout()


def plot_weighted_area_bar_over_time(class_probabilities, class_cmap, class_labels, start_step=0):

    num_classes = class_probabilities.shape[0]
    norm_class_area = np.ones([num_saves - start_step, 1000]) * num_classes
    total_area = class_probabilities.shape[2] * class_probabilities.shape[3]

    for ts in range(start_step, num_saves):
        cumulative_count = 0
        for c in range(num_classes):
            class_area = np.sum(class_probabilities[c, ts, :, :]) / total_area * 1000

            norm_class_area[ts - start_step, int(cumulative_count): int(cumulative_count + class_area)] = c
            cumulative_count += class_area

    # Plot
    norm_class_area = np.repeat(norm_class_area, int(1000 / num_saves), axis=0)
    fig, ax = plt.subplots()
    im_ratio = norm_class_area.shape[0] / norm_class_area.shape[1]
    cax = ax.matshow(norm_class_area, cmap=class_cmap)

    t_loc = np.arange(0, num_saves - start_step, 2) * int(1000 / num_saves) + int(1000 / num_saves / 2)
    t_lab = np.arange(start_step, num_saves, 2) * save_frequency
    plt.xticks([0, 200, 400, 600, 800, 1000], ['0', '20', '40', '60', '80', '100'])
    plt.yticks(t_loc, t_lab)
    plt.ylabel('Forecast Year')
    plt.xlabel('Percent of Total Area')

    tic = np.linspace(start=((num_classes - 1) / num_classes) / 2, stop=num_classes - 1 - ((num_classes - 1) / num_classes) / 2, num=num_classes)
    mcbar = fig.colorbar(cax, fraction=0.046 * im_ratio, ticks=tic)
    mcbar.ax.set_yticklabels(class_labels)


def plot_transitions_area_matrix(class_probabilities, class_labels, norm='class', start_step=0):

    num_classes = class_probabilities.shape[0]
    transition_matrix = np.zeros([num_classes, num_classes])

    start_class = np.argmax(class_probabilities[:, start_step, :, :], axis=0)  # Bin of most probable outcome
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
        cbar_label = "Proportion of Net Change in Area From State Transitions"
    elif norm == 'from':  # Area normalized based on initial area of from class
        for class_from in range(num_classes):
            for class_to in range(num_classes):
                if class_from == class_to:
                    transition_matrix[class_from, class_to] = 0
                else:
                    transition_matrix[class_from, class_to] = np.sum(np.logical_and(start_class == class_from, end_class == class_to)) / np.sum(start_class == class_from)
        cbar_label = "Proportional Net Change in Area of 'From' Class"
    elif norm == 'to':  # Area normalized based on initial area of from class
        for class_from in range(num_classes):
            for class_to in range(num_classes):
                if class_from == class_to:
                    transition_matrix[class_from, class_to] = 0
                else:
                    transition_matrix[class_from, class_to] = np.sum(np.logical_and(start_class == class_from, end_class == class_to)) / np.sum(start_class == class_to)
        cbar_label = "Proportional Net Change in Area of 'To' Class"
    else:
        raise ValueError("Invalid entry in norm field: must use 'class', 'from', or 'to'")

    mat_max = np.max(transition_matrix)  # 0.5946182772744985#

    fig, ax = plt.subplots()
    cax = ax.matshow(transition_matrix, cmap='binary', vmin=0, vmax=mat_max)
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

    s_to_ticks = ['', 'No Change', 'Beach-Shallow', 'Beach-Steep Beach', 'Dune', 'Unvegetated Interior', 'Interior']
    cmap1 = colors.ListedColormap(['white', 'black', 'tan', 'gold', 'saddlebrown', 'red', 'green'])
    ax_1 = Fig.add_subplot(231)
    cax_1 = ax_1.matshow(subaqueous_to[:, plot_xmin: plot_xmax], cmap=cmap1, vmin=0, vmax=len(s_to_ticks) - 1)
    tic = np.linspace(start=((len(s_to_ticks) - 1) / len(s_to_ticks)) / 2, stop=len(s_to_ticks) - 1 - ((len(s_to_ticks) - 1) / len(s_to_ticks)) / 2, num=len(s_to_ticks))
    mcbar = Fig.colorbar(cax_1, ticks=tic)
    mcbar.ax.set_yticklabels(s_to_ticks)
    plt.title('From Subaqueous to...')

    # Beach-Shallow to..
    beach_to = np.zeros([longshore, crossshore])
    beach_to[np.logical_and(most_likely_ts == 1, prev_most_likely == 1)] = 1
    beach_to[np.logical_and(most_likely_ts == 0, prev_most_likely == 1)] = 2
    beach_to[np.logical_and(most_likely_ts == 2, prev_most_likely == 1)] = 3
    beach_to[np.logical_and(most_likely_ts == 3, prev_most_likely == 1)] = 4
    beach_to[np.logical_and(most_likely_ts == 4, prev_most_likely == 1)] = 5
    beach_to[np.logical_and(most_likely_ts == 5, prev_most_likely == 1)] = 6

    b_to_ticks = ['', 'No Change', 'Subaqueous', 'Beach-Steep', 'Dune', 'Unvegetated Interior', 'Interior']
    cmap2 = colors.ListedColormap(['white', 'black', 'blue', 'gold', 'saddlebrown', 'red', 'green'])
    ax_2 = Fig.add_subplot(232)
    cax_2 = ax_2.matshow(beach_to[:, plot_xmin: plot_xmax], cmap=cmap2, vmin=0, vmax=len(b_to_ticks) - 1)
    tic = np.linspace(start=((len(b_to_ticks) - 1) / len(b_to_ticks)) / 2, stop=len(b_to_ticks) - 1 - ((len(b_to_ticks) - 1) / len(b_to_ticks)) / 2, num=len(b_to_ticks))
    mcbar = Fig.colorbar(cax_2, ticks=tic)
    mcbar.ax.set_yticklabels(b_to_ticks)
    plt.title('From Beach-Shallow to...')

    # Beach-Steep to..
    dune_to = np.zeros([longshore, crossshore])
    dune_to[np.logical_and(most_likely_ts == 2, prev_most_likely == 2)] = 1
    dune_to[np.logical_and(most_likely_ts == 0, prev_most_likely == 2)] = 2
    dune_to[np.logical_and(most_likely_ts == 1, prev_most_likely == 2)] = 3
    dune_to[np.logical_and(most_likely_ts == 3, prev_most_likely == 2)] = 4
    dune_to[np.logical_and(most_likely_ts == 4, prev_most_likely == 2)] = 5
    dune_to[np.logical_and(most_likely_ts == 5, prev_most_likely == 2)] = 6

    d_to_ticks = ['', 'No Change', 'Subaqueous', 'Beach-Shallow', 'Dune', 'Unvegetated Interior', 'Interior']
    cmap3 = colors.ListedColormap(['white', 'black', 'blue', 'tan', 'saddlebrown', 'red', 'green'])
    ax_3 = Fig.add_subplot(233)
    cax_3 = ax_3.matshow(dune_to[:, plot_xmin: plot_xmax], cmap=cmap3, vmin=0, vmax=len(d_to_ticks) - 1)
    tic = np.linspace(start=((len(d_to_ticks) - 1) / len(d_to_ticks)) / 2, stop=len(d_to_ticks) - 1 - ((len(d_to_ticks) - 1) / len(d_to_ticks)) / 2, num=len(d_to_ticks))
    mcbar = Fig.colorbar(cax_3, ticks=tic)
    mcbar.ax.set_yticklabels(d_to_ticks)
    plt.title('From Beach-Steep to...')

    # Dune to..
    washover_to = np.zeros([longshore, crossshore])
    washover_to[np.logical_and(most_likely_ts == 3, prev_most_likely == 3)] = 1
    washover_to[np.logical_and(most_likely_ts == 0, prev_most_likely == 3)] = 2
    washover_to[np.logical_and(most_likely_ts == 1, prev_most_likely == 3)] = 3
    washover_to[np.logical_and(most_likely_ts == 2, prev_most_likely == 3)] = 4
    washover_to[np.logical_and(most_likely_ts == 4, prev_most_likely == 3)] = 5
    washover_to[np.logical_and(most_likely_ts == 5, prev_most_likely == 3)] = 6

    w_to_ticks = ['', 'No Change', 'Subaqueous', 'Beach-Shallow', 'Beach-Steep', 'Unvegetated Interior', 'Interior']
    cmap4 = colors.ListedColormap(['white', 'black', 'blue', 'tan', 'gold', 'red', 'green'])
    ax_4 = Fig.add_subplot(234)
    cax_4 = ax_4.matshow(washover_to[:, plot_xmin: plot_xmax], cmap=cmap4, vmin=0, vmax=len(w_to_ticks) - 1)
    tic = np.linspace(start=((len(w_to_ticks) - 1) / len(w_to_ticks)) / 2, stop=len(w_to_ticks) - 1 - ((len(w_to_ticks) - 1) / len(w_to_ticks)) / 2, num=len(w_to_ticks))
    mcbar = Fig.colorbar(cax_4, ticks=tic)
    mcbar.ax.set_yticklabels(w_to_ticks)
    plt.title('From Dune to...')

    # Unvegetated Interior to..
    washover_to = np.zeros([longshore, crossshore])
    washover_to[np.logical_and(most_likely_ts == 4, prev_most_likely == 4)] = 1
    washover_to[np.logical_and(most_likely_ts == 0, prev_most_likely == 4)] = 2
    washover_to[np.logical_and(most_likely_ts == 1, prev_most_likely == 4)] = 3
    washover_to[np.logical_and(most_likely_ts == 2, prev_most_likely == 4)] = 4
    washover_to[np.logical_and(most_likely_ts == 3, prev_most_likely == 4)] = 5
    washover_to[np.logical_and(most_likely_ts == 5, prev_most_likely == 4)] = 6

    w_to_ticks = ['', 'No Change', 'Subaqueous', 'Beach-Shallow', 'Beach-Steep', 'Dune', 'Interior']
    cmap4 = colors.ListedColormap(['white', 'black', 'blue', 'tan', 'gold', 'saddlebrown', 'green'])
    ax_4 = Fig.add_subplot(235)
    cax_4 = ax_4.matshow(washover_to[:, plot_xmin: plot_xmax], cmap=cmap4, vmin=0, vmax=len(w_to_ticks) - 1)
    tic = np.linspace(start=((len(w_to_ticks) - 1) / len(w_to_ticks)) / 2, stop=len(w_to_ticks) - 1 - ((len(w_to_ticks) - 1) / len(w_to_ticks)) / 2, num=len(w_to_ticks))
    mcbar = Fig.colorbar(cax_4, ticks=tic)
    mcbar.ax.set_yticklabels(w_to_ticks)
    plt.title('From Unvegetated Interior to...')

    # Interior to..
    interior_to = np.zeros([longshore, crossshore])
    interior_to[np.logical_and(most_likely_ts == 5, prev_most_likely == 5)] = 1
    interior_to[np.logical_and(most_likely_ts == 0, prev_most_likely == 5)] = 2
    interior_to[np.logical_and(most_likely_ts == 1, prev_most_likely == 5)] = 3
    interior_to[np.logical_and(most_likely_ts == 2, prev_most_likely == 5)] = 4
    interior_to[np.logical_and(most_likely_ts == 3, prev_most_likely == 5)] = 5
    interior_to[np.logical_and(most_likely_ts == 4, prev_most_likely == 5)] = 6

    i_to_ticks = ['', 'No Change', 'Subaqueous', 'Beach-Shallow', 'Beach-Steep', 'Dune', 'Unvegetated Interior']
    cmap5 = colors.ListedColormap(['white', 'black', 'blue', 'tan', 'gold', 'saddlebrown', 'red'])
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


def plot_transitions_time_matrix(class_probabilities, start_step=1):

    num_classes = class_probabilities.shape[0]
    transition_domain = np.zeros([num_saves, class_probabilities.shape[2], class_probabilities.shape[3]])

    for ts in range(1, num_saves):

        start_class = np.argmax(class_probabilities[:, ts - 1, :, :], axis=0)  # Bin of most probable outcome
        end_class = np.argmax(class_probabilities[:, ts, :, :], axis=0)  # Bin of most probable outcome

        tnum = 1
        for class_from in range(num_classes):
            for class_to in range(num_classes):
                if class_to != class_from:
                    temp = np.logical_and(start_class == class_from, end_class == class_to)
                    transition_domain[ts, :, :][temp] = tnum
                tnum += 1

    transitions = transition_domain > 0
    sum_all_transitions_over_time = np.sum(np.sum(transitions, axis=2), axis=1)

    plt.figure()
    xx = np.arange(start_step, num_saves) * save_frequency
    transition_nums = [9, 14, 25, 26, 27, 30, 35]
    transition_labels = ['Shallow Beach to Steep Beach',
                         'Steep Beach to Shallow Beach',
                         'Unvegetated Interior to Subaqueous',
                         'Unvegetated Interior to Shallow Beach',
                         'Unvegetated Interior to Steep Beach',
                         'Unvegetated Interior to Vegetated Interior',
                         'Vegetated Interior to Unvegetated Interior']

    for tnum in transition_nums:
        transitions_count = np.zeros(xx.shape)
        for ts in range(start_step, num_saves):
            transitions_count[ts - start_step] = np.sum(transition_domain[ts, :, :] == tnum) * cellsize ** 2 / 1e6  # [km2]

        plt.plot(xx, transitions_count)

    plt.xlabel('Forecast Year')
    plt.ylabel('Transitioned Area [km2]')
    plt.legend(transition_labels)


def plot_transitions_intensity_alongshore(class_probabilities, dy=100, start_step=1, end_step=-1):

    dy = int(dy / cellsize)
    n_dy = int(longshore / dy)

    num_classes = class_probabilities.shape[0]
    transition_domain = np.zeros([class_probabilities.shape[2], class_probabilities.shape[3]])

    start_class = np.argmax(class_probabilities[:, start_step, :, :], axis=0)  # Bin of most probable outcome
    end_class = np.argmax(class_probabilities[:, end_step, :, :], axis=0)  # Bin of most probable outcome

    tnum = 1
    for class_from in range(num_classes):
        for class_to in range(num_classes):
            if class_to != class_from:
                temp = np.logical_and(start_class == class_from, end_class == class_to)
                transition_domain[:, :][temp] = tnum
            tnum += 1

    transition_nums = [35, 25, 31, 13, 26, 21, 34, 28]
    transition_nums_from = [5, 4, 5, 2, 4, 3, 5, 4]

    temp = np.zeros([n_dy, len(transition_nums)])
    for num in range(len(transition_nums)):
        summy = np.sum(transition_domain == transition_nums[num], axis=1, dtype=np.float32)
        summy_base = np.sum(start_class == transition_nums_from[num], axis=1, dtype=np.float32)

        for nn in range(n_dy):
            summy_dy = np.sum(summy[nn * dy: nn * dy + dy])
            summy_base_dy = np.sum(summy_base[nn * dy: nn * dy + dy])
            if summy_base_dy > 0:
                summy_prop = summy_dy / summy_base_dy  # Proportional to initial area
            else:
                summy_prop = 0  # Prevent divide by zero errors
            temp[nn, num] = summy_prop

    temp2 = np.fliplr(np.rot90(np.repeat(temp, int(n_dy / len(transition_nums)), axis=1), 3))
    plt.matshow(temp2, cmap='Purples')
    plt.colorbar()
    plt.xlabel('Alongshore Extent')


def plot_transitions_intensity_alongshore_2(class_probabilities, dy=100, start_step=1, end_step=-1):

    dy = int(dy / cellsize)
    n_dy = int(longshore / dy)

    num_classes = class_probabilities.shape[0]
    transition_domain = np.zeros([class_probabilities.shape[2], class_probabilities.shape[3]])

    start_class = np.argmax(class_probabilities[:, start_step, :, :], axis=0)  # Bin of most probable outcome
    end_class = np.argmax(class_probabilities[:, end_step, :, :], axis=0)  # Bin of most probable outcome

    tnum = 1
    for class_from in range(num_classes):
        for class_to in range(num_classes):
            if class_to != class_from:
                temp = np.logical_and(start_class == class_from, end_class == class_to)
                transition_domain[:, :][temp] = tnum
            tnum += 1

    transition_nums = [35, 25, 31, 13, 26, 21, 34, 28]

    temp = np.zeros([n_dy, len(transition_nums)])
    for num in range(len(transition_nums)):
        summy = np.sum(transition_domain == transition_nums[num], axis=1, dtype=np.float32)
        summy_base = np.sum(transition_domain == transition_nums[num], dtype=np.float32)

        for nn in range(n_dy):
            summy_dy = np.sum(summy[nn * dy: nn * dy + dy])
            summy_prop = summy_dy / summy_base  # Proportional to total area transitioned

            temp[nn, num] = summy_prop

    temp2 = np.fliplr(np.rot90(np.repeat(temp, int(n_dy / len(transition_nums)), axis=1), 3))
    plt.matshow(temp2, cmap='bone_r')
    plt.colorbar()
    plt.xlabel('Alongshore Extent')


def plot_transitions_intensity_over_time(class_probabilities, start_step=1, end_step=17):

    num_classes = class_probabilities.shape[0]
    transition_domain = np.zeros([class_probabilities.shape[2], class_probabilities.shape[3]])
    transition_nums = [25, 21, 35, 13, 11, 31, 26, 30]
    transition_nums_from = [4, 3, 5, 2, 1, 5, 4, 4]
    transition_over_time = np.zeros([end_step - start_step, len(transition_nums)])

    for ts in range(start_step, end_step):

        start_class = np.argmax(class_probabilities[:, ts - 1, :, :], axis=0)  # Bin of most probable outcome
        end_class = np.argmax(class_probabilities[:, ts, :, :], axis=0)  # Bin of most probable outcome

        tnum = 1
        for class_from in range(num_classes):
            for class_to in range(num_classes):
                if class_to != class_from:
                    temp = np.logical_and(start_class == class_from, end_class == class_to)
                    transition_domain[:, :][temp] = tnum
                tnum += 1

        for num in range(len(transition_nums)):
            summy = np.sum(transition_domain == transition_nums[num], dtype=np.float32)
            base = max(1, np.sum(start_class == transition_nums_from[num], dtype=np.float32))

            proportional_sum = summy / base  # Proportional to initial area

            transition_over_time[ts - start_step, num] = proportional_sum

    transition_over_time = np.fliplr(np.rot90(np.repeat(transition_over_time, save_frequency, axis=0), 3))
    plt.matshow(transition_over_time, cmap='Purples')
    plt.colorbar()
    plt.xlabel('Simulation Year')


def plot_transition_succession(class_probabilities, transition_num, class_labels, transition_num_label='', start_step=1):

    num_classes = class_probabilities.shape[0]
    transition_domain = np.zeros([num_saves, class_probabilities.shape[2], class_probabilities.shape[3]])

    for ts in range(start_step, num_saves):

        start_class = np.argmax(class_probabilities[:, ts - 1, :, :], axis=0)  # Bin of most probable outcome
        end_class = np.argmax(class_probabilities[:, ts, :, :], axis=0)  # Bin of most probable outcome

        tnum = 1  # No transition cells will be 0
        for class_from in range(num_classes):
            for class_to in range(num_classes):
                if class_to != class_from:
                    temp = np.logical_and(start_class == class_from, end_class == class_to)
                    transition_domain[ts, :, :][temp] = tnum
                tnum += 1

    N = transition_num  # Number of the transition in question
    transition_num_label = transition_num_label + ' (' + str(N) + ')'  # 'Shallow Beach to Dune'

    transition_counts = np.zeros([36])
    for ts in reversed(range(start_step, num_saves)):

        N_transitions = (transition_domain[ts, :, :] == N)  # Find cells where transition N took place at this time step

        for t in reversed(range(start_step, ts)):  # Find preceding transition to transition N, if any
            prev = transition_domain[t, :, :]
            transition_nums, counts = np.unique(prev[N_transitions], return_counts=True)
            N_transitions[np.logical_and(N_transitions > 0, prev > 0)] = False

            for x in range(1, len(transition_nums)):
                transition_counts[int(transition_nums[x])] += counts[x]

    # Normalize
    transition_counts /= np.sum(transition_counts)

    # Plot count of transitions preceding transition N
    transition_counts_matrix = np.roll(transition_counts, -1).reshape(num_classes, num_classes)
    fig, ax = plt.subplots()
    cax = ax.matshow(transition_counts_matrix, cmap='binary', vmin=0, vmax=np.max(transition_counts))
    tic_locs = np.arange(len(class_labels))
    plt.xticks(tic_locs, class_labels)
    plt.yticks(tic_locs, class_labels)
    plt.ylabel('From Class')
    plt.xlabel('To Class')
    plt.title(transition_num_label)
    fig.colorbar(cax, label='Proportion of Preceding State Transitions')


def plot_transect_history_most_likely_class(class_probabilities, lsx, start_step=1):

    class_history = np.zeros([num_saves - start_step, crossshore])

    for cs in range(crossshore):
        for ts in range(num_saves - start_step):

            class_history[ts, cs] = np.argmax(class_probabilities[:, ts + start_step, lsx, cs], axis=0)  # Bin of most probable outcome

    plt.matshow(np.repeat(class_history, 10, axis=0), cmap=state_class_cmap, vmin=0, vmax=5)
    plt.title('Most Likely State Class at X=' + str(lsx * cellsize) + ' (m alongshore)')
    plt.ylabel('Forecast Year')
    plt.xlabel('Cross-Shore Extent')


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
# SIM SPECIFICATIONS

# Classification Scheme - Choose from one or more of three currently available options: elevation, overwash_frequency, state
classification_scheme = ['state', 'elevation', 'overwash_frequency']

name = ''  # Name of simulation suite
sim_duration = 32  # [yr] Note: For probabilistic projections, use a duration that is divisible by the save_frequency
save_frequency = 2  # [yr] Time step for probability calculations
cellsize = 2  # [m]
plot_xmin = 0  # [m] Cross-shore min coordinate for plotting
plot_xmax = plot_xmin + 1500  # [m] Cross-shore max coordinate for plotting
plot_start_step = 3  # [save_frequency] Time step at which to start timeline plots (steps before this can be considered spin-up)

plot = True  # [bool]
animate = False  # [bool]

num_saves = int(np.floor(sim_duration/save_frequency)) + 1

# __________________________________________________________________________________________________________________________________
# LOAD PROBABILISTIC SIM DATA

# Specify filenames located in /Output/SimData
elev_class_probabilities_filename = ["ElevClassProbabilities_2-4Apr25_0-14000_meeb90-91-106.npy",
                                     "ElevClassProbabilities_4Apr25_14000-32000_meeb92-93-107.npy"]

state_class_probabilities_filename = ["HabitatStateClassProbabilities_2-4Apr25_0-14000_meeb90-91-106.npy",
                                      "HabitatStateClassProbabilities_4Apr25_14000-32000_meeb92-93-107.npy"]

overwash_class_probabilities_filename = ["OverwashFrequencyClassProbabilities_2-4Apr25_0-14000_meeb90-91-106.npy",
                                         "OverwashFrequencyClassProbabilities_4Apr25_14000-32000_meeb92-93-107.npy"]

# Specify extents of each file section
xmin = [225, 100]  # [cells] Cross-shore extent minimum of domain, for each output file
xmax = [675, 750]  # [cells] Cross-shore extent maximum of domain, for each output file
y_length = [7000, 9000]  # [cells] Alongshore length of domain, for each output file

# Load and resize array(s) to be equal in cross-shore dimensions
xmin_targ = min(xmin)
xmax_targ = max(xmax)
if 'elevation' in classification_scheme:
    elev_class_probabilities = np.load("Output/SimData/" + elev_class_probabilities_filename[0])
    add_front = np.zeros([elev_class_probabilities.shape[0], elev_class_probabilities.shape[1], y_length[0], (xmin[0] - xmin_targ)])
    add_back = np.zeros([elev_class_probabilities.shape[0], elev_class_probabilities.shape[1], y_length[0], (xmax_targ - xmax[0])])
    add_front[2, :, :, :] = 1
    add_back[2, :, :, :] = 1
    elev_class_probabilities = np.concatenate((add_front, elev_class_probabilities), axis=3)
    elev_class_probabilities = np.concatenate((elev_class_probabilities, add_back), axis=3)

    # Stich multiple sections of probabilistic runs together in space (if applicable)
    if len(elev_class_probabilities_filename) > 0:
        for n in range(1, len(elev_class_probabilities_filename)):
            elev_class_probabilities_addition = np.load("Output/SimData/" + elev_class_probabilities_filename[n])
            add_front = np.zeros([elev_class_probabilities_addition.shape[0], elev_class_probabilities_addition.shape[1], y_length[n], (xmin[n] - xmin_targ)])
            add_back = np.zeros([elev_class_probabilities_addition.shape[0], elev_class_probabilities_addition.shape[1], y_length[n], (xmax_targ - xmax[n])])
            add_front[2, :, :, :] = 1
            add_back[2, :, :, :] = 1
            elev_class_probabilities_addition = np.concatenate((add_front, elev_class_probabilities_addition), axis=3)
            elev_class_probabilities_addition = np.concatenate((elev_class_probabilities_addition, add_back), axis=3)

            elev_class_probabilities = np.concatenate((elev_class_probabilities, elev_class_probabilities_addition), axis=2)

    longshore = elev_class_probabilities.shape[2]
    crossshore = elev_class_probabilities.shape[3]

if 'state' in classification_scheme:
    state_class_probabilities = np.load("Output/SimData/" + state_class_probabilities_filename[0])
    add_front = np.zeros([state_class_probabilities.shape[0], state_class_probabilities.shape[1], y_length[0], (xmin[0] - xmin_targ)])
    add_back = np.zeros([state_class_probabilities.shape[0], state_class_probabilities.shape[1], y_length[0], (xmax_targ - xmax[0])])
    add_front[0, :, :, :] = 1
    add_back[0, :, :, :] = 1
    state_class_probabilities = np.concatenate((add_front, state_class_probabilities), axis=3)
    state_class_probabilities = np.concatenate((state_class_probabilities, add_back), axis=3)

    # Stich multiple sections of probabilistic runs together in space (if applicable)
    if len(state_class_probabilities_filename) > 0:
        for n in range(1, len(state_class_probabilities_filename)):
            state_class_probabilities_addition = np.load("Output/SimData/" + state_class_probabilities_filename[n])
            add_front = np.zeros([state_class_probabilities_addition.shape[0], state_class_probabilities_addition.shape[1], y_length[n], (xmin[n] - xmin_targ)])
            add_back = np.zeros([state_class_probabilities_addition.shape[0], state_class_probabilities_addition.shape[1], y_length[n], (xmax_targ - xmax[n])])
            add_front[0, :, :, :] = 1
            add_back[0, :, :, :] = 1
            state_class_probabilities_addition = np.concatenate((add_front, state_class_probabilities_addition), axis=3)
            state_class_probabilities_addition = np.concatenate((state_class_probabilities_addition, add_back), axis=3)

            state_class_probabilities = np.concatenate((state_class_probabilities, state_class_probabilities_addition), axis=2)

    longshore = state_class_probabilities.shape[2]
    crossshore = state_class_probabilities.shape[3]

if 'overwash_frequency' in classification_scheme:
    overwash_class_probabilities = np.load("Output/SimData/" + overwash_class_probabilities_filename[0])
    add_front = np.zeros([overwash_class_probabilities.shape[0], y_length[0], (xmin[0] - xmin_targ)])
    add_back = np.zeros([overwash_class_probabilities.shape[0], y_length[0], (xmax_targ - xmax[0])])
    overwash_class_probabilities = np.concatenate((add_front, overwash_class_probabilities), axis=2)
    overwash_class_probabilities = np.concatenate((overwash_class_probabilities, add_back), axis=2)

    # Stich multiple sections of probabilistic runs together in space (if applicable)
    if len(overwash_class_probabilities_filename) > 0:
        for n in range(1, len(overwash_class_probabilities_filename)):
            overwash_class_probabilities_addition = np.load("Output/SimData/" + overwash_class_probabilities_filename[n])
            add_front = np.zeros([overwash_class_probabilities_addition.shape[0], y_length[n], (xmin[n] - xmin_targ)])
            add_back = np.zeros([overwash_class_probabilities_addition.shape[0], y_length[n], (xmax_targ - xmax[n])])
            overwash_class_probabilities_addition = np.concatenate((add_front, overwash_class_probabilities_addition), axis=2)
            overwash_class_probabilities_addition = np.concatenate((overwash_class_probabilities_addition, add_back), axis=2)

            overwash_class_probabilities = np.concatenate((overwash_class_probabilities, overwash_class_probabilities_addition), axis=1)

    longshore = overwash_class_probabilities.shape[1]
    crossshore = overwash_class_probabilities.shape[2]

# __________________________________________________________________________________________________________________________________
# CLASS SPECIFICATIONS - Labels & color schemes

# Elevation
elev_class_labels = ['< -0.5', '-0.5 - -0.1', '-0.1 - 0.1', '0.1 - 0.5', '> 0.5']
elev_class_cmap = colors.ListedColormap(['#ca0020', '#f4a582', '#f7f7f7', '#92c5de', '#0571b0'])

# Habitat State
state_class_labels = ['Subaqueous', 'Beach-Shallow', 'Beach-Steep', 'Dune', 'Unveg. Interior', 'Veg. Interior']
state_class_colors = ['blue', 'tan', 'gold', 'saddlebrown', 'red', 'green']
state_class_cmap = colors.ListedColormap(state_class_colors)

# Class Probability
cmap_class_prob = plt.get_cmap('cividis', 5)

# Confidence
cmap_conf = plt.get_cmap('BuPu', 4)  # 4 discrete colors

# __________________________________________________________________________________________________________________________________
# PLOT RESULTS

print()
print(name)


if plot:
    if 'elevation' in classification_scheme:
        # plot_class_maps(elev_class_probabilities, elev_class_labels, it=-1)
        plot_most_probable_class(elev_class_probabilities, elev_class_cmap, elev_class_labels, it=-1, orientation='horizontal')
        if animate:
            bins_animation(elev_class_probabilities, elev_class_labels)
            most_likely_animation(elev_class_probabilities, elev_class_cmap, elev_class_labels, orientation='horizontal')

    if 'state' in classification_scheme:
        # plot_class_maps(state_class_probabilities, state_class_labels, it=-1)
        plot_most_probable_class(state_class_probabilities, state_class_cmap, state_class_labels, it=plot_start_step, orientation='horizontal')
        plot_most_probable_class(state_class_probabilities, state_class_cmap, state_class_labels, it=-1, orientation='horizontal')
        plot_probabilistic_class_area_change_over_time(state_class_probabilities, state_class_labels, norm='class', start_step=plot_start_step)
        plot_weighted_area_bar_over_time(state_class_probabilities, state_class_cmap, state_class_labels, start_step=plot_start_step)
        # plot_most_likely_transition_maps(state_class_probabilities)
        plot_transitions_area_matrix(state_class_probabilities, state_class_labels, norm='total', start_step=plot_start_step)
        plot_class_area_loss_gain_over_time(state_class_probabilities, state_class_labels, start_step=plot_start_step)
        # plot_transitions_time_matrix(state_class_probabilities, start_step=plot_start_step)
        plot_transitions_intensity_alongshore_2(state_class_probabilities, dy=500, start_step=plot_start_step, end_step=-1)
        plot_most_likely_ocean_shoreline(state_class_probabilities, plot_start_step)
        plot_most_likely_backbarrier_shoreline(state_class_probabilities, plot_start_step)
        plot_most_likely_barrier_width(state_class_probabilities, plot_start_step)
        plot_beach_and_barrier_width_change_box(state_class_probabilities, plot_start_step)
        # plot_transition_succession(state_class_probabilities, 14, state_class_labels, 'Steep to Shallow Beach', start_step=plot_start_step)
        # plot_transect_history_most_likely_class(state_class_probabilities, 500, start_step=plot_start_step)
        plot_transitions_intensity_over_time(state_class_probabilities, start_step=plot_start_step, end_step=17)
        plot_class_area_loss_gain_over_time_subplots(state_class_probabilities, state_class_labels, start_step=plot_start_step)
        plot_dune_alongshore_extent(state_class_probabilities, plot_start_step)
        if animate:
            bins_animation(state_class_probabilities, state_class_labels)
            most_likely_animation(state_class_probabilities, state_class_cmap, state_class_labels, orientation='horizontal')

    if 'overwash_frequency' in classification_scheme:
        plot_class_frequency(overwash_class_probabilities, it=-1, start_step=plot_start_step, class_label='Overwash Events', orientation='horizontal')
        if 'state' in classification_scheme:
            plot_overwash_intensity_over_time(overwash_class_probabilities, state_class_probabilities, dy=100)
        if animate:
            class_frequency_animation(overwash_class_probabilities, orientation='horizontal')

plt.show()
