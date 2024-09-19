"""
Script for plotting output from datafiles of probabilistic MEEB simulation.

IRBR 18 September 2024
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
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
    cax = ax.matshow(transition_matrix, cmap='binary')
    tic_locs = np.arange(len(class_labels))
    plt.xticks(tic_locs, class_labels)
    plt.yticks(tic_locs, class_labels)
    plt.ylabel('From Class')
    plt.xlabel('To Class')
    plt.title('Ecogeomorphic State Transitions')
    fig.colorbar(cax, label='Proportion of Net Change in Area From State Transitions')


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
        raise ValueError("plot_most_probable_class: orientation invalid, must use 'vertical' or 'horizontal'")

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


# __________________________________________________________________________________________________________________________________
# LOAD PROBABILISTIC SIM DATA

# Specify filenames
elev_class_probabilities_filename = ["ElevClassProbabilities_16Sep24_13000-21000.npy"]  # To stich together mulitiple probabilistic runs in space, list multiple filenames in spatial order

state_class_probabilities_filename = ["HabitatStateClassProbabilities_16Sep24_13000-21000.npy"]

inundation_class_probabilities_filename = ["InundationClassProbabilities_16Sep24_13000-21000.npy"]

# Load
elev_class_probabilities = np.load("Output/SimData/" + elev_class_probabilities_filename[0])
state_class_probabilities = np.load("Output/SimData/" + state_class_probabilities_filename[0])
inundation_class_probabilities = np.load("Output/SimData/" + inundation_class_probabilities_filename[0])

# Stich multiple sections of probabilistic runs together in space (if applicable)
if len(inundation_class_probabilities_filename) > 0:
    for n in range(1, len(inundation_class_probabilities_filename)):
        elev_class_probabilities_addition = np.load("Output/SimData/" + elev_class_probabilities_filename[n])
        state_class_probabilities_addition = np.load("Output/SimData/" + state_class_probabilities_filename[n])
        inundation_class_probabilities_addition = np.load("Output/SimData/" + inundation_class_probabilities_filename[n])
        elev_class_probabilities = np.concatenate((elev_class_probabilities, elev_class_probabilities_addition), axis=2)
        state_class_probabilities = np.concatenate((state_class_probabilities, state_class_probabilities_addition), axis=2)
        inundation_class_probabilities = np.concatenate((inundation_class_probabilities, inundation_class_probabilities_addition), axis=1)


# __________________________________________________________________________________________________________________________________
# CLASS SPECIFICATIONS

# Elevation
elev_class_labels = ['< -0.5', '-0.5 - -0.1', '-0.1 - 0.1', '0.1 - 0.5', '> 0.5']
elev_class_cmap = colors.ListedColormap(['#ca0020', '#f4a582', '#f7f7f7', '#92c5de', '#0571b0'])

# Habitat State
state_class_labels = ['Subaqueous', 'Beach-Steep', 'Beach-Shallow', 'Dune', 'Washover', 'Interior']
state_class_cmap = colors.ListedColormap(['blue', 'gold', 'tan', 'saddlebrown', 'red', 'green'])

# Class Probability
cmap_class_prob = plt.get_cmap('cividis', 5)

# Confidence
cmap_conf = plt.get_cmap('BuPu', 4)  # 4 discrete colors

# __________________________________________________________________________________________________________________________________
# SIM SPECIFICATIONS

name = '13000-21000, 16Sep24'  # Name of simulation suite
sim_duration = 32  # [yr] Note: For probabilistic projections, use a duration that is divisible by the save_frequency
save_frequency = 1  # [yr] Time step for probability calculations
cellsize = 2  # [m]
plot_xmin = 0  # [m] Cross-shore min coordinate for plotting
plot_xmax = plot_xmin + 600  # [m] Cross-shore max coordinate for plotting

plot = True
animate = False

longshore = elev_class_probabilities.shape[2]
crossshore = elev_class_probabilities.shape[3]
num_saves = int(np.floor(sim_duration/save_frequency)) + 1

# __________________________________________________________________________________________________________________________________
# PLOT RESULTS

if plot:
    plot_class_maps(elev_class_probabilities, elev_class_labels, it=-1)
    plot_class_maps(state_class_probabilities, state_class_labels, it=-1)
    plot_most_probable_class(elev_class_probabilities, elev_class_cmap, elev_class_labels, it=-1, orientation='horizontal')
    plot_most_probable_class(state_class_probabilities, state_class_cmap, state_class_labels, it=-1, orientation='horizontal')
    plot_class_probability(inundation_class_probabilities, it=-1, class_label='Inundation', orientation='horizontal')
    plot_class_area_change_over_time(state_class_probabilities, state_class_labels)
    plot_most_likely_transition_maps(state_class_probabilities)
    plot_transitions_area_matrix(state_class_probabilities, state_class_labels)
if animate:
    bins_animation(elev_class_probabilities, elev_class_labels)
    bins_animation(state_class_probabilities, state_class_labels)
    most_likely_animation(elev_class_probabilities, elev_class_cmap, elev_class_labels, orientation='horizontal')
    most_likely_animation(state_class_probabilities, state_class_cmap, state_class_labels, orientation='horizontal')
    class_probability_animation(inundation_class_probabilities, orientation='horizontal')
plt.show()

