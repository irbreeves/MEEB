"""__________________________________________________________________________________________________________________________________

Model Functions for BEEM

Barrier Explicit Evolution Model

IRB Reeves

Last update: 11 February 2023

__________________________________________________________________________________________________________________________________"""
import matplotlib.pyplot as plt
import numpy as np
import math
import copy
import scipy
from scipy import signal


def shadowzones2(topof, sh, lee, longshore, crossshore, direction):
    """Returns a logical map with all shadowzones identified as ones. Wind from left to right, along the +2 dimension.

    Parameters
    ----------
    topof : ndarray
        Topography map.
    sh : float
        Slab height.
    lee : float
        Shadow angle.
    longshore : int
        Alongshore length of domain
    crossshore : int
        Cross-shore width of domain
    direction : int
        Wind direction (1 east, 2 north, 3, west, 4 south).

    Returns
    -------
    inshade
        Logical map of shaded cells, now with open boundaries
    """

    # steplimit = math.floor(math.tan(lee * math.pi / 180) / sh)  # The maximum step difference allowed given the slabheight and shadowangle
    steplimit = math.tan(lee * math.pi / 180) / sh  # The maximum step difference allowed given the slabheight and shadowangle, not rounded yet

    # range = min(crossshore, max(topof) / steplimit)
    search_range = int(math.floor(np.max(topof) / steplimit))  # Identifies highest elevation and uses that to determine what the largest search distance needs to be

    inshade = np.zeros([longshore, crossshore]).astype(bool)  # Define the zeroed logical map

    for i in range(1, search_range + 1):
        if direction == 1:
            step = topof - np.roll(topof, i, axis=1)  # Shift across columns (2nd dimension; along a row)
            tempinshade = step < -math.floor(steplimit * i)  # Identify cells with too great a stepheight
            tempinshade[:, 0:i] = 0  # Part that is circshifted back into beginning of space is ignored
        elif direction == 2:
            step = topof - np.roll(topof, i, axis=0)  # Shift across columns (2nd dimension; along a row)
            tempinshade = step < -math.floor(steplimit * i)  # Identify cells with too great a stepheight
            tempinshade[0:i, :] = 0  # Part that is circshifted back into beginning of space is ignored
        elif direction == 3:
            step = topof - np.roll(topof, -i, axis=1)  # Shift across columns (2nd dimension; along a row)
            tempinshade = step < -math.floor(steplimit * i)  # Identify cells with too great a stepheight
            tempinshade[:, -1 - i:-1] = 0  # Part that is circshifted back into beginning of space is ignored
        elif direction == 4:
            step = topof - np.roll(topof, -i, axis=0)  # Shift across columns (2nd dimension; along a row)
            tempinshade = step < -math.floor(steplimit * i)  # Identify cells with too great a stepheight
            tempinshade[-1 - i:-1, :] = 0  # Part that is circshifted back into beginning of space is ignored
        inshade = np.bitwise_or(inshade, tempinshade)  # Merge with previous inshade zones

    return inshade


def erosprobs2(vegf, shade, sand, topof, groundw, p_er):
    """ Returns a map with erosion probabilities
    - vegf: map of combined vegetation effectiveness [veg] [0,1]
    - shade: logical map of shadowzones [shadowmap]
    - sand: logical map of sandy cells [sandmap]
    - topof: topography map [topo]
    - groundw: groundwater map [gw]
    - p_er: probability of erosion of base/sandy cell"""

    # r = np.logical_not(shade) * sand * (1 - vegf) * (topof > groundw) * p_er
    r = np.logical_not(shade) * sand * (p_er - vegf) * (topof > groundw)  # IRBR 30Nov22: Fixed to follow Keijsers paper
    r[r < 0] = 0

    return r


def depprobs(vegf, shade, sand, dep0, dep1):
    """Returns a map of deposition probabilities that can then be used to implement transport
    - vegf: map of combined vegetation effectiveness [veg] [0,1]
    - shade: logical map of shadowzones [shadowmap]
    - sand: logical map of sanded sites [sandmap]
    - dep0: deposition probability on base cell [p_dep_base]
    - dep1: deposition probability on sandy cell [p_dep_sand]"""

    # For base cells
    # temp1 = vegf * (1 - dep0) + dep0  # Deposition probabilities on base cells
    temp1 = vegf * dep0 + dep0  # Deposition probabilities on base cells  # IRBR 30Nov22: Fixed to follow Keijsers paper
    temp2 = np.logical_not(sand) * np.logical_not(shade) * temp1  # Apply to base cells outside shadows only

    # For sandy cells
    # temp3 = vegf * (1 - dep1) + dep1  # Deposition probabilities on sandy cells
    temp3 = vegf * dep1 + dep1  # Deposition probabilities on sandy cells  # IRBR 30Nov22: Fixed to follow Keijsers paper
    temp4 = sand * np.logical_not(shade) * temp3  # Apply to sandy cells outside shadows only

    r = temp2 + temp4 + shade  # Combine both types of cells + shadowzones

    return r


def shiftslabs3_open3(erosprobs, deposprobs, hop, contour, longshore, crossshore, direction, RNG):
    """Shifts the sand. Movement is from left to right, along +2 dimension, across columns, along rows. Returns a map of height changes [-,+].
    Open boundaries (modification by Alma), no feeding from the sea side.
    - erosprobs: map of erosion probabilities [erosmap]
    - deposprobs: map of deposition probabilities [deposmap]
    - hop: jumplength
    - direction: wind direction (1 east, 2 north, 3, west, 4 south)"""

    pickedup = RNG.random((longshore, crossshore)) < erosprobs  # True where slab is picked up
    # pickedup[:, -1 - hop: -1] = 0  # Do not pick any slab that are on the ladward boundary -- east only?

    totaldeposit = np.zeros([longshore, crossshore])
    inmotion = copy.deepcopy(pickedup)  # Make copy of original erosion map
    numshifted = 0  # [slabs] Number of shifted cells weighted for transport distance
    transportdist = 0  # Transport distance [slab lengths] or [hop length]
    sum_contour = np.zeros([len(contour)])

    while np.sum(inmotion) > 0:  # While still any slabs moving
        transportdist += 1  # Every time in the loop the slaps are transported one length further to the right
        if direction == 1:
            inmotion = np.roll(inmotion, hop, axis=1)  # Shift the moving slabs one hop length to the right
            transp_contour = np.nansum(inmotion[:, contour.astype(np.int64)], axis=0)  # Account ammount of slabs that are in motion in specific contours
            sum_contour = sum_contour + transp_contour  # Sum the slabs to the others accounted before
            depocells = RNG.random((longshore, crossshore)) < deposprobs  # True where slab should be deposited
            # depocells[:, -1 - hop: -1] = 1  # All slabs are deposited if they are transported over the landward edge
            deposited = inmotion * depocells  # True where a slab is available and should be deposited
            deposited[:, 0: hop] = 0  # Remove all slabs that are transported from the landward side to the seaward side (this changes the periodic boundaries into open ones)
        elif direction == 2:
            inmotion = np.roll(inmotion, hop, axis=0)  # Shift the moving slabs one hop length to the right
            transp_contour = np.nansum(inmotion[:, contour.astype(np.int64)], axis=0)  # Account ammount of slabs that are in motion in specific contours
            sum_contour = sum_contour + transp_contour  # Sum the slabs to the others accounted before
            depocells = RNG.random((longshore, crossshore)) < deposprobs  # True where slab should be deposited
            # depocells[0 : hop, :] = 1  # All slabs are deposited if they are transported over the landward edge
            deposited = inmotion * depocells  # True where a slab is available and should be deposited
            deposited[0: hop, :] = 0  # Remove all slabs that are transported from the landward side to the seaward side (this changes the periodic boundaries into open ones)
        elif direction == 3:
            inmotion = np.roll(inmotion, -hop, axis=1)  # Shift the moving slabs one hop length to the right
            transp_contour = np.nansum(inmotion[:, contour.astype(np.int64)], axis=0)  # Account ammount of slabs that are in motion in specific contours
            sum_contour = sum_contour + transp_contour  # Sum the slabs to the others accounted before
            depocells = RNG.random((longshore, crossshore)) < deposprobs  # True where slab should be deposited
            # depocells[:, -1 - hop: -1] = 1  # All slabs are deposited if they are transported over the landward edge
            deposited = inmotion * depocells  # True where a slab is available and should be deposited
            deposited[:, -1 - hop: -1] = 0  # Remove all slabs that are transported from the landward side to the seaward side (this changes the periodic boundaries into open ones)
        elif direction == 4:
            inmotion = np.roll(inmotion, -hop, axis=0)  # Shift the moving slabs one hop length to the right
            transp_contour = np.nansum(inmotion[contour.astype(np.int64), :], axis=1)  # Account ammount of slabs that are in motion in specific contours
            sum_contour = sum_contour + transp_contour  # Sum the slabs to the others accounted before
            depocells = RNG.random((longshore, crossshore)) < deposprobs  # True where slab should be deposited
            # depocells[0 : hop + 1, :] = 1  # All slabs are deposited if they are transported over the landward edge
            deposited = inmotion * depocells  # True where a slab is available and should be deposited
            deposited[-1 - hop: -1, :] = 0  # Remove all slabs that are transported from the landward side to the seaward side (this changes the periodic boundaries into open ones)

        inmotion[deposited] = False  # Left over in transport after this round of deposition
        numshifted = numshifted + np.sum(deposited) * transportdist  # Number of slabs deposited, weighted for transport distance
        totaldeposit = totaldeposit + deposited  # Total slabs deposited so far

    diff = totaldeposit - pickedup  # deposition - erosion

    return diff, numshifted, sum_contour


def enforceslopes2(topof, vegf, sh, anglesand, angleveg, th, RNG):
    """Function to enforce the angle of repose; open boundaries (18 oct 2010). Returns an updated topography.
    - topof         : topography map [topo]
    - vegf          : map of combined vegetation effectiveness [veg] [0,1]
    - sh            : slab height [slabheight]
    - anglesand     : angle of repose for bare sand [repose_bare]
    - angleveg      : angle of repose for vegetated cells [repose_veg]
    - th            : switching threshold [repose_threshold]"""

    steplimitsand = math.floor(math.tan(anglesand * math.pi / 180) / sh)  # Maximum allowed height difference for sandy cells
    steplimitsanddiagonal = math.floor(math.sqrt(2) * math.tan(anglesand * math.pi / 180) / sh)  # Maximum allowed height difference for sandy cells along diagonal
    steplimitveg = math.floor(math.tan(angleveg * math.pi / 180) / sh)  # Maximum allowed height difference for cells vegetated > threshold
    steplimitvegdiagonal = math.floor(math.sqrt(2) * math.tan(angleveg * math.pi / 180) / sh)  # Maximum allowed height difference for cells vegetated  along diagonal > threshold

    steplimit = (vegf < th) * steplimitsand + (vegf >= th) * steplimitveg  # Map with max height diff seen from each central cell
    steplimitdiagonal = (vegf < th) * steplimitsanddiagonal + (vegf >= th) * steplimitvegdiagonal  # Idem for diagonal

    M, N = topof.shape  # Retrieve dimensions of area
    slopes = np.zeros([M, N, 8])  # Pre-allocating matrix to improve calculation speed
    exceeds = np.zeros([M, N, 8])  # Pre-allocating matrix to improve calculation speed

    total = 0  # Number of avalanches
    slabsmoved = 1  # Initial number to get the loop going
    avcount = 0

    while slabsmoved > 0:
        # Padding with same cell to create open boundaries
        topof = np.column_stack((topof[:, 0], topof, topof[:, -1]))
        topof = np.row_stack((topof[0, :], topof, topof[-1, :]))

        # All height differences relative to centre cell (positive is sloping upward away, negative is sloping downward away)
        # Directions are clockwise starting from upward/North [-1,0]=1, and so-on
        central = topof[1: -1, 1: -1]  # (2: -1 -1) --> (1 : -1)
        slopes[:, :, 0] = topof[0: -2, 1: -1] - central  # dir 1 [-1,0]
        slopes[:, :, 4] = topof[2:, 1: -1] - central  # dir 5 [+1,0]
        slopes[:, :, 2] = topof[1: -1, 2:] - central  # dir 3 [0,+1]
        slopes[:, :, 6] = topof[1: -1, 0:-2] - central  # dir 7 [0,-1]
        slopes[:, :, 1] = topof[0:-2, 2:] - central  # dir 2 [-1,+1]
        slopes[:, :, 3] = topof[2:, 2:] - central  # dir 4 [+1,+1]
        slopes[:, :, 5] = topof[2:, 0:-2] - central  # dir 6 [+1,-1]
        slopes[:, :, 7] = topof[0:-2, 0:-2] - central  # dir 8 [-1,-1]

        minima = np.amin(slopes, axis=2)  # Identify for each cell the value of the lowest (negative) slope of all 8 directions

        # True (1) if steepest slope is in this direction & is more negative than the repose limit
        exceeds[:, :, 0] = (minima == slopes[:, :, 0]) * (slopes[:, :, 0] < -steplimit)
        exceeds[:, :, 1] = (minima == slopes[:, :, 1]) * (slopes[:, :, 1] < -steplimitdiagonal)
        exceeds[:, :, 2] = (minima == slopes[:, :, 2]) * (slopes[:, :, 2] < -steplimit)
        exceeds[:, :, 3] = (minima == slopes[:, :, 3]) * (slopes[:, :, 3] < -steplimitdiagonal)
        exceeds[:, :, 4] = (minima == slopes[:, :, 4]) * (slopes[:, :, 4] < -steplimit)
        exceeds[:, :, 5] = (minima == slopes[:, :, 5]) * (slopes[:, :, 5] < -steplimitdiagonal)
        exceeds[:, :, 6] = (minima == slopes[:, :, 6]) * (slopes[:, :, 6] < -steplimit)
        exceeds[:, :, 7] = (minima == slopes[:, :, 7]) * (slopes[:, :, 7] < -steplimitdiagonal)

        # If there are multiple equally steepest slopes that exceed the angle of repose, one of them needs to be assigned and the rest set to 0
        k = np.argwhere(np.sum(exceeds, axis=2) > 1)  # Identify cells with multiple equally steepest minima in different directions that all exceed repose angle
        if k.size != 0:
            for i in range(len(k)):
                row, col = k[i]  # Recover row and col #s from k
                a1 = RNG.random((1, 1, 8)) * exceeds[row, col, :]  # Give all equally steepest slopes in this cell a random number
                exceeds[row, col, :] = (a1 == np.max(a1))  # Pick the largest random number and set the rest to zero

        # Begin avalanching
        topof[1: -1, 1: -1] = topof[1: -1, 1: -1] \
                              - exceeds[:, :, 0] + np.roll(exceeds[:, :, 0], -1, axis=0) \
                              - exceeds[:, :, 4] + np.roll(exceeds[:, :, 4], 1, axis=0) \
                              - exceeds[:, :, 2] + np.roll(exceeds[:, :, 2], 1, axis=1) \
                              - exceeds[:, :, 6] + np.roll(exceeds[:, :, 6], -1, axis=1) \
                              - exceeds[:, :, 1] + np.roll(exceeds[:, :, 1], (-1, 1), axis=(0, 1)) \
                              - exceeds[:, :, 3] + np.roll(exceeds[:, :, 3], (1, 1), axis=(0, 1)) \
                              - exceeds[:, :, 5] + np.roll(exceeds[:, :, 5], (1, -1), axis=(0, 1)) \
                              - exceeds[:, :, 7] + np.roll(exceeds[:, :, 7], (-1, -1), axis=(0, 1))

        topof = topof[1: -1, 1: -1]  # Remove padding
        slabsmoved = np.sum(exceeds)  # Count moved slabs during this loop
        total = total + slabsmoved  # Total number of moved slabs since beginning of m-file

    return topof, total


def marine_processes3_diss3e(total_tide, msl, slabheight, cellsizef, topof, eqtopof, vegf, m26f, m27af, m28f, pwavemaxf, pwaveminf, depthlimitf, shelterf, pcurr, RNG):
    """Calculates the effects of high tide levels on the beach and foredune (with stochastic element)
    % Beachupdate in cellular automata fashion
    % Sets back a certain length of the profile to the equilibrium;
    % if waterlevel exceeds dunefoot, this lead to dune erosion;
    % otherwise, only the beach is reset.
    %
    % tide          : height of storm surge [slabs]
    % slabheight    : slabheight in m [slabheightm] [m]
    % cellsizef     : interpreted cell size [cellsize] [m]
    % topof         : topography map [topo] [slabs]
    % eqtopof       : equilibrium beach topography map [eqtopo] [slabs]
    % vegf          : map of combined vegetation effectiveness [veg] [0-1]
    % m26f          : parameter for dissipation strength ~[0.01 - 0.02]
    % m27af         : wave energy factor
    % m28f          : resistance of vegetation: 1 = full, 0 = none.
    % pwavemaxf     : maximum erosive strenght of waves (if >1: overrules vegetation)
    % pwaveminf     : in area with waves always potential for action
    % depthlimitf   : no erosive strength in very shallow water
    % shelterf      : exposure of sheltered cells: 1 = full shelter, 0 = no shelter.
    % phydro         : probability of erosion due to any other hydrodynamic process rather than waves"""

    # --------------------------------------
    # WAVE RUNUP
    # Offshore measured tide (sealevf) has to be converted to effective tide
    # level at the shoreline. The vertical limit of wave runup (R) is a function of
    # tide, slope and wave conditions. Since wave conditions are correlated
    # with tide level (higher waves associated with higher storms), we use a
    # simplified expression where R = f(tide, slope).
    #
    # Original expression of wave runup height controlled by foreshore slope
    # (Stockdon et al, 2006):
    #   Irribarren number = b/sqrt(H/L)
    #   for dissipative beaches (Irribarren number < 0.3):
    #       runup_m = 0.043 * sqrt(H*L);
    #   for intermediate or reflective beaches (Irribarren number >= 0.3):
    #       runup = 1.1*(.35*tan(b)*sqrt(H*L) + sqrt(H*L*(0.563*tan(b^2)+0.004))/2);

    # Derive gradient of eqtopo as an approximation of foreshore slope
    loc_upper_slope = np.argwhere(eqtopof[0, :] > 15)
    loc_lower_slope = np.argwhere(eqtopof[0, :] > -15)

    if len(loc_upper_slope) == 0 or len(loc_lower_slope) == 0:
        b = 0.01
    else:
        b = np.nanmean(np.gradient(eqtopof[0, loc_lower_slope[0]: loc_upper_slope[0]] * slabheight))

    # Tidal elevation above MSL
    tide = total_tide - msl
    tide_m = tide * slabheight
    msl_m = msl * slabheight

    # Calculate wave height and length based on empirical relationships between tide level and wave conditions
    H = max(0, -2.637 + 2.931 * tide_m)
    L = max(0, -30.59 + 46.74 * tide_m)

    # Runup as a function of wave conditions (H, L) and foreshore slope (b)
    if L == 0 and H > 0:  # Dissipative beach
        runup_m = 0.043 * math.sqrt(H * L)
    elif (L > 0 and H > 0) and b / math.sqrt(H / L) < 0.3:  # Dissipative beach
        runup_m = 0.043 * math.sqrt(H * L)
    else:  # Intermediate or reflective beach
        runup_m = 1.1 * (0.35 * math.tan(b) * math.sqrt(H * L) + math.sqrt(H * L * (0.563 * math.tan(b ** 2) + 0.004)) / 2)

    # Add runup to tide to arrive at total water level (=tide + runup + msl)
    totalwater_m = tide_m + runup_m + msl_m
    totalwater = totalwater_m / slabheight

    # --------------------------------------
    # IDENTIFY CELLS EXPOSED TO WAVES
    # by dunes and embryodunes, analogous to shadowzones but no angle

    toolow = topof < totalwater  # [0 1] Give matrix with cells that are potentially under water
    pexposed = np.ones(topof.shape)  # Initialise matrix of cells exposed [T] and not exposed [F] to high water
    for m20 in range(len(topof[:, 0])):  # Run along all the rows
        twlloc = np.argwhere(topof[m20, :] >= totalwater)
        if len(twlloc) > 0:  # If there are any sheltered cells
            m21 = twlloc[0][0]  # Finds for every row the first instance where the topography exceeds the sea level
            pexposed[m20, m21: -1] = 1 - shelterf  # Subtract shelter from exposedness: sheltered cells are less exposed

    # --------------------------------------
    # FILL TOPOGRAPHY TO EQUILIBRIUM

    inundatedf = copy.deepcopy(pexposed)  # Bool matrix of cells that exposed to sea water

    # --------------------------------------
    # WAVES

    waterdepth = (totalwater - topof) * pexposed * slabheight  # [m] Exposed is used to avoid negative waterdepths
    waterdepth[waterdepth <= depthlimitf] = depthlimitf  # This limits high dissipitation when depths are limited; remainder of energy is transferred landward

    # Initialise dissiptation matrices
    diss = np.zeros(topof.shape)
    cumdiss = np.zeros(topof.shape)

    # Calculate dissipation
    loc = np.argwhere(topof[0, :] > -10)  # Find location to start dissipation; limits dissipation to cells greater than -1 m in elevation (assuming 0.1 m slabheight)

    if len(loc) == 0:
        loc = [1]
    elif loc[0] == 0:
        loc[0] = 1

    for m25 in range(int(loc[0]), topof.shape[1]):  # Do for all columns
        diss[:, m25] = (cellsizef / waterdepth[:, m25]) - (cellsizef / waterdepth[:, loc[0]][:, 0])  # Dissipation corrected for cellsize; calcs difference is dissipation between cells in first row and cells in row m25
        cumdiss[:, m25] = diss[:, m25 - 1] + cumdiss[:, m25 - 1]  # Cumulative dissipation from the shore, excluding the current cell; lowers hydrodynamic energy for landward celss

    # --------------------------------------
    # CALCULATING PROBABILITY OF HYDRODYNAMIC EROSION (phydro = pwave - pinun)

    # Dissipation of wave strength across the topography (wave strength times dissiptation)
    pwave = (m27af * pwavemaxf - m26f * cumdiss)
    pwave_ero = (m27af * pwavemaxf - m26f * cumdiss)  # Wave used for dune attack

    # Normalize and remove negatives
    pwave_ero[pwave_ero < 0] = 0  # Set negative values to 0
    pwave_ero = pwave_ero / np.max(pwave_ero)  # Set max between 1 and 0
    pwave[pwave < 0] = 0  # Set negative values to 0
    pwave = pwave / np.max(pwave)  # Set max between 1 and 0

    pwave[np.logical_and.reduce((pwave < pwaveminf, topof < totalwater, pexposed > 0))] = pwaveminf  # In area with waves always potential for action

    pcurr = toolow * pcurr

    # Local reduction of erosive strength due to vegetation
    pbare = 1 - m28f * vegf  # If vegf = 1, still some chance for being eroded

    # Ssumming up erosion probabilities
    phydro = copy.deepcopy(pwave)
    phydro_ero = pwave_ero + pcurr

    # --------------------------------------
    # UPDATING THE TOPOGRAPHY

    pbeachupdate = pbare * phydro * (topof < totalwater) * pexposed  # Probability for beachupdate, also indication of strength of process (it does not do anything random)
    pbeachupdate[pbeachupdate < 0] = 0  # Keep probabilities to 0 - 1 range
    pbeachupdate[pbeachupdate > 1] = 1

    pbeachupdate_ero = pbare * phydro_ero * (topof > totalwater)
    pbeachupdate_ero[pbeachupdate_ero < 0] = 0  # Keep probabilities to 0 - 1 range
    pbeachupdate_ero[pbeachupdate_ero > 1] = 1

    # Stochastic element
    width, length = pbeachupdate.shape
    dbeachupdate = RNG.random((width, length)) < pbeachupdate  # Do beachupdate only for random cells
    dbeachupdate_ero = RNG.random((width, length)) < pbeachupdate_ero  # Do beachupdate only for random cells

    # # marine_processes3_diss3e
    topof = topof - ((topof - eqtopof) * dbeachupdate)  # Adjusting the topography
    topof[topof < -10] = eqtopof[topof < -10]  # Update those that migrated to the ocean by the open boundary

    # # Changed after revision (JK 21/01/2015)
    # # Limit filling up of topography to the maximum water level, so added (accreted) cells cannot be above the maximum water level
    # eqtopof[(eqtopof > totalwater)] = totalwater
    #
    # topof = topof - (topof - eqtopof) * dbeachupdate  # Adjusting the topography
    # topof[np.logical_and(topof > totalwater, dbeachupdate > 0)] = totalwater
    # # Changed after revision (FGS)
    # # Above proposed change has the counterpart of a lot of filling when beach width is short, since it starts to build-up dunes instead of eroding it

    if np.isnan(np.sum(dbeachupdate_ero)) is False:
        topof[np.logical_and(topof >= totalwater, dbeachupdate_ero > 0)] = topof[np.logical_and(topof >= totalwater, dbeachupdate_ero > 0)] - (topof[np.logical_and(topof >= totalwater, dbeachupdate_ero > 0)] - eqtopof[np.logical_and(topof >= totalwater, dbeachupdate_ero > 0)])

    return topof, inundatedf, pbeachupdate, diss, cumdiss, pwave


def marine_processes3_IRBRtest(total_tide, msl, slabheight, cellsizef, topof, eqtopof, vegf, m26f, m27af, m28f, pwavemaxf, pwaveminf, depthlimitf, shelterf, pcurr, RNG):
    """Calculates the effects of high tide levels on the beach and foredune (with stochastic element)
    % Beachupdate in cellular automata fashion
    % Sets back a certain length of the profile to the equilibrium;
    % if waterlevel exceeds dunefoot, this lead to dune erosion;
    % otherwise, only the beach is reset.
    %
    % tide          : height of storm surge [slabs]
    % slabheight    : slabheight in m [slabheightm] [m]
    % cellsizef     : interpreted cell size [cellsize] [m]
    % topof         : topography map [topo] [slabs]
    % eqtopof       : equilibrium beach topography map [eqtopo] [slabs]
    % vegf          : map of combined vegetation effectiveness [veg] [0-1]
    % m26f          : parameter for dissipation strength ~[0.01 - 0.02]
    % m27af         : wave energy factor
    % m28f          : resistance of vegetation: 1 = full, 0 = none.
    % pwavemaxf     : maximum erosive strenght of waves (if >1: overrules vegetation)
    % pwaveminf     : in area with waves always potential for action
    % depthlimitf   : no erosive strength in very shallow water
    % shelterf      : exposure of sheltered cells: 1 = full shelter, 0 = no shelter.
    % phydro         : probability of erosion due to any other hydrodynamic process rather than waves"""

    # --------------------------------------
    # WAVE RUNUP
    # Offshore measured tide (sealevf) has to be converted to effective tide
    # level at the shoreline. The vertical limit of wave runup (R) is a function of
    # tide, slope and wave conditions. Since wave conditions are correlated
    # with tide level (higher waves associated with higher storms), we use a
    # simplified expression where R = f(tide, slope).
    #
    # Original expression of wave runup height controlled by foreshore slope
    # (Stockdon et al, 2006):
    #   Irribarren number = b/sqrt(H/L)
    #   for dissipative beaches (Irribarren number < 0.3):
    #       runup_m = 0.043 * sqrt(H*L);
    #   for intermediate or reflective beaches (Irribarren number >= 0.3):
    #       runup = 1.1*(.35*tan(b)*sqrt(H*L) + sqrt(H*L*(0.563*tan(b^2)+0.004))/2);

    # Derive gradient of eqtopo as an approximation of foreshore slope
    # loc_upper_slope = np.argwhere(eqtopof[0, :] > 15)
    # loc_lower_slope = np.argwhere(eqtopof[0, :] > -15)
    loc_upper_slope = np.argwhere(eqtopof[0, :] > max(eqtopof[0, :]))  # IRBR 31Oct22 Test: bug fix for above
    loc_lower_slope = np.argwhere(eqtopof[0, :] > min(eqtopof[0, :]))  # IRBR 31Oct22 Test: bug fix for above

    if len(loc_upper_slope) == 0 or len(loc_lower_slope) == 0:
        b = 0.01
    else:
        b = np.nanmean(np.gradient(eqtopof[0, loc_lower_slope[0]: loc_upper_slope[0]] * slabheight))
    # b = np.nanmean(np.gradient(eqtopof[0, :] * slabheight))  # IRBR 31Oct22 Test: from preceeding marine_processes versions

    # Tidal elevation above MSL
    tide = total_tide - msl
    tide_m = tide * slabheight
    msl_m = msl * slabheight

    # Calculate wave height and length based on empirical relationships between tide level and wave conditions
    H = max(0, -2.637 + 2.931 * tide_m)
    L = max(0, -30.59 + 46.74 * tide_m)

    # Runup as a function of wave conditions (H, L) and foreshore slope (b)
    if L == 0 and H > 0:  # Dissipative beach
        runup_m = 0.043 * math.sqrt(H * L)
    elif (L > 0 and H > 0) and b / math.sqrt(H / L) < 0.3:  # Dissipative beach
        runup_m = 0.043 * math.sqrt(H * L)
    else:  # Intermediate or reflective beach
        runup_m = 1.1 * (0.35 * math.tan(b) * math.sqrt(H * L) + math.sqrt(H * L * (0.563 * math.tan(b ** 2) + 0.004)) / 2)

    # Add runup to tide to arrive at total water level (=tide + runup + msl)
    totalwater_m = tide_m + runup_m + msl_m
    totalwater = totalwater_m / slabheight

    # --------------------------------------
    # IDENTIFY CELLS EXPOSED TO WAVES
    # by dunes and embryodunes, analogous to shadowzones but no angle

    toolow = topof < totalwater  # [0 1] Give matrix with cells that are potentially under water
    pexposed = np.ones(topof.shape)  # Initialise matrix of cells exposed [T] and not exposed [F] to high water
    for m20 in range(len(topof[:, 0])):  # Run along all the rows
        twlloc = np.argwhere(topof[m20, :] >= totalwater)
        if len(twlloc) > 0:  # If there are any sheltered cells
            m21 = twlloc[0][0]  # Finds for every row the first instance where the topography exceeds the sea level
            pexposed[m20, m21: -1] = 1 - shelterf  # Subtract shelter from exposedness: sheltered cells are less exposed

    # --------------------------------------
    # FILL TOPOGRAPHY TO EQUILIBRIUM

    inundatedf = copy.deepcopy(pexposed)  # Bool matrix of cells that exposed to sea water

    # --------------------------------------
    # WAVES

    waterdepth = (totalwater - topof) * pexposed * slabheight  # [m] Exposed is used to avoid negative waterdepths
    waterdepth[waterdepth <= depthlimitf] = depthlimitf  # This limits high dissipitation when depths are limited; remainder of energy is transferred landward

    # Initialise dissiptation matrices
    diss = np.zeros(topof.shape)
    cumdiss = np.zeros(topof.shape)

    # Calculate dissipation
    loc = np.argwhere(topof[0, :] > -10)  # Find location to start dissipation; limits dissipation to cells greater than -1 m in elevation (assuming 0.1 m slabheight)

    if len(loc) == 0:
        loc = [1]
    elif loc[0] == 0:
        loc[0] = 1

    for m25 in range(int(loc[0]), topof.shape[1]):  # Do for all columns
        # diss[:, m25] = (cellsizef / waterdepth[:, m25]) - (cellsizef / waterdepth[:, loc[0]][:, 0])  # Dissipation corrected for cellsize; calcs difference is dissipation between cells in first row and cells in row m25
        diss[:, m25] = (cellsizef / waterdepth[:, m25])  # IRBR 31Oct22 Test: from preceeding marine_processes versions
        cumdiss[:, m25] = diss[:, m25 - 1] + cumdiss[:, m25 - 1]  # Cumulative dissipation from the shore, excluding the current cell; lowers hydrodynamic energy for landward celss

    # --------------------------------------
    # CALCULATING PROBABILITY OF HYDRODYNAMIC EROSION (phydro = pwave - pinun)

    # Dissipation of wave strength across the topography (wave strength times dissiptation)
    pwave = (m27af * pwavemaxf - m26f * cumdiss)
    pwave_ero = (m27af * pwavemaxf - m26f * cumdiss)  # Wave used for dune attack

    # Normalize and remove negatives
    pwave_ero[pwave_ero < 0] = 0  # Set negative values to 0
    pwave_ero = pwave_ero / np.max(pwave_ero)  # Set max between 1 and 0
    pwave[pwave < 0] = 0  # Set negative values to 0
    pwave = pwave / np.max(pwave)  # Set max between 1 and 0

    pwave[np.logical_and.reduce((pwave < pwaveminf, topof < totalwater, pexposed > 0))] = pwaveminf  # In area with waves always potential for action

    pcurr = toolow * pcurr

    # Local reduction of erosive strength due to vegetation
    pbare = 1 - m28f * vegf  # If vegf = 1, still some chance for being eroded

    # Ssumming up erosion probabilities
    phydro = copy.deepcopy(pwave)
    phydro_ero = pwave_ero + pcurr

    # --------------------------------------
    # UPDATING THE TOPOGRAPHY

    pbeachupdate = pbare * phydro * (topof < totalwater)  # * pexposed  # Probability for beachupdate, also indication of strength of process (it does not do anything random)
    pbeachupdate[pbeachupdate < 0] = 0  # Keep probabilities to 0 - 1 range
    pbeachupdate[pbeachupdate > 1] = 1

    pbeachupdate_ero = pbare * phydro_ero * (topof > totalwater)
    pbeachupdate_ero[pbeachupdate_ero < 0] = 0  # Keep probabilities to 0 - 1 range
    pbeachupdate_ero[pbeachupdate_ero > 1] = 1

    # Stochastic element
    width, length = pbeachupdate.shape
    dbeachupdate = RNG.random((width, length)) < pbeachupdate  # Do beachupdate only for random cells
    dbeachupdate_ero = RNG.random((width, length)) < pbeachupdate_ero  # Do beachupdate only for random cells
    # dbeachupdate = pbeachupdate * (RNG.random((width, length)) < pbeachupdate)  # IRBR 31Oct22 Test: from preceeding marine_processes versions
    # dbeachupdate_ero = pbeachupdate_ero * (RNG.random((width, length)) < pbeachupdate_ero)  # IRBR 31Oct22 Test: from preceeding marine_processes versions

    # # marine_processes3_diss3e
    # topof = topof - ((topof - eqtopof) * dbeachupdate)  # Adjusting the topography
    # topof[topof < -10] = eqtopof[topof < -10]  # Update those that migrated to the ocean by the open boundary
    # ----
    # marine_processes  # IRBR 31Oct22 Test: from marine_processes
    topof = topof - (topof - eqtopof) * pbeachupdate  # Adjusting the topography
    topof[np.logical_and(topof > totalwater, pbeachupdate > 0)] = totalwater
    topof[topof < -10] = eqtopof[topof < -10]  # Update those that migrated to the ocean by the open boundary
    # ----
    # # IRBR 31Oct22 Test: based on beachupdate_ero from below
    # if np.isnan(np.sum(dbeachupdate)) is False:
    #     topof[np.logical_and(topof < totalwater, dbeachupdate > 0)] = topof[np.logical_and(topof < totalwater, dbeachupdate > 0)] - (topof[np.logical_and(topof < totalwater, dbeachupdate > 0)] - eqtopof[np.logical_and(topof < totalwater, dbeachupdate > 0)])

    # # Changed after revision (JK 21/01/2015)
    # # Limit filling up of topography to the maximum water level, so added (accreted) cells cannot be above the maximum water level
    # eqtopof[(eqtopof > totalwater)] = totalwater
    #
    # topof = topof - (topof - eqtopof) * dbeachupdate  # Adjusting the topography
    # topof[np.logical_and(topof > totalwater, dbeachupdate > 0)] = totalwater
    # # Changed after revision (FGS)
    # # Above proposed change has the counterpart of a lot of filling when beach width is short, since it starts to build-up dunes instead of eroding it

    if np.isnan(np.sum(dbeachupdate_ero)) is False:
        topof[np.logical_and(topof >= totalwater, dbeachupdate_ero > 0)] = topof[np.logical_and(topof >= totalwater, dbeachupdate_ero > 0)] - (topof[np.logical_and(topof >= totalwater, dbeachupdate_ero > 0)] - eqtopof[np.logical_and(topof >= totalwater, dbeachupdate_ero > 0)])

    return topof, inundatedf, pbeachupdate, diss, cumdiss, pwave


def marine_processes(total_tide, msl, slabheight, cellsizef, topof, eqtopof, vegf, m26f, m27af, m28f, pwavemaxf, pwaveminf, depthlimitf, shelterf, TWLexcursion, veg_elev_limit, year, iterations_per_year):
    """Calculates the effects of high tide levels on the beach and foredune (with stochastic element)
    % Beachupdate in cellular automata fashion
    % Sets back a certain length of the profile to the equilibrium;
    % if waterlevel exceeds dunefoot, this lead to dune erosion;
    % otherwise, only the beach is reset.
    %
    % tide: height of storm surge [slabs]
    % slabheight: slabheight in m [slabheightm] [m]
    % cellsizef: interpreted cell size [cellsize] [m]
    % topof: topography map [topo] [slabs]
    % eqtopof: equilibrium beach topography map [eqtopo] [slabs]
    % vegf: map of combined vegetation effectiveness [veg] [0-1]
    % m26f: parameter for dissipation strength ~[0.01 - 0.02]
    % m27af: wave energy factor
    % m28f: resistance of vegetation: 1 = full, 0 = none.
    % pwavemaxf: maximum erosive strenght of waves (if >1: overrules vegetation)
    % pwaveminf: in area with waves always potential for action
    % depthlimitf: no erosive strength in very shallow water
    % shelterf: exposure of sheltered cells: 1 = full shelter, 0 = no shelter.
    % phydro: probability of erosion due to any other hydrodynamic process rather than waves"""

    # --------------------------------------
    # WAVE RUNUP
    # Offshore measured tide (sealevf) has to be converted to effective tide
    # level at the shoreline. The vertical limit of wave runup (R) is a function of
    # tide, slope and wave conditions. Since wave conditions are correlated
    # with tide level (higher waves associated with higher storms), we use a
    # simplified expression where R = f(tide, slope).
    #
    # Original expression of wave runup height controlled by foreshore slope
    # (Stockdon et al, 2006):
    #   Irribarren number = b/sqrt(H/L)
    #   for dissipative beaches (Irribarren number < 0.3):
    #       runup_m = 0.043 * sqrt(H*L);
    #   for intermediate or reflective beaches (Irribarren number >= 0.3):
    #       runup = 1.1*(.35*tan(b)*sqrt(H*L) + sqrt(H*L*(0.563*tan(b^2)+0.004))/2);

    # Derive gradient of eqtopo as an approximation of foreshore slope
    # b = np.nanmean(np.gradient(eqtopof[0, :] * slabheight))  # Old method

    dune_crest_loc = foredune_crest(topof, eqtopof, vegf)  # Cross-shore location of pre-storm dune crest
    slopes = beach_slopes(topof, msl, dune_crest_loc, slabheight)
    b = np.mean(slopes)

    # Tidal elevation above MSL
    tide = total_tide - msl  # [slabs]
    tide_m = tide * slabheight  # [m]
    msl_m = msl * slabheight  # [m]

    H = max(0, -2.637 + 2.931 * tide_m)
    L = max(0, -30.59 + 46.74 * tide_m)

    # Runup as a function of wave conditions (H, L) and foreshore slope (b)
    if L == 0 and H > 0:
        runup_m = 0.043 * math.sqrt(H * L)
    elif (L > 0 and H > 0) and b / math.sqrt(H / L) < 0.3:
        runup_m = 0.043 * math.sqrt(H * L)
    else:
        runup_m = 1.1 * (0.35 * math.tan(b) * math.sqrt(H * L) + math.sqrt(H * L * (0.563 * math.tan(b ** 2) + 0.004)) / 2)

    # Add runup to tide to arrive at total water level (=tide + runup + msl)
    totalwater_m = tide_m + runup_m + msl_m
    totalwater = totalwater_m / slabheight

    # --------------------------------------
    # IDENTIFY CELLS EXPOSED TO WAVES
    # by dunes and embryodunes, analogous to shadowzones but no angle

    # toolow = topof < totalwater  # [0 1] Give matrix with cells that are potentially under water
    pexposed = np.ones(topof.shape)  # Initialise matrix
    for m20 in range(len(topof[:, 0])):  # Run along all the rows
        twlloc = np.argmax(topof[m20, :] >= totalwater)  # Finds for every row the first instance where the topography exceeds the total water level
        if twlloc > 0:  # If there are any sheltered cells
            m21 = twlloc
            pexposed[m20, m21:] = 1 - shelterf  # Subtract shelter from exposedness: sheltered cells are less exposed

    # --------------------------------------
    # FILL TOPOGRAPHY TO EQUILIBRIUM

    inundatedf = copy.deepcopy(pexposed)  # Inundated is the area that really receives sea water

    # --------------------------------------
    # WAVES

    waterdepth = (totalwater - topof) * pexposed * slabheight  # [m] Exposed is used to avoid negative waterdepths
    waterdepth[waterdepth <= depthlimitf] = depthlimitf  # This limits high dissipitation when depths are limited; remainder of energy is transferred landward

    # Initialise dissiptation matrices
    diss = np.zeros(topof.shape)
    cumdiss = np.zeros(topof.shape)

    # Calculate dissipation
    diss[:, 0] = cellsizef / waterdepth[:, 0]  # Dissipation corrected for cellsize
    cumdiss[:, 0] = diss[:, 0]
    for m25 in range(1, topof.shape[1]):  # Do for all columns
        diss[:, m25] = cellsizef / waterdepth[:, m25]  # Dissipation corrected for cellsize
        cumdiss[:, m25] = diss[:, m25 - 1] + cumdiss[:, m25 - 1]  # Cumulative dissipation from the shore, excluding the current cell

    # Initial wave strength m27f
    m27f = m27af * totalwater

    # Dissipation of wave strength across the topography (wave strength times dissiptation)
    pwave = (pwavemaxf - m26f * cumdiss) * m27f
    pwave[np.logical_and.reduce((pwave < pwaveminf, topof < totalwater, pexposed > 0))] = pwaveminf  # In area with waves always potential for action
    # pwave[waterdepth < (slabheight * depthlimitf)] = 0  # No erosive strength in very shallow water

    # Local reduction of erosive strength due to vegetation
    pbare = 1 - m28f * vegf  # If vegf = 1, still some chance for being eroded

    # Updating the topography
    pbeachupdate = pbare * pwave * pexposed  # Probability for beachupdate, also indication of strength of process (it does not do anything random)  IRBR 1Nov22: Un-commented *pexposed
    pbeachupdate[pbeachupdate < 0] = 0  # Keep probabilities to 0 - 1 range
    pbeachupdate[pbeachupdate > 1] = 1

    # IRBR 1Dec22: Only allow update up to dune/berm crest
    crest_limit = np.ones(topof.shape)
    for ls in range(topof.shape[0]):
        crest_limit[ls, dune_crest_loc[ls]:] = 0
    pbeachupdate *= crest_limit

    # Changed after revision (JK 21/01/2015)
    topof = topof - (topof - eqtopof) * pbeachupdate  # Adjusting the topography
    topof[np.logical_and(topof > totalwater, pbeachupdate > 0)] = totalwater  # Limit filling up of topography to the maximum water level so added (accreted) cells cannot be above the maximum water level

    return topof, inundatedf, pbeachupdate, diss, cumdiss, pwave


def marine_processes_Rhigh(Rhigh, slabheight, cellsizef, topof, eqtopof, vegf, m26f, m27af, m28f, pwavemaxf, pwaveminf, depthlimitf, shelterf, crestline):
    """Calculates the effects of high tide levels on the beach and foredune (with stochastic element) - IRBR 1Dec22: This version takes already-computed TWL as direct input.
    % Beachupdate in cellular automata fashion
    % Sets back a certain length of the profile to the equilibrium;
    % if waterlevel exceeds dunefoot, this lead to dune erosion;
    % otherwise, only the beach is reset.
    %
    % Rhigh: height of total water level [slabs]
    % slabheight: slabheight in m [slabheightm] [m]
    % cellsizef: interpreted cell size [cellsize] [m]
    % topof: topography map [topo] [slabs]
    % eqtopof: equilibrium beach topography map [eqtopo] [slabs]
    % vegf: map of combined vegetation effectiveness [veg] [0-1]
    % m26f: parameter for dissipation strength ~[0.01 - 0.02]
    % m27af: wave energy factor
    % m28f: resistance of vegetation: 1 = full, 0 = none.
    % pwavemaxf: maximum erosive strenght of waves (if >1: overrules vegetation)
    % pwaveminf: in area with waves always potential for action
    % depthlimitf: no erosive strength in very shallow water
    % shelterf: exposure of sheltered cells: 1 = full shelter, 0 = no shelter.
    % phydro: probability of erosion due to any other hydrodynamic process rather than waves"""

    totalwater = np.mean(Rhigh)

    # --------------------------------------
    # IDENTIFY CELLS EXPOSED TO WAVES
    # by dunes and embryodunes, analogous to shadowzones but no angle

    # toolow = topof < totalwater  # [0 1] Give matrix with cells that are potentially under water
    pexposed = np.ones(topof.shape)  # Initialise matrix
    for m20 in range(len(topof[:, 0])):  # Run along all the rows
        twlloc = np.argmax(topof[m20, :] >= totalwater)  # Finds for every row the first instance where the topography exceeds the total water level
        if twlloc > 0:  # If there are any sheltered cells
            m21 = twlloc
            pexposed[m20, m21:] = 1 - shelterf  # Subtract shelter from exposedness: sheltered cells are less exposed

    # --------------------------------------
    # FILL TOPOGRAPHY TO EQUILIBRIUM

    inundatedf = copy.deepcopy(pexposed)  # Inundated is the area that really receives sea water

    # --------------------------------------
    # WAVES

    waterdepth = (totalwater - topof) * pexposed * slabheight  # [m] Exposed is used to avoid negative waterdepths
    waterdepth[waterdepth <= depthlimitf] = depthlimitf  # This limits high dissipitation when depths are limited; remainder of energy is transferred landward

    # Initialise dissiptation matrices
    diss = np.zeros(topof.shape)
    cumdiss = np.zeros(topof.shape)

    # Calculate dissipation
    diss[:, 0] = cellsizef / waterdepth[:, 0]  # Dissipation corrected for cellsize
    cumdiss[:, 0] = diss[:, 0]
    for m25 in range(1, topof.shape[1]):  # Do for all columns
        diss[:, m25] = cellsizef / waterdepth[:, m25]  # Dissipation corrected for cellsize
        cumdiss[:, m25] = diss[:, m25 - 1] + cumdiss[:, m25 - 1]  # Cumulative dissipation from the shore, excluding the current cell

    # Initial wave strength m27f
    m27f = m27af * totalwater

    # Dissipation of wave strength across the topography (wave strength times dissiptation)
    pwave = (pwavemaxf - m26f * cumdiss) * m27f
    pwave[np.logical_and.reduce((pwave < pwaveminf, topof < totalwater, pexposed > 0))] = pwaveminf  # In area with waves always potential for action
    # pwave[waterdepth < (slabheight * depthlimitf)] = 0  # No erosive strength in very shallow water

    # Local reduction of erosive strength due to vegetation
    pbare = 1 - m28f * vegf  # If vegf = 1, still some chance for being eroded

    # Updating the topography
    pbeachupdate = pbare * pwave * pexposed  # Probability for beachupdate, also indication of strength of process (it does not do anything random)  IRBR 1Nov22: Un-commented *pexposed
    pbeachupdate[pbeachupdate < 0] = 0  # Keep probabilities to 0 - 1 range
    pbeachupdate[pbeachupdate > 1] = 1

    # Only allow beach update up to the dune/berm crest; IRBR 1Dec22
    crest_limit = np.ones(topof.shape)
    for ls in range(topof.shape[0]):
        crest_limit[ls, crestline[ls] + 1:] = 0
    pbeachupdate *= crest_limit

    # Changed after revision (JK 21/01/2015)
    newtopof = topof - (topof - eqtopof) * pbeachupdate  # Adjusting the topography
    topof[np.logical_and(topof > totalwater, pbeachupdate > 0)] = totalwater  # Limit filling up of topography to the maximum water level so added (accreted) cells cannot be above the maximum water level

    crestline_change = topof[np.arange(topof.shape[0]), crestline] - newtopof[np.arange(topof.shape[0]), crestline]

    return newtopof, inundatedf, pbeachupdate, diss, cumdiss, pwave, crestline_change


def growthfunction1_sens(spec, sed, A1, B1, C1, D1, E1, P1):
    """Input is the vegetation map, and the sedimentation balance in units of cell dimension (!), i.e. already adjusted by the slabheight
    Output is the change in vegetation effectiveness
    Ammophilia-like vegetation"""

    # Physiological range (needs to be specified)
    minimum = 0.0
    maximum = 1.0

    # Vertices (these need to be specified)
    x1 = A1  # was x1 = -1.4
    x2 = B1  # was x2 = 0.1
    x3 = C1  # was x3 = 0.45
    x4 = D1  # was x4 = 0.85
    x5 = E1  # was x5 = 1.4
    y1 = -1.0  # y1 = -1.0; y1 = -1.0
    y2 = 0.0  # y2 = 0.0; y2 = 0.0
    y3 = P1  # y3 = 0.4; y3 = P1;
    y4 = 0.0  # y4 = 0.0; y4 = 0.0
    y5 = -1.0  # y5 = -1.0; y5 = -1.0

    # Slopes between vertices (calculated from vertices)
    s12 = (y2 - y1) / (x2 - x1)
    s23 = (y3 - y2) / (x3 - x2)
    s34 = (y4 - y3) / (x4 - x3)
    s45 = (y5 - y4) / (x5 - x4)

    leftextension = (sed < x1) * -1
    firstleg = (sed >= x1) * (sed < x2) * ((sed - x1) * s12 + y1)
    secondleg = (sed >= x2) * (sed < x3) * ((sed - x2) * s23 + y2)
    thirdleg = (sed >= x3) * (sed < x4) * ((sed - x3) * s34 + y3)
    fourthleg = (sed >= x4) * (sed < x5) * ((sed - x4) * s45 + y4)
    rightextension = (sed >= x5) * -1

    spec = spec + leftextension + firstleg + secondleg + thirdleg + fourthleg + rightextension

    spec[spec < minimum] = minimum
    spec[spec > maximum] = maximum

    return spec


def growthfunction2_sens(spec, sed, A2, B2, D2, E2, P2):
    """Input is the vegetation map, and the sedimentation balance in units of cell dimension (!), i.e. already adjusted by the slabheight
    Output is the change in vegetation effectiveness
    Buckthorn-type vegetation"""

    # Physiological range (needs to be specified)
    minimum = 0.0  # was -0.5
    maximum = 1.0  # was 1.5

    # Vertices (these need to be specified)
    x1 = A2  # was x1 = -1.3
    x2 = B2  # was x2 = -0.65
    x3 = 0  # was x3 = 0
    x4 = D2  # was x4 = 0.2
    x5 = E2  # was x5 = 2.2
    y1 = -1.0  # y1 = -1.0
    y2 = 0.0  # y2 = 0.0
    y3 = P2  # y3 = 0.1
    y4 = 0.0  # y4 = 0.0
    y5 = -1.0  # y5 = -1.0

    # Slopes between vertices (calculated from vertices)
    s12 = (y2 - y1) / (x2 - x1)
    s23 = (y3 - y2) / (x3 - x2)
    s34 = (y4 - y3) / (x4 - x3)
    s45 = (y5 - y4) / (x5 - x4)

    leftextension = (sed < x1) * -1
    firstleg = (sed >= x1) * (sed < x2) * ((sed - x1) * s12 + y1)
    secondleg = (sed >= x2) * (sed < x3) * ((sed - x2) * s23 + y2)
    thirdleg = (sed >= x3) * (sed < x4) * ((sed - x3) * s34 + y3)
    fourthleg = (sed >= x4) * (sed < x5) * ((sed - x4) * s45 + y4)
    rightextension = (sed >= x5) * -1

    spec = spec + leftextension + firstleg + secondleg + thirdleg + fourthleg + rightextension

    spec[spec < minimum] = minimum
    spec[spec > maximum] = maximum

    return spec


def lateral_expansion(veg, dist, prob, RNG):
    """LATERAL_EXPANSION implements lateral expansion of existing vegetation patches.
    1) mark cells that lie within <dist> of existing patches: probability for new vegetated cells = 1
    2) cells not adjacent to existing patches get probability depending on elevation: pioneer most likely to establish between 3 and 5 m + sea level.
    Returns logical array of which cells veg has successfully expanded into."""

    # Pad vegetation matrix with zeros for rolling
    veg = veg > 0
    vegpad = np.zeros(np.add(veg.shape, (2, 2)))
    vegpad[1: -1, 1: -1] = veg
    veg3 = copy.deepcopy(vegpad)

    # Add shifted matrices to initial matrix to include boundaries
    for i in [-dist, 0, dist]:
        for j in [-dist, 0, dist]:
            # Only 4 neighbouring cells (Von Neumann neighbourhood)
            # if i*3+5+j % 2 == 0:
            veg2 = np.roll(vegpad, (i, j), axis=(0, 1))
            veg3 = veg3 + veg2

    veg3[veg3 > 1] = 1
    lateral_expansion_possible = veg3[1: -1, 1: -1]

    # Lateral expansion only takes place in a fraction <prob> of the possible cells
    width, length = veg.shape
    lateral_expansion_allowed = RNG.random((width, length)) < (lateral_expansion_possible * prob)
    # Include existing vegetation to incorporate growth or decay of existing patches
    lateral_expansion_allowed = lateral_expansion_allowed + veg
    lateral_expansion_allowed = lateral_expansion_allowed > 0

    return lateral_expansion_allowed


def establish_new_vegetation(topof, mht, prob, RNG):
    """Establishment of pioneer veg in previously bare cells.
    Represents the germination and development of veg from seeds or rhizome fragments distributed by the sea or wind.
    Returns logical array of which cells new veg has successfully established in."""

    # Convert topography to height above current sea level
    topof = topof - mht

    # Vertices (these need to be specified)
    x1 = 0.0
    x2 = 2.0
    x3 = 3.0
    x4 = 4.5
    x5 = 20.0
    y1 = prob
    y2 = prob
    y3 = prob
    y4 = prob
    y5 = prob

    # Slopes between vertices (calculated from vertices)
    s12 = (y2 - y1) / (x2 - x1)
    s23 = (y3 - y2) / (x3 - x2)
    s34 = (y4 - y3) / (x4 - x3)
    s45 = (y5 - y4) / (x5 - x4)

    leftextension = (topof < x1) * 0
    firstleg = (topof >= x1) * (topof < x2) * ((topof - x1) * s12 + y1)
    secondleg = (topof >= x2) * (topof < x3) * ((topof - x2) * s23 + y2)
    thirdleg = (topof >= x3) * (topof < x4) * ((topof - x3) * s34 + y3)
    fourthleg = (topof >= x4) * (topof < x5) * ((topof - x4) * s45 + y4)
    rightextension = (topof >= x5) * 0

    pioneer_establish_prob = leftextension + firstleg + secondleg + thirdleg + fourthleg + rightextension
    width, length = topof.shape
    pioneer_established = RNG.random((width, length)) < pioneer_establish_prob

    return pioneer_established


def shoreline_change(
        topof,
        longshore,
        d_sf,
        k_sf,
        s_sf_eq,
        RSLR,
        Qat,
        Qbe,
        OWflux,
        DuneLoss,
        x_s,
        x_t,
        MHW,
        cellsize,
        slabheight,
):
    """Finds shoreline change for modeled time step following Lorenzo-Trueba and Ashton (2014).

    Parameters
    ----------
    topof : ndarray
        [m] Present elevation domain
    longshore : flaot
        [m] Alongshore length of domain
    d_sf : float
        [m] Shoreface depth
    k_sf : flaot
        [] Shoreface flux constant
    s_sf_eq : flaot
        Shoreface equilibrium slope
    RSLR :  float
        [m/ts] Relative sea-level rise rate
    Qat : float
        [m^3/m/ts] Volume of sediment removed from (or added to) the upper shoreface by alongshore transport
    Qbe : float
        [m^3/m/ts] Volume of sediment removed from (or added to) the upper shoreface by fairweather beach change
    OWflux : float
        [m^3/ts] Volume of sediment removed from (or added to) the upper shoreface by overwash (not yet normalized alongshore)
    DuneLoss : float
        [m^3/m/ts] Dune volume lost from storm erosion
    x_s : float
        [m] Cross-shore shoreline position relative to start of simulation
    x_t : float
        [m] Cross-shore shoreface toe position relative to start of simulation
    MHW : float
        [slabs] Present mean high water
    cellsize : float
        [m] Horizontal dimension of model grid cells
    slabheight : float
        [m] Vertical dimension of model grid cells

    Returns
    ----------
    x_s
        [m] New cross-shore shoreline position relative to start of simulation
    x_t
        [m] New cross-shore shoreface toe position relative to start of simulation
    """

    # Find volume of shoreface/beach/dune sand deposited in island interior and back-barrier
    Qow = OWflux * cellsize * slabheight / longshore  # [m^3/m/ts] Volume of sediment lost from shoreface/beach by overwash
    if Qow < 0:
        Qow = 0
    if DuneLoss < Qow:
        Qow = Qow - DuneLoss  # Account for dune contribution to overwash volume; dune loss volume subtracted from Qow because this volume is not coming from the shoreface/beach and therefor does not contribute to shoreline/shoreface change
    else:
        Qow = 0  # Assumes all sediment deposited on barrier interior came from the dunes and therefore nothing came from shoreface/beach; excess DuneLoss assumed lost offshore/downshore

    # DefineParams
    h_b = np.average(topof[topof >= MHW]) * slabheight  # [m] Average height of barrier

    # Shoreface Flux
    s_sf = d_sf / (x_s - x_t)
    Qsf = k_sf * (s_sf_eq - s_sf)  # [m^3/m/ts]

    # Toe, Shoreline, and island base elevation changes
    x_t_dt = (4 * Qsf * (h_b + d_sf) / (d_sf * (2 * h_b + d_sf))) + (2 * RSLR / s_sf)
    x_s_dt = 2 * (Qow + Qbe + Qat) / ((2 * h_b) + d_sf) - (4 * Qsf * (h_b + d_sf) / (((2 * h_b) + d_sf) ** 2))  # Dune growth and alongshore transport added to LTA14 formulation

    # Record changes
    x_t = x_t + x_t_dt
    x_s = x_s + x_s_dt

    return x_s, x_t  # [m]


def shoreline_change2(
        topof,
        d_sf,
        k_sf,
        s_sf_eq,
        RSLR,
        Qat,
        Qbe,
        OWflux,
        DuneLoss,
        x_s,
        x_t,
        MHW,
        cellsize,
        slabheight,
):
    """Finds shoreline change for modeled time step following Lorenzo-Trueba and Ashton (2014). This version is alongshore variable!

    Parameters
    ----------
    topof : ndarray
        [m] Present elevation domain
    d_sf : float
        [m] Shoreface depth
    k_sf : flaot
        [] Shoreface flux constant
    s_sf_eq : flaot
        Shoreface equilibrium slope
    RSLR :  float
        [m/ts] Relative sea-level rise rate
    Qat : ndarray
        [m^3/m/ts] Volume of sediment removed from (or added to) the upper shoreface by alongshore transport
    Qbe : ndarray
        [m^3/m/ts] Volume of sediment removed from (or added to) the upper shoreface by fairweather beach change
    OWflux : ndarray
        [m^3/ts] Volume of sediment removed from (or added to) the upper shoreface by overwash (not yet normalized alongshore)
    DuneLoss : ndarray
        [m^3/m/ts] Dune volume lost from storm erosion
    x_s : ndarray
        [m] Cross-shore shoreline position relative to start of simulation
    x_t : ndarray
        [m] Cross-shore shoreface toe position relative to start of simulation
    MHW : float
        [slabs] Present mean high water
    cellsize : float
        [m] Horizontal dimension of model grid cells
    slabheight : float
        [m] Vertical dimension of model grid cells

    Returns
    ----------
    x_s
        [m] New cross-shore shoreline position relative to start of simulation
    x_t
        [m] New cross-shore shoreface toe position relative to start of simulation
    """

    # Find volume of shoreface/beach/dune sand deposited in island interior and back-barrier
    Qow = OWflux * cellsize * slabheight  # [m^3/m/ts] Volume of sediment lost from shoreface/beach by overwash
    Qow[Qow < 0] = 0
    DuneLoss[DuneLoss < Qow] = Qow[DuneLoss < Qow] - DuneLoss[DuneLoss < Qow]  # Account for dune contribution to overwash volume; dune loss volume subtracted from Qow because this volume is not coming from the shoreface/beach and therefor does not contribute to shoreline/shoreface change
    DuneLoss[DuneLoss >= Qow] = 0  # Assumes all sediment deposited on barrier interior came from the dunes and therefore nothing came from shoreface/beach; excess DuneLoss assumed lost offshore/downshore

    # DefineParams
    h_b = np.average(topof[topof >= MHW], axis=0) * slabheight  # [m] Average height of barrier

    # Shoreface Flux
    s_sf = d_sf / (x_s - x_t)
    Qsf = k_sf * (s_sf_eq - s_sf)  # [m^3/m/ts]

    # Toe, Shoreline, and island base elevation changes
    x_t_dt = (4 * Qsf * (h_b + d_sf) / (d_sf * (2 * h_b + d_sf))) + (2 * RSLR / s_sf)
    x_s_dt = 2 * (Qow + Qbe + Qat) / ((2 * h_b) + d_sf) - (4 * Qsf * (h_b + d_sf) / (((2 * h_b) + d_sf) ** 2))  # Dune growth and alongshore transport added to LTA14 formulation

    # Record changes
    x_t = x_t + x_t_dt
    x_s = x_s + x_s_dt

    return x_s, x_t  # [m]


def ocean_shoreline(topof, MHW):
    """Returns location of the ocean shoreline.

    Parameters
    ----------
    topof : ndarray
        [m] Present elevation domain.
    MHW : float
        [slabs] Mean high water elevation.

    Returns
    ----------
    shoreline : ndarray
        Cross-shore location of the ocean shoreline for each row alongshore.
    """

    shoreline = np.argmax(topof >= MHW, axis=1)

    return shoreline


def backbarrier_shoreline(topof, MHW):
    """Returns location of the back-barrier shoreline.

    Parameters
    ----------
    topof : ndarray
        [m] Present elevation domain.
    MHW : float
        [slabs] Mean high water elevation.

    Returns
    ----------
    BBshoreline : ndarray
        Cross-shore location of the back-barrier shoreline for each row alongshore.
    """

    BBshoreline = topof.shape[1] - np.argmax(np.flip(topof) >= MHW, axis=1) - 1

    return BBshoreline


def foredune_crest(topo, eqtopo, veg):
    """Finds and returns the location of the foredune crest for each grid column alongshore."""

    # Step 1: Find locations of nth percentile of all vegetated cells
    crestline = np.zeros([len(topo)])
    buff1 = 50
    percentile = 0.98
    veg = veg > 0
    Lveg_longshore = np.zeros(len(topo))
    for r in range(len(topo)):
        veg_profile = veg[r, :]
        sum_veg = np.sum(veg_profile)
        target = sum_veg * (1 - percentile)
        CumVeg = np.cumsum(veg_profile) * veg_profile
        Lveg = np.argmax(CumVeg >= target)
        Lveg_longshore[r] = Lveg
        Lveg_longshore[Lveg_longshore == len(topo)] = float('NaN')

    # Step 2: Apply smoothening to Lveg
    Lveg_longshore = np.round(scipy.signal.savgol_filter(Lveg_longshore, 25, 1)).astype(int)

    # Step 3: Find peaks within buffer of location of smoothed Lveg
    for r in range(len(topo)):
        if Lveg_longshore[r] > 0:
            mini = max(0, Lveg_longshore[r] - buff1)
            maxi = min(topo.shape[1] - 1, Lveg_longshore[r] + buff1)
            loc = find_peak(topo[r, mini: maxi] * 0.1, 0, 0.6, 0.1)  # TODO: Replace second input with MHW
            if np.isnan(loc):
                crestline[r] = crestline[r - 1]
            else:
                crestline[r] = int(mini + loc)
        else:
            crestline[r] = int(np.argmax(eqtopo[r, :]))
        # TODO: Set secondary for cases in which pre-existing dunes or vegetation are minimal/nonexistant

    # Step 4: Apply smoothening to crestline
    crestline = np.round(scipy.signal.savgol_filter(crestline, 25, 1)).astype(int)

    # Step 5: Relocate peaks within smaller buffer of smoothened crestline
    buff2 = 5
    for r in range(len(topo)):
        mini = int(crestline[r] - buff2)
        maxi = int(crestline[r] + buff2)
        try:
            loc = np.argmax(topo[r, mini: maxi])
        except:
            raise Exception("Crestline Error")
        if np.isnan(loc):
            crestline[r] = crestline[r - 1]
        else:
            crestline[r] = int(mini + loc)

    return crestline.astype(int)


def foredune_heel(topof, crestline, threshold, slabheight_m):
    """Finds and returns the location of the foredune heel for each grid column alongshore."""

    longshore, crossshore = topof.shape
    heelline = np.zeros([longshore])

    for ls in range(longshore):
        # Loop landward from the crest
        idx = crestline[ls]
        while idx < crossshore:

            # Check the elevation difference
            elevation_difference = topof[ls, crestline[ls]] - topof[ls, idx]
            if idx + 1 >= crossshore:
                break
            elif topof[ls, idx + 1] < topof[ls, idx]:
                idx += 1
            elif elevation_difference >= (threshold / slabheight_m):  # Convert threshold to slabs
                break
            elif topof[ls, idx + 1] * 0.95 >= topof[ls, idx]:
                break
            else:
                idx += 1

        heelline[ls] = idx

    return heelline.astype(int)


def find_peak(profile, MHW, threshold, crest_pct):
    """Finds foredune peak of profile following Automorph (Itzkin et al., 2021). Returns Nan if no dune peak found on profile."""

    # Find peaks on the profile. The indices increase landwards
    pks_idx, _ = scipy.signal.find_peaks(profile)

    # Remove peaks below MHW
    if len(pks_idx) > 0:
        pks_idx = pks_idx[profile[pks_idx] > MHW]

    # If there aren't any peaks just take the maximum value
    if len(pks_idx) == 0:
        idx = np.argmax(profile)

    else:

        peak_found = False

        # Loop through the peaks
        for idx in pks_idx:
            backshore_drop = 0

            # Set the current peak elevation
            curr_elevation = profile[idx]

            # Loop landwards across the profile
            check_idx = idx
            while check_idx < len(profile):

                # Check that the next landward point is lower
                if check_idx + 1 >= len(profile):
                    break
                elif profile[check_idx + 1] > profile[idx]:
                    break
                else:

                    # Check the elevation distance
                    check_elevation = profile[check_idx + 1]
                    backshore_drop = curr_elevation - check_elevation
                    if backshore_drop >= threshold:
                        peak_found = True
                        break
                    else:
                        check_idx += 1

            if backshore_drop >= threshold:

                # Check the seaward peaks
                lo = profile[idx] * (1 - crest_pct)
                pks_idx = pks_idx[profile[pks_idx] > lo]
                if len(pks_idx) > 0:
                    idx = pks_idx[0]
                break

        # If there aren't any peaks with backshore drop past threshold, set to NaN
        if not peak_found:
            idx = float('NaN')

    return idx


def beach_slopes(eqtopo, MHW, crestline, slabheight_m):
    """Finds beach slope based on EQ beach profile for each cell alongshore, from MHW to location of dune crest.

    Parameters
    ----------
    eqtopo : ndarray
        [slabs] Current equilibrium beach topography.
    MHW : float
        [slabs] Current mean high water level.
    crestline : ndarray
        Location of the foredune crest for each cell alongshore
    slabheight_m : float
        [m] Height of slabs.

    Returns
    ----------
    slopes
        Array of beach slopes for each cell alongshore
    """

    longshore = eqtopo.shape[0]
    slopes = np.zeros([longshore])

    # Derive gradient of eqtopo as an approximation of foreshore slope
    for ls in range(longshore):
        loc_upper_slope = crestline[ls]
        loc_lower_slope = np.argmax(eqtopo[ls, :] >= MHW)
        slopes[ls] = np.nanmean(np.gradient(eqtopo[ls, loc_lower_slope: loc_upper_slope] * slabheight_m))

    return slopes


def stochastic_storm(pstorm, iteration, storm_list, beach_slope, RNG):
    """Stochastically determines whether a storm occurs for this timestep, and, if so, stochastically determines the relevant characteristics of the storm (i.e., water levels, duration).

    Parameters
    ----------
    pstorm : ndarray
        Empirical probability of storm occuring for each iteration of the year.
    iteration : int
        Present iteration for the year.
    storm_list : ndarray
        List of synthetic storms (rows), with wave and tide statistics (columns) desccribing each storm.
    beach_slope : float
        Beach slope from MHW to dune toe.
    RNG :
        Random number generator, either seeded or unseeded

    Returns
    ----------
    storm
        Bool whether or not storm occurs this time step.
    Rhigh
        [m MHW] Highest elevation of the landward margin of runup (i.e. total water level).
    Rlow
        [m MHW] Lowest elevation of the landward margin of runup.
    dur
        [hrs] Duration of storm.
    """

    # Determine if storm will occur this iteration
    storm = RNG.random() < pstorm[iteration]

    if storm:
        # Randomly select storm from list of synthetic storms
        n = RNG.integers(0, len(storm_list))  # Randomly selected storm
        Hs = storm_list[n, 0]  # Significant wave height
        dur = storm_list[n, 1]  # Duration
        NTR = storm_list[n, 3]  # Non-tidal residual
        Tp = storm_list[n, 4]  # Wave period
        AT = storm_list[n, 5]  # Tidal amplitude

        # Calculate simulated R2% and add to SL to get the simulated TWL
        L0 = (9.8 * Tp ** 2) / (2 * math.pi)  # Wavelength
        Setup = 0.35 * beach_slope * np.sqrt(Hs * L0)
        Sin = 0.75 * beach_slope * np.sqrt(Hs * L0)  # Incident band swash
        Sig = 0.06 * math.sqrt(Hs * L0)  # Infragravity band swash
        Swash = np.sqrt((Sin ** 2) + (Sig ** 2))  # Total swash
        R2 = 1.1 * (Setup + (Swash / 2))  # R2%

        # Calculate storm water levels
        Rhigh = NTR + R2 + AT
        Rlow = (Rhigh - (Swash / 2))

    else:
        Rhigh = 0
        Rlow = 0
        dur = 0

    return storm, Rhigh, Rlow, dur


def overwash_processes(
        topof,
        topof_change_remainder,
        crestline,
        Rhigh,
        Rlow,
        dur,
        slabheight_m,
        threshold_in,
        Rin_i,
        Rin_r,
        Cx,
        AvgSlope,
        nn,
        MaxUpSlope,
        Qs_min,
        Kr,
        Ki,
        mm,
        MHW,
        Cbb_i,
        Cbb_r,
        Qs_bb_min,
        substep_i,
        substep_r,
):
    """Overwashes barrier interior where storm water levels exceed pre-storm dune crests.

    Parameters
    ----------
    topof : ndarray
        [slabs] Current elevation domain.
    topof_change_remainder : ndarray
        [slabs] Portion of elevation change unused in previous time step (i.e., partial slabs).
    crestline : ndarray
        Cross-shore location of foredune crest cells.
    Rhigh : ndarray
        [m MHW] Highest elevation of the landward margin of runup (i.e. total water level).
    Rlow : ndarray
        [m MHW] Lowest elevation of the landward margin of runup.
    dur: ndarray
        [hrs] Duration of storm.
    slabheight_m : float
        [m] Height of slabs.
    threshold_in : float
        [%] Threshold percentage of overtopped dune cells needed to be in inundation overwash regime.
    Rin_i : float
        [m^3/hr] Flow infiltration and drag parameter, inundation overwash regime.
    Rin_r : float
        [m^3/hr] Flow infiltration and drag parameter, run-up overwash regime.
    Cx : float
        Constant for representing flow momentum for sediment transport in inundation overwash regime.
    AvgSlope : float
        Average slope of the barrier interior; invariable.
    nn : float
        Flow routing constant.
    MaxUpSlope : float
        Maximum slope water can flow uphill.
    Qs_min : float
        [m^3/hr] Minimum discharge out of cell needed to transport sediment.
    Kr : float
        Sediment transport coefficient for run-up overwash regime.
    Ki : float
        Sediment transport coefficient for inundation overwash regime.
    mm : float
        Inundation overwash constant.
    MHW : float
        [slabs] Mean high water.
    Cbb_i : float
        [%] Coefficient for exponential decay of sediment load entering back-barrier bay, inundation regime.
    Cbb_r : float
        [%] Coefficient for exponential decay of sediment load entering back-barrier bay, run-up regime.
    Qs_bb_min : float
        [m^3/hr] Minimum discharge out of subaqueous back-barrier cell needed to transport sediment.
    substep_i : int
        Number of substeps to run for each hour in inundation overwash regime (e.g., 3 substeps means discharge/elevation updated every 20 minutes)
    substep_r : int
        Number of substeps to run for each hour in run-up overwash regime (e.g., 3 substeps means discharge/elevation updated every 20 minutes)

    Returns
    ----------
    topof
        [slabs] Updated elevation domain.
    topof_change_effective
        [slabs] Array of topographic change from storm overwash processes, in units of full slabs.
    topof_change_remainder
        [slabs] Array of portion of topographic change leftover from conversion to slab units.
    OWloss
        [m^3] Volume of overwash deposition landward of dune crest for each cell unit alongshore.
    """

    longshore, crossshore = topof.shape
    topof_prestorm = copy.deepcopy(topof)
    dune_crest_loc_prestorm = crestline  # Cross-shore location of pre-storm dune crest
    dune_crest_height_prestorm_m = topof_prestorm[np.arange(len(topof_prestorm)), dune_crest_loc_prestorm] * slabheight_m  # [m] Height of dune crest alongshore
    MHW_m = MHW * slabheight_m  # Convert from slabs to meters

    # --------------------------------------
    # DUNE EROSION

    # --------------------------------------
    # OVERWASH

    OWloss = np.zeros([longshore])  # [m^3] Initialize aggreagate volume of overwash deposition landward of dune crest for this storm

    # Find overwashed dunes and gaps
    inundation_regime = Rlow > dune_crest_height_prestorm_m  # [bool] Identifies rows alongshore where dunes crest is overwashed in inundation regime
    runup_regime = np.logical_and(Rlow <= dune_crest_height_prestorm_m, Rhigh > dune_crest_height_prestorm_m)  # [bool] Identifies rows alongshore where dunes crest is overwashed in run-up regime
    overwash = np.logical_or(inundation_regime, runup_regime)  # [bool] Identifies rows alongshore where dunes crest is overwashed
    inundation_regime_count = np.count_nonzero(inundation_regime)
    runup_regime_count = np.count_nonzero(runup_regime)

    if inundation_regime_count > 0 or runup_regime_count > 0:  # Determine if there is any overwash for this storm

        # Calculate discharge through each dune cell
        Rexcess = (Rhigh - dune_crest_height_prestorm_m) * overwash  # [m] Height of storm water level above dune crest cells
        Vdune = np.sqrt(2 * 9.8 * Rexcess)  # [m/s] Velocity of water over each dune crest cell (Larson et al., 2004)
        Qdune = Vdune * Rexcess * 3600  # [m^3/hr] Discharge at each overtopped dune crest cell

        # Determine Sediment And Water Routing Rules Based on Overwash Regime
        if inundation_regime_count / (inundation_regime_count + runup_regime_count) >= threshold_in:  # If greater than threshold % of overtopped dune cells are inunundation regime -> inundation overwash regime
            inundation = True
            substep = substep_i
            Rin = Rin_i
            C = Cx * AvgSlope  # Momentum constant
            print("  INUNDATION OVERWASH")
        else:  # Run-up overwash regime
            inundation = False
            substep = substep_r
            Rin = Rin_r
            C = Cx * AvgSlope  # Momentum constant
            print("  RUN-UP OVERWASH")

        # Modify based on number of substeps
        fluxLimit = 1 / substep  # [m/hr] Maximum elevation change during one storm hour allowed
        Qs_min /= substep
        Qs_bb_min /= substep

        # Set Domain
        iterations = int(math.floor(dur) * substep)
        domain_width_start = int(np.min(dune_crest_loc_prestorm))  # Remove topo seaward of first dune crest cell from overwash routing domain
        domain_width_end = int(crossshore)  # + bay_routing_width  # [m]
        domain_width = domain_width_end - domain_width_start
        Elevation = np.zeros([iterations, longshore, domain_width])
        # Bay = np.ones([bay_routing_width, longshore]) * -BayDepth
        domain_topo_start = (topof[:, domain_width_start:] + topof_change_remainder[:, domain_width_start:]) * slabheight_m  # [m] Incorporate leftover topochange from PREVIOUS storm)
        Elevation[0, :, :] = domain_topo_start  # np.vstack([Dunes, self._InteriorDomain, Bay])

        # Initialize Memory Storage Arrays
        Discharge = np.zeros([iterations, longshore, domain_width])
        SedFluxIn = np.zeros([iterations, longshore, domain_width])
        SedFluxOut = np.zeros([iterations, longshore, domain_width])

        # Set Discharge at Dune Crest
        Discharge[:, np.arange(longshore), dune_crest_loc_prestorm - domain_width_start] = Qdune

        # Run Flow Routing Algorithm
        for TS in range(iterations):

            if TS > 0:
                Elevation[TS, :, :] = Elevation[TS - 1, :, :]  # Begin timestep with elevation from end of last

            Rin_eff = 1

            for d in range(domain_width - 1):
                # Reduce discharge across row via infiltration
                if d > 0:
                    Discharge[TS, :, d][Discharge[TS, :, d] > 0] -= Rin  # Constant Rin, old method
                    # Rin_r = 0.075   # [1/m] Logistic growth rate
                    # Rin_eff += Rin_r * Rin_eff * (1 - (Rin_eff / Rin))  # Increase Rin logistically with distance across domain
                    # Discharge[TS, :, d][Discharge[TS, :, d] > 0] -= Rin_eff  # New logistic growth of Rin parameter towards maximum, limits deposition where discharge is introduced
                Discharge[TS, :, d][Discharge[TS, :, d] < 0] = 0

                for i in range(longshore):
                    if Discharge[TS, i, d] > 0:

                        Q0 = Discharge[TS, i, d]

                        # Calculate Slopes
                        if i > 0:
                            S1 = (Elevation[TS, i, d] - Elevation[TS, i - 1, d + 1]) / (math.sqrt(2))
                            S1 = np.nan_to_num(S1)
                        else:
                            S1 = float('NaN')

                        S2 = Elevation[TS, i, d] - Elevation[TS, i, d + 1]
                        S2 = np.nan_to_num(S2)

                        if i < (longshore - 1):
                            S3 = (Elevation[TS, i, d] - Elevation[TS, i + 1, d + 1]) / (math.sqrt(2))
                            S3 = np.nan_to_num(S3)
                        else:
                            S3 = float('NaN')

                        # Calculate Discharge To Downflow Neighbors

                        # One or more slopes positive
                        if S1 > 0 or S2 > 0 or S3 > 0:

                            S1e = np.nan_to_num(S1)
                            S2e = np.nan_to_num(S2)
                            S3e = np.nan_to_num(S3)

                            if S1e < 0:
                                S1e = 0
                            if S2e < 0:
                                S2e = 0
                            if S3e < 0:
                                S3e = 0

                            Q1 = (Q0 * S1e ** nn / (S1e ** nn + S2e ** nn + S3e ** nn))
                            Q2 = (Q0 * S2e ** nn / (S1e ** nn + S2e ** nn + S3e ** nn))
                            Q3 = (Q0 * S3e ** nn / (S1e ** nn + S2e ** nn + S3e ** nn))

                            Q1 = np.nan_to_num(Q1)
                            Q2 = np.nan_to_num(Q2)
                            Q3 = np.nan_to_num(Q3)

                        # No slopes positive, one or more equal to zero
                        elif S1 == 0 or S2 == 0 or S3 == 0:

                            pos = 0
                            if S1 == 0:
                                pos += 1
                            if S2 == 0:
                                pos += 1
                            if S3 == 0:
                                pos += 1

                            Qx = Q0 / pos
                            Qx = np.nan_to_num(Qx)

                            if S1 == 0 and i > 0:
                                Q1 = Qx
                            else:
                                Q1 = 0
                            if S2 == 0:
                                Q2 = Qx
                            else:
                                Q2 = 0
                            if S3 == 0 and i < (longshore - 1):
                                Q3 = Qx
                            else:
                                Q3 = 0

                        # All slopes negative
                        else:

                            if np.isnan(S1):
                                Q1 = 0
                                Q2 = (Q0 * abs(S2) ** (-nn) / (abs(S2) ** (-nn) + abs(S3) ** (-nn)))
                                Q3 = (Q0 * abs(S3) ** (-nn) / (abs(S2) ** (-nn) + abs(S3) ** (-nn)))
                            elif np.isnan(S3):
                                Q1 = (Q0 * abs(S1) ** (-nn) / (abs(S1) ** (-nn) + abs(S2) ** (-nn)))
                                Q2 = (Q0 * abs(S2) ** (-nn) / (abs(S1) ** (-nn) + abs(S2) ** (-nn)))
                                Q3 = 0
                            else:
                                Q1 = (Q0 * abs(S1) ** (-nn) / (abs(S1) ** (-nn) + abs(S2) ** (-nn) + abs(S3) ** (-nn)))
                                Q2 = (Q0 * abs(S2) ** (-nn) / (abs(S1) ** (-nn) + abs(S2) ** (-nn) + abs(S3) ** (-nn)))
                                Q3 = (Q0 * abs(S3) ** (-nn) / (abs(S1) ** (-nn) + abs(S2) ** (-nn) + abs(S3) ** (-nn)))

                            Q1 = np.nan_to_num(Q1)
                            Q2 = np.nan_to_num(Q2)
                            Q3 = np.nan_to_num(Q3)

                            if abs(S1) > MaxUpSlope:
                                Q1 = 0
                            else:
                                Q1 = Q1 * (1 - (abs(S1) / MaxUpSlope))

                            if abs(S2) > MaxUpSlope:
                                Q2 = 0
                            else:
                                Q2 = Q2 * (1 - (abs(S2) / MaxUpSlope))

                            if abs(S3) > MaxUpSlope:
                                Q3 = 0
                            else:
                                Q3 = Q3 * (1 - (abs(S3) / MaxUpSlope))

                        # Save Discharge
                        # if Shrubs:  # Reduce Overwash Through Shrub Cells
                        #     # Insert shrub effects here
                        # else:
                        # Cell 1
                        if i > 0:
                            Discharge[TS, i - 1, d + 1] += Q1

                        # Cell 2
                        Discharge[TS, i, d + 1] += Q2

                        # Cell 3
                        if i < (longshore - 1):
                            Discharge[TS, i + 1, d + 1] += Q3

                        # Calculate Sed Movement

                        # Run-up Regime
                        if not inundation:
                            if Q1 > Qs_min:  # and S1 >= 0:
                                Qs1 = max(0, Kr * Q1)
                                # Qs1 = max(0, Kr * Q1 * (S1 + C))
                            else:
                                Qs1 = 0

                            if Q2 > Qs_min:  # and S2 >= 0:
                                Qs2 = max(0, Kr * Q2)
                                # Qs2 = max(0, Kr * Q2 * (S2 + C))
                            else:
                                Qs2 = 0

                            if Q3 > Qs_min:  # and S3 >= 0:
                                Qs3 = max(0, Kr * Q3)
                                # Qs3 = max(0, Kr * Q3 * (S3 + C))
                            else:
                                Qs3 = 0

                        # Inundation Regime - Murray and Paola (1994, 1997) Rule 3 with flux limiter
                        else:
                            if Q1 > Qs_min:
                                Qs1 = Ki * (Q1 * (S1 + C)) ** mm
                                if Qs1 < 0:
                                    Qs1 = 0
                            else:
                                Qs1 = 0

                            if Q2 > Qs_min:
                                Qs2 = Ki * (Q2 * (S2 + C)) ** mm
                                if Qs2 < 0:
                                    Qs2 = 0
                            else:
                                Qs2 = 0

                            if Q3 > Qs_min:
                                Qs3 = Ki * (Q3 * (S3 + C)) ** mm
                                if Qs3 < 0:
                                    Qs3 = 0
                            else:
                                Qs3 = 0

                        Qs1 = np.nan_to_num(Qs1)
                        Qs2 = np.nan_to_num(Qs2)
                        Qs3 = np.nan_to_num(Qs3)

                        # Calculate Net Erosion/Accretion
                        if Elevation[TS, i, d] > MHW_m or any(z > MHW_m for z in Elevation[TS, i, d + 1: d + 10]):  # If cell is subaerial, elevation change is determined by difference between flux in vs. flux out
                            if i > 0:
                                SedFluxIn[TS, i - 1, d + 1] += Qs1

                            SedFluxIn[TS, i, d + 1] += Qs2

                            if i < (longshore - 1):
                                SedFluxIn[TS, i + 1, d + 1] += Qs3

                            Qs_out = Qs1 + Qs2 + Qs3
                            SedFluxOut[TS, i, d] = Qs_out

                        else:  # If cell is subaqeous, exponentially decay dep. of remaining sed across bay

                            if inundation:
                                Cbb = Cbb_r
                            else:
                                Cbb = Cbb_i

                            Qs0 = SedFluxIn[TS, i, d] * Cbb

                            Qs1 = Qs0 * Q1 / (Q1 + Q2 + Q3)
                            Qs2 = Qs0 * Q2 / (Q1 + Q2 + Q3)
                            Qs3 = Qs0 * Q3 / (Q1 + Q2 + Q3)

                            Qs1 = np.nan_to_num(Qs1)
                            Qs2 = np.nan_to_num(Qs2)
                            Qs3 = np.nan_to_num(Qs3)

                            if Qs1 < Qs_bb_min:
                                Qs1 = 0
                            if Qs2 < Qs_bb_min:
                                Qs2 = 0
                            if Qs3 < Qs_bb_min:
                                Qs3 = 0

                            if i > 0:
                                SedFluxIn[TS, i - 1, d + 1] += Qs1

                            SedFluxIn[TS, i, d + 1] += Qs2

                            if i < (longshore - 1):
                                SedFluxIn[TS, i + 1, d + 1] += Qs3

                            Qs_out = Qs1 + Qs2 + Qs3
                            SedFluxOut[TS, i, d] = Qs_out

                        # # Shrub Saline Flooding
                        # if Shrubs:
                        #     # Insert shrub saline flooding here

            # Update Elevation After Every Storm Hour
            ElevationChange = (SedFluxIn[TS, :, :] - SedFluxOut[TS, :, :]) / substep
            ElevationChange[ElevationChange > fluxLimit] = fluxLimit
            ElevationChange[ElevationChange < -fluxLimit] = -fluxLimit
            ElevationChange[np.arange(longshore), dune_crest_loc_prestorm - domain_width_start] = 0  # Do not update elevation change at dune crest cell where discharge was introduced
            Elevation[TS, :, :] = Elevation[TS, :, :] + ElevationChange

            # Calculate and save volume of sediment deposited on/behind the island for every hour
            OWloss = OWloss + np.sum(ElevationChange, axis=1)  # [m^3] For each cell alongshore

            # # Shrub Burial/Erosion
            # if Shrubs:
            #     # Insert calculation of burial/erosion for each shrub

        # Update Interior Domain After Storm
        domain_topo_change_m = Elevation[-1, :, :] - domain_topo_start  # [m] Change in elevation of routing domain

        # Update interior domain
        topof_change = np.hstack((np.zeros([longshore, domain_width_start + 1]), domain_topo_change_m[:, 1:])) / slabheight_m  # [slabs] Add back in beach cells (zero topo change) and convert to units of slabs
        topof_change_effective = np.zeros(topof_change.shape)
        topof_change_effective[topof_change >= 0] = np.floor(topof_change[topof_change >= 0])  # Round to whole slab unit
        topof_change_effective[topof_change < 0] = np.ceil(topof_change[topof_change < 0])  # Round to whole slab unit
        topof_change_remainder = topof_change - topof_change_effective  # [slabs] Portion of elevation change unused this time step because can't have partial slabs; will be incorporated at the start of next storm (see above)
        # topof += topof_change_effective  # [slabs] Rounded to nearest slab (i.e., 0.1 m)
        topof += topof_change  # [slabs] Not rounded
        topof_change_remainder *= 0  # Temp! Is remainder no longer needed?

        netDischarge = np.hstack((np.zeros([longshore, domain_width_start]), np.sum(Discharge, axis=0)))

    else:
        topof_change_effective = np.zeros(topof.shape)  # No topo change if no overwash
        netDischarge = np.zeros(topof.shape)

    return topof, topof_change_effective, topof_change_remainder, OWloss, netDischarge


def storm_processes(
        topof,
        topof_change_remainder,
        eqtopof,
        vegf,
        Rhigh,
        Rlow,
        dur,
        slabheight_m,
        threshold_in,
        Rin_i,
        Rin_r,
        Cx,
        AvgSlope,
        nn,
        MaxUpSlope,
        Qs_min,
        Kr,
        Ki,
        mm,
        MHW,
        Cbb_i,
        Cbb_r,
        Qs_bb_min,
        substep_i,
        substep_r,
):
    """Resolves topographical change from storm events. Landward of dune crest: overwashes barrier interior where storm water levels exceed
    pre-storm dune crests following Barrier3D (Reeves et al., 2021) flow routing. Seaward of dune crest: determines topographic change of beach
    and dune face following the Coastal Dune Model (v2.0; Duran Vinent & Moore, 2015).

    Parameters
    ----------
    topof : ndarray
        [slabs] Current elevation domain.
    topof_change_remainder : ndarray
        [slabs] Portion of elevation change unused in previous time step (i.e., partial slabs).
    eqtopof : ndarray
        [slabs] Equilibrium beach topography map.
    vegf : ndarray
        [0-1] Map of combined vegetation effectiveness
    Rhigh : ndarray
        [m MHW] Highest elevation of the landward margin of runup (i.e. total water level).
    Rlow : ndarray
        [m MHW] Lowest elevation of the landward margin of runup.
    dur: ndarray
        [hrs] Duration of storm.
    slabheight_m : float
        [m] Height of slabs.
    threshold_in : float
        [%] Threshold percentage of overtopped dune cells needed to be in inundation overwash regime.
    Rin_i : float
        [m^3/hr] Flow infiltration and drag parameter, inundation overwash regime.
    Rin_r : float
        [m^3/hr] Flow infiltration and drag parameter, run-up overwash regime.
    Cx : float
        Constant for representing flow momentum for sediment transport in inundation overwash regime.
    AvgSlope : float
        Average slope of the barrier interior; invariable.
    nn : float
        Flow routing constant.
    MaxUpSlope : float
        Maximum slope water can flow uphill.
    Qs_min : float
        [m^3/hr] Minimum discharge out of cell needed to transport sediment.
    Kr : float
        Sediment transport coefficient for run-up overwash regime.
    Ki : float
        Sediment transport coefficient for inundation overwash regime.
    mm : float
        Inundation overwash constant.
    MHW : float
        [slabs] Mean high water.
    Cbb_i : float
        [%] Coefficient for exponential decay of sediment load entering back-barrier bay, inundation regime.
    Cbb_r : float
        [%] Coefficient for exponential decay of sediment load entering back-barrier bay, run-up regime.
    Qs_bb_min : float
        [m^3/hr] Minimum discharge out of subaqueous back-barrier cell needed to transport sediment.
    substep_i : int
        Number of substeps to run for each hour in inundation overwash regime (e.g., 3 substeps means discharge/elevation updated every 20 minutes)
    substep_r : int
        Number of substeps to run for each hour in run-up overwash regime (e.g., 3 substeps means discharge/elevation updated every 20 minutes)

    Returns
    ----------
    topof
        [slabs] Updated elevation domain.
    topof_change_effective
        [slabs] Array of topographic change from storm overwash processes, in units of full slabs.
    topof_change_remainder
        [slabs] Array of portion of topographic change leftover from conversion to slab units.
    OWloss
        [m^3] Volume of overwash deposition landward of dune crest for each cell unit alongshore.
    netDischarge
        [m^3] Map of discharge aggregated for duration of entire storm.
    """

    longshore, crossshore = topof.shape
    topof_prestorm = copy.deepcopy(topof)
    dune_crest_loc = foredune_crest(topof, eqtopof, vegf)  # Cross-shore location of pre-storm dune crest
    dune_crest_height_prestorm_m = topof_prestorm[np.arange(len(topof_prestorm)), dune_crest_loc] * slabheight_m  # [m] Height of dune crest alongshore
    MHW_m = MHW * slabheight_m  # Convert from slabs to meters

    # --------------------------------------
    # OVERWASH

    iterations = int(math.floor(dur) * substep_r)

    # Set Up Flow Routing Domain
    domain_width_start = 0
    domain_width_end = int(crossshore)  # [m]
    domain_width = domain_width_end - domain_width_start
    Elevation = np.zeros([iterations, longshore, domain_width])
    domain_topo_start = (topof[:, domain_width_start:] + topof_change_remainder[:, domain_width_start:]) * slabheight_m  # [m] Incorporate leftover topochange from PREVIOUS storm)
    Elevation[0, :, :] = domain_topo_start  # TODO: Allow for open boundary conditions?

    # Initialize Memory Storage Arrays
    Discharge = np.zeros([iterations, longshore, domain_width])
    SedFluxIn = np.zeros([iterations, longshore, domain_width])
    SedFluxOut = np.zeros([iterations, longshore, domain_width])
    OWloss = np.zeros([longshore])  # [m^3] Aggreagate volume of overwash deposition landward of dune crest for this storm

    # Run Storm
    for TS in range(iterations):

        # Find dune crest locations and heights for this storm iteration
        dune_crest_loc = foredune_crest(topof, eqtopof, vegf)  # Cross-shore location of pre-storm dune crest
        # dune_crest_loc[75: 128] = 153  # for 1685-1885 TEMP!!!

        dune_crest_height_m = topof_prestorm[np.arange(len(topof_prestorm)), dune_crest_loc] * slabheight_m  # [m] Height of dune crest alongshore
        overwash = Rhigh > dune_crest_height_m  # [bool] Identifies rows alongshore where dunes crest is overwashed

        if any(overwash):  # Determine if there is any overwash for this storm

            # Calculate discharge through each dune cell for this storm iteration
            Rexcess = (Rhigh - dune_crest_height_m) * overwash  # [m] Height of storm water level above dune crest cells
            Vdune = np.sqrt(2 * 9.8 * Rexcess)  # [m/s] Velocity of water over each dune crest cell (Larson et al., 2004)
            Qdune = Vdune * Rexcess * 3600  # [m^3/hr] Discharge at each overtopped dune crest cell

            # Determine Sediment And Water Routing Rules Based on Overwash Regime
            inundation_regime = Rlow > dune_crest_height_prestorm_m  # [bool] Identifies rows alongshore where dunes crest is overwashed in inundation regime
            runup_regime = np.logical_and(Rlow <= dune_crest_height_prestorm_m, Rhigh > dune_crest_height_prestorm_m)  # [bool] Identifies rows alongshore where dunes crest is overwashed in run-up regime
            inundation_regime_count = np.count_nonzero(inundation_regime)
            runup_regime_count = np.count_nonzero(runup_regime)
            if inundation_regime_count / (inundation_regime_count + runup_regime_count) >= threshold_in:  # If greater than threshold % of overtopped dune cells are inunundation regime -> inundation overwash regime
                inundation = True  # TODO: Inundation regime parameterization needs work...
                substep = substep_r  # Temp keep substeps same between regimes
                Rin = Rin_i
                C = Cx * AvgSlope  # Momentum constant
                print("  INUNDATION OVERWASH")
            else:  # Run-up overwash regime
                inundation = False
                substep = substep_r  # Temp keep substeps same between regimes
                Rin = Rin_r
                C = Cx * AvgSlope  # Momentum constant
                # print("  RUN-UP OVERWASH")

            # Modify based on number of substeps
            fluxLimit = 1 / substep  # [m/hr] Maximum elevation change during one storm hour allowed
            Qs_min /= substep
            Qs_bb_min /= substep

            # Set Discharge at Dune Crest
            Discharge[TS, np.arange(longshore), dune_crest_loc - domain_width_start] = Qdune  # TODO: Vary Rhigh over storm duration

            # Begin timestep with elevation from end of last
            if TS > 0:
                Elevation[TS, :, :] = Elevation[TS - 1, :, :]

            Rin_eff = 1  # TEMP

            for d in range(domain_width - 1):

                # Reduce discharge across row via infiltration
                if d > 0:
                    Discharge[TS, :, d][Discharge[TS, :, d] > 0] -= Rin  # Constant Rin, old method
                    # Rin_r = 0.075   # [1/m] Logistic growth rate
                    # Rin_eff += Rin_r * Rin_eff * (1 - (Rin_eff / Rin))  # Increase Rin logistically with distance across domain
                    # Discharge[TS, :, d][Discharge[TS, :, d] > 0] -= Rin_eff  # New logistic growth of Rin parameter towards maximum, limits deposition where discharge is introduced
                Discharge[TS, :, d][Discharge[TS, :, d] < 0] = 0

                for i in range(longshore):
                    if Discharge[TS, i, d] > 0:

                        Q0 = Discharge[TS, i, d]

                        # Calculate Slopes
                        if i > 0:
                            S1 = (Elevation[TS, i, d] - Elevation[TS, i - 1, d + 1]) / (math.sqrt(2))
                            S1 = np.nan_to_num(S1)
                        else:
                            S1 = float('NaN')

                        S2 = Elevation[TS, i, d] - Elevation[TS, i, d + 1]
                        S2 = np.nan_to_num(S2)

                        if i < (longshore - 1):
                            S3 = (Elevation[TS, i, d] - Elevation[TS, i + 1, d + 1]) / (math.sqrt(2))
                            S3 = np.nan_to_num(S3)
                        else:
                            S3 = float('NaN')

                        # Calculate Discharge To Downflow Neighbors

                        # One or more slopes positive
                        if S1 > 0 or S2 > 0 or S3 > 0:

                            S1e = np.nan_to_num(S1)
                            S2e = np.nan_to_num(S2)
                            S3e = np.nan_to_num(S3)

                            if S1e < 0:
                                S1e = 0
                            if S2e < 0:
                                S2e = 0
                            if S3e < 0:
                                S3e = 0

                            Q1 = (Q0 * S1e ** nn / (S1e ** nn + S2e ** nn + S3e ** nn))
                            Q2 = (Q0 * S2e ** nn / (S1e ** nn + S2e ** nn + S3e ** nn))
                            Q3 = (Q0 * S3e ** nn / (S1e ** nn + S2e ** nn + S3e ** nn))

                            Q1 = np.nan_to_num(Q1)
                            Q2 = np.nan_to_num(Q2)
                            Q3 = np.nan_to_num(Q3)

                        # No slopes positive, one or more equal to zero
                        elif S1 == 0 or S2 == 0 or S3 == 0:

                            pos = 0
                            if S1 == 0:
                                pos += 1
                            if S2 == 0:
                                pos += 1
                            if S3 == 0:
                                pos += 1

                            Qx = Q0 / pos
                            Qx = np.nan_to_num(Qx)

                            if S1 == 0 and i > 0:
                                Q1 = Qx
                            else:
                                Q1 = 0
                            if S2 == 0:
                                Q2 = Qx
                            else:
                                Q2 = 0
                            if S3 == 0 and i < (longshore - 1):
                                Q3 = Qx
                            else:
                                Q3 = 0

                        # All slopes negative
                        else:

                            if np.isnan(S1):
                                Q1 = 0
                                Q2 = (Q0 * abs(S2) ** (-nn) / (abs(S2) ** (-nn) + abs(S3) ** (-nn)))
                                Q3 = (Q0 * abs(S3) ** (-nn) / (abs(S2) ** (-nn) + abs(S3) ** (-nn)))
                            elif np.isnan(S3):
                                Q1 = (Q0 * abs(S1) ** (-nn) / (abs(S1) ** (-nn) + abs(S2) ** (-nn)))
                                Q2 = (Q0 * abs(S2) ** (-nn) / (abs(S1) ** (-nn) + abs(S2) ** (-nn)))
                                Q3 = 0
                            else:
                                Q1 = (Q0 * abs(S1) ** (-nn) / (abs(S1) ** (-nn) + abs(S2) ** (-nn) + abs(S3) ** (-nn)))
                                Q2 = (Q0 * abs(S2) ** (-nn) / (abs(S1) ** (-nn) + abs(S2) ** (-nn) + abs(S3) ** (-nn)))
                                Q3 = (Q0 * abs(S3) ** (-nn) / (abs(S1) ** (-nn) + abs(S2) ** (-nn) + abs(S3) ** (-nn)))

                            Q1 = np.nan_to_num(Q1)
                            Q2 = np.nan_to_num(Q2)
                            Q3 = np.nan_to_num(Q3)

                            if abs(S1) > MaxUpSlope:
                                Q1 = 0
                            else:
                                Q1 = Q1 * (1 - (abs(S1) / MaxUpSlope))

                            if abs(S2) > MaxUpSlope:
                                Q2 = 0
                            else:
                                Q2 = Q2 * (1 - (abs(S2) / MaxUpSlope))

                            if abs(S3) > MaxUpSlope:
                                Q3 = 0
                            else:
                                Q3 = Q3 * (1 - (abs(S3) / MaxUpSlope))

                        # Save Discharge
                        # if Shrubs:  # Reduce Overwash Through Shrub Cells
                        #     # Insert shrub effects here
                        # else:
                        # Cell 1
                        if i > 0:
                            Discharge[TS, i - 1, d + 1] += Q1

                        # Cell 2
                        Discharge[TS, i, d + 1] += Q2

                        # Cell 3
                        if i < (longshore - 1):
                            Discharge[TS, i + 1, d + 1] += Q3

                        # Calculate Sed Movement

                        # Run-up Regime
                        if not inundation:
                            if Q1 > Qs_min:  # and S1 >= 0:
                                # Qs1 = max(0, Kr * Q1)
                                Qs1 = max(0, Kr * Q1 * (S1 + C))
                            else:
                                Qs1 = 0

                            if Q2 > Qs_min:  # and S2 >= 0:
                                # Qs2 = max(0, Kr * Q2)
                                Qs2 = max(0, Kr * Q2 * (S2 + C))
                            else:
                                Qs2 = 0

                            if Q3 > Qs_min:  # and S3 >= 0:
                                # Qs3 = max(0, Kr * Q3)
                                Qs3 = max(0, Kr * Q3 * (S3 + C))
                            else:
                                Qs3 = 0

                        # Inundation Regime - Murray and Paola (1994, 1997) Rule 3 with flux limiter
                        else:
                            if Q1 > Qs_min:
                                Qs1 = Ki * (Q1 * (S1 + C)) ** mm
                                if Qs1 < 0:
                                    Qs1 = 0
                            else:
                                Qs1 = 0

                            if Q2 > Qs_min:
                                Qs2 = Ki * (Q2 * (S2 + C)) ** mm
                                if Qs2 < 0:
                                    Qs2 = 0
                            else:
                                Qs2 = 0

                            if Q3 > Qs_min:
                                Qs3 = Ki * (Q3 * (S3 + C)) ** mm
                                if Qs3 < 0:
                                    Qs3 = 0
                            else:
                                Qs3 = 0

                        Qs1 = np.nan_to_num(Qs1)
                        Qs2 = np.nan_to_num(Qs2)
                        Qs3 = np.nan_to_num(Qs3)

                        # Calculate Net Erosion/Accretion
                        if Elevation[TS, i, d] > MHW_m or any(z > MHW_m for z in Elevation[TS, i, d + 1: d + 10]):  # If cell is subaerial, elevation change is determined by difference between flux in vs. flux out
                            if i > 0:
                                SedFluxIn[TS, i - 1, d + 1] += Qs1

                            SedFluxIn[TS, i, d + 1] += Qs2

                            if i < (longshore - 1):
                                SedFluxIn[TS, i + 1, d + 1] += Qs3

                            Qs_out = Qs1 + Qs2 + Qs3
                            SedFluxOut[TS, i, d] = Qs_out

                        else:  # If cell is subaqeous, exponentially decay dep. of remaining sed across bay

                            if inundation:
                                Cbb = Cbb_r
                            else:
                                Cbb = Cbb_i

                            Qs0 = SedFluxIn[TS, i, d] * Cbb

                            Qs1 = Qs0 * Q1 / (Q1 + Q2 + Q3)
                            Qs2 = Qs0 * Q2 / (Q1 + Q2 + Q3)
                            Qs3 = Qs0 * Q3 / (Q1 + Q2 + Q3)

                            Qs1 = np.nan_to_num(Qs1)
                            Qs2 = np.nan_to_num(Qs2)
                            Qs3 = np.nan_to_num(Qs3)

                            if Qs1 < Qs_bb_min:
                                Qs1 = 0
                            if Qs2 < Qs_bb_min:
                                Qs2 = 0
                            if Qs3 < Qs_bb_min:
                                Qs3 = 0

                            if i > 0:
                                SedFluxIn[TS, i - 1, d + 1] += Qs1

                            SedFluxIn[TS, i, d + 1] += Qs2

                            if i < (longshore - 1):
                                SedFluxIn[TS, i + 1, d + 1] += Qs3

                            Qs_out = Qs1 + Qs2 + Qs3
                            SedFluxOut[TS, i, d] = Qs_out

                        # # Shrub Saline Flooding
                        # if Shrubs:
                        #     # Insert shrub saline flooding here

        # Update Elevation After Every Storm Hour of Overwash
        ElevationChange = (SedFluxIn[TS, :, :] - SedFluxOut[TS, :, :]) / substep
        ElevationChange[ElevationChange > fluxLimit] = fluxLimit
        ElevationChange[ElevationChange < -fluxLimit] = -fluxLimit
        ElevationChange[np.arange(longshore), dune_crest_loc - domain_width_start] = 0  # Do not update elevation change at dune crest cell where discharge was introduced
        Elevation[TS, :, :] = Elevation[TS, :, :] + ElevationChange

        # Calculate and save volume of sediment deposited on/behind the island for every hour
        OWloss = OWloss + np.sum(ElevationChange, axis=1)  # [m^3] For each cell alongshore

        # # Shrub Burial/Erosion
        # if Shrubs:
        #     # Insert calculation of burial/erosion for each shrub

        # Beach and Duneface Change
        Elevation[TS, :, :], dV = calc_dune_erosion_TS(
            Elevation[TS, :, :],
            1,
            dune_crest_loc,
            MHW_m,
            Rhigh,
            1 / substep)

    # Update Interior Domain After Storm
    domain_topo_change_m = Elevation[-1, :, :] - domain_topo_start  # [m] Change in elevation of routing domain

    # Update interior domain
    topof_change = np.hstack((np.zeros([longshore, domain_width_start + 1]), domain_topo_change_m[:, 1:])) / slabheight_m  # [slabs] Add back in beach cells (zero topo change) and convert to units of slabs
    topof_change_effective = np.zeros(topof_change.shape)
    topof_change_effective[topof_change >= 0] = np.floor(topof_change[topof_change >= 0])  # Round to whole slab unit
    topof_change_effective[topof_change < 0] = np.ceil(topof_change[topof_change < 0])  # Round to whole slab unit
    topof_change_remainder = topof_change - topof_change_effective  # [slabs] Portion of elevation change unused this time step because can't have partial slabs; will be incorporated at the start of next storm (see above)
    # topof += topof_change_effective  # [slabs] Rounded to nearest slab (i.e., 0.1 m)
    topof += topof_change  # [slabs] Not rounded
    topof_change_remainder *= 0  # Temp! Is remainder no longer needed?

    netDischarge = np.hstack((np.zeros([longshore, domain_width_start]), np.sum(Discharge, axis=0)))

    return topof, topof_change_effective, topof_change_remainder, OWloss, netDischarge


def calc_dune_erosion_TS(topo,
                         dx,
                         crestline,
                         MHW,
                         Rhigh,
                         dT,
                         ):
    """Dune erosion model from CDM v2.0 (Duran Vinent & Moore, 2015). Returns updated topography for one storm iteration.

        Parameters
        ----------
        topo : ndarray
            [slabs] Elevation domain.
        dx : float
            [m] Cell dimension.
        crestline : ndarray
            Alongshore array of dune crest locations.
        MHW : float
            [slabs] Mean high water
        Rhigh : ndarray
            [m MHW] Highest elevation of the landward margin of runup (i.e. total water level).
        dT : float
            [hr] Time step length.

        Returns
        ----------
        topo
            Updated elevation domain.
        dV
            [m^3/m] Dune & beach volumetric change.
        """

    # TODO: Move these to main model inputs
    Beq = 0.02  # Equilibrium beach slope
    Tc = 1  # Timescale constant: larger Tc = lesser erosion
    # Time
    substeps = 10
    Q = dT / Tc  # Timescale

    # Initialize
    longshore, crossshore = topo.shape  # Domain dimensions
    x_s = ocean_shoreline(topo, MHW)
    fluxMap = np.zeros(topo.shape)  # Initialize map of sediment flux aggregated over duration of storm

    topoPrestorm = copy.copy(topo)

    # Loop through each substep of each iteration
    for step in range(substeps):

        # Loop through each cell alongshore
        for y in range(longshore):

            Rh = Rhigh  # Total water level
            xStart = x_s[y]  # Start
            xFinish = crestline[y] + 1

            cont = True

            # Loop through each cell in domain from ocean shoreline to back-barrier bay shoreline
            for x in range(xStart, xFinish):

                # Cell to operate on
                hi = topo[y, x]

                # Definition of boundary conditions
                if x == 0:
                    hprev = MHW
                else:
                    hprev = topo[y, x - 1]

                if x == xFinish - 1:
                    hnext = topo[y, xFinish]
                else:
                    hnext = topo[y, x + 1]

                if hi <= Rh < topo[y, x + 1]:
                    hnext = Rh
                    cont = False

                # Auxiliar
                Bl = 0.5 * (hnext - hprev) / dx  # Local topo gradient
                hxx = (hnext - 2 * hi + hprev) / dx / dx
                Rexcess = Rh - hi  # Height of water above elevation surface

                # Flux
                fluxMap[y, x] += (Beq - Bl) * Rexcess * Rexcess

                # Flux divergence
                divq = Rexcess * (hxx * Rexcess + 2 * (Beq - Bl) * Bl)

                # Evolve topography
                topo[y, x] += Q * divq / substeps

                # Break if next cell not inundated
                if not cont:
                    break

    topoChange = topo - topoPrestorm
    dV = np.sum(topoChange, axis=1) * dx ** 3  # [m^3] Dune/beach volume change: (-) loss, (+) gain

    return topo, dV