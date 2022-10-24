"""__________________________________________________________________________________________________________________________________

Main Model Script for PyDUBEVEG

Python version of the DUne, BEach, and VEGetation (DUBEVEG) model from Keijsers et al. (2016) and Galiforni Silva et al. (2018, 2019)

Translated from Matlab by IRB Reeves

Last update: 20 October 2022

__________________________________________________________________________________________________________________________________"""

import numpy as np
import math


def shadowzones2(topof, sh, lee, longshore, crossshore, direction):
    """Returns a logical map with all shadowzones identified as ones
    Wind from left to right, along the +2 dimension
    - topof: topography map [topo]
    - sh: slab height [slabheight]
    - lee: shadow angle [shadowangle]
    - longshore:
    - crossshore:
    - direction: wind direction (1 east, 2 north, 3, west, 4 south)
    Returns a logical map of shaded cells, now with open boundaries"""

    # steplimit = math.floor(math.tan(lee * math.pi / 180) / sh)  # The maximum step difference allowed given the slabheight and shadowangle
    steplimit = math.tan(lee * math.pi / 180) / sh  # The maximum step difference allowed given the slabheight and shadowangle, not rounded yet

    # range = min(crossshore, max(topof) / steplimit)
    search_range = int(math.floor(np.max(topof) / steplimit))  # Identifies highest elevation and uses that to determine what the largest search distance needs to be

    inshade = np.zeros([longshore, crossshore])  # Define the zeroed logical map

    for i in range(1, search_range + 1):
        if direction == 1:
            step = topof - np.roll(topof, [0, i])  # Shift across columns (2nd dimension; along a row)
            tempinshade = step < -math.floor(steplimit * i)  # Identify cells with too great a stepheight
            tempinshade[:, 0:i] = 0  # Part that is circshifted back into beginning of space is ignored
        elif direction == 2:
            step = topof - np.roll(topof, [i, 0])  # Shift across columns (2nd dimension; along a row)
            tempinshade = step < -math.floor(steplimit * i)  # Identify cells with too great a stepheight
            tempinshade[0:i, :] = 0  # Part that is circshifted back into beginning of space is ignored
        elif direction == 3:
            step = topof - np.roll(topof, [0, -i])  # Shift across columns (2nd dimension; along a row)
            tempinshade = step < -math.floor(steplimit * i)  # Identify cells with too great a stepheight
            tempinshade[:, -1 - i:-1] = 0  # Part that is circshifted back into beginning of space is ignored
        elif direction == 4:
            step = topof - np.roll(topof, [-i, 0])  # Shift across columns (2nd dimension; along a row)
            tempinshade = step < -math.floor(steplimit * i)  # Identify cells with too great a stepheight
            tempinshade[-1 - i:-1, :] = 0  # Part that is circshifted back into beginning of space is ignored
        inshade[tempinshade is True] = True  # Merge with previous inshade zones

    return inshade


def erosprobs2(vegf, shade, sand, topof, groundw, p_er):
    """% Returns a map with erosion probabilities
    - vegf: map of combined vegetation effectiveness [veg] [0,1]
    - shade: logical map of shadowzones [shadowmap]
    - sand: logical map of sandy cells [sandmap]
    - topof: topography map [topo]
    - groundw: groundwater map [gw]
    - p_er: probability of erosion of bare cell"""

    r = np.logical_not(shade) * sand * (1 - vegf) * (topof > groundw) * p_er

    return r


def depprobs(vegf, shade, sand, dep0, dep1):
    """Returns a map of deposition probabilities that can then be used to implement transport
    - vegf: map of combined vegetation effectiveness [veg] [0,1]
    - shade: logical map of shadowzones [shadowmap]
    - sand: logical map of sanded sites [sandmap]
    - dep0: deposition probability on bare cell [p_dep_bare]
    - dep1: deposition probability on sandy cell [p_dep_sand]"""

    # For bare cells
    temp1 = vegf * (1 - dep0) + dep0  # Deposition probabilities on bare cells
    temp2 = np.logical_not(sand) * np.logical_not(shade) * temp1  # Apply to bare cells outside shadows only

    # For sandy cells
    temp3 = vegf * (1 - dep1) + dep1  # Deposition probabilities on sandy cells
    temp4 = sand * np.logical_not(shade) * temp3  # Apply to sandy cells outside shadows only

    r = temp2 + temp4 + shade  # Combine both types of cells + shadowzones

    return r


def shiftslabs3_open3(erosprobs, deposprobs, hop, contour, longshore, crossshore, direction):
    """Shifts the sand. Movement is from left to right, along +2 dimension, across columns, along rows. Returns a map of height changes [-,+].
    Open boundaries (modification by Alma), no feeding from the sea side.
    - erosprobs: map of erosion probabilities [erosmap]
    - deposprobs: map of deposition probabilities [deposmap]
    - hop: jumplength
    - direction: wind direction (1 east, 2 north, 3, west, 4 south)"""

    pickedup = np.random.rand(longshore, crossshore) < erosprobs  # True where slab is picked up
    # pickedup[:, -1 - hop: -1] = 0  # Do not pick any slab that are on the ladward boundary -- east only?

    totaldeposit = np.zeros([longshore, crossshore])
    inmotion = pickedup  # Make copy of original erosion map
    numshifted = 0  # [slabs] Number of shifted cells weighted for transport distance
    transportdist = 0  # Transport distance [slab lengths] or [hop length]
    sum_contour = np.zeros([len(contour)])

    while np.sum(inmotion) > 0:  # While still any slabs moving
        transportdist += 1  # Every time in the loop the slaps are transported one length further to the right
        if direction == 1:
            inmotion = np.roll(inmotion, [0, hop])  # Shift the moving slabs one hop length to the right
            transp_contour = np.nansum(inmotion[:, contour.astype(np.int64)], axis=0)  # Account ammount of slabs that are in motion in specific contours
            sum_contour = sum_contour + transp_contour  # Sum the slabs to the others accounted before
            depocells = np.random.rand(longshore, crossshore) < deposprobs  # True where slab should be deposited
            # depocells[:, -1 - hop: -1] = 1  # All slabs are deposited if they are transported over the landward edge
            deposited = inmotion * depocells  # True where a slab is available and should be deposited
            deposited[:, 0: hop] = 0  # Remove  all slabs that are transported from the landward side to the seaward side (this changes the periodic boundaries into open ones)
        elif direction == 2:
            inmotion = np.roll(inmotion, [hop, 0])  # Shift the moving slabs one hop length to the right
            transp_contour = np.nansum(inmotion[:, contour.astype(np.int64)], axis=0)  # Account ammount of slabs that are in motion in specific contours
            sum_contour = sum_contour + transp_contour  # Sum the slabs to the others accounted before
            depocells = np.random.rand(longshore, crossshore) < deposprobs  # True where slab should be deposited
            # depocells[0 : hop, :] = 1  # All slabs are deposited if they are transported over the landward edge
            deposited = inmotion * depocells  # True where a slab is available and should be deposited
            deposited[0: hop, :] = 0  # Remove  all slabs that are transported from the landward side to the seaward side (this changes the periodic boundaries into open ones)
            # # IRBR 21Oct22: Should the above actually be like below?
            # inmotion = np.roll(inmotion, [hop, 0])  # Shift the moving slabs one hop length to the right
            # transp_contour = np.nansum(inmotion[contour.astype(np.int64), :], axis=1)  # Account ammount of slabs that are in motion in specific contours
            # sum_contour = sum_contour + transp_contour  # Sum the slabs to the others accounted before
            # depocells = np.random.rand(longshore, crossshore) < deposprobs  # True where slab should be deposited
            # # depocells[0 : hop, :] = 1  # All slabs are deposited if they are transported over the landward edge
            # deposited = inmotion * depocells  # True where a slab is available and should be deposited
            # deposited[0: hop, :] = 0  # Remove  all slabs that are transported from the landward side to the seaward side (this changes the periodic boundaries into open ones)
        elif direction == 3:
            inmotion = np.roll(inmotion, [0, -hop])  # Shift the moving slabs one hop length to the right
            transp_contour = np.nansum(inmotion[:, contour.astype(np.int64)], axis=0)  # Account ammount of slabs that are in motion in specific contours
            sum_contour = sum_contour + transp_contour  # Sum the slabs to the others accounted before
            depocells = np.random.rand(longshore, crossshore) < deposprobs  # True where slab should be deposited
            # depocells[:, -1 - hop: -1] = 1  # All slabs are deposited if they are transported over the landward edge
            deposited = inmotion * depocells  # True where a slab is available and should be deposited
            deposited[:, -1 - hop: -1] = 0  # Remove  all slabs that are transported from the landward side to the seaward side (this changes the periodic boundaries into open ones)
        elif direction == 4:
            inmotion = np.roll(inmotion, [-hop, 0])  # Shift the moving slabs one hop length to the right
            transp_contour = np.nansum(inmotion[contour.astype(np.int64), :], axis=1)  # Account ammount of slabs that are in motion in specific contours
            sum_contour = sum_contour + transp_contour  # Sum the slabs to the others accounted before
            depocells = np.random.rand(longshore, crossshore) < deposprobs  # True where slab should be deposited
            # depocells[0 : hop + 1, :] = 1  # All slabs are deposited if they are transported over the landward edge
            deposited = inmotion * depocells  # True where a slab is available and should be deposited
            deposited[-1 - hop: -1, :] = 0  # Remove  all slabs that are transported from the landward side to the seaward side (this changes the periodic boundaries into open ones)

        inmotion[deposited] = False  # Left over in transport after this round of deposition
        numshifted = numshifted + np.sum(deposited) * transportdist  # Number of slabs deposited, weighted for transport distance
        totaldeposit = totaldeposit + deposited  # Total slabs deposited so far

    diff = totaldeposit - pickedup  # deposition - erosion

    return diff, numshifted, sum_contour


def enforceslopes3(topof, vegf, sh, anglesand, angleveg, th):
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
                a1 = np.random.rand(1, 1, 8) * exceeds[row, col, :]  # Give all equally steepest slopes in this cell a random number
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


def marine_processes3_diss3e(total_tide, msl, slabheight, cellsizef, topof, eqtopof, vegf, m26f, m27af, m28f, pwavemaxf, pwaveminf, depthlimitf, shelterf, pcurr):
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

    # # Activate this to test standalone
    # [topo, veg] = read_profile_from_netcdf(3, 20000, 1997,0);
    # [topo, eqtopo, veg1, veg2] = create_topographies(topo, veg);
    # slabheight   = 0.1;
    # depthlimitf  = 0.4;     % strongly controls the retreat distance
    # cellsizef    = 1;
    # topof        = topo./slabheight;
    # old_topof    = topof;
    # eqtopof      = eqtopo./slabheight;
    # vegf         = veg1+veg2;
    # test         = 1;
    # m26f         = 0.013;   % dissipation strength
    # m27af        = 1;       % wave energy factor (redundant?)
    # m28f         = 0.0;     % resistance of vegetation (1 = full)
    # pwavemaxf    = 1;
    # pwaveminf    = 1;       % if 1: always erosion if inundated; if 0: no erosion if all energy has been dissipated
    # shelterf     = 1.0;     % exposure to waves in sheltered cells (1 = full shelter, 0 = no shelter)
    # total_tide   = 45; % offshore tide level [slabs]
    # msl          = 0;

    # --------------------------------------
    # WAVE RUNUP
    # Offshore measured tide (sealevf) has to be converted to effective tide
    # level at the shoreline. The vertical limit of wave runup (R) is a function of
    # tide, slope and wave conditions. Since wave conditions are correlated
    # with tide level (higher waves associated with higher storms, we use a
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

    H = max(0, -2.637 + 2.931 * tide_m)
    L = max(0, -30.59 + 46.74 * tide_m)

    # Runup as a function of wave conditions (H, L) and foreshore slope (b)
    if b / math.sqrt(H / L) < 0.3:
        runup_m = 0.043 * math.sqrt(H * L)
    else:
        runup_m = 1.1 * (0.35 * math.tan(b) * math.sqrt(H * L) + math.sqrt(H * L * (0.563 * math.tan(b**2) + 0.004)) / 2)

    # Add runup to tide to arrive at total water level (=tide + runup + msl)
    totalwater_m = tide_m + runup_m + msl_m
    totalwater = totalwater_m / slabheight

    # --------------------------------------
    # IDENTIFY CELLS EXPOSED TO WAVES
    # by dunes and embryodunes, analogous to shadowzones but no angle

    toolow = topof < totalwater  # [0 1] Give matrix with cells that are potentially under water
    pexposed = np.ones(topof.shape)  # Initialise matrix
    for m20 in range(len(topof[:, 0])):  # Run along all the rows
        m21 = np.argwhere(topof[m20, :] >= totalwater)[0][0]
        pexposed[m20, m21: -1] = 1 - shelterf

    # --------------------------------------
    # FILL TOPOGRAPHY TO EQUILIBRIUM

    inundatedf = pexposed  # Inundated is the area that really receives sea water

    # --------------------------------------
    # WAVES

    waterdepth = (totalwater - topof) * pexposed * slabheight  # [m] Exposed is used to avoid negative waterdepths
    waterdepth[waterdepth <= depthlimitf] = depthlimitf  # This limits high dissipitation when depths are limited; remainder of energy is transferred landward

    # Initialise dissiptation matrices
    diss = np.zeros(topof.shape)
    cumdiss = np.zeros(topof.shape)

    # Calculate dissipation
    loc = np.argwhere(topof[0, :] > -10)  # Find location to start dissipation

    if len(loc) == 0:
        loc[0] = 1
    elif loc[0] == 0:
        loc[0] = 1

    for m25 in range(int(loc[0]), topof.shape[1]):  # Do for all columns
        diss[:, m25] = (cellsizef / waterdepth[:, m25]) - (cellsizef / waterdepth[:, loc[0]][:, 0])  # Dissipation corrected for cellsize
        cumdiss[:, m25] = diss[:, m25 - 1] + cumdiss[:, m25 - 1]  # Cumulative dissipation from the shore, excluding the current cell

    # --------------------------------------
    # CALCULATING PROBABILITY OF HYDRODYNAMIC EROSION (phydro = pwave - pinun)



    print()





    return 0, 0, 0, 0, 0, 0















