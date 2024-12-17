"""__________________________________________________________________________________________________________________________________

Model Functions for MEEB

Mesoscale Explicit Ecogeomorphic Barrier model

IRB Reeves

Last update: 20 November 2024

__________________________________________________________________________________________________________________________________"""

import matplotlib.colors as mcolors
import numpy as np
import math
import copy
import scipy
import joblib
import contextlib
from scipy import signal
from AST.alongshore_transporter import calc_alongshore_transport_k
from AST.waves import WaveAngleGenerator
from numba import njit
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt


def shadowzones(topof, shadowangle, direction, MHW, cellsize):
    """Returns a boolean map with all shadowzones identified as ones. Wind from left to right, along the +2 dimension.

    Parameters
    ----------
    topof : ndarray
        [m NAV88] Topography map.
    shadowangle : float
        Shadow angle.
    direction : int
        Wind direction (1 right, 2 down, 3 left, 4 up).
    MHW : float
        [m NAVD88] Mean high water.
    cellsize : float
        [m] Model cell dimensions.

    Returns
    -------
    inshade
        [Bool] Map of cells in shadow zones
    """

    init_topo_copy = topof.copy()
    x_s_min = np.min(ocean_shoreline(topof, MHW))
    x_b_max = np.max(backbarrier_shoreline(topof, MHW))

    topof = topof[:, x_s_min: x_b_max]

    longshore, crossshore = topof.shape
    steplimit = math.tan(shadowangle * math.pi / 180) * cellsize  # The maximum step difference allowed given the shadowangle

    search_range = int(math.ceil(np.max(topof) / steplimit))  # Identifies highest elevation and uses that to determine what the largest search distance needs to be
    inshade = np.zeros([longshore, crossshore]).astype(bool)  # Define the zeroed logical map

    for i in range(1, search_range + 1):
        if direction == 1:
            step = topof - np.roll(topof, i, axis=1)  # Shift across columns (2nd dimension; along a row)
            tempinshade = step < -(steplimit * i)  # Identify cells with too great a stepheight (IRBR 4May23: Removed floor rounding of steplimit)
            tempinshade[:, 0:i] = 0  # Part that is rolled back into beginning of space is ignored
        elif direction == 2:
            step = topof - np.roll(topof, i, axis=0)  # Shift across columns (2nd dimension; along a row)
            tempinshade = step < -(steplimit * i)  # Identify cells with too great a stepheight (IRBR 4May23: Removed floor rounding of steplimit)
            tempinshade[0:i, :] = 0  # Part that is rolled back into beginning of space is ignored
        elif direction == 3:
            step = topof - np.roll(topof, -i, axis=1)  # Shift across columns (2nd dimension; along a row)
            tempinshade = step < -(steplimit * i)  # Identify cells with too great a stepheight (IRBR 4May23: Removed floor rounding of steplimit)
            tempinshade[:, -1 - i:-1] = 0  # Part that is rolled back into beginning of space is ignored
        else:
            step = topof - np.roll(topof, -i, axis=0)  # Shift across columns (2nd dimension; along a row)
            tempinshade = step < -(steplimit * i)  # Identify cells with too great a stepheight (IRBR 4May23: Removed floor rounding of steplimit)
            tempinshade[-1 - i:-1, :] = 0  # Part that is rolled back into beginning of space is ignored
        inshade = np.bitwise_or(inshade, tempinshade)  # Merge with previous inshade zones

    inshade = np.hstack((np.zeros([longshore, x_s_min]), inshade, np.zeros([longshore, init_topo_copy.shape[1] - x_b_max])))

    return inshade


def erosprobs(vegf, shade, sand, topof, groundw, p_er, entrainment_veg_limit, slabheight, mhw):
    """ Returns a map with erosion probabilities.

    Parameters
    ----------
    vegf : ndarray
        [%] Map of combined vegetation effectiveness.
    shade : ndarray
        [bool] Map of shadowzones.
    sand : ndarray
        [bool] Map of sandy cells.
    topof : ndarray
        [m NAVD88] Topography map.
    groundw : ndarray
        [m NAVD88] Groundwater elevation map.
    p_er : float
        Probability of erosion of base/sandy cell with zero vegetation.
    entrainment_veg_limit : float
        [%] Percent of vegetation cover (effectiveness) beyond which aeolian sediment entrainment is no longer possible.
    slabheight : float
        [m] Slab height.
    mhw : float
        [m] Mean high water.

    Returns
    -------
    Pe
        Map of effective erosion probabilities across landscape.
    """

    Pe = np.logical_not(shade) * sand * (topof > groundw) * ((topof - mhw) > slabheight) * (p_er - (p_er / entrainment_veg_limit * vegf))
    Pe *= (Pe >= 0)

    return Pe


def depprobs(vegf, shade, sand, dep_base, dep_sand, dep_sand_MaxVeg, topof, groundw):
    """Returns a map of deposition probabilities that can then be used to implement transport.

    Parameters
    ----------
    vegf : ndarray
        [%] Map of combined vegetation effectiveness.
    shade : ndarray
        [bool] Map of shadowzones.
    sand : ndarray
        [bool] Map of sandy cells.
    dep_base : float
        Probability of deposition in base cell with zero vegetation.
    dep_sand : float
        Probability of deposition in sandy cell with zero vegetation.
    dep_sand_MaxVeg : float
        Probability of deposition in sandy cell with 100% vegetation cover.
    topof : ndarray
        [m NAVD88] Topography map.
    groundw : ndarray
        [m NAVD88] Groundwater elevation map.

    Returns
    -------
    Pd
        Map of effective deposition probabilities across landscape.
    """

    # For base cells
    pdb_veg = dep_base + ((dep_sand_MaxVeg - dep_base) * vegf)  # Deposition probabilities on base cells, greater with increasing vegetation cover
    pdb = np.logical_not(sand) * np.logical_not(shade) * pdb_veg  # Apply to base cells outside shadows only
    pdb[(topof <= groundw)] = 1  # 100% probability of deposition in where groundwater level is at or above surfae level

    # For sandy cells
    pds_veg = dep_sand + ((dep_sand_MaxVeg - dep_sand) * vegf)  # Deposition probabilities on base cells, greater with increasing for veg
    pds = sand * np.logical_not(shade) * pds_veg  # Apply to sandy cells outside shadows only
    pds[(topof <= groundw)] = 1  # 100% probability of deposition in where groundwater level is at or above surfae level

    Pd = pdb + pds + shade  # Combine both types of cells + shadowzones
    Pd[Pd > 1] = 1

    return Pd


def shiftslabs(Pe, Pd, hop_avg, hop_rand_deviation, vegf, vegf_lim, direction, random_hoplength, topo, saltation_slope_limit, MHW, cellsize, RNG):
    """Shifts the sand from wind. Returns a map of surface elevation change. Open boundaries, no feeding from the sea side.

    Follows modifications by Teixeira et al. (2023) that allow larger hop lengths while still accounting for vegetation interactions
    over the course of the hop trajectory, which results in a saltation transport mode rather than a ripple migration transport mode.

    Includes option to vary the hop length randomly around a mean. Note: the random distribution must be centered around mean, otherwise
    the overall sediment flux will vary according to whether or not the hop length is random (not ideal).

    Parameters
    ----------
    Pe : ndarray
        Map of erosion probabilities.
    Pd : ndarray
        Map of deposition probabilities.
    hop_avg : int
        [cells] Slab hop length.
    hop_rand_deviation : int
        [cells] Deviation around average hop length for random uniform distribution. Must be smaller than hop_avg.
    vegf : ndarray
        Map of combined vegetation effectiveness.
    vegf_lim : float
        Threshold vegetation effectiveness needed for a cell along a slab saltation path needed to be considered vegetated.
    direction : int
        Wind direction (1 right, 2 down, 3 left, 4 up).
    random_hoplength : bool
        When True, hop length varies randomly +/- 2 around the average hop length.
    topo : ndarray
        [m NAV88] Topography.
    saltation_slope_limit : float
        [deg] Surface slope at and beyond which saltation transport of sand slabs cannot occur (i.e., too steep)
    MHW : float
        [m NAVD88] Mean high water.
    cellsize : float
        [m] Horizontal cell dimensions.
    RNG
        Random Number Generator object.

    Returns
    -------
    elevation_change
        [slabs] Net change in surface elevation in vertical units of slabs."""

    x_s_min = np.min(ocean_shoreline(topo, MHW))
    x_b_max = np.max(backbarrier_shoreline(topo, MHW))

    Pe = Pe[:, x_s_min: x_b_max]
    Pd = Pd[:, x_s_min: x_b_max]
    vegf = vegf[:, x_s_min: x_b_max]
    topof = topo.copy()[:, x_s_min: x_b_max]

    longshore, crossshore = vegf.shape

    shift = 1  # [cell length] Shift increment

    if random_hoplength:
        hop = RNG.integers(hop_avg - hop_rand_deviation, hop_avg + hop_rand_deviation + 1)  # Draw random hop length from uniform randomn distribution for each iteration
    else:
        hop = hop_avg  # Or, use the same representative hoplength (average) every iteration

    pickedup = RNG.random((longshore, crossshore)) < Pe  # True where slab is picked up

    totaldeposit = np.zeros([longshore, crossshore])
    inmotion = copy.deepcopy(pickedup)  # Make copy of original erosion map
    transportdist = 0  # [cell length] Transport distance counter

    slope_transport_limit = np.tan(saltation_slope_limit * np.pi / 180) * cellsize  # Height difference beyond which slab transport does not occur

    while np.sum(inmotion) > 0:  # While still any slabs moving
        transportdist += 1  # Every time in the loop the slaps are transported one slab length
        if direction == 1:
            inmotion = np.roll(inmotion, shift, axis=1)  # Shift the moving slabs one hop length to the right
            topo_offset = np.roll(topof.copy(), -transportdist, axis=1) - topof.copy()
            if transportdist % hop == 0:  # If cell is at hop target, poll for deposition
                depocells = np.logical_or(RNG.random((longshore, crossshore)) < Pd, topo_offset > slope_transport_limit * transportdist)  # True where slab should be deposited
            else:  # If cell is inbetween slab origin and hop target (i.e., on its saltation path), only poll for deposition if vegetation (above a threshold density) is present
                depocells = np.logical_or(np.logical_and(RNG.random((longshore, crossshore)) < Pd, vegf >= vegf_lim), topo_offset > slope_transport_limit * transportdist)  # True where slab should be deposited
            deposited = inmotion * depocells  # True where a slab is available and should be deposited
            deposited[:, 0: hop] = 0  # Remove all slabs that are transported from the landward side to the seaward side (this changes the periodic boundaries into open ones)
        elif direction == 2:
            inmotion = np.roll(inmotion, shift, axis=0)  # Shift the moving slabs one hop length to the down
            topo_offset = np.roll(topof.copy(), -transportdist, axis=0) - topof.copy()
            if transportdist % hop == 0:  # If cell is at hop target, poll for deposition
                depocells = np.logical_or(RNG.random((longshore, crossshore)) < Pd, topo_offset > slope_transport_limit * transportdist)  # True where slab should be deposited
            else:  # If cell is inbetween slab origin and hop target (i.e., on its saltation path), only poll for deposition if vegetation (above a threshold density) is present
                depocells = np.logical_or(np.logical_and(RNG.random((longshore, crossshore)) < Pd, vegf >= vegf_lim), topo_offset > slope_transport_limit * transportdist)  # True where slab should be deposited
            deposited = inmotion * depocells  # True where a slab is available and should be deposited
            deposited[0: hop, :] = 0  # Remove all slabs that are transported from the landward side to the seaward side (this changes the periodic boundaries into open ones)
        elif direction == 3:
            inmotion = np.roll(inmotion, -shift, axis=1)  # Shift the moving slabs one hop length to the left
            topo_offset = np.roll(topof.copy(), transportdist, axis=1) - topof.copy()
            if transportdist % hop == 0:  # If cell is at hop target, poll for deposition
                depocells = np.logical_or(RNG.random((longshore, crossshore)) < Pd, topo_offset > slope_transport_limit * transportdist)  # True where slab should be deposited
            else:  # If cell is inbetween slab origin and hop target (i.e., on its saltation path), only poll for deposition if vegetation (above a threshold density) is present
                depocells = np.logical_or(np.logical_and(RNG.random((longshore, crossshore)) < Pd, vegf >= vegf_lim), topo_offset > slope_transport_limit * transportdist)  # True where slab should be deposited
            deposited = inmotion * depocells  # True where a slab is available and should be deposited
            deposited[:, -1 - hop: -1] = 0  # Remove all slabs that are transported from the landward side to the seaward side (this changes the periodic boundaries into open ones)
        else:
            inmotion = np.roll(inmotion, -shift, axis=0)  # Shift the moving slabs one hop length to the up
            topo_offset = np.roll(topof.copy(), transportdist, axis=0) - topof.copy()
            if transportdist % hop == 0:  # If cell is at hop target, poll for deposition
                depocells = np.logical_or(RNG.random((longshore, crossshore)) < Pd, topo_offset > slope_transport_limit * transportdist)  # True where slab should be deposited
            else:  # If cell is inbetween slab origin and hop target (i.e., on its saltation path), only poll for deposition if vegetation (above a threshold density) is present
                depocells = np.logical_or(np.logical_and(RNG.random((longshore, crossshore)) < Pd, vegf >= vegf_lim), topo_offset > slope_transport_limit * transportdist)  # True where slab should be deposited
            deposited = inmotion * depocells  # True where a slab is available and should be deposited
            deposited[-1 - hop: -1, :] = 0  # Remove all slabs that are transported from the landward side to the seaward side (this changes the periodic boundaries into open ones)

        inmotion[deposited] = False  # Left over in transport after this round of deposition

        totaldeposit = totaldeposit + deposited  # Total slabs deposited so far

    elevation_change = totaldeposit - pickedup  # [slabs] Deposition - erosion

    elevation_change = np.hstack((np.zeros([longshore, x_s_min]), elevation_change, np.zeros([longshore, topo.shape[1] - x_b_max])))

    return elevation_change


def enforceslopes(topo, vegf, sh, anglesand, angleveg, th, MHW, cellsize, RNG):
    """Function to enforce the angle of repose, with open boundaries.

    Parameters
    ----------
    topo : ndarray
        [m NAVD88] Elevation domain.
    vegf : ndarray
        Map of combined vegetation effectiveness.
    sh : float
        [m] Slab height.
    anglesand : int
        [deg] Angle of repose for bare sand cells.
    angleveg : int
        [deg] Angle of repose for vegetated cells.
    th : float
        Vegetation effectiveness threshold for applying vegetated angle of repose (versus bare angle of repose).
    MHW : float
        [m NAVD88] Mean high water.
    cellsize : float
        [m] Model cell dimensions.
    RNG :
        Random Number Generator object.

    Returns
    ----------
    topo :
        Topography updated with enforced angles of repose.
    moved_slabs:
        Number of slabs moved this iteration
    """

    x_s_min = np.min(ocean_shoreline(topo, MHW))
    x_b_max = np.max(backbarrier_shoreline(topo, MHW))

    topof = topo.copy()[:, x_s_min: x_b_max] / sh  # [slabs] Convert from m to slabs NAVD88
    vegf = vegf[:, x_s_min: x_b_max]
    subaerial = topof > MHW

    steplimitsand = np.floor(np.tan(anglesand * np.pi / 180) * cellsize / sh)  # Maximum allowed height difference for sandy cells
    steplimitsanddiagonal = np.floor(np.sqrt(2) * np.tan(anglesand * np.pi / 180) * cellsize / sh)  # Maximum allowed height difference for sandy cells along diagonal
    steplimitveg = np.floor(np.tan(angleveg * np.pi / 180) * cellsize / sh)  # Maximum allowed height difference for cells vegetated > threshold
    steplimitvegdiagonal = np.floor(np.sqrt(2) * np.tan(angleveg * np.pi / 180) * cellsize / sh)  # Maximum allowed height difference for cells vegetated  along diagonal > threshold

    steplimit = (vegf < th) * steplimitsand + (vegf >= th) * steplimitveg  # Map with max height diff seen from each central cell
    steplimitdiagonal = (vegf < th) * steplimitsanddiagonal + (vegf >= th) * steplimitvegdiagonal  # Idem for diagonal

    M, N = topof.shape  # Retrieve dimensions of area
    slopes = np.zeros([M, N, 8])  # Initialize
    exceeds = np.zeros([M, N, 8])  # Initialize

    avalanched_cells = 0  # Number of avalanched cells
    slabsmoved = 1  # Initial number to get the loop going

    while slabsmoved > 0:
        # Padding with same cell to create open boundaries
        topof = np.column_stack((topof[:, 0], topof, topof[:, -1]))
        topof = np.vstack((topof[0, :], topof, topof[-1, :]))

        # All height differences relative to center cell (positive is sloping upward away, negative is sloping downward away)
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

        minima = np.min(slopes, axis=2)  # Identify for each cell the value of the lowest (negative) slope of all 8 directions

        # True (1) if steepest slope is in this direction & is more negative than the repose limit
        exceeds[:, :, 0] = (minima == slopes[:, :, 0]) * (slopes[:, :, 0] < -steplimit) * subaerial
        exceeds[:, :, 1] = (minima == slopes[:, :, 1]) * (slopes[:, :, 1] < -steplimitdiagonal) * subaerial
        exceeds[:, :, 2] = (minima == slopes[:, :, 2]) * (slopes[:, :, 2] < -steplimit) * subaerial
        exceeds[:, :, 3] = (minima == slopes[:, :, 3]) * (slopes[:, :, 3] < -steplimitdiagonal) * subaerial
        exceeds[:, :, 4] = (minima == slopes[:, :, 4]) * (slopes[:, :, 4] < -steplimit) * subaerial
        exceeds[:, :, 5] = (minima == slopes[:, :, 5]) * (slopes[:, :, 5] < -steplimitdiagonal) * subaerial
        exceeds[:, :, 6] = (minima == slopes[:, :, 6]) * (slopes[:, :, 6] < -steplimit) * subaerial
        exceeds[:, :, 7] = (minima == slopes[:, :, 7]) * (slopes[:, :, 7] < -steplimitdiagonal) * subaerial

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
        avalanched_cells = avalanched_cells + slabsmoved  # Total number of moved slabs this iteration

    topof_updated = np.hstack((topo[:, :x_s_min], topof.copy() * sh, topo[:, x_b_max:]))  # [m NAVD88] Convert back to m

    return topof_updated, avalanched_cells


def growthfunction1_sens(species, sed, A1, B1, C1, D1, E1, P1):
    """Grows (or reduces) vegetation by updating the vegetation effectiveness. Growth/reduction is based on sedimentation balance and growth curve.

    For Ammophilia-like dune-growing grasses.

    Parameters
    ----------
    species : ndarray
        Map of vegetation effectiveness for specific species.
    sed : ndarray
        [m] Map of sedimentation balance (i.e., aggregate change in elevation) from last vegetation iteration.
    A1
    B1
    C1
    D1
    E1
    P1

    Returns
    ----------
    spec :
        Topography updated with enforced angles of repose.
  """

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

    species = species + leftextension + firstleg + secondleg + thirdleg + fourthleg + rightextension

    species[species < minimum] = minimum
    species[species > maximum] = maximum

    return species


def growthfunction2_sens(species, sed, A2, B2, C2, D2, E2, P2):
    """Grows (or reduces) vegetation by updating the vegetation effectiveness. Growth/reduction is based on sedimentation balance and growth curve.

    For Buckthorn-like woody vegetation.

    Parameters
    ----------
    species : ndarray
        Map of vegetation effectiveness for specific species.
    sed : ndarray
        [m] Map of sedimentation balance (i.e., aggregate change in elevation) from last vegetation iteration.
    A2
    B2
    C2
    D2
    E2
    P2
    
    Returns
    ----------
    spec :
        Topography updated with enforced angles of repose.
    """

    # Physiological range (needs to be specified)
    minimum = 0.0  # was -0.5
    maximum = 1.0  # was 1.5

    # Vertices (these need to be specified)
    x1 = A2  # was x1 = -1.3
    x2 = B2  # was x2 = -0.65
    x3 = C2  # was x3 = 0
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

    species = species + leftextension + firstleg + secondleg + thirdleg + fourthleg + rightextension

    species[species < minimum] = minimum
    species[species > maximum] = maximum

    return species


def lateral_expansion(veg, dist, prob, RNG):
    """Implements lateral expansion of existing vegetation patches. Marks cells that lie within specified distance of existing vegetated cells and
    probabilistically determines whether veg can expanded to each of those cells.

    Parameters
    ----------
    veg : ndarray
        Map of vegetation effectiveness for specific species.
    dist : float
        [cell_length] Distance vegetation can expand laterally over one vegetation iteration.
    prob : float
        Probability of lateral expansion of existing vegetation.
    RNG :
        Random Number Generator object.
    Returns
    ----------
    lateral_expansion_allowed : ndarray
        [bool] Cells into which vegetation has successfully expanded.
    """

    # Pad vegetation matrix with zeros for rolling
    veg = veg > 0
    vegpad = np.zeros(np.add(veg.shape, (2, 2)))
    vegpad[1: -1, 1: -1] = veg
    veg3 = vegpad.copy()

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
    lateral_expansion_allowed = lateral_expansion_allowed > 0

    return lateral_expansion_allowed


def establish_new_vegetation(topof, MHW, prob, RNG):
    """Establishes pioneer vegetation in previously bare cells. Represents the germination and development of veg from seeds or rhizome fragments
    distributed by water or wind.

    Parameters
    ----------
    topof : ndarray
        [m] Elevation domain.
    MHW : float
        [m] Mean high water.
    prob : float
        Probability of new pioneer vegetation establishing.
    RNG :
        Random Number Generator object.
    Returns
    ----------
    pioneer_established : ndarray
        [bool] Cells into which new vegetation has successfully established.
    """

    # Convert topography to height above current sea level
    topof = topof - MHW

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


def spec2_zonation(spec2, topo, min_elev, max_elev, dune_threshold, MHW, cellsize):
    """Determines where across the model domain species2 can establish, based on elevation,
    dune crestline, and elevation of fronting dune."""

    dune_crestline, not_gap = foredune_crest(topo, MHW, cellsize)
    longshore, crossshore = spec2.shape

    # ---------
    # Must be landward of the foredune crest
    landward_of_dunecrest = np.zeros(topo.shape, dtype=bool)
    for ls in range(longshore):
        landward_of_dunecrest[ls, dune_crestline[ls]:] = True

    # ---------
    # Must be within elevation range
    elevation = np.logical_and(topo - MHW > min_elev, topo - MHW <= max_elev)

    # ---------
    # Must be fronted by sufficiently tall dune
    index = np.argmax(topo - MHW > dune_threshold, axis=1)
    index[index == 0] = -1
    fronting_low_dune = np.zeros(topo.shape, dtype=bool)
    for ls in range(longshore):
        fronting_low_dune[ls, :index[ls]] = True

    # ---------
    # Determine where spec2 can establish
    shrub_allowed = landward_of_dunecrest * elevation * ~fronting_low_dune

    return shrub_allowed


def shoreline_change_from_CST(
        d_sf,
        k_sf,
        s_sf_eq,
        RSLR,
        Qbe,
        Qow,
        x_s,
        x_t,
        dy,
        storm_iterations_per_year,
        cellsize,
):
    """Shoreline change from cross-shore sediment transport following Lorenzo-Trueba and Ashton (2014).

    Parameters
    ----------
    d_sf : float
        [m] Shoreface depth.
    k_sf : flaot
        [k^m/m/yr] Shoreface flux constant.
    s_sf_eq : float
        Shoreface equilibrium slope.
    RSLR :  float
        [m/yr] Relative sea-level rise rate.
    Qbe : ndarray
        [m^3/m/ts] Volume of sediment removed from (or added to) the upper shoreface by fairweather beach/duneface change.
    Qow : ndarray
        [m^3/ts] Volume of sediment removed from the upper shoreface by overwash.
    x_s : ndarray
        [m] Cross-shore shoreline position relative to start of simulation.
    x_t : ndarray
        [m] Cross-shore shoreface toe position relative to start of simulation.
    MHW : float
        [m NAVD88] Present mean high water.
    dy : int
        [m] Alongshore length between shoreline nodes, i.e. alongshore section length.
    storm_iterations_per_year : int
        Number of storm/shoreline change iterations in a model year.
    cellsize : float
        [m] Model cell dimensions.

    Returns
    ----------
    x_s
        [m] New cross-shore shoreline position relative to start of simulation for each cell length alongshore.
    x_t
        [m] New cross-shore shoreface toe position relative to start of simulation for each cell length alongshore.
    s_sf
        [m/m] Slope of the shoreface for each row cell length alongshore.
    """

    RSLR /= storm_iterations_per_year  # [m] Convert from m/year to m/timestep (timestep typically 0.04 yr)
    k_sf /= storm_iterations_per_year  # [m^3/yr] Convert from m^3/year to m^3/timestep (timestep typically 0.04 yr)
    Qow[Qow < 0] = 0

    # Shoreface Flux
    s_sf = d_sf / (x_s - x_t)
    Qsf = k_sf * (s_sf_eq - s_sf)  # [m^3/m/ts]

    # Toe, Shoreline, and island base elevation changes
    x_t_dt_temp = (4 * Qsf / d_sf) + (2 * RSLR / s_sf)
    x_s_dt_temp = (2 * (Qow + Qbe) / d_sf) - (4 * Qsf / d_sf)  # Beach/dune change (Qbe) added to LTA14 formulation, barrier height removed

    # Find mean change in x_s and x_t for every dy meters alongshore
    x_t_dt_dy_mean = np.nanmean(np.pad(x_t_dt_temp.copy().astype(float), (0, 0 if x_t_dt_temp.copy().size % dy == 0 else dy - x_t_dt_temp.copy().size % dy), mode='constant', constant_values=np.NaN).reshape(-1, dy), axis=1)
    x_s_dt_dy_mean = np.nanmean(np.pad(x_s_dt_temp.copy().astype(float), (0, 0 if x_s_dt_temp.copy().size % dy == 0 else dy - x_s_dt_temp.copy().size % dy), mode='constant', constant_values=np.NaN).reshape(-1, dy), axis=1)

    x_t_dt = np.repeat(x_t_dt_dy_mean, dy)[:len(x_t)] / cellsize
    x_s_dt = np.repeat(x_s_dt_dy_mean, dy)[:len(x_s)] / cellsize

    # Record changes
    x_t = x_t + x_t_dt
    x_s = x_s + x_s_dt

    return x_s, x_t, s_sf  # [m]


@njit
def ocean_shoreline(topof, MHW):
    """Returns location of the ocean shoreline.

    Parameters
    ----------
    topof : ndarray
        [unit] Present elevation domain.
    MHW : float
        [unit] Mean high water elevation. Must be same units as topof.

    Returns
    ----------
    shoreline : ndarray
        Cross-shore location of the ocean shoreline for each row alongshore.
    """

    shoreline = np.argmax(topof >= MHW, axis=1)

    return shoreline


@njit
def backbarrier_shoreline(topof, MHW):
    """Returns location of the back-barrier shoreline.

    Parameters
    ----------
    topof : ndarray
        [m] Present elevation domain.
    MHW : float
        [m] Mean high water elevation.

    Returns
    ----------
    BBshoreline : ndarray
        Cross-shore location of the back-barrier shoreline for each row alongshore.
    """

    BBshoreline = topof.shape[1] - np.argmax(np.fliplr(topof) >= MHW, axis=1) - 1

    return BBshoreline


@njit
def maintain_equilibrium_backbarrier_depth(topo, eq_depth, MHW):
    """Adjusts all back-barrier bay cells below equilubrium depth to equilibrium depth and returns updated elevation domain."""

    # Back-barrier shoreline
    bb_shoreline = backbarrier_shoreline(topo, MHW)

    # Adjust all bay cells at or below eq depth to eq depth
    for y in range(topo.shape[0]):
        topo[y, bb_shoreline[y]:][topo[y, bb_shoreline[y]:] < MHW - eq_depth] = MHW - eq_depth

    return topo


def foredune_crest(topo, MHW, cellsize, buffer=25, window_XL=150, window_large=75, window_small=11):
    """Finds and returns the location of the foredune crest for each grid column alongshore.

    Parameters
    ----------
    topo : ndarray
        [m] Present elevation domain.
    MHW : ndarray
        [m] Mean high water.
    cellsize : float
        [m] Model cell dimensions.
    buffer : int
        [m] Buffer for searching for foredune crests around rough estimate of dune location (dune crest trendline).
    window_XL : int
        [m] Window size for alongshore moving average of topography to find dune crest trendline.
    window_large : int
        [m] Window size for primary broad savgol smoothing of dune crest line.
    window_small : int
        [m] Window size for secondary narrow savgol smoothing of dune crest line.

    Returns
    ----------
    crestline : ndarray
        Cross-shore location of the dune crest for each row alongshore.
    not_gap : ndarry
        Boolean array of alongshore length where True represents cross-shore profiles where a dune crest is present, and False where no dune crest is present (i.e., dune gap)
    """

    x_s = ocean_shoreline(topo, MHW)
    x_b = backbarrier_shoreline(topo, MHW)
    x_s_min = np.min(x_s)
    x_b_max = np.max(x_b)

    topo = topo[:, x_s_min: x_b_max]

    # Parameters
    buffer = int(buffer / cellsize)  # [m]
    window_XL = int(window_XL / cellsize)  # Window size for alongshore moving average of topography
    window_large = int(window_large / cellsize)  # Window size for primary broad savgol smoothening
    window_small = int(window_small / cellsize)  # Window size for secondary narrow savgol smoothening
    if window_XL > topo.shape[0]:
        window_XL = topo.shape[0]
    if window_large > topo.shape[0]:
        window_large = topo.shape[0]

    # Step 1: Find cross-shore location of maximum elevation for each cell alongshore of averaged topography
    moving_avg_elevation = scipy.ndimage.uniform_filter1d(topo.copy(), axis=0, size=window_XL)  # Rolling average in alongshore direction
    crestline = find_max_elev(moving_avg_elevation, x_s, x_s_min, x_b, cellsize)

    # Step 2: Broad smoothing of maximum-elevation line. This gives a rough area of where the dunes are or should be
    crestline = np.round(signal.savgol_filter(crestline, window_large, 1)).astype(int)

    # Step 3: Find peaks with buffer of broadly-smoothened line. If no peak is found, location is marked as gap
    crestline, not_gap = find_crest_buffer(topo, crestline, crestline, buffer, MHW)

    # Step 4: Narrow smoothing of peak-buffer line
    crestline = np.round(signal.savgol_filter(crestline, window_small, 1)).astype(int)

    # Step 5: Add back x-coordinate removed from original domain (to speed up function)
    crestline += x_s_min

    # Step 6: Ensure crestline is landward of ocean shoreline
    crestline[crestline <= 0] = 1
    crestline[crestline <= x_s] = x_s[crestline <= x_s] + 1

    return crestline.astype(int), not_gap


@njit
def find_max_elev(topo_avg, x_s, x_s_min, x_b, cellsize, threshold_width=300):
    """Finds cross-shore location of maximum elevation of averaged topography, with cross-shore search boundaries specified by barrier width."""

    x_s -= x_s_min
    x_b -= x_s_min
    barrier_width = (x_b - x_s)
    crestline = np.zeros(topo_avg.shape[0])
    for ls in range(topo_avg.shape[0]):
        if barrier_width[ls] < threshold_width / cellsize:
            xmax = int(max(x_s[ls] + 100, x_b[ls]))  # If barrier narrower than threshold, search in first fraction of barrier, with fraction increasing to 1 (whole barrier) at 100 m width)
            crestline[ls] = np.argmax(topo_avg[ls, x_s[ls]: xmax])
        else:
            crestline[ls] = np.argmax(topo_avg[ls, x_s[ls]: x_s[ls] + int(barrier_width[ls] * 0.3)])  # If barrier wider than threshold, search only in first third of barrier
    crestline += x_s

    return crestline


@njit
def find_crest_buffer(topo, line_init, crestline, buffer, MHW):
    """Find peaks within buffer of location of line_init; return new crestline."""

    threshold = 0.6  # [m] Threshold backshore drop for peak detection
    crest_pct = 0.1

    not_gap = np.ones(line_init.shape)  # Array indicating which cells in crestline returned an index for an actual peak (i.e., not gap, [1]) and which cells for which no peak was found (i.e., gap, [0])

    for r in range(len(topo)):
        if line_init[r] > 0:
            mini = max(0, line_init[r] - buffer)
            maxi = min(topo.shape[1] - 1, line_init[r] + buffer)
            loc, peak_found = find_crests(topo[r, mini: maxi], MHW, threshold, crest_pct)
            if not peak_found:
                not_gap[r] = 0  # Gap
                if np.isnan(loc):
                    crestline[r] = crestline[r - 1]
                else:
                    crestline[r] = int(mini + loc)
            else:
                crestline[r] = int(mini + loc)
        else:
            crestline[r] = int(np.argmax(topo[r, :]))

    return crestline, not_gap


def foredune_heel(topof, crestline, not_gap, cellsize, threshold, window_small=11):
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
            elif elevation_difference >= threshold:
                break
            elif topof[ls, idx + 1] * 0.95 >= topof[ls, idx]:
                break
            else:
                idx += 1

        # Make a reference copy of the profile with a straight line from the Shoreline to Crest
        z = topof[ls, :]  # Elevation profile
        crest_idx = crestline[ls]
        z_ref = z.copy()
        z_ref[crest_idx: idx] = np.linspace(start=z[crest_idx],
                                            stop=z[idx],
                                            num=idx - crest_idx)

        # Subtract the reference from the original profile and idenitfy the maximum point
        z_diff = z_ref - z
        heel_idx = np.argmax(z_diff)

        heelline[ls] = heel_idx

    # Fill in gaps with linear interpolation
    x = np.arange(len(heelline))
    xp = np.nonzero(heelline * not_gap)[0]
    if len(xp) > 0:  # If there are any gaps
        fp = heelline[xp]
        heelline = np.interp(x, xp, fp)  # Interpolate

    window_small = int(window_small / cellsize)  # Window size for savgol smoothening
    heelline = np.round(signal.savgol_filter(heelline, window_small, 1)).astype(int)

    return heelline


def foredune_toe(topo, dune_crest_loc, MHW, not_gap, cellsize, window_small=11):
    """Finds the dune toe locationd using the stretched sheet method from Mitasova et al. (2011), based on AUTOMORPH (Itzkin et al., 2021)"""

    longshore, crossshore = topo.shape
    shoreline_loc = ocean_shoreline(topo, MHW)

    dune_toe_loc = np.zeros([longshore])  # Initialize

    for ls in range(longshore):
        z = topo[ls, :]  # Elevation profile
        crest_idx = dune_crest_loc[ls]
        shoreline_idx = shoreline_loc[ls]

        if crest_idx > shoreline_idx:

            # Make a reference copy of the profile with a straight line from the Shoreline to Crest
            z_ref = z.copy()
            z_ref[shoreline_idx:crest_idx] = np.linspace(start=z[shoreline_idx],
                                                         stop=z[crest_idx],
                                                         num=crest_idx - shoreline_idx)

            # Subtract the reference from the original profile and idenitfy the maximum point
            z_diff = z_ref - z
            toe_idx = np.argmax(z_diff)
        else:
            toe_idx = crest_idx

        # Store the toe location
        dune_toe_loc[ls] = toe_idx

    # Fill in gaps with linear interpolation
    x = np.arange(len(dune_toe_loc))
    xp = np.nonzero(dune_toe_loc * not_gap)[0]
    if len(xp) > 0:  # If there are any gaps
        fp = dune_toe_loc[xp]
        dune_toe_loc = np.interp(x, xp, fp)  # Interpolate

    window_small = int(window_small / cellsize)  # Window size for savgol smoothening
    dune_toeline = np.round(signal.savgol_filter(dune_toe_loc, window_small, 1)).astype(int)

    return dune_toeline


@njit
def find_local_maxima(profile):
    """Finds and returns an array of indices of local maxima from an elevation profile."""

    pks_idx = []
    r = 1
    while r < len(profile) - 1:
        back = profile[r - 1]
        current = profile[r]
        forward_idx = r + 1
        while profile[forward_idx] == current and forward_idx <= len(profile):
            forward_idx += 1
        front = profile[forward_idx]

        if np.logical_and(back < current, front < current):
            pks_idx.append(r)

        r = forward_idx

    return np.asarray(pks_idx)


@njit
def find_crests(profile, MHW, threshold, crest_pct):
    """Finds foredune peak of profile following Automorph (Itzkin et al., 2021). Returns NaN if no dune peak found on profile."""

    # Find peaks on the profile. The indices increase landwards
    pks_idx = find_local_maxima(profile)

    # Remove peaks below MHW
    if len(pks_idx) > 0:
        pks_idx = pks_idx[profile[pks_idx] > MHW]

    peak_found = False

    # If there aren't any peaks, return NaN
    if len(pks_idx) == 0:
        idx = np.nan

    else:

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
            idx = np.argmax(profile)

    return idx, peak_found


def stochastic_storm(pstorm, iteration, storm_list, beach_slope, longshore, MHW, RNG):
    """Stochastically determines whether a storm occurs for this timestep, and, if so, stochastically determines the relevant characteristics of the storm (i.e., water levels, duration).

    Parameters
    ----------
    pstorm : ndarray
        Empirical probability of storm occuring for each iteration of the year.
    iteration : int
        Present iteration for the year.
    storm_list : ndarray
        List of synthetic storms (rows), with wave and tide statistics (columns) desccribing each storm. Order of statistics: Hs, dur, TWL, SL, Tp, R2, Rlow. Elevations in NAVD88.
    beach_slope : float
        Equilibrium beach slope.
    longshore :
        [m] Longshore length of model domain.
    MHW : float
        [m NAVD88] Mean high water elevation.
    RNG :
        Random number generator, either seeded or unseeded.

    Returns
    ----------
    storm
        Bool whether or not storm occurs this time step.
    Rhigh
        [m NAVD88] Highest elevation of the landward margin of runup (i.e. total water level).
    Rlow
        [m NAVD88] Lowest elevation of the landward margin of runup.
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
        SL = storm_list[n, 3]  # Sea-level [m NAVD88]
        Tp = storm_list[n, 4]  # Wave period

        # Calculate simulated R2% and add to SL to get the simulated TWL
        # Recalculated here instead of using TWL value from synthetic storm list because beach slope varies alongshore in MEEB
        L0 = (9.8 * Tp ** 2) / (2 * np.pi)  # Wavelength
        Setup = 0.35 * beach_slope * np.sqrt(Hs * L0)
        Sin = 0.75 * beach_slope * np.sqrt(Hs * L0)  # Incident band swash
        Sig = 0.06 * np.sqrt(Hs * L0)  # Infragravity band swash
        Swash = np.sqrt((Sin ** 2) + (Sig ** 2))  # Total swash
        R2 = 1.1 * (Setup + (Swash / 2))  # R2%

        # Calculate storm water levels
        Rhigh = SL + R2
        Rlow = (Rhigh - (Swash / 2))

        # No storm if TWL < MHW
        if (Rhigh <= MHW).all():
            storm = False
    else:
        Rhigh = 0
        Rlow = 0
        dur = 0

    Rhigh = Rhigh * np.ones(longshore)
    Rlow = Rlow * np.ones(longshore)

    return storm, Rhigh, Rlow, int(dur)


def get_storm_timeseries(storm_timeseries, it, longshore, MHW, hindcast_start):
    """Returns storm characteristics for this model iteration from an empirical storm timeseries.

    Parameters
    ----------
    storm_timeseries : ndarray
        Table of observed storm events.
    it : int
        Current model iteration.
    longshore : int
        [m] Longshore length of model domain.
    MHW : float
        [m NAVD88] Mean high water elevation.
    hindcast_start :
        [week] Week (1/50 year) to start hindcast from.

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

    it_effective = it + hindcast_start

    if it_effective in storm_timeseries[:, 0]:
        idx = np.where(storm_timeseries[:, 0] == it_effective)[0][0]
        Rhigh = storm_timeseries[idx, 1] * np.ones(longshore)
        Rlow = storm_timeseries[idx, 2] * np.ones(longshore)
        dur = storm_timeseries[idx, 3]
        if storm_timeseries[idx, 1] <= MHW:
            storm = False  # No storm if TWL < MHW
        else:
            storm = True
    else:
        storm = False
        Rhigh = np.zeros(longshore)
        Rlow = np.zeros(longshore)
        dur = 0

    return storm, Rhigh, Rlow, int(dur)


def storm_processes(
        topof,
        Rhigh,
        dur,
        Rin,
        Cs,
        nn,
        MaxUpSlope,
        fluxLimit,
        Qs_min,
        Kow,
        mm,
        MHW,
        Cbb,
        Qs_bb_min,
        substep,
        beach_equilibrium_slope,
        swash_erosive_timescale,
        beach_substeps,
        x_s,
        cellsize,
        spec1,
        spec2,
        flow_reduction_max_spec1,
        flow_reduction_max_spec2,
):
    """Resolves topographical change from storm events. Landward of dune crest: overwashes barrier interior where storm water levels exceed
        pre-storm dune crests following Barrier3D (Reeves et al., 2021) flow routing. Seaward of dune crest: determines topographic change of beach
        and dune face following swash transport equations from Larson et al. (2004).

        Parameters
        ----------
        topof : ndarray
            [m NAVD88] Current elevation domain.
        Rhigh : ndarray
            [m NAVD88] Highest elevation of the landward margin of runup (i.e. total water level).
        dur: int
            [hrs] Duration of storm.
        Rin : float
            [m^3/hr] Flow infiltration and drag parameter for overwash.
        Cs : float
            Constant for representing flow momentum for sediment transport in inundation overwash regime.
        nn : float
            Flow routing constant.
        MaxUpSlope : float
            Maximum slope water can flow uphill.
        fluxLimit : float
            [m/hr] Maximum elevation change allowed per time step (prevents instabilities)
        Qs_min : float
            [m^3/hr] Minimum discharge out of cell needed to transport sediment.
        Kow : float
            Sediment transport coefficient for overwash.
        mm : float
            Inundation overwash constant.
        MHW : float
            [m NAVD88] Mean high water.
        Cbb : float
            [%] Coefficient for exponential decay of sediment load entering back-barrier bay for overwash.
        Qs_bb_min : float
            [m^3/hr] Minimum discharge out of subaqueous back-barrier cell needed to transport sediment.
        substep : int
            Number of substeps to run for each hour in run-up overwash regime (e.g., 3 substeps means discharge/elevation updated every 20 minutes).
        beach_equilibrium_slope : float
            Beach equilibrium slope.
        swash_erosive_timescale : float
            Non-dimensional erosive timescale coefficient for beach/duneface sediment transport.
        beach_substeps : int
            Number of substeps per iteration of beach/duneface model; instabilities will occur if too low.
        x_s : ndarray
            Alongshore array of ocean shoreline locations.
        cellsize : float
            [m] Horizontal dimension of model grid cells
        spec1 : ndarray
            [%] Map of  vegetation effectiveness for species 1
        spec2 : ndarray
            [%] Map of  vegetation effectiveness for species 2
        flow_reduction_max_spec1 : float
            Proportion of overwash flow reduction through a cell populated with species 1 at full effectiveness (i.e., full density)
        flow_reduction_max_spec2 : float
            Proportion of overwash flow reduction through a cell populated with species 2 at full effectiveness (i.e., full density)

        Returns
        ----------
        topof
            [m NAVD88] Updated elevation domain.
        topof_change
            [m] Array of topographic change from storm overwash processes.
        OWloss
            [m^3] Volume of overwash deposition landward of dune crest for each cell unit alongshore.
        netDischarge
            [m^3] Map of discharge aggregated for duration of entire storm.
        inundated
            [bool] Map of cells inundated during storm event
        BeachDune_Volume_Change
            [m^3/m] Dune & beach volumetric change summed for each row alongshore.
        """

    longshore, crossshore = topof.shape

    # Set Up Flow Routing Domain
    domain_width_start = 0  # [cells]
    domain_width_end = int(crossshore)  # [cells]
    domain_width = domain_width_end - domain_width_start  # [cells]
    domain_topo_start = topof[:, domain_width_start:].copy()  # [m NAVD88]
    Elevation = domain_topo_start.copy()  # [m NAVD88]
    dune_crest_loc, not_gap = foredune_crest(Elevation, MHW, cellsize)  # Cross-shore location of pre-storm dune crest

    # Initialize Memory Storage Arrays
    OWloss = np.zeros([longshore])  # [m^3] Aggreagate volume of overwash deposition landward of dune crest for this storm

    # Modify based on number of substeps
    fluxLimit /= substep  # [m/hr] Maximum elevation change during one storm hour allowed
    Qs_min /= substep * (cellsize ** 2)
    Qs_bb_min /= substep * (cellsize ** 2)
    iterations = int(math.floor(dur) * substep)

    BeachDune_Volume_Change = np.zeros([longshore])  # [m^3] Initialize dune/beach volume change: (-) loss, (+) gain
    inundated = np.zeros(topof.shape).astype(bool)  # Initialize

    # Run Storm
    for TS in range(iterations):

        # Find dune crest locations and heights for this storm iteration
        if TS % substep == 0:  # Update every storm hour (not every substep) for speed
            dune_crest_loc, not_gap = foredune_crest(Elevation, MHW, cellsize)  # Cross-shore location of pre-storm dune crest

        # ----------------------
        # Landward of Dune Crest

        # Route Overwash
        Discharge, SedFluxIn, SedFluxOut = route_overwash(
            Elevation,
            dune_crest_loc,
            MHW,
            domain_width,
            domain_width_start,
            longshore,
            Rhigh,
            Rin,
            Cs,
            nn,
            mm,
            MaxUpSlope,
            Kow,
            Qs_min,
            Cbb,
            Qs_bb_min,
            spec1,
            spec2,
            flow_reduction_max_spec1,
            flow_reduction_max_spec2,
            cellsize,
        )

        # Update Elevation After Every Storm Hour of Overwash
        ElevationChangeLandward = (SedFluxIn - SedFluxOut) / cellsize / cellsize / substep  # [m] Net elevation change
        ElevationChangeLandward[ElevationChangeLandward > fluxLimit] = fluxLimit  # Constrain to flux limit
        ElevationChangeLandward[ElevationChangeLandward < -fluxLimit] = -fluxLimit  # Constrain to flux limit
        ElevationChangeLandward[np.arange(longshore), dune_crest_loc - domain_width_start] = 0  # Do not yet update elevation change at dune crest

        # Calculate and save volume of sediment deposited on/behind the barrier interior for every hour
        OWloss = OWloss + np.sum(ElevationChangeLandward, axis=1) * (cellsize ** 2)  # [m^3] For each cell alongshore

        # Record cells inundated from overwash
        overwash_inundated = Discharge > 0
        inundated[domain_width_start:, :] = np.logical_or(inundated[domain_width_start:, :], overwash_inundated)  # Update inundated map with cells landward of dune crest

        # ---------------------
        # Seaward of Dune Crest

        # Beach-Dune Change
        ElevationChangeSeaward, dV, beach_inundated = calc_beach_dune_change(
            Elevation.copy(),
            cellsize,
            dune_crest_loc,
            x_s,
            MHW,
            Rhigh,
            beach_equilibrium_slope,
            swash_erosive_timescale,
            beach_substeps,
        )

        ElevationChangeSeaward /= substep
        BeachDune_Volume_Change += dV / substep

        inundated = np.logical_or(inundated, beach_inundated)  # Update inundated map with cells seaward of dune crest

        # ----------------
        # Update Elevation

        Elevation += ElevationChangeSeaward
        Elevation += ElevationChangeLandward

    # Update Elevation Domain After Storm
    topo_change = Elevation - domain_topo_start  # [m] Change in elevation of barrier
    topof += topo_change  # [m NAVD88]

    return topof, topo_change, OWloss, inundated, BeachDune_Volume_Change


@njit
def calc_beach_dune_change(topo,
                           dx,
                           crestline,
                           x_s,
                           MHW,
                           Rhigh,
                           Beq,
                           Te,
                           substeps,
                           ):
    """Updates the topography seaward of the dune crest after one storm iteration using a cross-shore sand flux equation
        from the Coast Dune Model (Duran Vinent & Moore, 2015), which is based off of SBEACH from Larson et al. (2004). This function
        determines elevation change up to the dune crest and is coupled with the overwash flow routing function (which modifies
        the landscape landward of the dune crest).

        Parameters
        ----------
        topo : ndarray
            [m] Elevation domain.
        dx : float
            [m] Cell horizontal dimension.
        crestline : ndarray
            Alongshore array of dune crest locations.
        x_s : ndarray
            Alongshore array of ocean shoreline locations.
        MHW : float
            [m] Mean high water.
        Rhigh : ndarray
            [m MHW] Highest elevation of the landward margin of runup (i.e. total water level).
        Beq : float
            Beach equilibrium slope.
        Te : float
            Non-dimensional erosive timescale coefficient.
        substeps : int
            Number of substeps per iteration of beach/duneface model; instabilities will occur if too low.

        Returns
        ----------
        topoChange
            [m] Change in elevation for one storm iteration (1 hr).
        dV
            [m^3/m] Dune & beach volumetric change summed for each row alongshore.
        wetMap
            [bool] Map of beach/duneface cells inundated this storm iteration
        """

    Q = Te / substeps  # Erosive timescale flux multiplier

    # Initialize
    longshore, crossshore = topo.shape  # Domain dimensions
    wetMap = np.logical_and(np.zeros(topo.shape), False)  # Initialize map of beach/duneface cells inundated this storm iteration
    topoPrestorm = topo.copy()

    for t in range(substeps):

        # Loop through each cell alongshore
        for y in range(longshore):

            xD = crestline[y]  # [m] Dune crest location
            Rh = Rhigh[y]  # [m NAVD88] Total water level
            xStart = int(x_s[y])  # [m] Start loction
            xFinish = xD + 1

            Qsize = xFinish - xStart

            if Qsize > 0:
                cont = True
                flux = np.zeros(Qsize)  # Array of sediment flux for this substep at cross-shore position y

                wetMap[y, :xStart] = True  # All cells seaward of shoreline marked as inundated

                # Loop through each cell in domain from ocean shoreline to back-barrier bay shoreline
                for x in range(xStart, xFinish):

                    # Cell to operate on
                    zi = topo[y, x]

                    # Definition of boundary conditions
                    if x == xStart:
                        hprev = MHW
                    else:
                        hprev = topo[y, x - 1]

                    if x == xFinish - 1:
                        hnext = topo[y, xFinish]
                    else:
                        hnext = topo[y, x + 1]

                    if zi <= Rh < topo[y, x + 1]:
                        hnext = Rh
                        cont = False

                    # Local topo gradient
                    Bl = 0.5 * (hnext - hprev) / dx
                    Rexcess = Rh - zi

                    qs = (Beq - Bl) * Rexcess * Rexcess

                    # Store Sediment Flux
                    flux[x - xStart] = qs  # Update array of sediment flux for this substep at cross-shore position y

                    # Break if next cell is not inundated
                    if not cont:
                        break
                    else:
                        wetMap[y, x] = True  # Record cell as inundated

                divq = gradient(flux, dx)  # [m/s] Flux divergence
                dzdt = divq * Q  # [m/substep] Change in elevation for this timestep
                topo[y, xStart: xFinish] -= dzdt  # [m/substep] Update elevation for this substep with the flux multiplier

    # Determine topographic change for storm iteration
    topoChange = topo - topoPrestorm
    dV = np.sum(topoChange, axis=1) * (dx ** 2)  # [m^3] Change in volume for the beach/dune system from the storm iteration: (-) loss, (+) gain

    return topoChange, dV, wetMap


@njit
def gradient(y, dx=1.0):
    """Computes the gradient using second order accurate central differences in the interior points
    and first order accurate one-sides (forward or backwards) differences at the boundaries. The returned gradient
    hence has the same shape as the input array. Same as numpy.gradient(), but this one works with numba.

    Parameters
    ----------
    y : ndarray
        Array to compute gradient on.
    dx : float, optional
        Spacing between points in y.

    Returns
    ----------
    grad
        Gradient, in array same size as input.
    """

    stop = len(y) - 1
    grad = np.zeros(y.shape)
    grad[0] = (y[0 + 1] - y[0]) / dx  # Boundary
    grad[-1] = (y[-1] - y[-1 - 1]) / dx  # Boundary
    for i in range(1, stop):
        grad[i] = (y[i + 1] - y[i - 1]) / (2 * dx)

    return grad


@njit
def route_overwash(
        Elevation,
        dune_crest_loc,
        MHW,
        domain_width,
        domain_width_start,
        longshore,
        Rhigh,
        Rin,
        Cs,
        nn,
        mm,
        MaxUpSlope,
        Kow,
        Qs_min,
        Cbb,
        Qs_bb_min,
        spec1,
        spec2,
        flow_reduction_max_spec1,
        flow_reduction_max_spec2,
        cellsize,
):
    """Routes overwash and sediment for one storm iteration based off of Barrier3D (Reeves et al., 2021)"""

    Discharge = np.zeros(Elevation.shape)
    SedFluxIn = np.zeros(Elevation.shape)
    SedFluxOut = np.zeros(Elevation.shape)

    # Find height of dune crest alongshore
    dune_crest_height_m = np.zeros(longshore)
    for ls in range(longshore):
        dune_crest_height_m[ls] = Elevation[ls, dune_crest_loc[ls]]  # [m NAVD88]

    overwash = Rhigh > dune_crest_height_m  # [bool] Identifies rows alongshore where dunes crest is overwashed

    if np.any(overwash):  # Determine if there is any overwash for this storm iteration

        # Calculate discharge through each dune cell for this storm iteration
        Rexcess = (Rhigh - dune_crest_height_m) * overwash  # [m] Height of storm water level above dune crest cells
        Vdune = np.sqrt(2 * 9.8 * Rexcess)  # [m/s] Velocity of water over each dune crest cell (Larson et al., 2004)
        Qdune = Vdune * (Rexcess * cellsize) * 3600  # [m^3/hr] Discharge at each overtopped dune crest cell

        # Set Discharge at Dune Crest
        for ls in range(longshore):
            Discharge[ls, dune_crest_loc[ls] - domain_width_start] = Qdune[ls]

        flow_start = int(np.min(dune_crest_loc))

        for d in range(flow_start, domain_width - 1):
            #  Break out of flow routing if negligible discharge enters next landward row
            if d > np.max(dune_crest_loc) + 2 and np.sum(Discharge[:, d]) <= 0:
                break

            # Reduce discharge across row via infiltration
            if d > 0:
                Discharge[:, d][Discharge[:, d] > 0] -= Rin * cellsize ** 2  # Constant Rin, old method

            Discharge[:, d][Discharge[:, d] < 0] = 0

            for i in range(longshore):
                if Discharge[i, d] > 0:

                    Q0 = Discharge[i, d]

                    # Calculate Slopes
                    if i > 0:
                        S1 = (Elevation[i, d] - Elevation[i - 1, d + 1]) / (math.sqrt(2) * cellsize)
                        if np.isnan(S1) or np.isinf(S1):
                            S1 = 0
                    else:
                        S1 = np.nan

                    S2 = (Elevation[i, d] - Elevation[i, d + 1]) / cellsize
                    if np.isnan(S2) or np.isinf(S2):
                        S2 = 0

                    if i < (longshore - 1):
                        S3 = (Elevation[i, d] - Elevation[i + 1, d + 1]) / (math.sqrt(2) * cellsize)
                        if np.isnan(S3) or np.isinf(S3):
                            S3 = 0
                    else:
                        S3 = np.nan

                    # Calculate Discharge To Downflow Neighbors

                    # One or more slopes positive
                    if S1 > 0 or S2 > 0 or S3 > 0:

                        if S1 < 0 or np.isnan(S1):
                            S1e = 0
                        else:
                            S1e = S1
                        if S2 < 0 or np.isnan(S2):
                            S2e = 0
                        else:
                            S2e = S2
                        if S3 < 0 or np.isnan(S3):
                            S3e = 0
                        else:
                            S3e = S3

                        Q1 = (Q0 * S1e ** nn / (S1e ** nn + S2e ** nn + S3e ** nn))
                        Q2 = (Q0 * S2e ** nn / (S1e ** nn + S2e ** nn + S3e ** nn))
                        Q3 = (Q0 * S3e ** nn / (S1e ** nn + S2e ** nn + S3e ** nn))

                        if np.isnan(Q1):
                            Q1 = 0
                        if np.isnan(Q2):
                            Q2 = 0
                        if np.isnan(Q3):
                            Q3 = 0

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
                        if np.isnan(Qx):
                            Qx = 0

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
                            if not np.isinf(S2) and not np.isinf(S3):
                                Q1 = 0
                                Q2 = (Q0 * abs(S2) ** (-nn) / (abs(S2) ** (-nn) + abs(S3) ** (-nn)))
                                Q3 = (Q0 * abs(S3) ** (-nn) / (abs(S2) ** (-nn) + abs(S3) ** (-nn)))
                            else:
                                Q1 = 0
                                Q2 = Q0 / 2
                                Q3 = Q0 / 2
                        elif np.isnan(S3):
                            if not np.isinf(S1) and not np.isinf(S2):
                                Q1 = (Q0 * abs(S1) ** (-nn) / (abs(S1) ** (-nn) + abs(S2) ** (-nn)))
                                Q2 = (Q0 * abs(S2) ** (-nn) / (abs(S1) ** (-nn) + abs(S2) ** (-nn)))
                                Q3 = 0
                            else:
                                Q1 = Q0 / 2
                                Q2 = Q0 / 2
                                Q3 = 0
                        else:
                            if not np.isinf(S1) and not np.isinf(S2) and not np.isinf(S3):
                                Q1 = (Q0 * abs(S1) ** (-nn) / (abs(S1) ** (-nn) + abs(S2) ** (-nn) + abs(S3) ** (-nn)))
                                Q2 = (Q0 * abs(S2) ** (-nn) / (abs(S1) ** (-nn) + abs(S2) ** (-nn) + abs(S3) ** (-nn)))
                                Q3 = (Q0 * abs(S3) ** (-nn) / (abs(S1) ** (-nn) + abs(S2) ** (-nn) + abs(S3) ** (-nn)))
                            else:
                                Q1 = Q0 / 3
                                Q2 = Q0 / 3
                                Q3 = Q0 / 3

                        if np.isnan(Q1):
                            Q1 = 0
                        if np.isnan(Q2):
                            Q2 = 0
                        if np.isnan(Q3):
                            Q3 = 0

                        if Elevation[i, d] > MHW:  # If subaerial
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
                    # Cell 1
                    if i > 0:
                        if spec1[i - 1, d] > 0 and spec2[i - 1, d] > 0:
                            flow_reduction_max_proportional = flow_reduction_max_spec1 * spec1[i - 1, d] / (spec1[i - 1, d] + spec2[i - 1, d]) + flow_reduction_max_spec2 * spec2[i - 1, d] / (spec1[i - 1, d] + spec2[i - 1, d])
                            Q1 = Q1 * (1 - (flow_reduction_max_proportional * (spec1[i - 1, d] + spec2[i - 1, d])))
                        elif spec1[i - 1, d] > 0:
                            Q1 = Q1 * (1 - (flow_reduction_max_spec1 * spec1[i - 1, d]))
                        else:
                            Q1 = Q1 * (1 - (flow_reduction_max_spec2 * spec2[i - 1, d]))
                        Discharge[i - 1, d + 1] += Q1

                    # Cell 2
                    if spec1[i, d] > 0 and spec2[i, d] > 0:
                        flow_reduction_max_proportional = flow_reduction_max_spec1 * spec1[i, d] / (spec1[i, d] + spec2[i, d]) + flow_reduction_max_spec2 * spec2[i, d] / (spec1[i, d] + spec2[i, d])
                        Q1 = Q1 * (1 - (flow_reduction_max_proportional * (spec1[i, d] + spec2[i, d])))
                    elif spec1[i, d] > 0:
                        Q2 = Q2 * (1 - (flow_reduction_max_spec1 * spec1[i, d]))
                    else:
                        Q2 = Q2 * (1 - (flow_reduction_max_spec2 * spec2[i, d]))
                    Discharge[i, d + 1] += Q2

                    # Cell 3
                    if i < (longshore - 1):
                        if spec1[i + 1, d] > 0 and spec2[i + 1, d] > 0:
                            flow_reduction_max_proportional = flow_reduction_max_spec1 * spec1[i + 1, d] / (spec1[i + 1, d] + spec2[i + 1, d]) + flow_reduction_max_spec2 * spec2[i + 1, d] / (spec1[i + 1, d] + spec2[i + 1, d])
                            Q1 = Q1 * (1 - (flow_reduction_max_proportional * (spec1[i + 1, d] + spec2[i + 1, d])))
                        elif spec1[i + 1, d] > 0:
                            Q3 = Q3 * (1 - (flow_reduction_max_spec1 * spec1[i + 1, d]))
                        else:
                            Q3 = Q3 * (1 - (flow_reduction_max_spec2 * spec2[i + 1, d]))
                        Discharge[i + 1, d + 1] += Q3

                    # Calculate Sed Movement
                    if Q1 > Qs_min * cellsize ** 2:
                        Qs1 = max(0, Kow * (Q1 * (S1 + Cs)) ** mm)
                    else:
                        Qs1 = 0

                    if Q2 > Qs_min * cellsize ** 2:
                        Qs2 = max(0, Kow * (Q2 * (S2 + Cs)) ** mm)
                    else:
                        Qs2 = 0

                    if Q3 > Qs_min * cellsize ** 2:
                        Qs3 = max(0, Kow * (Q3 * (S3 + Cs)) ** mm)
                    else:
                        Qs3 = 0

                    if np.isnan(Qs1):
                        Qs1 = 0
                    if np.isnan(Qs2):
                        Qs2 = 0
                    if np.isnan(Qs3):
                        Qs3 = 0

                    # Calculate Net Erosion/Accretion
                    # If cell is subaerial, elevation change is determined by difference between flux in vs. flux out
                    if Elevation[i, d] > MHW or np.sum(np.greater(Elevation[i, d + 1: d + 10], MHW)) > 0:
                        if i > 0:
                            SedFluxIn[i - 1, d + 1] += Qs1

                        SedFluxIn[i, d + 1] += Qs2

                        if i < (longshore - 1):
                            SedFluxIn[i + 1, d + 1] += Qs3

                        Qs_out = Qs1 + Qs2 + Qs3
                        SedFluxOut[i, d] = Qs_out

                    # If cell is subaqeous, exponentially decay dep. of remaining sed across bay
                    else:
                        Qs0 = SedFluxIn[i, d] * Cbb

                        if Q1 + Q2 + Q3 != 0:
                            Qs1 = Qs0 * Q1 / (Q1 + Q2 + Q3)
                            Qs2 = Qs0 * Q2 / (Q1 + Q2 + Q3)
                            Qs3 = Qs0 * Q3 / (Q1 + Q2 + Q3)
                        else:
                            Qs1 = 0
                            Qs2 = 0
                            Qs3 = 0

                        if Qs1 < Qs_bb_min * cellsize ** 2 or np.isnan(Qs1):
                            Qs1 = 0
                        if Qs2 < Qs_bb_min * cellsize ** 2 or np.isnan(Qs2):
                            Qs2 = 0
                        if Qs3 < Qs_bb_min * cellsize ** 2 or np.isnan(Qs3):
                            Qs3 = 0

                        if i > 0:
                            SedFluxIn[i - 1, d + 1] += Qs1

                        SedFluxIn[i, d + 1] += Qs2

                        if i < (longshore - 1):
                            SedFluxIn[i + 1, d + 1] += Qs3

                        Qs_out = Qs1 + Qs2 + Qs3
                        SedFluxOut[i, d] = Qs_out

    return Discharge, SedFluxIn, SedFluxOut


def init_AST_environment(wave_asymmetry,
                         wave_high_angle_fraction,
                         mean_wave_height,
                         mean_wave_period,
                         DShoreface,
                         dy,
                         alongshore,
                         n_bins=181,
                         ):
    """Initialize alongshore tranport environment, i.e. the average coastal diffusivity based on wave climate.
    From CASCADE (Anarde et al., 2023), stemming from BRIE (Nienhuis & Lorenzo-Trueba, 2019) and CEM (Ashton & Murray, 2006).

    Parameters
    ----------
    wave_asymmetry: float
        Fraction of waves approaching from the left (when looking offshore).
    wave_high_angle_fraction: float
        Fraction of waves approaching at angles higher than 45 degrees from shore normal.
    mean_wave_height: float
        [m] Mean offshore significant wave height.
    mean_wave_period: float
        [s] Mean offshore wave period.
    DShoreface: float
        [m] Shoreface depth.
    dy: int
        [m] Alongshore width of shoreline sections.
    alongshore: int
        [m] Alongshore dimension of model domain.
    n_bins: int, optional
        The number of bins used for the wave resolution: if 181 and [-90,90] in angle array, the wave angles are in the middle of the bins, symmetrical about zero, spaced by 1 degree

    Returns
    ----------
    coast_diff:
        Wave-climate-averaged coastal diffusivity for each section alongshore.
    di
        Timestepping for implicit diffusion equation.
    dj
        Spacestepping for implicit diffusion equation.
    ny
        Number of alongshore sections in domain.
    """

    ny = int(math.ceil(alongshore / dy))  # Alongshore section count

    angle_array, step = np.linspace(-np.pi / 2.0, np.pi / 2.0, n_bins, retstep=True)  # Array of resolution angles for wave climate [radians]

    k = calc_alongshore_transport_k()

    angles = WaveAngleGenerator(asymmetry=wave_asymmetry, high_fraction=wave_high_angle_fraction)  # Wave angle generator for each time step for calculating Qs_in

    wave_pdf = angles.pdf(angle_array) * step  # Wave climate PDF

    diff = (-(k
              / DShoreface
              * mean_wave_height ** 2.4
              * mean_wave_period ** 0.2
              )
            * 365
            * 24
            * 3600
            * (np.cos(angle_array) ** 0.2)
            * (1.2 * np.sin(angle_array) ** 2 - np.cos(angle_array) ** 2)
            )

    conv = np.convolve(wave_pdf, diff, mode="full")
    npad = len(diff) - 1
    first = npad - npad // 2
    coast_diff = conv[first: first + len(wave_pdf)]

    di = (np.r_[ny, np.arange(2, ny + 1), np.arange(1, ny + 1), np.arange(1, ny), 1] - 1)  # timestepping implicit diffusion equation (KA: -1 for python indexing)

    dj = (np.r_[1, np.arange(1, ny), np.arange(1, ny + 1), np.arange(2, ny + 1), ny] - 1)

    return coast_diff, di, dj, ny


def shoreline_change_from_AST(x_s,
                              coast_diffusivity,
                              di,
                              dj,
                              dy,
                              dt,  # [] Time step
                              ny,
                              nbins=181,
                              ):
    """Determine change in shoreline position via alongshore sediment transport. From CASCADE (Anarde et al., 2023), stemming from BRIE (Nienhuis & Lorenzo-Trueba, 2019) and CEM (Ashton & Murray, 2006).

    Parameters
    ----------
    x_s : array of float
        Cross-shore coordinates for shoreline position.
    coast_diffusivity : array of float
        Wave-climate-averaged coastal diffusivity for each section alongshore.
    di : ndarray
        Timestepping for implicit diffusion equation.
    dj: ndarray
        Spacestepping for implicit diffusion equation.
    dy : int
        [m] Alongshore width of shoreline sections used in the shoreline diffusion calculations.
    dt: float
        [yr] Time step length for shoreline change.
    ny: int
        Number of alongshore sections in domain.
    nbins: int, optional
        The number of bins used for the wave resolution: if 181 and [-90,90] in angle array, the wave angles are in the middle of the bins, symmetrical about zero, spaced by 1 degree

    Returns
    ----------
    x_s_updated
        Cross-shore coordinates for shoreline position updated for alongshore sediment transport.
    """

    # Take shoreline position every dy [m] alongshore
    x_s_ast = x_s[0::dy]

    # Find shoreline angles
    shoreline_angles = (180 * (np.arctan2((x_s_ast[np.r_[1: len(x_s_ast), 0]] - x_s_ast), dy)) / np.pi)

    r_ipl = np.maximum(0, (coast_diffusivity[np.maximum(1, np.minimum(nbins, np.round(90 - shoreline_angles).astype(int)))] * dt / 2 / dy ** 2))

    # Set non-periodic boundary conditions
    r_ipl[0] = 0
    r_ipl[-1] = 0

    dv = np.r_[-r_ipl[-1], -r_ipl[1:], 1 + 2 * r_ipl, -r_ipl[0:-1], -r_ipl[0]]

    A = csr_matrix((dv, (di, dj)))

    RHS = (x_s_ast + r_ipl * (x_s_ast[np.r_[1: ny, 0]] - 2 * x_s_ast + x_s_ast[np.r_[ny - 1, 0: ny - 1]]))

    # Solve for new shoreline position
    new_x_s = spsolve(A, RHS)

    x_s_updated = np.repeat(new_x_s, dy)[:len(x_s)]

    return x_s_updated


def init_ocean_shoreline(topo, MHW, dy):
    """Takes raw shoreline position and converts to average of every dy meters."""

    x_s_raw = ocean_shoreline(topo, MHW)

    # Find average shoreline position of every dy [m] alongshore
    x_s_dy_mean = np.nanmean(np.pad(x_s_raw.copy().astype(float), (0, 0 if x_s_raw.copy().size % dy == 0 else dy - x_s_raw.copy().size % dy), mode='constant', constant_values=np.NaN).reshape(-1, dy), axis=1)

    # Expand to full shoreline length
    x_s_init = np.repeat(x_s_dy_mean, dy)[:topo.shape[0]]

    return x_s_init


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=-1):
    if n == -1:
        n = cmap.N
    new_cmap = mcolors.LinearSegmentedColormap.from_list(
        'trunc({name},{a:.2f},{b:.2f})'.format(name=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def get_MEEB_colormap():
    """Return colormap for plotting elevation in MEEB. Designed for (vmin = -1, vmax = 7)."""

    # [deep sea blue, ocean blue, azure, floral white, light tan, sandy, chocolate, linen]
    colors = ["#015482", "#03719c", "azure", "floralwhite", "#fbeeac", "#f1da7a", "#3d1c02", "linen"]
    nodes = [0.0, 0.08, 0.125, 0.13, 0.18, 0.3, 0.75, 1.0]
    MEEB_cmap = mcolors.LinearSegmentedColormap.from_list("MEEB_cmap", list(zip(nodes, colors)))

    return MEEB_cmap


def brier_skill_score(simulated, observed, baseline, mask):
    """Computes a Brier Skill Score for simulated and observed variables.

    Parameters
    ----------
    simulated : ndarray
        Array of simulated data.
    observed : ndarray
        Matching array of observed data.
    baseline : ndarray
        Assumption of no norphological change (i.e., and array of zeros if considering the variable of elevation change)
    mask : ndarray
        [bool] Map of cells to perform analysis for. If no mask, use np.ones(simulated.shape).astype(bool)

    Returns
    ----------
    BSS
        Brier Skill Score.

    """

    MSE = np.nanmean(np.square(np.abs(np.subtract(simulated[mask], observed[mask]))))
    MSEref = np.nanmean(np.square(np.abs(np.subtract(baseline[mask], observed[mask]))))

    BSS = 1 - MSE / MSEref

    return BSS


@njit
def adjust_ocean_shoreline(
        topo,
        new_shoreline,
        prev_shoreline,
        MHW,
        shoreface_slope,
        RSLR,
        storm_iterations_per_year,
        cellsize,
):
    """Adjust topography domain to according to amount of shoreline change.

    Parameters
    ----------
    topo : ndarray
        [m NAVD88] Current elevation domain.
    new_shoreline : ndarray float
        [m] Cross-shore location of new target ocean shoreline.
    prev_shoreline : ndarray float
        [m] Cross-shore location of ocean shoreline from previous time step.
    MHW : float
        [m NAVD88] Mean high water.
    shoreface_slope : ndarray
        [m/m] Active slope of the shoreface for each cell alongshore.
    RSLR : float
        [m/yr] Relative sea-level rise rate.
    storm_iterations_per_year : int
        Number of storm/shoreline change iterations in a model year.
    cellsize : float
        [m] Horizontal cell dimensions.

    Returns
    ----------
    topo
        [m NAVD88] Topobathy updated for ocean shoreline change.
    """

    RSLR /= storm_iterations_per_year  # [m] Convert from m/year to m/timestep (timestep typically 0.04 yr)

    target_xs = np.floor(new_shoreline).astype(np.int64)
    prev_xs = np.floor(prev_shoreline).astype(np.int64)
    shoreline_change = target_xs - prev_xs  # [cells] (+) erosion, (-) accretion

    erosion = shoreline_change > 0  # [bool]
    accretion = shoreline_change < 0  # [bool]

    for ls in range(len(target_xs)):
        target = target_xs[ls]
        prev = prev_xs[ls]
        if erosion[ls]:  # Erode the shoreline
            if target < topo.shape[1]:  # Shoreline is within model domain
                # Adjust shoreline
                topo[ls, target] = min(MHW - ((MHW - topo[ls, prev]) / 2), MHW - RSLR)  # [m NAVD88]  # Old beach cell elevations set halfway between first subaqeuous cell MHW or to RSLR below MHW, whichever's lowest
                # Adjust shoreface
                shoreface = np.arange(-target, 0) * (shoreface_slope[ls] * cellsize) + topo[ls, target]  # New shoreface cells
                topo[ls, :target] = shoreface  # Insert into domain
            else:  # Shoreline is outside model domain
                shoreface = np.arange(-target, 0) * (shoreface_slope[ls] * cellsize) + MHW  # New shoreface cells
                shoreface = shoreface[-topo.shape[1]:]  # Trim to size of model domain
                topo[ls, :] = shoreface
        elif accretion[ls]:  # Prograde the shoreline
            if 0 < target <= topo.shape[1]:  # Shoreline is within model domain
                # Adjust shoreline  TODO: Account for case where shoreline previously beyond domain progrades back into domain
                topo[ls, target: prev] = np.mean(topo[ls, prev: prev + 5]) + RSLR  # [m NAVD88]  # New beach cell elevations set to average of previous 5 most-oceanward beach cells
                # Adjust shoreface
                shoreface = np.arange(-target, 0) * (shoreface_slope[ls] * cellsize) + MHW  # New shoreface cells
                topo[ls, :target] = shoreface  # Insert into domain
            elif target < 0:
                raise ValueError("Out-Of-Bounds: Ocean shoreline prograded beyond simulation domain boundary.")

    return topo


def reduce_raster_resolution(raster, reduction_factor):
    """Reduces raster resolutions by reduction factor using averaging.

    Parameters
    ----------
    raster : ndarray
        2D numpy array raster.
    reduction_factor : int
        Fraction to reduce raster resolution by (i.e., 10 will reduce 500x300 raster to 50x30).

    Returns
    ----------
    raster
        Reduced resolution raster.
    """

    shape = tuple(int(ti / reduction_factor) for ti in raster.shape)
    sh = shape[0], raster.shape[0] // shape[0], shape[1], raster.shape[1] // shape[1]

    return raster.reshape(sh).mean(-1).mean(1)


def replace_nans_infs(arr):
    """Replaces nans and infs in array with the real value of the neighboring cell one cell alongshore.

    Parameters
    ----------
    arr : ndarray
        2D numpy array raster.

    Returns
    ----------
    arr
        2D numpy array with nans and infs replaced by real values.
    """

    nans_or_infs = True

    while nans_or_infs:

        nan_bool = np.isnan(arr)  # Bool location of nans
        inf_bool = np.isinf(arr)  # Bool location of nans
        replacements = np.roll(arr, shift=1, axis=0)  # Values to replace nans with (alongshore neighboring cell)

        arr[nan_bool] = replacements[nan_bool]  # Replace nans with alongshore neighboring cell value
        arr[inf_bool] = replacements[inf_bool]  # Replace nans with alongshore neighboring cell value

        if not np.isnan(np.sum(arr)):  # Check if any nans remain
            if not np.isinf(np.sum(arr)):
                nans_or_infs = False  # No more nans or infs, exit loop

    return arr


@njit
def calculate_beach_slope(topof, x_s, dune_crest_loc, average_dune_toe_height, MHW, cellsize):
    """Finds the beach slope for each cell alongshore. Slope is calculated using the average dune toe height.

    Parameters
    ----------
    topof : ndarray
        [m] Elevation grid.
    x_s : array of float
        [m] Cross-shore position of the ocean shoreline for each cell alongshore.
    dune_crest_loc : array of float
        [m] Cross-shore position of the foredune crest for each cell alongshore.
    average_dune_toe_height : float
        [m] Time- and space-averaged dune toe height above MHW.
    MHW : float
        [m] Mean high water elevation.
    cellsize : float
        [m] Model cell dimensions.

    Returns
    ----------
    beach_slopes
        Slope of the beach for each cell alongshore.
    """
    beach_slopes = np.zeros(topof.shape[0])  # Initialize
    toe_locs = np.zeros(topof.shape[0])  # Initialize

    # Find local active beach slope
    for ls in range(topof.shape[0]):
        toe_loc = dune_crest_loc[ls] - np.argmax(np.flip(topof[ls, :dune_crest_loc[ls]]) <= MHW + average_dune_toe_height)
        toe_locs[ls] = toe_loc
        beach_slopes[ls] = topof[ls, toe_loc] / ((toe_loc - x_s[ls]) * cellsize)

    return beach_slopes


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()
