"""__________________________________________________________________________________________________________________________________

Main Model Script for PyDUBEVEG

Python version of the DUne, BEach, and VEGetation (DUBEVEG) model from Keijsers et al. (2016) and Galiforni Silva et al. (2018, 2019)

Translated from Matlab by IRB Reeves

Last update: 20 October 2022

__________________________________________________________________________________________________________________________________"""

import numpy as np
import math


def shaowzones2(topof, sh, lee, longshore, crossshore):
    """Returns a logical map with all shadowzones identified as ones
    % wind from left to right, along the +2 dimension
    % topof     : topography map [topo]
    % sh        : slab height [slabheight]
    % lee       : shadow angle [shadowangle]
    % returns a logical map of shaded cells
    % now with open boundaries"""

    # steplimit = math.floor(math.tan(lee * math.pi / 180) / sh)  # The maximum step difference allowed given the slabheight and shadowangle
    steplimit = math.tan(lee * math.pi / 180) / sh  # The maximum step difference allowed given the slabheight and shadowangle, not rounded yet

    # range = min(crossshore, max(topof) / steplimit)
    search_range = int(math.floor(np.max(topof) / steplimit))  # Identifies highest elevation and uses that to determine what the largest search distance needs to be

    inshade = np.zeros([longshore, crossshore])  # Define the zeroed logical map

    for i in range(1, search_range + 1):
        step = topof - np.roll(topof, [0, i])  # Shift across columns (2nd dimension; along a row)
        print()
        # tempinshade = bsxfun(@lt, step, -math.floor(steplimit * i))  # Identify cells with too great a stepheight
        tempinshade = step < -math.floor(steplimit * i)  # Identify cells with too great a stepheight
        tempinshade[:, 0:i] = 0  # Part that is circshifted back into beginning of space is ignored
        inshade[tempinshade is True] = True  # Merge with previous inshade zones
        print()



    return inshade















