""" Converts .tif elevation, vegetation type, and vegetation density raster files into a numpy array initial conditions file for
    use in MEEB (Mesoscale Explicit Ecogeomorphic Barrier model).

    IRB Reeves
    Last update: 7 October 2024
"""

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import routines_meeb as routine
from skimage.transform import resize


def populateVeg(topo,
                veggie,
                mhw,
                dune_crest_loc,
                density_slope,
                density_intercept,
                elevation_mean,
                elevation_tau,
                elevation_maximum,
                distance_r,
                distance_u,
                distance_maximum,
                density_weight,
                elevation_weight,
                distance_weight,
                ):
    """
    Stochastically populates model domain with 2 species types of vegetation (grass & shrub) using an observed veg cover map categorized by estimated relatively density (low, medium, high). Placement of species types depends
    probabilistically on observed density, elevation, and distance from dune crest. Initial percent cover depends probablistically on estimated relative density.

    :param topo: [m NAVD88] Initial elevation domain.
    :param veggie: Initial vegetation density map.
    :param mhw: [NAVD88] Mean high water.
    :param dune_crest_loc: Cross-shore locations for dune crest for each m alongshore.
    :param density_slope: Slope of line capturing probabilistic relationship between observed density and species type.
    :param density_intercept: Intercept of line capturing probabilistic relationship between observed density and species type.
    :param elevation_mean: Mean of normal distribution of shrub elevation capturing probabilistic relationship between elevation and species type.
    :param elevation_tau: Spread (Std Dev) of normal distribution of shrub elevation; controls elevation tolerance bounds.
    :param elevation_maximum: Maximum probability of a cell being a shrub at the optimal elevation.
    :param distance_r: Steepness of logistic curve capturing probabilistic relationship between distance from the dune crest and species type.
    :param distance_u: Distance from dune crest where probability of shrub = 50% on the logistic curve.
    :param distance_maximum: Maximum probability of a cell being a shrub at the optimal distance from the dune crest.
    :param density_weight: Relative weight factor for observed density probability for calculating a weighted composite probability.
    :param elevation_weight: Relative weight factor for elevation probability for calculating a weighted composite probability.
    :param distance_weight: Relative weight factor for distance from dune crest probability for calculating a weighted composite probability.
    :return: s1: Species 1 (grass) percent cover map.
    :return: s2: Species 2 (shrub) percent cover map.
    """

    longshore, crossshore = topo.shape

    # Define Topographic Parameters
    elev_mhw = topo - mhw  # [MHW] Elevation relative to MHW

    # Find height of dune crest alongshore
    dune_crest_height = np.zeros(longshore)
    for ls in range(longshore):
        dune_crest_height[ls] = topo[ls, dune_crest_loc[ls]]  # [m]

    # Calculate Distance from Dune Crest
    dist_to_dune_crest = np.zeros(topo.shape)
    x_vals = np.arange(crossshore)
    for i in range(longshore):
        dist_to_dune_crest[i, :] = x_vals - dune_crest_loc[i]

    # Calculate Probability of Shrub Across Model Domain Based on 3 Factors
    prob_density = (density_slope * veggie + density_intercept) / 100  # Probability of shrub based on estimated "density" of vegetation
    prob_density[prob_density < 0] = 0
    prob_elevation = bell_curve(elev_mhw, elevation_mean, elevation_tau, elevation_maximum)  # Probability of shrub based on elevation
    prob_distance = logistic_curve(dist_to_dune_crest, distance_r, distance_u, distance_maximum)  # Probability of shrub based on distance to dune crest

    weighted_composite_probability = (prob_density * density_weight + prob_elevation * elevation_weight + prob_distance * distance_weight) / (
                density_weight + elevation_weight + distance_weight)

    # Stochastically Distribute Shrubs and Grass Species Types Across Domain According to Calculated Probabilities
    s1_bool = np.zeros(topo.shape, dtype=bool)  # Species 1 (grass)
    s2_bool = np.zeros(topo.shape, dtype=bool)  # Species 2 (shrub)
    randDistribution = np.random.rand(longshore, crossshore)
    s1_bool[np.logical_and(randDistribution > weighted_composite_probability, veggie > 0)] = True
    s2_bool[np.logical_and(randDistribution <= weighted_composite_probability, veggie > 0)] = True

    # Stochastically Assign Initial Density Following Estimated Observed Relative Density Classification
    s1 = np.zeros(topo.shape)  # Species 1 (grass)
    s2 = np.zeros(topo.shape)  # Species 2 (shrub)
    randDensity = np.random.rand(longshore, crossshore)

    s1[np.logical_and(s1_bool, veggie == 1)] = 0.2 * randDensity[np.logical_and(s1_bool, veggie == 1)] + 0.1  # Assign random initial density between 0.1 and 0.3 for low density cells
    s2[np.logical_and(s2_bool, veggie == 1)] = 0.2 * randDensity[np.logical_and(s2_bool, veggie == 1)] + 0.1  # Assign random initial density between 0.1 and 0.3 for low density cells

    s1[np.logical_and(s1_bool, veggie == 2)] = 0.3 * randDensity[np.logical_and(s1_bool, veggie == 2)] + 0.3  # Assign random initial density between 0.3 and 0.6 for medium density cells
    s2[np.logical_and(s2_bool, veggie == 2)] = 0.3 * randDensity[np.logical_and(s2_bool, veggie == 2)] + 0.3  # Assign random initial density between 0.3 and 0.6 for medium density cells

    s1[np.logical_and(s1_bool, veggie == 3)] = 0.3 * randDensity[np.logical_and(s1_bool, veggie == 3)] + 0.6  # Assign random initial density between 0.6 and 0.9 for high density cells
    s2[np.logical_and(s2_bool, veggie == 3)] = 0.3 * randDensity[np.logical_and(s2_bool, veggie == 3)] + 0.6  # Assign random initial density between 0.6 and 0.9 for high density cells

    # Artificially & Stochastically Thin Out Coverage of Low & Medium Densities
    if Thin:
        randThin = np.random.rand(longshore, crossshore)
        s1[np.logical_and(randThin < 0.25, veggie == 1)] = 0
        s2[np.logical_and(randThin < 0.25, veggie == 1)] = 0
        s1[np.logical_and(randThin < 0.1, veggie == 2)] = 0
        s2[np.logical_and(randThin < 0.1, veggie == 2)] = 0

    return s1, s2


def bell_curve(x, mean, tau, maximum):
    """
    :param x: multiplier
    :param mean: mean of normal distribution (optimum)
    :param tau: parameter controlling tolerance bounds (i.e., spread)
    :param maximum: maximum probability at optimum value (from 0 to 1)
    :return: probability
    """
    return np.exp(-(tau * (x - mean)) ** 2) * maximum


def logistic_curve(x, r, u, maximum):
    """
    :param x: multiplier
    :param r: parameter controlling slope of logistic curve
    :param u: value where probability = 0.5
    :param maximum: maximum probability at optimum value (from 0 to 1)
    :return: probability
    """
    return 1 / (1 + np.exp(r * (u - x))) * maximum


def densityVeg(veggie, den):
    """Assigns initial veg density based on negligible, low, medium, and high classification, with random noise perturbations."""

    s1 = np.zeros(veggie.shape)  # Species 1 (grass)
    s2 = np.zeros(veggie.shape)  # Species 2 (shrub)

    s1[veggie == 1] = 1
    s2[veggie == 2] = 1

    # Apply density
    s1[den <= 0] *= 0.4
    s1[den == 1] *= 0.6
    s1[den == 2] *= 0.8
    s1[den == 3] *= 1.0

    s2[den <= 0] *= 0.4
    s2[den == 1] *= 0.6
    s2[den == 2] *= 0.8
    s2[den == 3] *= 1.0

    s1[np.logical_and(np.logical_and(s1 == 0, s2 == 0), den > 0)] = 0.4

    # Add +/- 10% density random noise to vegetated cells
    randNoise = np.random.uniform(-0.1, 0.1, s1.shape) * (s1 > 0)
    s1 += randNoise
    s1[s1 > 1] = 1
    s1[s1 < 0] = 0
    randNoise = np.random.uniform(-0.1, 0.1, s2.shape) * (s2 > 0)
    s2 += randNoise
    s2[s2 > 1] = 1
    s2[s2 < 0] = 0

    if Thin:
        randThin = np.random.rand(s1.shape[0], s1.shape[1])
        s1[np.logical_and(den <= 0, randThin < 0.25)] = 0
        s2[np.logical_and(den <= 0, randThin < 0.25)] = 0

    return s1, s2


# ================================================================================================================
# SPECIFICATIONS

# Elevation (NAVD88)
tif_file = ""  # Path to raw input tif file with 1 m resolution
MHW = 0.39  # [m initial datum] Mean high water, for finding ocean shoreline
BB_thresh = 0.08  # [m initial datum] Threshold elevation for finding back-barrier marsh shoreline
BB_depth = 1.5 - MHW  # [m MHW] Back-barrier bay depth
BB_slope_length = 30  # [m] Length of slope from back-barrier shoreline into bay
SF_slope = 0.0082  # Equilibrium shoreface slope
buffer = 3000  #

# Domain
xmin = 7000  # [m] Alongshore coordinate for start of domain
xmax = 35750  # [m] Alongshore coordinate for end of domain
ymin = 100  # [m] Cross-shore coordinate for start of domain
ymax = 1800  # [m] Cross-shore coordinate for end of domain
zmax = 10  # [m NAVD88] Maximum elevation to remove outlier errors
zmin = -10  # [m NAVD88] Minimum elevation to remove outlier errors
bay = 0  # [m] Additional width of bay to add to domain
cellsize = 1  # [m] Cell dimensions; if greater than 1, script resizes arrays to fit new resolution; cellsizes less than 1 m not advised

# Vegetation
Veggie = True  # [bool] Whether or not to load & convert a contemporaneous init veg raster
VeggiePop = False  # [bool] Whether or not to use stoachstic population of vegetation or init veg density raster
Thin = True  # [bool] Whether to artificiallly and randomly thin out vegetation cover
veg_tif_file = ""  # Path to raw input tif file with 1 m resolution
vegden_tif_file = ""  # Path to raw input tif file with 1 m resolution
veg_min = 0.5  # [m] Minimum elevation for vegetation

# Save
save = False  # [bool] Whether or not to save finished arrays
savename = "NCB-NewDrum-Ocracoke_2018_PostFlorence"


# ================================================================================================================
# MAKE INIT

RNG = np.random.default_rng(seed=13)  # Seeded random numbers for reproducibility (e.g., model development/testing)

# Import
with rasterio.open(tif_file,  'r') as ds:
    full_dem = ds.read()
full_dem = full_dem[0, :, :]

if Veggie:
    with rasterio.open(veg_tif_file,  'r') as ds:
        full_veg = ds.read()
    full_veg = full_veg[0, :, :]
    with rasterio.open(vegden_tif_file,  'r') as ds:
        full_vegden = ds.read()
    full_vegden = full_vegden[0, :, :]
else:
    full_veg = full_dem.copy() * 0
    full_vegden = full_dem.copy() * 0

# Trim
dem = full_dem[max(0, ymin): ymax, xmin: xmax]
veg = full_veg[max(0, ymin): ymax, xmin: xmax]
vegden = full_vegden[max(0, ymin): ymax, xmin: xmax]
dem[dem > zmax] = 0  # Remove any very large values (essentially NaNs)
dem[dem < zmin] = 0  # Remove any very large values (essentially NaNs)
veg[veg > zmax] = 0  # Remove any very large values (essentially NaNs)
veg[veg < 0] = 0  # Remove any very large values (essentially NaNs)
vegden[vegden > zmax] = 0  # Remove any very large values (essentially NaNs)
vegden[vegden < 0] = 0  # Remove any very large values (essentially NaNs)
cs, ls = dem.shape
if ymin < 0:
    dem = np.vstack((np.ones([abs(ymin), ls]) * BB_depth * -1, dem))  # Add more bay
    veg = np.vstack((np.zeros([abs(ymin), ls]), veg))  # Add more bay (without veg)
    vegden = np.vstack((np.zeros([abs(ymin), ls]), vegden))  # Add more bay (without veg)
cs, ls = dem.shape

# Set back-barrier TODO: Account for barrier profiles that are entirely under MHW (i.e., inlets)
BB_shoreline = []
BB_slope = np.arange(1, BB_slope_length + 1) * BB_depth / BB_slope_length
BB_profile = np.hstack((BB_slope, np.ones([buffer]) * BB_depth)) * -1
for l in range(ls):
    BB_loc = np.argmax(dem[:, l] > BB_thresh)
    BB_shoreline.append(BB_loc)
    dem[:BB_loc, l] = np.flip(BB_profile[:BB_loc])

# Set shoreface
dem = np.rot90(dem, 2)  # Flip upside down
ocean_shoreline = []
SF_profile = np.arange(1, buffer + 1) * SF_slope * -1 + MHW
for l in range(ls):
    shoreline_loc = np.argmax(dem[:, l] > MHW)
    ocean_shoreline.append(shoreline_loc)
    dem[:shoreline_loc, l] = np.flip(SF_profile[:shoreline_loc])

add_bay_dem = np.ones([bay, ls]) * -BB_depth
add_bay_veg = np.zeros([bay, ls])
dem = np.vstack([dem, add_bay_dem])
veg = np.vstack([veg, add_bay_veg])
vegden = np.vstack([vegden, add_bay_veg])
cs, ls = dem.shape

# Rotate
dem = np.rot90(dem, 1)
veg = np.rot90(veg, 3)
vegden = np.rot90(vegden, 3)

# Veg
veg[dem < veg_min] = 0  # Remove veg below minimum elevation threshold
vegden[dem < veg_min] = 0  # Remove veg below minimum elevation threshold

dune_crest, not_gap = routine.foredune_crest(dem, MHW, cellsize=cellsize)

if VeggiePop:
    spec1, spec2 = populateVeg(dem,
                               veg,
                               MHW,
                               dune_crest,
                               density_slope=40,
                               density_intercept=-30,
                               elevation_mean=1.29,
                               elevation_tau=3,
                               elevation_maximum=0.75,
                               distance_r=0.1,
                               distance_u=50,
                               distance_maximum=0.9,
                               density_weight=0.5,
                               elevation_weight=0.25,
                               distance_weight=0.25,
                               )
else:
    spec1, spec2 = densityVeg(veg, vegden)

veg = spec1 + spec2

# Resize
if cellsize > 1:
    orig_size = dem.shape
    new_size = (int(orig_size[0] / cellsize), int(orig_size[1] / cellsize))
    dem = resize(dem, new_size)
    spec1 = resize(spec1, new_size)
    spec2 = resize(spec2, new_size)
    veg = resize(veg, new_size)
    dune_crest = routine.foredune_crest(dem, MHW, cellsize)

# ================================================================================================================
# PLOT & SAVE

# Plot
vmin = -1  # np.min(dem)
vmax = 5  # np.max(dem)
fig = plt.figure()
ax_1 = fig.add_subplot(111)
ax_1.matshow(dem, cmap='terrain', vmin=vmin, vmax=vmax)
plt.tight_layout()
# ax_1.plot(np.arange(len(ocean_shoreline)), ocean_shoreline, c='red')

fig = plt.figure()
ax_1 = fig.add_subplot(111)
ax_1.matshow(spec1, cmap='YlGn')
plt.plot(dune_crest, np.arange(len(dune_crest)))
plt.tight_layout()

fig = plt.figure()
ax_1 = fig.add_subplot(111)
ax_1.matshow(spec2, cmap='YlGn')
plt.tight_layout()

fig = plt.figure()
ax_1 = fig.add_subplot(111)
ax_1.matshow(veg, cmap='YlGn')
plt.tight_layout()

plt.show()

# Save
if save:
    Init = np.zeros([3, dem.shape[0], dem.shape[1]])
    Init[0, :, :] = dem
    Init[1, :, :] = spec1
    Init[2, :, :] = spec2

    name = "Init_" + savename
    outloc = "Input/" + name
    np.save(outloc, Init)
