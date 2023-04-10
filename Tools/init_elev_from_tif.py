""" This script converts a .tif elevation raster file to a numpy array initial elevation file suitable for
    use in BEEM (Barrier Explicit Evolution Model).

    IRB Reeves
    Last update: 9 January 2023
"""

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import copy


# ================================================================================================================
# SPECIFICATIONS

# Elevation (NAVD88)
tif_file = "/Volumes/IRBR256/USGS/NCB_Data/NCB_Full/PostIrene_2011829_NOAA_FullNCB.tif"
MHW = 0.3  # [m initial datum] For finding back-barrier shoreline (0.36, Duke Marine Lab, Beaufort)
BB_depth = 1.5  # [m] Back-barrier bay depth
BB_slope_length = 30  # [m] Length of slope from back-barrier shorline into bay
SF_slope = 0.02  # Equilibrium shoreface slope
buffer = 3000  #

# Domain
xmin = 343  # [m] Alongshore coordinate for start of domain
xmax = 35783  # [m] Alongshore coordinate for end of domain
ymin = 0  # [m] Cross-shore coordinate for start of domain
ymax = 1949  # [m] Cross-shore coordinate for end of domain
bay = 100  # [m] Additional width of bay to ad to domain

# Vegetation
Veggie = True  # [bool] Whether or not to load & convert a contemporaneous init veg raster
veg_tif_file = "/Volumes/IRBR256/USGS/NCB_Data/NCB_Full/ModSAVI_20190830_FullNCB.tif"
veg_min = 0.45  # [m] Minimum elevation for vegetation

# Save
save = False  # [bool] Whether or not to save finished arrays
savename = "Init_NCB_2009_PreIrene"


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
else:
    full_veg = full_dem.copy() * 0

# Trim
dem = full_dem[max(0, ymin): ymax, xmin: xmax]
veg = full_veg[max(0, ymin): ymax, xmin: xmax]
dem[dem > 30] = 0  # Remove any very large values (essentially NaNs)
dem[dem < -30] = 0  # Remove any very large values (essentially NaNs)
veg[veg > 30] = 0  # Remove any very large values (essentially NaNs)
veg[veg < -30] = 0  # Remove any very large values (essentially NaNs)
cs, ls = dem.shape
if ymin < 0:
    dem = np.vstack((np.ones([abs(ymin), ls]) * BB_depth * -1, dem))  # Add more bay
    veg = np.vstack((np.zeros([abs(ymin), ls]), veg))  # Add more bay (without veg)
cs, ls = dem.shape

# Set back-barrier TODO: Account for barrier profiles that are entirely under MHW
BB_shoreline = []
BB_slope = np.arange(1, BB_slope_length + 1) * BB_depth / BB_slope_length
BB_profile = np.hstack((BB_slope, np.ones([buffer]) * BB_depth)) * -1
for l in range(ls):
    BB_loc = np.argmax(dem[:, l] > MHW)
    BB_shoreline.append(BB_loc)
    dem[:BB_loc, l] = np.flip(BB_profile[:BB_loc])

# Set shoreface
dem = np.rot90(dem, 2)  # Flip upside down
ocean_shoreline = []
SF_profile = np.arange(1, buffer + 1) * SF_slope * -1
for l in range(ls):
    shoreline_loc = np.argmax(dem[:, l] > 0)
    ocean_shoreline.append(shoreline_loc)
    dem[:shoreline_loc, l] = np.flip(SF_profile[:shoreline_loc])

add_bay_dem = np.ones([bay, ls]) * -BB_depth
add_bay_veg = np.zeros([bay, ls])
dem = np.vstack([dem, add_bay_dem])
veg = np.vstack([veg, add_bay_veg])
cs, ls = dem.shape

# Rotate
dem = np.rot90(dem, 1)
veg = np.rot90(veg, 3)

# Veg
veg[dem < veg_min] = 0  # Remove veg below minimum elevation threshold
spec1 = np.zeros(veg.shape)
spec2 = np.zeros(veg.shape)
spec_rand = RNG.random(spec1.shape)  # Randomly decide species, weighted by density
spec1[veg == 1] = (spec_rand[veg == 1] < 0.75) * 0.2
spec1[veg == 2] = (spec_rand[veg == 2] < 0.50) * 0.4
spec1[veg == 3] = (spec_rand[veg == 3] < 0.25) * 0.6
spec2[veg == 1] = (spec_rand[veg == 1] > 0.75) * 0.2
spec2[veg == 2] = (spec_rand[veg == 2] > 0.50) * 0.4
spec2[veg == 3] = (spec_rand[veg == 3] > 0.25) * 0.6


# ================================================================================================================
# PLOT & SAVE

# Plot
vmin = -1  # np.min(dem)
vmax = 5  # np.max(dem)
fig = plt.figure()
ax_1 = fig.add_subplot(111)
ax_1.matshow(dem, cmap='terrain', vmin=vmin, vmax=vmax)
plt.tight_layout
# ax_1.plot(np.arange(len(ocean_shoreline)), ocean_shoreline, c='red')

fig = plt.figure()
ax_1 = fig.add_subplot(111)
ax_1.matshow(spec1, cmap='YlGn')
plt.tight_layout

fig = plt.figure()
ax_1 = fig.add_subplot(111)
ax_1.matshow(spec2, cmap='YlGn')
plt.tight_layout

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
