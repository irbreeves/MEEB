""" This script converts a .tif elevation raster file to a numpy array initial elevation file suitable for
    use in the coupled DUBEVEG/Barrier3D model.

    IRB Reeves
    Last update: 9 January 2023
"""

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import copy

# Specifications
tif_file = "/Users/reevesi/Desktop/WHOI-USGS/Data/InitConditions/NCB_20190830_DEM.tif"
xlim1 = 22000 +900 # [m]
xlim2 = xlim1 + 1000  # [m]
ylim1 = -50  # [m]
ylim2 = 800  # [m]
MHW = 0.2  # [m initial datum]
BB_depth = 1.5  # [m] Back-barrier bay depth
BB_slope_length = 15  # [m] Length of slope from back-barrier shorline into bay
SF_slope = 0.02  # Shoreface slope
buffer = 500
EQ_berm_elev = 2.1  # [m] Berm elevation of equilibrium topography
EQ_beach_width = 90  # [m] Beach width of equilibrium topography
EQ_barrier_width = 650  # [m] Barrier width (including beach) of equilibrium topography
Veggie = True  # [bool] Whether or not to load & convert a contemporaneous init veg raster
veg_tif_file = "/Users/reevesi/Desktop/WHOI-USGS/Data/InitConditions/NCB_20190830_ModSAVI.tif"
veg_min = 0.45  # [m] Minimum elevation for vegetation
save = False  # [bool] Whether or not to save finished arrays
savename = "NCB_20190830_1000m_22900_OWflat"

RNG = np.random.default_rng(seed=13)  # Seeded random numbers for reproducibility (e.g., model development/testing)

if Veggie:
    # Import
    with rasterio.open(tif_file,  'r') as ds:
        full_dem = ds.read()
    full_dem = full_dem[0, :, :]
    with rasterio.open(veg_tif_file,  'r') as ds:
        full_veg = ds.read()
    full_veg = full_veg[0, :, :]

    # Trim
    dem = full_dem[max(0, ylim1): ylim2, xlim1: xlim2]
    veg = full_veg[max(0, ylim1): ylim2, xlim1: xlim2]
    dem -= MHW  # Convert to MHW datum
    dem[dem > 30] = 0  # Remove any very large values (essentially NaNs)
    dem[dem < -30] = 0  # Remove any very large values (essentially NaNs)
    veg[veg > 30] = 0  # Remove any very large values (essentially NaNs)
    veg[veg < -30] = 0  # Remove any very large values (essentially NaNs)
    cs, ls = dem.shape
    if ylim1 < 0:
        dem = np.vstack((np.ones([abs(ylim1), ls]) * BB_depth * -1, dem))  # Add more bay
        veg = np.vstack((np.zeros([abs(ylim1), ls]), veg))  # Add more bay (without veg)
    cs, ls = dem.shape

    # Set back-barrier
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

    # Equilibrium topography
    EQ_dem = copy.deepcopy(dem)
    EQ_beach = np.arange(EQ_beach_width) * EQ_berm_elev / EQ_beach_width
    EQ_interior = np.arange(EQ_barrier_width - EQ_beach_width) * EQ_berm_elev / (EQ_barrier_width - EQ_beach_width)
    EQ_profile = np.hstack((EQ_beach, np.flip(EQ_interior)))
    for l in range(ls):
        shoreline_loc = ocean_shoreline[l]
        if shoreline_loc + len(EQ_profile) >= cs:
            trim = shoreline_loc + len(EQ_profile) - cs
            EQ_profile_trim = EQ_profile[:-trim]
            EQ_dem[shoreline_loc:, l] = EQ_profile_trim
        else:
            add = cs - shoreline_loc - len(EQ_profile)
            EQ_dem[shoreline_loc:shoreline_loc + len(EQ_profile), l] = EQ_profile
            EQ_dem[shoreline_loc + len(EQ_profile):, l] = BB_profile[:add]

    # Rotate
    dem = np.rot90(dem, 1)
    EQ_dem = np.rot90(EQ_dem, 1)
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

    # Plot
    vmin = np.min(dem)
    vmax = np.max(dem)
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

    fig = plt.figure()
    ax_1 = fig.add_subplot(111)
    ax_1.matshow(EQ_dem, cmap='terrain', vmin=vmin, vmax=vmax)
    plt.tight_layout
    # ax_1.plot(np.arange(len(ocean_shoreline)), ocean_shoreline, c='red')

    plt.show()

    # Save
    if save:
        Init = np.zeros([4, dem.shape[0], dem.shape[1]])
        Init[0, :, :] = dem
        Init[1, :, :] = EQ_dem
        Init[2, :, :] = spec1
        Init[3, :, :] = spec2

        name = "Init_" + savename
        outloc = "Input/" + name
        np.save(outloc, Init)

        # name = "EQTopo_" + savename
        # outloc = "Input/" + name
        # np.save(outloc, EQ_dem)
        #
        # name = "Spec1_" + savename
        # outloc = "Input/" + name
        # np.save(outloc, spec1)
        #
        # name = "Spec2_" + savename
        # outloc = "Input/" + name
        # np.save(outloc, spec2)

else:
    # Import
    with rasterio.open(tif_file, 'r') as ds:
        full_dem = ds.read()
    full_dem = full_dem[0, :, :]

    # Trim
    dem = full_dem[max(0, ylim1): ylim2, xlim1: xlim2]
    dem -= MHW  # Convert to MHW datum
    dem[dem > 30] = 0  # Remove any very large values (essentially NaNs)
    dem[dem < -30] = 0  # Remove any very large values (essentially NaNs)
    cs, ls = dem.shape
    if ylim1 < 0:
        dem = np.vstack((np.ones([abs(ylim1), ls]) * BB_depth * -1, dem))  # Add more bay
    cs, ls = dem.shape

    # Set back-barrier
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

    # Equilibrium topography
    EQ_dem = copy.deepcopy(dem)
    EQ_beach = np.arange(EQ_beach_width) * EQ_berm_elev / EQ_beach_width
    EQ_interior = np.arange(EQ_barrier_width - EQ_beach_width) * EQ_berm_elev / (EQ_barrier_width - EQ_beach_width)
    EQ_profile = np.hstack((EQ_beach, np.flip(EQ_interior)))
    for l in range(ls):
        shoreline_loc = ocean_shoreline[l]
        if shoreline_loc + len(EQ_profile) >= cs:
            trim = shoreline_loc + len(EQ_profile) - cs
            EQ_profile_trim = EQ_profile[:-trim]
            EQ_dem[shoreline_loc:, l] = EQ_profile_trim
        else:
            add = cs - shoreline_loc - len(EQ_profile)
            EQ_dem[shoreline_loc:shoreline_loc + len(EQ_profile), l] = EQ_profile
            EQ_dem[shoreline_loc + len(EQ_profile):, l] = BB_profile[:add]

    # Rotate
    dem = np.rot90(dem, 1)
    EQ_dem = np.rot90(EQ_dem, 1)

    # Plot
    vmin = np.min(dem)
    vmax = np.max(dem)
    fig = plt.figure()
    ax_1 = fig.add_subplot(111)
    ax_1.matshow(dem, cmap='terrain', vmin=vmin, vmax=vmax)
    plt.tight_layout
    # ax_1.plot(np.arange(len(ocean_shoreline)), ocean_shoreline, c='red')

    fig = plt.figure()
    ax_1 = fig.add_subplot(111)
    ax_1.matshow(EQ_dem, cmap='terrain', vmin=vmin, vmax=vmax)
    plt.tight_layout
    # ax_1.plot(np.arange(len(ocean_shoreline)), ocean_shoreline, c='red')

    plt.show()

    # Save
    if save:
        name = "Topo_" + savename
        outloc = "Input/" + name
        np.save(outloc, dem)

        name = "EQTopo_" + savename
        outloc = "Input/" + name
        np.save(outloc, EQ_dem)