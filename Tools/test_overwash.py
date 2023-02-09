"""
Script for testing overwash function.
IRBR 6 Jan 2023
"""

import numpy as np
import matplotlib.pyplot as plt
import routines_dubeveg as routine
import copy
import time
from scipy.interpolate import RegularGridInterpolator


# _____________________________________________
def regrid(data, out_x, out_y):
    m = max(data.shape[0], data.shape[1])
    y = np.linspace(0, 1.0 / m, data.shape[0])
    x = np.linspace(0, 1.0 / m, data.shape[1])
    interpolating_function = RegularGridInterpolator((y, x), data)

    yv, xv = np.meshgrid(np.linspace(0, 1.0 / m, out_y), np.linspace(0, 1.0 / m, out_x))

    return interpolating_function((xv, yv))


start_time = time.time()  # Record time at start of run

# _____________________________________________
# Define Variables
Rhigh = 3.3
Rlow = 0.9
dur = 35
slabheight_m = 0.1
MHT = 0
# name = "ss=5, Rin=50, MaxUpSlope=0.5, YES Uphill, Kr=5e-07, fluxlimit=1, Rhigh=3.3, Q(S+C)**2"
#
# print(name)

# Initial topo
compare = True  # Compare overwash simulation change to observed change from 2 DEMs?
# Init = np.load("Input/Init_NCB_20190830_500m_20200_LinearRidge.npy")
# Init = np.load("Input/Init_NCB_20190830_250m_22550_LrgGap.npy")
Init = np.load("../Input/Init_NCB_2017_2000m_12000_GapsPreFlorence.npy")
if compare:
    Init2 = np.load("../Input/Init_NCB_2018_2000m_12000_GapsPostFlorence.npy")


# _____________________________________________
# Conversions & Initializations
xmin = 85  # 1685  # 85  # 1550
xmax = 385  # 1885  # 385  # 1675
topo_init = Init[0, xmin: xmax, :]
eqtopo_init = Init[1, xmin: xmax, :]
spec1 = Init[2, xmin: xmax, :]
spec2 = Init[3, xmin: xmax, :]
noise = np.ones(topo_init.shape) + np.random.uniform(-0.25, 0.25, topo_init.shape)
topo0 = topo_init / slabheight_m  # + noise  # [slabs] Transform from m into number of slabs
topo = copy.deepcopy(topo0)  # np.round(topo0)  # [slabs] Initialise the topography map
eqtopo0 = eqtopo_init / slabheight_m  # [slabs] Transform from m into number of slabs
eqtopo = copy.deepcopy(eqtopo0)  # np.round(eqtopo0)  # [slabs] Initialise the topography map
if compare:
    topo_init2 = Init2[0, xmin:xmax, :]
    topo0_2 = topo_init2 / slabheight_m  # [slabs] Transform from m into number of slabs
    topo2 = copy.deepcopy(topo0_2)  # np.round(topo0_2)  # [slabs] Initialise the topography map
    obs_change = topo0_2 - topo0

# Temp
# x, y = topo.shape
# newx = int(x/10)
# newy = int(y/10)
# topo = regrid(topo, newx, newy)
# eqtopo = regrid(eqtopo, newx, newy)
# spec1 = regrid(spec1, newx, newy)
# spec2 = regrid(spec2, newx, newy)
#
# # Temp
# topo[5, 10] = 20.2
# dune_crest = np.ones([100]) * 113
# topo[:, 113] = 38
# topo[45: 55, 113] = 21.15
# dune_crest = dune_crest.astype(int)

# Set veg
veg = spec1 + spec2  # Determine the initial cumulative vegetation effectiveness
veg[veg > 1] = 1  # Cumulative vegetation effectiveness cannot be negative or larger than one
veg[veg < 0] = 0

# Find Dune Crest, Beach Slopes
dune_crest = routine.foredune_crest(topo, eqtopo, veg)
dune_heel = routine.foredune_heel(topo, dune_crest, 0.6, slabheight_m)
slopes = routine.beach_slopes(eqtopo, MHT, dune_crest, slabheight_m)

# dune_crest[75: 128] = 153  # for 1685-1885
# dune_crest[77 + 135: 127 + 135] = 157  # for 1685-1885
# dune_crest[77 - 30: 127 - 30] = 157  # 1715-1845


# _____________________________________________
# Beach & Dune Change
topo_prebeach = copy.deepcopy(topo)

# xx = 53
# # topo[xx, 16:69] = np.linspace(0, topo[xx, 69], 69 - 16)
# topo_post, sedflux = routine.calc_dune_erosion3(topo[xx:xx + 2, :],
#                                                 dx=1,
#                                                 crestline=dune_crest[xx:xx + 2],
#                                                 MHW=0,
#                                                 Rhigh=[Rhigh, Rhigh],
#                                                 T=12,
#                                                 duration=24,
#                                                 veg=veg[xx:xx + 2, :],
#                                                 eqtopo=eqtopo,
#                                                 RNG=np.random.default_rng(seed=13))
# topo_post, sedflux = routine.calc_dune_erosion(topo[xx:xx + 2, :],
#                                                dx=1,
#                                                MHW=0,
#                                                Rhigh=[Rhigh, Rhigh],
#                                                Q=0.5,
#                                                storm_iter=100,)
# Fig = plt.figure(figsize=(13, 6))
# ax1 = Fig.add_subplot(211)
# ax1.plot(topo_prebeach[xx, :200])
# ax1.plot(topo_post[0, :200])
# ax1.set_ylabel("Elevation [m]")
# ax2 = Fig.add_subplot(212)
# ax2.plot(sedflux[0, :200])
# ax2.set_ylabel("Sediment Flux")
# print('Done!')
# plt.tight_layout()
# plt.show()

# # Beach Update
# topo, inundated, pbeachupdate, diss, cumdiss, pwave, crestline_change = routine.marine_processes_Rhigh(
#     Rhigh / slabheight_m,  # Convert to slabs
#     slabheight_m,
#     1,
#     topo,
#     eqtopo,
#     veg,
#     m26f=0.012,
#     m27af=1.0,
#     m28f=0.8,
#     pwavemaxf=1.0,
#     pwaveminf=0.1,
#     depthlimitf=0.01,
#     shelterf=1.0,
#     crestline=dune_crest,
# )


# _____________________________________________
# Overwash
name = "ss=6, Rin=50, MaxUpSlope=1, Kr=5e-04, fluxlimit=1, QSC Cx=15, Time-varying Qdc"
print(name)

topo_preoverwash = copy.deepcopy(topo)

endtopo, topo_change_overwash, topo_change_leftover, OWflux, netDischarge = routine.storm_processes(
    topo,
    np.zeros(topo.shape),
    eqtopo,
    veg,
    Rhigh,
    Rlow,
    dur,
    slabheight_m,
    threshold_in=0.25,
    Rin_i=5,
    Rin_r=50,  # was 1.2
    Cx=15,  # was 10
    AvgSlope=2/200,
    nn=0.5,
    MaxUpSlope=1,  # was 0.25
    Qs_min=1,
    Kr=5e-04,  # was 7.5e-05   1e-4
    Ki=1e-06,  # was 7.5e-06
    mm=2,
    MHT=MHT,
    Cbb_i=0.85,
    Cbb_r=0.7,
    Qs_bb_min=1,
    substep_i=6,
    substep_r=6,  # 6 seems ideal
)

topo_change_prebeach = endtopo - topo_prebeach
topo_change_preoverwash = endtopo - topo_preoverwash

SimDuration = time.time() - start_time
print()
print("Elapsed Time: ", SimDuration, "sec")

# _____________________________________________
# NSE
if compare:
    A = 0
    B = 0
    longshore, crossshore = topo.shape

    # obs_change[netDischarge <= 0] = 0

    obs_change_sum = 0
    count = 0
    for y in range(longshore):
        crest = dune_crest[y]
        for x in range(crest + 1, crossshore):
            obs_change_sum += obs_change[y, x]
            count += 1
    obs_change_mean = obs_change_sum / count

    for y in range(longshore):
        crest = dune_crest[y]
        for x in range(crest + 1, crossshore):
            A += (obs_change[y, x] - topo_change_preoverwash[y, x]) ** 2
            B += (obs_change[y, x] - obs_change_mean) ** 2

    NSE = 1 - A / B
    print("  --> NSE", NSE)


# _____________________________________________
# Plot

pxmin = 0
pxmax = 500
if not compare:
    # Pre-Overwash
    Fig = plt.figure(figsize=(14, 7.5))
    ax1 = Fig.add_subplot(221)
    cax1 = ax1.matshow(topo_prebeach[:, pxmin: pxmax] * 0.1, cmap='terrain', vmin=-1.2, vmax=5.0)
    ax1.plot(dune_crest, np.arange(len(dune_crest)), c='black', alpha=0.6)
    # cbar = Fig.colorbar(cax1)
    # cbar.set_label('Start Elevation [m MHW]', rotation=270, labelpad=20)
    plt.title(name)

    ax2 = Fig.add_subplot(222)
    cax2 = ax2.matshow(endtopo[:, pxmin: pxmax] * 0.1, cmap='terrain', vmin=-1.2, vmax=5.0)
    # cbar = Fig.colorbar(cax2)
    # cbar.set_label('End Elevation [m MHW]', rotation=270, labelpad=20)

    maxxx = max(abs(np.min(topo_change_prebeach) * 0.1), abs(np.max(topo_change_prebeach) * 0.1))
    ax3 = Fig.add_subplot(223)
    cax3 = ax3.matshow(topo_change_prebeach[:, pxmin: pxmax] * 0.1, cmap='bwr', vmin=-maxxx, vmax=maxxx)
    ax3.plot(dune_crest, np.arange(len(dune_crest)), c='black', alpha=0.6)
    # cbar = Fig.colorbar(cax3)
    # cbar.set_label('Change [m]', rotation=270, labelpad=20)

    ax4 = Fig.add_subplot(224)
    cax4 = ax4.matshow(netDischarge[:, pxmin: pxmax], cmap='viridis')
    ax4.plot(dune_crest, np.arange(len(dune_crest)), c='black', alpha=0.6)
    # cbar = Fig.colorbar(cax4)
    # cbar.set_label('Net Discharge [m^3]', rotation=270, labelpad=20)
    plt.tight_layout()

    plt.figure(figsize=(14, 7.5))
    plt.plot(np.sum(netDischarge, axis=0))
    plt.ylabel("Cumulative Discharge")

    print()
    print("Complete.")
    plt.show()

else:
    # Pre Overwash
    Fig = plt.figure(figsize=(14, 7.5))
    ax1 = Fig.add_subplot(221)
    cax1 = ax1.matshow(topo_preoverwash[:, pxmin: pxmax] * 0.1, cmap='terrain', vmin=-1.2, vmax=5.0)
    ax1.plot(dune_crest, np.arange(len(dune_crest)), c='black', alpha=0.6)
    # cbar = Fig.colorbar(cax1)
    # cbar.set_label('Start Elevation [m MHW]', rotation=270, labelpad=20)
    plt.title(name)

    ax2 = Fig.add_subplot(222)
    cax2 = ax2.matshow(endtopo[:, pxmin: pxmax] * 0.1, cmap='terrain', vmin=-1.2, vmax=5.0)
    # cbar = Fig.colorbar(cax2)
    # cbar.set_label('End Elevation [m MHW]', rotation=270, labelpad=20)

    maxx = max(abs(np.min(obs_change)) * 0.1, abs(np.max(obs_change)) * 0.1)
    maxxx = max(abs(np.min(topo_change_preoverwash) * 0.1), abs(np.max(topo_change_preoverwash) * 0.1))
    maxxxx = 1  # max(maxx, maxxx)
    ax3 = Fig.add_subplot(223)
    cax3 = ax3.matshow(topo_change_preoverwash[:, pxmin: pxmax] * 0.1, cmap='bwr', vmin=-maxxxx, vmax=maxxxx)
    ax3.plot(dune_crest, np.arange(len(dune_crest)), c='black', alpha=0.6)
    # cbar = Fig.colorbar(cax3)
    # cbar.set_label('Change [m]', rotation=270, labelpad=20)

    ax4 = Fig.add_subplot(224)
    cax4 = ax4.matshow(obs_change[:, pxmin: pxmax] * 0.1, cmap='bwr', vmin=-maxxxx, vmax=maxxxx)
    ax4.plot(dune_crest, np.arange(len(dune_crest)), c='black', alpha=0.6)
    # cbar = Fig.colorbar(cax4)
    # cbar.set_label('Net Discharge [m^3]', rotation=270, labelpad=20)
    plt.tight_layout()

    plt.figure(figsize=(14, 7.5))
    plt.plot(np.sum(netDischarge, axis=0))
    plt.ylabel("Cumulative Discharge")

    print()
    print("Complete.")
    plt.show()
