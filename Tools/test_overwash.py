"""
Script for testing overwash function.
IRBR 9 Feb 2023
"""

import numpy as np
import matplotlib.pyplot as plt
import routines_dubeveg as routine
import copy
import time

start_time = time.time()  # Record time at start of run


# _____________________________________________
# Define Variables
Rhigh = 3.3
Rlow = 0.9
dur = 35
slabheight_m = 0.1
MHT = 0

# Initial Observed Topo
Init = np.load("Input/Init_NCB_2017_2000m_12000_GapsPreFlorence.npy")
# Final Observed
End = np.load("Input/Init_NCB_2018_2000m_12000_GapsPostFlorence.npy")

# Define Alongshore Coordinates of Domain
xmin = 1550  # 1685  # 85  # 1550
xmax = 1675  # 1885  # 385  # 1675


# _____________________________________________
# Conversions & Initializations

# Transform Initial Observed Topo
topo_init = Init[0, xmin: xmax, :]  # [m]
topo0 = topo_init / slabheight_m  # [slabs] Transform from m into number of slabs
topo = copy.deepcopy(topo0)  # np.round(topo0)  # [slabs] Initialise the topography map

# Transform Final Observed Topo
topo_final = End[0, xmin:xmax, :]  # [m]
topox = topo_final / slabheight_m  # [slabs] Transform from m into number of slabs
obs_change = topox - topo0  # [slabs] Observed change

# Equilibrium Topo (Temp)
eqtopo_init = Init[1, xmin: xmax, :]  # [m]
eqtopo0 = eqtopo_init / slabheight_m  # [slabs] Transform from m into number of slabs
eqtopo = copy.deepcopy(eqtopo0)  # np.round(eqtopo0)  # [slabs] Initialise the topography map

# Set Veg Domain
spec1 = Init[2, xmin: xmax, :]
spec2 = Init[3, xmin: xmax, :]
veg = spec1 + spec2  # Determine the initial cumulative vegetation effectiveness
veg[veg > 1] = 1  # Cumulative vegetation effectiveness cannot be negative or larger than one
veg[veg < 0] = 0

# Find Dune Crest, Beach Slopes
dune_crest = routine.foredune_crest(topo, eqtopo, veg)

# TEMP
# dune_crest[75: 128] = 153  # for 1685-1885
# dune_crest[77 + 135: 127 + 135] = 157  # for 1685-1885
# dune_crest[77 - 30: 127 - 30] = 157  # 1715-1845


# _____________________________________________
# Overwash, Beach, & Dune Change
topo_prestorm = copy.deepcopy(topo)

name = "ss=3, Rin=50, MaxUpSlope=1, Kr=1e-04, fluxlimit=1, Q"
print(name)

sim_topo_final, topo_change_overwash, topo_change_leftover, OWflux, netDischarge = routine.storm_processes(
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
    Kr=1e-04,  # was 7.5e-05   1e-4
    Ki=1e-06,  # was 7.5e-06
    mm=2,
    MHT=MHT,
    Cbb_i=0.85,
    Cbb_r=0.7,
    Qs_bb_min=1,
    substep_i=6,
    substep_r=3,  # 6 seems ideal
)

topo_change_prestorm = sim_topo_final - topo_prestorm

SimDuration = time.time() - start_time
print()
print("Elapsed Time: ", SimDuration, "sec")


# _____________________________________________
# NSE

Florence_Overwash_Mask = np.load("Input/FlorenceOverwashMask.npy")
OW_Mask = Florence_Overwash_Mask[xmin: xmax, :]

obs_change_masked = obs_change * slabheight_m * OW_Mask  # [m]
sim_change_masked = topo_change_prestorm * slabheight_m * OW_Mask  # [m]

A = 0
B = 0
longshore, crossshore = sim_topo_final.shape

obs_masked_sum = 0
count = 0
for y in range(longshore):
    crest = dune_crest[y]
    for x in range(crest + 1, crossshore):
        obs_masked_sum += obs_change_masked[y, x]
        count += 1
obs_change_mean = obs_masked_sum / count

for y in range(longshore):
    crest = dune_crest[y]
    for x in range(crest + 1, crossshore):
        A += (obs_change_masked[y, x] - sim_change_masked[y, x]) ** 2
        B += (obs_change_masked[y, x] - obs_change_mean) ** 2

NSE = 1 - A / B
print("  --> NSE", NSE)

# _____________________________________________
# Objective Function

OW_Mask_NoBeach = copy.deepcopy(OW_Mask)
for l in range(longshore):
    for c in range(crossshore):
        if c < dune_crest[l]:
            OW_Mask_NoBeach[l, c] = False

obs_final_m = topo_final * slabheight_m  # [m] Observed final topo
sim_final_m = sim_topo_final * slabheight_m  # [m] Simulated final topo

wi = 1 / (1 ** 2 * sim_final_m.size)  # Weight factor for model misfit score; Barnhart et al. (2020a) Eqn 3
OF = np.sum(wi * (obs_final_m - sim_final_m) ** 2) ** 2  # Model misfit score; Barnhart et al. (2020a) Eqn 4

print("  --> OF", OF)

wi_masked = np.sum(OW_Mask_NoBeach)
# obs_final_m_masked = obs_final_m * OW_Mask  # [m] Observed final topo
# sim_final_m_masked = sim_final_m * OW_Mask  # [m] Simulated final topo

OF_masked = np.sum(wi_masked * (obs_final_m[OW_Mask_NoBeach] - sim_final_m[OW_Mask_NoBeach]) ** 2) ** 2  # Model misfit score; Barnhart et al. (2020a) Eqn 4

print("  --> OF_masked", OF_masked)

# _____________________________________________
# Plot

pxmin = 0
pxmax = 500

# Pre Storm (Observed) Topo
Fig = plt.figure(figsize=(14, 7.5))
ax1 = Fig.add_subplot(221)
cax1 = ax1.matshow(topo_prestorm[:, pxmin: pxmax] * 0.1, cmap='terrain', vmin=-1.2, vmax=5.0)
ax1.plot(dune_crest, np.arange(len(dune_crest)), c='black', alpha=0.6)
# cbar = Fig.colorbar(cax1)
# cbar.set_label('Start Elevation [m MHW]', rotation=270, labelpad=20)
plt.title(name)

# Post-Storm (Simulated) Topo
ax2 = Fig.add_subplot(222)
cax2 = ax2.matshow(sim_topo_final[:, pxmin: pxmax] * 0.1, cmap='terrain', vmin=-1.2, vmax=5.0)
# cbar = Fig.colorbar(cax2)
# cbar.set_label('End Elevation [m MHW]', rotation=270, labelpad=20)

# Simulated Topo Change
maxx = max(abs(np.min(obs_change_masked)), abs(np.max(obs_change_masked)))
maxxx = max(abs(np.min(sim_change_masked)), abs(np.max(sim_change_masked)))
maxxxx = 1  # max(maxx, maxxx)
ax3 = Fig.add_subplot(223)
cax3 = ax3.matshow(sim_change_masked[:, pxmin: pxmax], cmap='bwr', vmin=-maxxxx, vmax=maxxxx)
ax3.plot(dune_crest, np.arange(len(dune_crest)), c='black', alpha=0.6)
# cbar = Fig.colorbar(cax3)
# cbar.set_label('Change [m]', rotation=270, labelpad=20)

# Observed Topo Change
ax4 = Fig.add_subplot(224)
cax4 = ax4.matshow(obs_change_masked[:, pxmin: pxmax], cmap='bwr', vmin=-maxxxx, vmax=maxxxx)
ax4.plot(dune_crest, np.arange(len(dune_crest)), c='black', alpha=0.6)
# cbar = Fig.colorbar(cax4)
# cbar.set_label('Net Discharge [m^3]', rotation=270, labelpad=20)
plt.tight_layout()

# Cumulative Discharge
plt.figure(figsize=(14, 7.5))
plt.plot(np.sum(netDischarge, axis=0))
plt.ylabel("Cumulative Discharge")

print()
print("Complete.")
plt.show()
