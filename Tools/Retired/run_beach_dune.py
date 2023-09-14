"""
Script for running MEEB dune/beach function.
IRBR 14 September 2023
"""

import numpy as np
import matplotlib.pyplot as plt
import routines_meeb as routine
import copy
import time
from matplotlib import colors

start_time = time.time()  # Record time at start of run

# _____________________________________________
# Define Variables
Rhigh = 0.4
Rlow = -0.3
dur = 70
slabheight_m = 0.02
MHW = 0.0
repose_bare = 20  # [deg]
repose_veg = 30  # [deg]
repose_threshold = 0.3

# Initial Observed Topo
Init = np.load("Input/Init_NCB-NewDrum-Ocracoke_2016_PostMatthew.npy")

# Define Alongshore Coordinates of Domain
xmin = 6300
xmax = 6500

# _____________________________________________
# Conversions & Initializations

# Transform Initial Observed Topo
topo_init = Init[0, xmin: xmax, :]  # * slabheight_m  # [m]
topo0 = topo_init  # [m]
topo = copy.deepcopy(topo0)  # [m] Initialise the topography map

# Set Veg Domain
spec1 = Init[1, xmin: xmax, :]
spec2 = Init[2, xmin: xmax, :]
veg = spec1 + spec2  # Determine the initial cumulative vegetation effectiveness
veg[veg > 1] = 1  # Cumulative vegetation effectiveness cannot be negative or larger than one
veg[veg < 0] = 0

# Find Dune Crest, Shoreline
dune_crest = routine.foredune_crest(topo, MHW)
x_s = routine.ocean_shoreline(topo, MHW)
print("Start XS:", x_s[84])

# Transform water levels to vectors
Rhigh = Rhigh * np.ones(topo_init.shape[0])
Rlow = Rlow * np.ones(topo_init.shape[0])

dune_beach_volume_change = np.zeros([topo.shape[0]])
RNG = np.random.default_rng(seed=13)


# _____________________________________________
# Overwash, Beach, & Dune Change
topo_prestorm = copy.deepcopy(topo)

name = ""
print(name)

for t in range(dur):
    topo, dV, wetMap = routine.calc_dune_erosion_TS(
        topo=topo,
        dx=1,
        crestline=dune_crest,
        x_s=x_s,
        MHW=MHW,
        Rhigh=Rhigh,
        Beq=0.02,
        Et=2.73,
        substeps=22,
    )

    dune_beach_volume_change += dV

# topo = routine.enforceslopes2(topo, veg, slabheight_m, repose_bare, repose_veg, repose_threshold, RNG)[0]

sim_topo_final = topo
topo_change_prestorm = sim_topo_final - topo_prestorm

SimDuration = time.time() - start_time
print()
print("Elapsed Time: ", SimDuration, "sec")

# _____________________________________________
# Model Skill: Comparisons to Observations

longshore, crossshore = sim_topo_final.shape

# Final Elevations
sim_final_m = sim_topo_final  # [m] Simulated final topo

# # Final Elevation Changes
sim_change_m = topo_change_prestorm  # [m] Simulated change


# _____________________________________________
# Plot

ymin = 700
ymax = 1100

# Change Comparisons
cmap1 = routine.truncate_colormap(copy.copy(plt.cm.get_cmap("terrain")), 0.5, 0.9)  # Truncate colormap
cmap1.set_bad(color='dodgerblue', alpha=0.5)  # Set cell color below MHW to blue

# Pre Storm (Observed) Topo
Fig = plt.figure(figsize=(11, 7.5))
ax1 = Fig.add_subplot(221)
topo1 = topo_prestorm[:, ymin: ymax]  # [m]
topo1 = np.ma.masked_where(topo1 < MHW, topo1)  # Mask cells below MHW
cax1 = ax1.matshow(topo1, cmap=cmap1, vmin=0, vmax=5.0)
ax1.plot(dune_crest - ymin, np.arange(len(dune_crest)), c='black', alpha=0.6)
plt.title(name)

# Post-Storm (Simulated) Topo
ax2 = Fig.add_subplot(222)
topo2 = sim_topo_final[:, ymin: ymax]  # [m]
topo2 = np.ma.masked_where(topo2 < MHW, topo2)  # Mask cells below MHW
cax2 = ax2.matshow(topo2, cmap=cmap1, vmin=0, vmax=5.0)
# cbar = Fig.colorbar(cax2)
# cbar.set_label('Elevation [m MHW]', rotation=270, labelpad=20)

# Simulated Topo Change
maxx = max(abs(np.min(sim_change_m)), abs(np.max(sim_change_m)))
maxxx = max(abs(np.min(sim_change_m)), abs(np.max(sim_change_m)))
maxxxx = 1  # max(maxx, maxxx)
ax3 = Fig.add_subplot(223)
cax3 = ax3.matshow(sim_change_m[:, ymin: ymax], cmap='bwr', vmin=-maxxxx, vmax=maxxxx)
ax3.plot(dune_crest - ymin, np.arange(len(dune_crest)), c='black', alpha=0.6)
# cbar = Fig.colorbar(cax3)
# cbar.set_label('Change [m]', rotation=270, labelpad=20)

# xx = 84
# proffig = plt.figure(figsize=(11, 7.5))
# plt.plot(topo_prestorm[xx, 50:pxmax], c='black')
# plt.plot(sim_topo_final[xx, 50:pxmax], c='red')
# plt.legend(["Pre", "Post Sim"])

plt.title(name)

print()
print("Complete.")
plt.show()
