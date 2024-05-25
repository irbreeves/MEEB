"""
Script for testing MEEB storm beach/dune function.
IRBR 26 October 2023
"""

import numpy as np
import matplotlib.pyplot as plt
import routines_meeb as routine
import copy
import time
import scipy

start_time = time.time()  # Record time at start of run

# _____________________________________________
# Define Variables
Rhigh = 3.32  # [m NAVD88]
Rlow = 1.90  # [m NAVD88]
dur = 83  # [hr]
cellsize = 1  # [m]
MHW = 0.39  # [m NAVD88]

# Initial Observed Topo
Init = np.load("Input/Init_NCB-NewDrum-Ocracoke_2017_PreFlorence.npy")
# Final Observed
End = np.load("Input/Init_NCB-NewDrum-Ocracoke_2018_PostFlorence-Plover.npy")

# Define Alongshore Coordinates of Domain
xmin = 18950  # 20525  # 19825  # 18950
xmax = 19250  # 20725  # 20275  # 19250

ymin = 900
ymax = ymin + 500

ymin_p = ymin + 115
ymax_p = ymin + 270

# _____________________________________________
# Conversions & Initializations

# Initial Observed Topo
topo0 = Init[0, xmin: xmax, :]  # [m NAVD88]
topo = copy.deepcopy(topo0)  # [m NAVD88] Initialise the topography map

# Final Observed Topo
obs_topo_final = End[0, xmin:xmax, :]  # [m NAVD88]

# Set Veg Domain
spec1 = Init[1, xmin: xmax, :]
spec2 = Init[2, xmin: xmax, :]
veg = spec1 + spec2  # Determine the initial cumulative vegetation effectiveness
veg[veg > 1] = 1  # Cumulative vegetation effectiveness cannot be negative or larger than one
veg[veg < 0] = 0

# Find Dune Crest, Shoreline Positions
dune_crest, not_gap = routine.foredune_crest(topo, MHW)
x_s = routine.ocean_shoreline(topo, MHW)

crestflux = np.ones(len(dune_crest)) * -2.548136316998706e-06

# Transform water levels to vectors
Rhigh = Rhigh * np.ones(obs_topo_final.shape[0])
Rlow = Rlow * np.ones(obs_topo_final.shape[0])

RNG = np.random.default_rng(seed=13)

name = "dur=83, Et=80, ss=30, xD + 1"
print(name)

# _____________________________________________
# Overwash, Beach, & Dune Change
topo_prestorm = copy.deepcopy(topo)  # [m NAVD88]
for ts in range(dur):
    topoChange, dV, inundated = routine.calc_beach_dune_change_OLD(topo=topo,
                                                                   dx=1,
                                                                   crestline=dune_crest,
                                                                   crestflux=crestflux,
                                                                   x_s=x_s,
                                                                   MHW=MHW,
                                                                   Rhigh=Rhigh,
                                                                   Beq=0.027,
                                                                   Kc=1e-3,
                                                                   Tp=9.4,
                                                                   substeps=20,
                                                                   )

    # topo,
    #                            dx,
    #                            crestline,
    #                            crestflux,
    #                            x_s,
    #                            MHW,
    #                            Rhigh,
    #                            Beq,
    #                            Kc,
    #                            Tp,
    #                            substeps,
    topoChange_smooth = scipy.ndimage.gaussian_filter(topoChange, sigma=3, mode='constant', axes=[0])  # Smooth beach/dune topo change in alongshore direction (diffuse transects alongshore)
    topo = topo + topoChange_smooth  # Update topography

sim_topo_final = topo.copy()
# sim_topo_final = routine.enforceslopes(topo.copy(), veg, sh=0.02, anglesand=20, angleveg=30, th=0.3, RNG=RNG)[0]

SimDuration = time.time() - start_time
print()
print("Elapsed Time: ", SimDuration, "sec")

# _____________________________________________
# Model Skill: Comparisons to Observations

longshore, crossshore = sim_topo_final.shape

# Final Elevation Change
obs_change_m = obs_topo_final - topo_prestorm  # [m] Observed change
sim_change_m = sim_topo_final - topo_prestorm  # [m] Simulated change

# PLOTTING

# Change Comparisons
cmap1 = routine.truncate_colormap(copy.copy(plt.colormaps.get_cmap("terrain")), 0.5, 0.9)  # Truncate colormap
cmap1.set_bad(color='dodgerblue', alpha=0.5)  # Set cell color below MHW to blue

# Post-Storm (Observed) Topo
Fig = plt.figure(figsize=(14, 7.5))
ax1 = Fig.add_subplot(221)
topo1 = obs_topo_final[:, ymin: ymax]  # [m] Post-storm
cax1 = ax1.matshow(topo1, cmap=cmap1, vmin=0, vmax=5.0)
ax1.plot(dune_crest - ymin, np.arange(len(dune_crest)), c='black', alpha=0.6)
plt.title('Observed')
plt.suptitle(name)

# Post-Storm (Simulated) Topo
ax2 = Fig.add_subplot(222)
topo2 = sim_topo_final[:, ymin: ymax]  # [m]
cax2 = ax2.matshow(topo2, cmap=cmap1, vmin=0, vmax=5.0)
plt.title('Simulated')
# cbar = Fig.colorbar(cax2)
# cbar.set_label('Elevation [m MHW]', rotation=270, labelpad=20)

# Simulated Topo Change
maxx = max(abs(np.min(obs_change_m)), abs(np.max(obs_change_m)))
maxxx = max(abs(np.min(sim_change_m)), abs(np.max(sim_change_m)))
maxxxx = 1  # max(maxx, maxxx)
ax3 = Fig.add_subplot(224)
cax3 = ax3.matshow(sim_change_m[:, ymin: ymax], cmap='bwr', vmin=-maxxxx, vmax=maxxxx)
ax3.plot(dune_crest - ymin, np.arange(len(dune_crest)), c='black', alpha=0.6)
# cbar = Fig.colorbar(cax3)
# cbar.set_label('Change [m]', rotation=270, labelpad=20)

# Observed Topo Change
ax4 = Fig.add_subplot(223)
obs_change = obs_change_m.copy()
cax4 = ax4.matshow(obs_change[:, ymin: ymax], cmap='bwr', vmin=-maxxxx, vmax=maxxxx)
ax4.plot(dune_crest - ymin, np.arange(len(dune_crest)), c='black', alpha=0.6)
# cbar = Fig.colorbar(cax4)
# cbar.set_label('Elevation Change [m]', rotation=270, labelpad=20)
plt.tight_layout()

# Profile Change
x1 = 118
proffig = plt.figure(figsize=(11, 7.5))
plt.plot(topo_prestorm[x1, x_s[x1] - 1: ymax_p], c='black')
plt.plot(obs_topo_final[x1, x_s[x1] - 1: ymax_p], c='green')
plt.plot(sim_topo_final[x1, x_s[x1] - 1: ymax_p], c='red')
plt.legend(["Pre", "Post Obs", "Post Sim"])
plt.title(x1)

# Profile Change
x2 = 150
proffig2 = plt.figure(figsize=(11, 7.5))
plt.plot(topo_prestorm[x2, x_s[x2] - 1: ymax_p], c='black')
plt.plot(sim_topo_final[x2, x_s[x2] - 1: ymax_p], c='red')
plt.legend(["Pre", "Post Sim"])
plt.title(x2)

plt.matshow(inundated)
plt.title("Inundated")

print()
print("Complete.")
plt.show()
