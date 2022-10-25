"""__________________________________________________________________________________________________________________________________

Main Model Script for PyDUBEVEG

Python version of the DUne, BEach, and VEGetation (DUBEVEG) model from Keijsers et al. (2016) and Galiforni Silva et al. (2018, 2019)

Translated from Matlab by IRB Reeves

Last update: 25 October 2022

__________________________________________________________________________________________________________________________________"""

import routines_dubeveg as routine
import numpy as np
import scipy.io
import math
import random
import matplotlib.pyplot as plt
import time

# __________________________________________________________________________________________________________________________________
# USER INPUTS

# GENERAL
simnum = 2  # Reference number of the simulation. Used for personal reference.
MHT = 0  # [m] Sea-level reference
MHTrise = 0  # [m/yr] Sea-level rise rate
qpotseries = 2  # Number reference to calculate how many iterations represent one year. 4 is standard year of 100 iterations, corresponds to qpot (#1 = 25 it, #2 = 50 it, #3 = 75 it, #4 = 100 it, #5 = 125 it) (*25 = qpot)
writeyear = 1  # Write results to disc every n years
simulation_time_y = 15  # [yr] Length of the simulation time
cellsize = 1  # [m] Interpreted cell size
slabheight = 0.1  # Ratio of cell dimension 0.1 (0.077 - 0.13 (Nield and Baas, 2007))
slabheight_m = cellsize * slabheight  # [m] Slab height
repl = 3  # Number of replicates
inputloc = "Input/"  # Input file directory
outputloc = "Output/"  # Output file directory

# AEOLIAN
groundwater_depth = 0.8  # Proportion of the equilibrium profile used to set groundwater profile.
direction = np.random.choice(np.tile([1, 1, 1, 1, 1], (1, 2000))[0, :], 10000, replace=False)  # Direction (from) of the slab movement . 1 = west, 2 = north, 3 = east and 4 = south
p_dep_sand = 0.1  # [0-1] Probability of deposition of sandy cells
p_dep_bare = 0.1  # [0-1] Probability of deposition of base cells
p_ero_bare = 0.5  # [0-1] Probability of erosion of bare/sandy cells
shadowangle = 15  # [deg]
repose_bare = 20  # [deg] - orig:30
repose_veg = 30  # [deg] - orig:35
repose_threshold = 0.3  # Vegetation threshold for applying repose_veg
jumplength = 1  # [slabs] Hop length for slabs
clim = 0.5  # Vegetation cover that limits erosion
n_contour = 10  # Number of contours to be used to calculate fluxes. Multiples of 10

# HYDRODYNAMIC
m26 = 0.012  # Parameter for dissipation strength ~[0.01 - 0.02]
wave_energy = 1  # Parameter for initial wave strength ~[1 - 10]
depth_limit = 0.01  # Depth limit up to where dissipation is calculated. For depths smaller than "depth_limit", the program sets the value as "pwaveminf"
pcurr = 0  # Probability of erosion due to any hydrodynamic forcing rather that waves
m28f = 0.8  # Resistance of vegetation: 1 = full, 0 = none
pwavemaxf = 1  # Maximum erosive strenght of waves (if >1: overrules vegetation)
pwaveminf = 0.1  # In area with waves always potential for action (this can never be 0, otherwise the beachupdate is shut down)
shelterf = 1  # Exposure of sheltered cells: 0 = no shelter, 1 = full shelter

# VEGETATION
sp1_a = -1.4  # Vertice a, spec1. vegetation growth based on Nield and Baas (2008)
sp1_b = 0.2  # Vertice b, spec1. vegetation growth based on Nield and Baas (2008)
sp1_c = 0.6  # Vertice c, spec1. vegetation growth based on Nield and Baas (2008)
sp1_d = 2  # Vertice d, spec1. vegetation growth based on Nield and Baas (2008)
sp1_e = 2.2  # Vertice e, spec1. vegetation growth based on Nield and Baas (2008)

sp2_a = -1.4  # Vertice a, spec2. vegetation growth based on Nield and Baas (2008)
sp2_b = -0.65  # Vertice b, spec2. vegetation growth based on Nield and Baas (2008)
sp2_c = 0  # Vertice c, spec2. vegetation growth based on Nield and Baas (2008)
sp2_d = 0.2  # Vertice d, spec2. vegetation growth based on Nield and Baas (2008)
sp2_e = 2.8  # Vertice e, spec2. vegetation growth based on Nield and Baas (2008)

sp1_peak = 0.2  # Growth peak, spec1
sp2_peak = 0.05  # Growth peak, spec2

VGR = 0  # [%] Growth reduction by end of period
lateral_probability = 0.2  # Probability of lateral expansion of existing vegetation
pioneer_probability = 0.05  # Probability of occurrence of new pioneering vegetation

maxvegeff = 1.0  # [0-1] Value of maximum vegetation effectiveness allowed

# TOPOGRAPHY AND TIME-SERIES
topo_initial = scipy.io.loadmat(inputloc + "topo_west.mat")["topo_final"]  # [m] 2D-matrix with initial topography
eqtopo_initial = scipy.io.loadmat(inputloc + "eqtopo_west.mat")["topo_final"]  # [m] 2D-matrix or 3D-matrix with equilibrium profile. For 3D-matrix, the third matrix relates to time

no_timeseries = 0

wl_timeseries = scipy.io.loadmat(inputloc + "wl_max_texel.mat")["wl_max_texel"]  # [m] Only when "no_timeseries" = 0. Waterlevel time-series. Length and frequency in relation to "simulation_time_y" and "qpotseries"
# wl_probcum                  = scipy.io.loadmat('prob_wl.mat')  # Only when "no_timeseries" = 1. Waterlevel probabilities values. To be used only when no_timeseries = 1

spec1 = scipy.io.loadmat(inputloc + "spec1.mat")["vegf"]  # [0-1] 2D-matrix of vegetation effectiveness for spec1
spec2 = scipy.io.loadmat(inputloc + "spec2.mat")["vegf"]  # [0-1] 2D-matrix of vegetation effectiveness for spec2

start_time = time.time()  # Record time at start of simulation to track duration


# __________________________________________________________________________________________________________________________________
# RUN DUBEVEG

for repl_cont in range(repl):  # Start replicate loop

    # __________________________________________________________________________________________________________________________________
    # SET INITIAL CONDITIONS

    # TIME
    vegetationupdate = round(qpotseries * 25)
    iterations_per_cycle = round(qpotseries * 25)  # Number of iterations that is regarded as 1 year (was 50) [iterations/year]
    beachreset = round(qpotseries * 1)

    # TOPOGRAPHY
    topo0 = topo_initial / slabheight_m  # [slabs] Transform from m into number of slabs
    topo = np.round(topo0)  # [slabs] Initialise the topography map

    if eqtopo_initial.ndim == 3:
        eqtopo = np.round(np.squeeze(eqtopo_initial[:, :, 0]) / slabheight_m)  # [slabs] Transform from m into number of slabs
    else:
        eqtopo = np.round(eqtopo_initial / slabheight_m)  # [slabs] Transform from m into number of slabs

    eqtopo_i = eqtopo
    longshore, crossshore = topo0.shape * cellsize  # [m] Cross-shore/alongshore size of topography
    x = np.linspace(1, crossshore, num=crossshore)  # Shore normal axis for plotting
    y = np.linspace(1, longshore, num=longshore)  # Shore parallel axis for plotting

    gw = np.round(eqtopo_i * groundwater_depth)  # GW lies under beach with less steep angle
    gw[gw >= topo] = topo[gw >= topo]

    beachslopeslabs = (eqtopo_i[0, -1] - eqtopo_i[0, 0]) / crossshore  # [slabs/m] Slope of equilibrium beach
    offbeachslabs = eqtopo_i[0, 0] - beachslopeslabs * cellsize  # [slabs] Offset for calculating moving equilibirum beach

    # HYDRODYNAMIC
    if no_timeseries == 0:
        waterlevels = wl_timeseries[:, 0]
    else:
        raise ValueError('No water level time-series has been loaded. Functionality to automatically build a WL time-series from cumulative probabilities has not yet been incorporated into this model version.')

    # VEGETATION
    veg = spec1 + spec2  # Determine the initial cumulative vegetation effectiveness
    veg[veg > maxvegeff] = maxvegeff  # Cumulative vegetation effectiveness cannot be negative or larger than one
    veg[veg < 0] = 0

    growth_reduction_timeseries = np.linspace(0, VGR / 100, simulation_time_y)

    # MODEL PARAMETERS
    timewaterlev = np.linspace(beachreset / iterations_per_cycle, len(waterlevels) * beachreset / iterations_per_cycle, num=len(waterlevels))
    waterlevels = ((timewaterlev * MHTrise) + (waterlevels + MHT)) / slabheight_m  # [slabs]
    slabheight = round(slabheight_m * 100) / 100
    balance = topo * 0  # Initialise the sedimentation balance map [slabs]
    stability = topo * 0  # Initialise the stability map [slabs]
    sp1_peak_at0 = sp1_peak  # Store initial peak growth of sp. 1
    sp2_peak_at0 = sp2_peak  # Store initial peak growth of sp. 2
    inundated = np.zeros([longshore, crossshore])  # Initial area of wave/current action
    inundatedcum = np.zeros([longshore, crossshore])  # Initial area of sea action
    pbeachupdatecum = np.zeros([longshore, crossshore])  # Matrix for cumulative effect of hydrodynamics
    beachcount = 1
    vegcount = 1

    # __________________________________________________________________________________________________________________________________
    # MODEL OUPUT CONFIGURATION
    """Select matrices to be calculated during the simulation. CAUTION WITH THE SIZE OF YOUR OUTPUT, YOU MAY GET AN OUT-OF-MEMORY ERROR."""

    # MANDATORY
    iterations = iterations_per_cycle * simulation_time_y  # Number of iterations
    timeits = np.linspace(1, iterations, iterations)  # Time vector for budget calculations
    seainput_slabs = np.zeros([longshore, crossshore])  # Inititalise vector for sea-transported slab

    # OPTIONAL
    # flux_contour = np.zeros([len(timeits), n_contour + 1])  # Inititalise vector for sea-transported slabs
    # seainput_tot = np.empty([longshore, crossshore, len(timeits)])  # 3-D seainput matrices
    # diss_tot = np.empty([longshore, crossshore, len(timeits)])  # 3-D dissipation matrices
    # cumdiss_tot = np.empty([longshore, crossshore, len(timeits)])  # 3-D cummulative dissipation matrices
    # pwave_tot = np.empty([longshore, crossshore, len(timeits)])  # 3-D pwave matrices
    # pbeachupdate_tot = np.empty([longshore, crossshore, len(timeits)])  # 3-D pbeachupdate matrices
    # balancea_tot = np.empty([longshore, crossshore, len(timeits)])  # 3-D balancea matrices
    # balanceb_tot = np.empty([longshore, crossshore, len(timeits)])  # 3-D balanceb matrices
    # stabilitya_tot = np.empty([longshore, crossshore, len(timeits)])  # 3-D stabilitya matrices
    # stabilityb_tot = np.empty([longshore, crossshore, len(timeits)])  # 3-D stabilityb matrices
    # erosmap_tot = np.empty([longshore, crossshore, len(timeits)])  # 3-D erosmap matrices
    # deposmap_tot = np.empty([longshore, crossshore, len(timeits)])  # 3-D deposmap matrices
    # shadowmap_tot = np.zeros([longshore, crossshore])  # 3-D shadowmap matrice
    erosmap_sum = np.zeros([longshore, crossshore])  # Sum of all erosmaps
    deposmap_sum = np.zeros([longshore, crossshore])  # Sum of all deposmaps
    seainput_sum = np.zeros([longshore, crossshore])  # Sum of all seainput maps
    balancea_sum = np.zeros([longshore, crossshore])  # Sum of all balancea maps
    balanceb_sum = np.zeros([longshore, crossshore])  # Sum of all balanceb maps
    stabilitya_sum = np.zeros([longshore, crossshore])  # Sum of all stabilitya maps
    stabilityb_sum = np.zeros([longshore, crossshore])  # Sum of all stabilityb maps

    # Inititalise vectors for transport activity
    windtransp_slabs = np.zeros([len(timeits)])
    landward_transport = np.zeros([len(timeits)])
    avalanches = np.zeros([len(timeits)])


    # __________________________________________________________________________________________________________________________________
    # MAIN ITERATION LOOP

    for it in range(iterations):

        # Print time step to screen
        print("\r", "Time Step: ", (it / iterations_per_cycle), "yrs", end="")

        year = math.ceil(it / iterations_per_cycle)

        if eqtopo_initial.ndim == 3:
            eqtopo = np.squeeze(eqtopo_initial[:, :, it]) / slabheight_m

        # --------------------------------------
        # SAND TRANSPORT

        before = topo
        gw = eqtopo * groundwater_depth
        sandmap = topo > MHT  # Boolean array, Returns True (1) for sandy cells

        shadowmap = routine.shadowzones2(topo, slabheight, shadowangle, longshore, crossshore, direction=direction[it])  # Returns map of True (1) for in shadow, False (2) not in shadow

        erosmap = routine.erosprobs2(veg, shadowmap, sandmap, topo, gw, p_ero_bare)  # Returns map of erosion probabilities
        deposmap = routine.depprobs(veg, shadowmap, sandmap, p_dep_bare, p_dep_sand)  # Returns map of deposition probabilities

        if 'erosmap_sum' in locals():
            erosmap_sum = erosmap_sum + erosmap

        if 'deposmap_sum' in locals():
            deposmap_sum = deposmap_sum + deposmap

        if 'erosmap_tot' in locals():
            erosmap_tot[:, :, it] = erosmap

        if 'deposmap_tot' in locals():
            deposmap_tot[:, :, it] = deposmap

        if 'shadowmap_tot' in locals():
            shadowmap_tot[:, :, it] = shadowmap

        if direction[it] == 1 or direction[it] == 3:  # East or west wind direction
            contour = np.linspace(0, round(crossshore) - 1, n_contour + 1)  # Contours to account for transport
            changemap, slabtransp, sum_contour = routine.shiftslabs3_open3(erosmap, deposmap, jumplength, contour, longshore, crossshore, direction=direction[it])  # Returns map of height changes
        else:  # North or south wind direction
            contour = np.linspace(0, round(longshore) - 1, n_contour + 1)  # Contours to account for transport  #  IRBR 21Oct22: This produces slightly different results than Matlab version
            changemap, slabtransp, sum_contour = routine.shiftslabs3_open3(erosmap, deposmap, jumplength, contour, longshore, crossshore, direction=direction[it])  # Returns map of height changes

        topo = topo + changemap  # Changes applied to the topography
        topo, aval = routine.enforceslopes3(topo, veg, slabheight, repose_bare, repose_veg, repose_threshold)  # Enforce angles of repose: avalanching
        balance = balance + (topo - before)  # Update the sedimentation balance map
        stability = stability + abs(topo - before)


        # ADD ADDITIONAL DATA SAVING



        # --------------------------------------
        # BEACH UPDATE

        if it % beachreset == 0:  # Update the inundated part of the beach

            inundatedold = inundated  # Make a copy for later use
            before1 = topo  # Copy of topo before it is changed

            topo, inundated, pbeachupdate, diss, cumdiss, pwave = routine.marine_processes3_diss3e(
                waterlevels[beachcount - 1],
                MHTrise,
                slabheight_m,
                cellsize,
                topo,
                eqtopo,
                veg,
                m26,
                wave_energy,
                m28f,
                pwavemaxf,
                pwaveminf,
                depth_limit,
                shelterf,
                pcurr,
            )

            seainput = topo - before1  # Sand added to the beach by the sea
            # seainput_slabs[it] = np.nansum(seainput)  # IRBR 24OCt22: Bug here. Why is this even needed?

            if 'seainput_total' in locals():
                seainput_total[:, :, it] = seainput

            if 'diss_total' in locals():
                diss_total[:, :, it] = diss

            if 'cumdiss_total' in locals():
                cumdiss_total[:, :, it] = cumdiss

            if 'pwave_total' in locals():
                pwave_total[:, :, it] = pwave

            if 'pbeachupdate_total' in locals():
                pbeachupdate_total[:, :, it] = pbeachupdate

            if 'seainput_sum' in locals():
                seainput_sum = seainput_sum + seainput

            inundated = np.logical_or(inundated, inundatedold)  # Combine updated area from individual loops
            pbeachupdatecum = pbeachupdate + pbeachupdatecum  # Cumulative beachupdate probabilities
            topo = routine.enforceslopes3(topo, veg, slabheight, repose_bare, repose_veg, repose_threshold)[0]  # Enforce angles of repose again
            balance = balance + (topo - before1)
            stability = stability + abs(topo - before1)

            if 'balanceb_sum' in locals():
                balanceb_sum = balanceb_sum + balance

            if 'balanceb_tot' in locals():
                balanceb_tot[:, :, it] = balance

            if 'stabilityb_sum' in locals():
                stabilityb_sum = stabilityb_sum + stability

            if 'stabilityb' in locals():
                stabilityb[:, :, it] = balance

            beachcount += 1  # Update counter


        # --------------------------------------
        # VEGETATION

        if it % iterations_per_cycle == 0 and it > 0:  # Update the vegetation

            veg_multiplier = (1 + growth_reduction_timeseries[vegcount])  # For the long term reduction.
            sp1_peak = sp1_peak_at0 * veg_multiplier
            sp2_peak = sp2_peak_at0 * veg_multiplier
            spec1_old = spec1
            spec2_old = spec2
            spec1 = routine.growthfunction1_sens(spec1, balance * slabheight_m, sp1_a, sp1_b, sp1_c, sp1_d, sp1_e, sp1_peak)
            spec2 = routine.growthfunction2_sens(spec2, balance * slabheight_m, sp2_a, sp2_b, sp2_d, sp1_e, sp2_peak)

            # Lateral Expansion & Veg Establishment
            lateral1 = routine.lateral_expansion(spec1_old, 1, lateral_probability * veg_multiplier)  # Species 1
            lateral2 = routine.lateral_expansion(spec2_old, 1, lateral_probability * veg_multiplier)  # Species 1
            pioneer1 = routine.establish_new_vegetation(topo * slabheight_m, MHT, pioneer_probability * veg_multiplier) * (spec1_old <= 0)
            pioneer2 = routine.establish_new_vegetation(topo * slabheight_m, MHT, pioneer_probability * veg_multiplier) * (spec2_old <= 0) * (stability == 0)

            spec1_diff = spec1 - spec1_old  # Determine changes in vegetation cover
            spec2_diff = spec2 - spec2_old  # Determine changes in vegetation cover
            spec1_growth = spec1_diff * (spec1_diff > 0)  # Split cover changes into into gain and loss
            spec1_loss = spec1_diff * (spec1_diff < 0)  # Split cover changes into into gain and loss
            spec2_growth = spec2_diff * (spec2_diff > 0)
            spec2_loss = spec2_diff * (spec2_diff < 0)

            spec1_change_allowed = np.minimum(1 - veg, spec1_growth) * (lateral1 | pioneer1)  # Only allow growth in adjacent or pioneer cells
            spec2_change_allowed = np.minimum(1 - veg, spec2_growth) * (lateral2 | pioneer2)  # Only allow growth in adjacent or pioneer cells
            spec1 = spec1_old + spec1_change_allowed + spec1_loss  # Re-assemble gain and loss and add to original vegetation cover
            spec2 = spec2_old + spec2_change_allowed + spec2_loss  # Re-assemble gain and loss and add to original vegetation cover

            pbeachupdatecum[pbeachupdatecum < 0] = 0
            pbeachupdatecum[pbeachupdatecum > 1] = 1

            spec1 = spec1 * (1 - pbeachupdatecum)  # Remove species where beach is reset
            spec2 = spec2 * (1 - pbeachupdatecum)  # Remove species where beach is reset

            # Limit to geomorphological range
            spec1_geom = spec1
            spec1_geom[spec1 < 0] = 0
            spec1_geom[spec1 > 1] = 1
            spec2_geom = spec2
            spec2_geom[spec2 < 0] = 0
            spec2_geom[spec2 > 1] = 1
            veg = spec1_geom + spec2_geom  # Update vegmap
            veg[veg > maxvegeff] = maxvegeff  # Limit to effective range
            veg[veg < 0] = 0

            vegcount = vegcount + 1  # Update counter


        # --------------------------------------
        # RESET DOMAINS
        balance_copy = balance
        balance[:] = 0  # Reset the balance map
        inundated[:] = 0  # Reset part of beach with wave/current action
        pbeachupdatecum[:] = 0
        stability[:] = 0  # Reset the balance map

        # IRBR 25Oct22: In Matlab version, topo is reset to initial after each replicate, but veg (i.e., spec1 & spec2) is not reset, meaning veg
        # cover picks up where the previous replicate left off (but on top of initial topo). Not sure why this is done; clearly not a "replicate."


# __________________________________________________________________________________________________________________________________
# SIMULATION END

# Print elapsed time of simulation
print()
SimDuration = time.time() - start_time
print()
print("Elapsed Time: ", SimDuration, "sec")

# Temp Plot
plt.matshow(topo * slabheight,
            cmap='terrain',
            )
plt.title("Elev TMAX")

plt.matshow(veg,
            cmap='YlGn',
            )
plt.title("Veg TMAX")
plt.show()
