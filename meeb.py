"""__________________________________________________________________________________________________________________________________

Main Model Script for MEEB

Mesoscale Explicit Ecogeomorphic Barrier model

IRB Reeves

Last update: 11 August 2023

__________________________________________________________________________________________________________________________________"""

import numpy as np
import math
import dill
import matplotlib.pyplot as plt
import time
import imageio
import scipy
import copy
from datetime import datetime, timedelta

import routines_meeb as routine


class MEEB:
    def __init__(
            self,

            # GENERAL
            name="default",
            simnum=1,  # Reference number of the simulation. Used for personal reference.
            MHW=0,  # [m NAVD88] Initial mean high water
            RSLR=0.000,  # [m/yr] Relative sea-level rise rate
            aeolian_iterations_per_year=50,  # Number of aeolian updates in 1 model year
            storm_iterations_per_year=25,  # Number of storm and shoreline change updates in 1 model year
            vegetation_iterations_per_year=1,  # Number of vegetation updates in 1 model year
            save_frequency=0.5,  # [years] Save results every n years
            simulation_time_yr=15.0,  # [yr] Length of the simulation time
            cellsize=1,  # [m] Interpreted cell size
            slabheight=0.02,  # Ratio of cell dimension (0.02, Teixeira et al. 2023)
            alongshore_domain_boundary_min=0,  # [m] Alongshore minimum boundary location for model domain
            alongshore_domain_boundary_max=10e7,  # [m] Alongshore maximum boundary location for model domain; if left to this default value, it will automatically adjust to the actual full length of the domain
            inputloc="Input/",  # Input file directory (end string with "/")
            outputloc="Output/",  # Output file directory (end string with "/")
            init_filename="Init_NorthernNCB_2017_PreFlorence.npy",  # [m NVD88] Name of initial topography and vegetation input file
            hindcast=False,  # [bool] Determines whether the model is run with the default stochastisity generated storms [hindcast=False], or an empirical storm, wind, wave, temp timeseries [hindcast=True]
            simulation_start_date='20040716',  # [date] Date from which to start hindcast; must be string in format 'yyyymmdd'
            hindcast_timeseries_start_date='19790101',  # [date] Start date of hindcast timeseries input data; format 'yyyymmdd'
            seeded_random_numbers=True,  # [bool] Determines whether to use seeded random number generator for reproducibility
            save_data=False,

            # AEOLIAN
            groundwater_depth=0.8,  # Proportion of the smoothend topography used to set groundwater profile
            wind_rose=(0.8, 0, 0.2, 0),  # Proportion of wind TOWARDS (right, down, left, up)
            p_dep_sand=0.1,  # [0-1] Probability of deposition in sandy cells with 0% vegetation cover
            p_dep_sand_VegMax=0.2,  # [0-1] Probability of deposition in sandy cells with 100% vegetation cover. Must be greater than or equal to p_dep_sand/p_dep_base.
            p_dep_base=0.1,  # [0-1] Probability of deposition of base cells
            p_ero_sand=0.5,  # [0-1] Probability of erosion of bare/sandy cells
            entrainment_veg_limit=0.5,  # [0-1] Percent of vegetation cover beyond which aeolian sediment entrainment is no longer possible.
            shadowangle=15,  # [deg]
            repose_bare=20,  # [deg]
            repose_veg=30,  # [deg]
            repose_threshold=0.3,  # Vegetation threshold for applying repose_veg
            saltation_veg_limit=0.25,  # Threshold vegetation effectiveness needed for a cell along a slab saltation path needed to be considered vegetated
            jumplength=5,  # [cell length] Hop length for slabs (5, Teixeira et al. 2023)
            clim=0.5,  # Vegetation cover that limits erosion

            # SHOREFACE, BEACH, & SHORELINE
            beach_equilibrium_slope=0.039,  # Equilibrium slope of the beach
            beach_erosiveness=1.75,  # Beach erosiveness timescale constant: larger (smaller) Et == lesser (greater) storm erosiveness
            beach_substeps=40,  # Number of substeps per iteration of beach/duneface model; instabilities will occur if too low
            shoreface_flux_rate=5000,  # [m3/m/yr] Shoreface flux rate coefficient
            shoreface_equilibrium_slope=0.02,  # Equilibrium slope of the shoreface
            shoreface_depth=10,  # [m] Depth to shoreface toe (i.e. depth of ‘closure’)
            shoreface_length_init=500,  # [m] Initial length of shoreface
            wave_asymetry=0.5,  # Fraction of waves approaching from the left (when looking offshore)
            wave_high_angle_fraction=0,  # Fraction of waves approaching at angles higher than 45 degrees from shore normal
            mean_wave_height=1.0,  # [m] Mean offshore significant wave height
            mean_wave_period=10,  # [s] Mean wave period
            alongshore_section_length=25,  # [m] Distance alongshore between shoreline positions used in the shoreline diffusion calculations

            # VEGETATION
            sp1_a=-1.3,  # Vertice a, spec1. vegetation growth based on Nield and Baas (2008)
            sp1_b=-0.1,  # Vertice b, spec1. vegetation growth based on Nield and Baas (2008)
            sp1_c=0.5,  # Vertice c, spec1. vegetation growth based on Nield and Baas (2008)
            sp1_d=1.5,  # Vertice d, spec1. vegetation growth based on Nield and Baas (2008)
            sp1_e=2.5,  # Vertice e, spec1. vegetation growth based on Nield and Baas (2008)
            sp2_a=-1.4,  # Vertice a, spec2. vegetation growth based on Nield and Baas (2008)
            sp2_b=-0.65,  # Vertice b, spec2. vegetation growth based on Nield and Baas (2008)
            sp2_c=0.0,  # Vertice c, spec2. vegetation growth based on Nield and Baas (2008)
            sp2_d=0.2,  # Vertice d, spec2. vegetation growth based on Nield and Baas (2008)
            sp2_e=2.2,  # Vertice e, spec2. vegetation growth based on Nield and Baas (2008)
            sp1_peak=0.2,  # Growth peak, spec1
            sp2_peak=0.05,  # Growth peak, spec2
            VGR=0,  # [%] Growth reduction by end of period
            lateral_probability=0.2,  # Probability of lateral expansion of existing vegetation
            pioneer_probability=0.05,  # Probability of occurrence of new pioneering vegetation
            maxvegeff=1.0,  # [0-1] Value of maximum vegetation effectiveness allowed
            Spec1_elev_min=0.25,  # [m MHW] Minimum elevation (relative to MHW) for species 1 (1 m MHW for A. brevigulata from Young et al., 2011)
            Spec2_elev_min=0.25,  # [m MHW] Minimum elevation (relative to MHW) for species 2
            flow_reduction_max_spec1=0.05,  # Proportion of overwash flow reduction through a cell populated with species 1 at full density
            flow_reduction_max_spec2=0.20,  # Proportion of overwash flow reduction through a cell populated with species 2 at full density

            # STORM OVERWASH AND DUNE EROSION
            storm_list_filename="VCRStormList.npy",
            storm_timeseries_filename="",  # Only needed if running hindcast simulations (i.e., without stochastic storms)
            threshold_in=0.25,  # [%] Threshold percentage of overtopped dune cells exceeded by Rlow needed to be in inundation overwash regime
            Rin_in=5,  # [m^3/hr] Flow infiltration and drag parameter, inundation overwash regime
            Rin_ru=325,  # [m^3/hr] Flow infiltration and drag parameter, run-up overwash regime
            Cx=25,  # Constant for representing flow momentum for sediment transport in inundation overwash regime
            nn=0.5,  # Flow routing constant
            MaxUpSlope=1,  # Maximum slope water can flow uphill
            fluxLimit=1,  # [m/hr] Maximum elevation change allowed per time step (prevents instabilities)
            Qs_min=1.0,  # [m^3/hr] Minimum discharge out of cell needed to transport sediment
            K_ru=5.15e-05,  # Sediment transport coefficient for run-up overwash regime
            K_in=5e-04,  # Sediment transport coefficient for inundation overwash regime
            mm=1.0,  # Inundation overwash constant
            Cbb_in=0.85,  # [%] Coefficient for exponential decay of sediment load entering back-barrier bay, inundation regime
            Cbb_ru=0.7,  # [%] Coefficient for exponential decay of sediment load entering back-barrier bay, run-up regime
            Qs_bb_min=1,  # [m^3/hr] Minimum discharge out of subaqueous back-barrier cell needed to transport sediment
            substep_in=3,  # Number of substeps to run for each hour in inundation overwash regime (e.g., 3 substeps means discharge/elevation updated every 20 minutes)
            substep_ru=5,  # Number of substeps to run for each hour in run-up overwash regime (e.g., 3 substeps means discharge/elevation updated every 20 minutes)

    ):
        """MEEB: Mesoscale Explicit Ecogeomorphic Barrier model"""

        self._name = name
        self._simnum = simnum
        self._MHW = MHW
        self._RSLR = RSLR
        self._aeolian_iterations_per_year = aeolian_iterations_per_year
        self._storm_iterations_per_year = storm_iterations_per_year
        self._vegetation_iterations_per_year = vegetation_iterations_per_year
        self._save_frequency = save_frequency
        self._simulation_time_yr = simulation_time_yr
        self._cellsize = cellsize
        self._slabheight = slabheight
        self._alongshore_domain_boundary_min = alongshore_domain_boundary_min
        self._alongshore_domain_boundary_max = alongshore_domain_boundary_max
        self._inputloc = inputloc
        self._outputloc = outputloc
        self._hindcast = hindcast
        self._simulation_start_date = simulation_start_date
        self._hindcast_timseries_start_date = hindcast_timeseries_start_date
        self._save_data = save_data
        self._groundwater_depth = groundwater_depth
        self._wind_rose = wind_rose
        self._p_dep_sand = p_dep_sand
        self._p_dep_sand_VegMax = p_dep_sand_VegMax
        self._p_dep_base = p_dep_base
        self._p_ero_sand = p_ero_sand
        self._entrainment_veg_limit = entrainment_veg_limit
        self._shadowangle = shadowangle
        self._repose_bare = repose_bare
        self._repose_veg = repose_veg
        self._repose_threshold = repose_threshold
        self._saltation_veg_limit = saltation_veg_limit
        self._jumplength = jumplength
        self._clim = clim
        self._beach_equilibrium_slope = beach_equilibrium_slope
        self._beach_erosiveness = beach_erosiveness
        self._beach_substeps = beach_substeps
        self._k_sf = shoreface_flux_rate
        self._s_sf_eq = shoreface_equilibrium_slope
        self._DShoreface = shoreface_depth
        self._LShoreface = shoreface_length_init
        self._wave_asymetry = wave_asymetry
        self._wave_high_angle_fraction = wave_high_angle_fraction
        self._mean_wave_height = mean_wave_height
        self._mean_wave_period = mean_wave_period
        self._alongshore_section_length = alongshore_section_length
        self._sp1_a = sp1_a
        self._sp1_b = sp1_b
        self._sp1_c = sp1_c
        self._sp1_d = sp1_d
        self._sp1_e = sp1_e
        self._sp2_a = sp2_a
        self._sp2_b = sp2_b
        self._sp2_c = sp2_c
        self._sp2_d = sp2_d
        self._sp2_e = sp2_e
        self._sp1_peak = sp1_peak
        self._sp2_peak = sp2_peak
        self._VGR = VGR
        self._lateral_probability = lateral_probability
        self._pioneer_probability = pioneer_probability
        self._maxvegeff = maxvegeff
        self._Spec1_elev_min = Spec1_elev_min
        self._Spec2_elev_min = Spec2_elev_min
        self._flow_reduction_max_spec1 = flow_reduction_max_spec1
        self._flow_reduction_max_spec2 = flow_reduction_max_spec2
        self._threshold_in = threshold_in
        self._Rin_in = Rin_in
        self._Rin_ru = Rin_ru
        self._Cx = Cx
        self._nn = nn
        self._MaxUpSlope = MaxUpSlope
        self._fluxLimit = fluxLimit
        self._Qs_min = Qs_min
        self._K_ru = K_ru
        self._K_in = K_in
        self._mm = mm
        self._Cbb_in = Cbb_in
        self._Cbb_ru = Cbb_ru
        self._Qs_bb_min = Qs_bb_min
        self._substep_in = substep_in
        self._substep_ru = substep_ru

        # __________________________________________________________________________________________________________________________________
        # SET INITIAL CONDITIONS

        # SEEDED RANDOM NUMBER GENERATOR
        if seeded_random_numbers:
            self._RNG = np.random.default_rng(seed=13)  # Seeded random numbers for reproducibility (e.g., model development/testing)
        else:
            self._RNG = np.random.default_rng()  # Non-seeded random numbers (e.g., model simulations)

        # TIME
        self._iterations_per_cycle = self._aeolian_iterations_per_year  # [iterations/year] Number of iterations in 1 model year
        self._vegetation_update_frequency = round(self.iterations_per_cycle / self._vegetation_iterations_per_year)  # Frequency of vegetation updates (i.e., every n iterations)
        self._storm_update_frequency = round(self.iterations_per_cycle / self._storm_iterations_per_year)  # Frequency of storm updates (i.e., every n iterations)
        self._iterations = int(self._iterations_per_cycle * self._simulation_time_yr)  # Total number of iterations
        self._simulation_start_date = datetime.strptime(self._simulation_start_date, '%Y%m%d').date()  # Convert to datetime
        if hindcast:
            self._hindcast_timeseries_start_date = datetime.strptime(self._hindcast_timseries_start_date, '%Y%m%d').date()  # Convert to datetime
            self._simulation_start_iteration = ((self._simulation_start_date.year - self._hindcast_timeseries_start_date.year) * self._iterations_per_cycle) + \
                                           math.floor(self._simulation_start_date.timetuple().tm_yday / 365 * self._iterations_per_cycle)  # Iteration, realtive to timeseries start, from which to begin hindcast
            if self._simulation_start_iteration % 2 != 0:
                self._simulation_start_iteration -= 1  # Round simulation start iteration to even number
        else:
            self._simulation_start_iteration = 0
        self._iteration_dates = [self._simulation_start_date + timedelta(minutes=10512 * x) for x in range(self._iterations)]  # List of dates corresponding to each model iteration

        # TOPOGRAPHY
        Init = np.load(inputloc + init_filename)
        self._alongshore_domain_boundary_max = min(self._alongshore_domain_boundary_max, Init[0, :, :].shape[0])
        self._topo_initial = Init[0, self._alongshore_domain_boundary_min: self._alongshore_domain_boundary_max, :]  # [m NAVD88] 2D array of initial topography
        self._topo = self._topo_initial.copy()  # [m NAVD88] Initialise the topography
        self._longshore, self._crossshore = self._topo.shape * self._cellsize  # [m] Cross-shore/alongshore size of topography
        self._groundwater_elevation = np.zeros(self._topo.shape)  # [m NAVD88] Initialize

        # SHOREFACE & SHORELINE
        self._x_s = routine.ocean_shoreline(self._topo, self._MHW)  # [m] Start locations of shoreline according to initial topography and MHW
        self._x_t = self._x_s - self._LShoreface  # [m] Start locations of shoreface toe

        # VEGETATION
        self._spec1 = Init[1, self._alongshore_domain_boundary_min: self._alongshore_domain_boundary_max, :]  # [0-1] 2D array of vegetation effectiveness for spec1
        self._spec2 = Init[2, self._alongshore_domain_boundary_min: self._alongshore_domain_boundary_max, :]  # [0-1] 2D array of vegetation effectiveness for spec2
        self._veg = self._spec1 + self._spec2  # Determine the initial cumulative vegetation effectiveness
        self._veg[self._veg > self._maxvegeff] = self._maxvegeff  # Cumulative vegetation effectiveness cannot be negative or larger than one
        self._veg[self._veg < 0] = 0
        self._effective_veg = scipy.ndimage.filters.gaussian_filter(self._veg, [1.5, 1.5], mode='constant')  # Effective vegetation cover represents effect of nearby vegetation on local wind
        self._growth_reduction_timeseries = np.linspace(0, self._VGR / 100, int(np.ceil(self._simulation_time_yr)))

        # STORMS
        self._StormList = np.load(inputloc + storm_list_filename)
        self._storm_timeseries = np.load(inputloc + storm_timeseries_filename)
        self._pstorm = [0.787878787878788, 0.393939393939394, 0.454545454545455, 0.727272727272727, 0.575757575757576, 0.484848484848485, 0.363636363636364, 0.363636363636364, 0.151515151515152, 0.0606060606060606, 0.181818181818182,
                        0.0606060606060606, 0.0606060606060606, 0, 0.0303030303030303, 0.121212121212121, 0.393939393939394, 0.272727272727273, 0.424242424242424, 0.424242424242424, 0.545454545454545, 0.424242424242424,
                        0.272727272727273, 0.484848484848485, 0.484848484848485]  # Empirical probability of storm occurance for each 1/25th (~biweekly) iteration of the year, from 1980-2013 VCR storm record
        if self._hindcast and self._iterations > (self._storm_timeseries[-1, 0] - self._simulation_start_iteration):
            raise ValueError("Simulation length is greater than hindcast timeSeries length.")

        # MODEL PARAMETERS
        self._MHW_init = copy.deepcopy(self._MHW)
        self._wind_direction = np.zeros([self._iterations], dtype=int)
        self._slabheight = round(self._slabheight, 2)  # Round slabheight to 2 decimals
        self._sedimentation_balance = self._topo * 0  # [m] Initialize map of the sedimentation balance: difference between erosion and deposition for 1 model year; (+) = net deposition, (-) = net erosion
        self._topographic_change = self._topo * 0  # [m] Map of the absolute value of topographic change over 1 model year (i.e., a measure of if the topography is changing or stable)
        self._x_s_TS = [self._x_s]  # Initialize storage array for shoreline position
        self._x_t_TS = [self._x_t]  # Initialize storage array for shoreface toe position
        self._sp1_peak_at0 = copy.deepcopy(self._sp1_peak)  # Store initial peak growth of sp. 1
        self._sp2_peak_at0 = copy.deepcopy(self._sp2_peak)  # Store initial peak growth of sp. 2
        self._vegcount = 0
        self._shoreline_change_aggregate = np.zeros([self._longshore])
        self._OWflux = np.zeros([self._longshore])  # [m^3]
        self._StormRecord = np.empty([5])  # Record of each storm that occurs in model: Year, iteration, Rhigh, Rlow, duration

        # __________________________________________________________________________________________________________________________________
        # MODEL OUPUT CONFIGURATION

        self._topo_TS = np.empty([self._longshore, self._crossshore, int(np.floor(self._simulation_time_yr / self._save_frequency)) + 1])  # Array for saving each topo map at specified frequency
        self._topo_TS[:, :, 0] = self._topo
        self._spec1_TS = np.empty([self._longshore, self._crossshore, int(np.floor(self._simulation_time_yr / self._save_frequency)) + 1])  # Array for saving each spec1 map at specified frequency
        self._spec1_TS[:, :, 0] = self._spec1
        self._spec2_TS = np.empty([self._longshore, self._crossshore, int(np.floor(self._simulation_time_yr / self._save_frequency)) + 1])  # Array for saving each spec2 map at specified frequency
        self._spec2_TS[:, :, 0] = self._spec2
        self._veg_TS = np.empty([self._longshore, self._crossshore, int(np.floor(self._simulation_time_yr / self._save_frequency)) + 1])  # Array for saving each veg map at specified frequency
        self._veg_TS[:, :, 0] = self._veg
        self._erosmap_sum = np.zeros([self._longshore, self._crossshore])  # Sum of all erosion probability maps
        self._deposmap_sum = np.zeros([self._longshore, self._crossshore])  # Sum of all deposition probability maps
        self._sedimentation_balance_sumA = np.zeros([self._longshore, self._crossshore])  # Sum of all balancea maps
        self._sedimentation_balance_sumB = np.zeros([self._longshore, self._crossshore])  # Sum of all balanceb maps
        self._topographic_change_sumA = np.zeros([self._longshore, self._crossshore])  # Sum of all stabilitya maps over the simulation period
        self._topographic_change_sumB = np.zeros([self._longshore, self._crossshore])  # Sum of all stabilityb maps over the simulation period
        self._avalanched_cells = np.zeros([self._iterations])  # Initialize; number of avalanched cells for each iteration

    # __________________________________________________________________________________________________________________________________
    # MAIN ITERATION LOOP

    def update(self, it):
        """Update MEEB by a single time step"""

        year = math.ceil(it / self._iterations_per_cycle)

        # Update sea level for this iteration
        self._MHW += self._RSLR / self._iterations_per_cycle  # [m NAVD88]

        # --------------------------------------
        # AEOLIAN
        self._wind_direction[it] = self._RNG.choice(np.arange(1, 5), p=self._wind_rose).astype(int)  # Randomly select and record wind direction for this iteration
        topo_iteration_start = copy.deepcopy(self._topo)

        # Get present groundwater elevations
        self._groundwater_elevation = scipy.ndimage.gaussian_filter(self._topo, sigma=12) * self._groundwater_depth  # [m NAVD88] Groundwater based on smoothed topography
        self._groundwater_elevation[self._groundwater_elevation < self._MHW] = self._MHW  # [m NAVD88]

        # Find subaerial and shadow cells
        subaerial = self._topo > self._MHW  # [bool] True for subaerial cells
        wind_shadows = routine.shadowzones(self._topo, self._shadowangle, direction=self._wind_direction[it])  # [bool] Map of True for in shadow, False not in shadow

        # Erosion/Deposition Probabilities
        aeolian_erosion_prob = routine.erosprobs(self._effective_veg, wind_shadows, subaerial, self._topo, self._groundwater_elevation, self._p_ero_sand, self._entrainment_veg_limit)  # Returns map of erosion probabilities
        aeolian_deposition_prob = routine.depprobs(self._effective_veg, wind_shadows, subaerial, self._p_dep_base, self._p_dep_sand, self._p_dep_sand_VegMax, self._topo, self._groundwater_elevation)  # Returns map of deposition probabilities

        # Move sand slabs
        if self._wind_direction[it] == 1 or self._wind_direction[it] == 3:  # Left or Right wind direction
            aeolian_elevation_change = routine.shiftslabs(aeolian_erosion_prob, aeolian_deposition_prob, self._jumplength, self._effective_veg, self._saltation_veg_limit, self._wind_direction[it], True, self._RNG)  # Returns map of height changes in units of slabs
        else:  # Up or Down wind direction
            aeolian_elevation_change = routine.shiftslabs(aeolian_erosion_prob, aeolian_deposition_prob, self._jumplength, self._effective_veg, self._saltation_veg_limit, self._wind_direction[it], True, self._RNG)  # Returns map of height changes in units of slabs

        # Apply changes, make calculations
        self._topo += aeolian_elevation_change * self._slabheight  # [m NAVD88] Changes applied to the topography; convert aeolian_elevation_change from slabs to meters
        self._topo, aval = routine.enforceslopes(self._topo, self._veg, self._slabheight, self._repose_bare, self._repose_veg, self._repose_threshold, self._RNG)  # Enforce angles of repose: avalanching
        self._sedimentation_balance = self._sedimentation_balance + (self._topo - topo_iteration_start)  # [m] Update the sedimentation balance map
        balance_init = self._sedimentation_balance + (self._topo - topo_iteration_start)
        self._topographic_change = self._topographic_change + abs(self._topo - topo_iteration_start)  # [m]
        stability_init = self._topographic_change + abs(self._topo - topo_iteration_start)

        # --------------------------------------
        # STORMS

        if it % self._storm_update_frequency == 0:
            iteration_year = np.floor(it % self._iterations_per_cycle / 2).astype(int)  # Iteration of the year (e.g., if there's 50 iterations per year, this represents the week of the year)

            # Generate Storms Stats
            if self._hindcast:  # Empirical storm time series
                storm, Rhigh, Rlow, dur = routine.get_storm_timeseries(self._storm_timeseries, it, self._longshore, self._simulation_start_iteration)  # [m NAVD88]
            else:  # Stochastic storm model
                storm, Rhigh, Rlow, dur = routine.stochastic_storm(self._pstorm, iteration_year, self._StormList, self._beach_equilibrium_slope, self._RNG)  # [m initial MSL]
                # Account for change in mean sea-level on synthetic storm elevations by adding aggregate RSLR since simulation start (i.e., convert from initial MSL to m NAVD88)
                Rhigh += self._MHW - self._MHW_init  # [m NAVD88] Add change in sea level to storm water levels, which were in elevation relative to initial sea level
                Rlow += self._MHW - self._MHW_init  # [m NAVD88] Add change in sea level to storm water levels, which were in elevation relative to initial sea level

            if storm:

                # Storm Processes: Beach/duneface change, overwash
                self._StormRecord = np.vstack((self._StormRecord, [year, iteration_year, np.max(Rhigh), np.max(Rlow), dur]))
                self._topo, topo_change, self._OWflux, netDischarge, inundated, Qbe = routine.storm_processes(
                    topof=self._topo,
                    Rhigh=Rhigh,
                    Rlow=Rlow,
                    dur=dur,
                    threshold_in=self._threshold_in,
                    Rin_i=self._Rin_in,
                    Rin_r=self._Rin_ru,
                    Cx=self._Cx,
                    AvgSlope=2 / 200,  # Representative average slope of interior (made static - representative of 200-m-wide barrier interior)
                    nn=self._nn,
                    MaxUpSlope=self._MaxUpSlope,
                    fluxLimit=self._fluxLimit,
                    Qs_min=self._Qs_min,
                    Kr=self._K_ru,
                    Ki=self._K_in,
                    mm=self._mm,
                    MHW=self._MHW,
                    Cbb_i=self._Cbb_in,
                    Cbb_r=self._Cbb_ru,
                    Qs_bb_min=self._Qs_bb_min,
                    substep_i=self._substep_in,
                    substep_r=self._substep_ru,
                    beach_equilibrium_slope=self._beach_equilibrium_slope,
                    beach_erosiveness=self._beach_erosiveness,
                    beach_substeps=self._beach_substeps,
                    x_s=self._x_s,
                    cellsize=self._cellsize,
                    spec1=self._spec1,
                    spec2=self._spec2,
                    flow_reduction_max_spec1=self._flow_reduction_max_spec1,
                    flow_reduction_max_spec2=self._flow_reduction_max_spec2,
                )

                # Enforce angles of repose again after overwash
                self._topo = routine.enforceslopes(self._topo, self._veg, self._slabheight, self._repose_bare, self._repose_veg, self._repose_threshold, self._RNG)[0]  # [m NAVD88]

                # Update vegetation from storm effects
                self._spec1[inundated] = 0  # Remove species where beach is inundated
                self._spec2[inundated] = 0  # Remove species where beach is inundated

            else:
                self._OWflux = np.zeros([self._longshore])  # [m^3] No overwash if no storm
                topo_change = np.zeros(self._topo.shape)  # [m NAVD88]
                Qbe = np.zeros([self._longshore])  # [m^3] No overwash if no storm

            self._sedimentation_balance = self._sedimentation_balance + topo_change  # [m]
            self._topographic_change = self._topographic_change + abs(topo_change)  # [m]

            # --------------------------------------
            # SHORELINE CHANGE

            # Update Shoreline Position from Cross-Shore Sediment Transport (i.e., RSLR, overwash, beach/dune change)
            self._x_s, self._x_t, shoreface_slope = routine.shoreline_change_from_CST(
                self._topo,
                self._DShoreface,
                self._k_sf,
                self._s_sf_eq,
                self._RSLR,
                Qbe,  # [m^3] Volume of sediment imported from (+) or exported to (-) the upper shoreface by storm beach/duneface change
                self._OWflux,
                self._x_s,
                self._x_t,
                self._MHW,
            )

            # Update Shoreline Position from Alongshore Sediment Transport (i.e., alongshore wave diffusion)
            self._x_s = routine.shoreline_change_from_AST(self._x_s,
                                                          self._wave_asymetry,
                                                          self._wave_high_angle_fraction,  # Note: High-angle waves seem to create problems if the alongshore domain length is very small (< 200m or so)
                                                          self._mean_wave_height,
                                                          self._mean_wave_period,
                                                          self._alongshore_section_length,
                                                          self._storm_update_frequency / self._iterations_per_cycle
                                                          )

            prev_shoreline = self._x_s_TS[-1].astype(int)  # Shoreline positions from previous time step
            shoreline_change = (self._x_s - prev_shoreline) * self._cellsize  # [cellsize] Shoreline change from last time step

            self._shoreline_change_aggregate += shoreline_change  # Add to aggregate of shoreline change over whole simulation

            shoreline_change[self._shoreline_change_aggregate >= 1] = np.floor(self._shoreline_change_aggregate[self._shoreline_change_aggregate >= 1]).astype(int)  # Remove partial shoreline change for this iteration
            self._shoreline_change_aggregate[self._shoreline_change_aggregate >= 1] -= shoreline_change[self._shoreline_change_aggregate >= 1]  # Subtract this iteration's shoreline change from aggregate

            shoreline_change[self._shoreline_change_aggregate <= -1] = np.ceil(self._shoreline_change_aggregate[self._shoreline_change_aggregate <= -1]).astype(int)  # Remove partial shoreline change for this iteration
            self._shoreline_change_aggregate[self._shoreline_change_aggregate <= -1] -= shoreline_change[self._shoreline_change_aggregate <= -1]  # Subtract this iteration's shoreline change from aggregate

            shoreline_change[np.logical_and(-1 < shoreline_change, shoreline_change < 1)] = 0  # Remove partial shoreline change for this iteration

            # Adjust topography domain to according to shoreline change
            if np.sum(shoreline_change) != 0:
                self._topo = routine.adjust_ocean_shoreline(
                    self._topo,
                    shoreline_change,
                    prev_shoreline,
                    self._MHW,
                    shoreface_slope,
                    self._RSLR,
                )

            # Store shoreline and shoreface toe locations
            self._x_s_TS = np.vstack((self._x_s_TS, self._x_s))  # Store
            self._x_t_TS = np.vstack((self._x_t_TS, self._x_t))  # Store


        # --------------------------------------
        # VEGETATION

        if it % self._iterations_per_cycle == 0 and it > 0:  # Update the vegetation

            # Growth and Decline
            veg_multiplier = (1 + self._growth_reduction_timeseries[self._vegcount])  # For the long term reduction
            self._sp1_peak = self._sp1_peak_at0 * veg_multiplier
            self._sp2_peak = self._sp2_peak_at0 * veg_multiplier
            spec1_prev = copy.deepcopy(self._spec1)
            spec2_prev = copy.deepcopy(self._spec2)
            self._spec1 = routine.growthfunction1_sens(self._spec1, self._sedimentation_balance, self._sp1_a, self._sp1_b, self._sp1_c, self._sp1_d, self._sp1_e, self._sp1_peak)
            self._spec2 = routine.growthfunction2_sens(self._spec2, self._sedimentation_balance, self._sp2_a, self._sp2_b, self._sp2_d, self._sp2_e, self._sp2_peak)

            # Lateral Expansion
            lateral1 = routine.lateral_expansion(spec1_prev, 1, self._lateral_probability * veg_multiplier, self._RNG)
            lateral2 = routine.lateral_expansion(spec2_prev, 1, self._lateral_probability * veg_multiplier, self._RNG)
            lateral1[self._topo <= self._MHW] = False  # Constrain to subaerial
            lateral2[self._topo <= self._MHW] = False

            # Pioneer Establishment
            pioneer1 = routine.establish_new_vegetation(self._topo, self._MHW, self._pioneer_probability * veg_multiplier, self._RNG) * (spec1_prev <= 0)
            pioneer2 = routine.establish_new_vegetation(self._topo, self._MHW, self._pioneer_probability * veg_multiplier, self._RNG) * (spec2_prev <= 0) * (self._topographic_change == 0)
            pioneer1[self._topo <= self._MHW] = False  # Constrain to subaerial
            pioneer2[self._topo <= self._MHW] = False

            # Update Vegetation Cover
            spec1_diff = self._spec1 - spec1_prev  # Determine changes in vegetation cover
            spec2_diff = self._spec2 - spec2_prev  # Determine changes in vegetation cover
            spec1_growth = spec1_diff * (spec1_diff > 0)  # Split cover changes into into gain and loss
            spec1_loss = spec1_diff * (spec1_diff < 0)
            spec2_growth = spec2_diff * (spec2_diff > 0)  # Split cover changes into into gain and loss
            spec2_loss = spec2_diff * (spec2_diff < 0)

            spec1_change_allowed = np.minimum(1 - self._veg, spec1_growth) * np.logical_or(lateral1, pioneer1)  # Only allow growth in adjacent or pioneer cells
            spec2_change_allowed = np.minimum(1 - self._veg, spec2_growth) * np.logical_or(lateral2, pioneer2)  # Only allow growth in adjacent or pioneer cells
            self._spec1 = spec1_prev + spec1_change_allowed + spec1_loss  # Re-assemble gain and loss and add to original vegetation cover
            self._spec2 = spec2_prev + spec2_change_allowed + spec2_loss  # Re-assemble gain and loss and add to original vegetation cover

            Spec1_elev_min_mhw = self._Spec1_elev_min + self._MHW  # [m MHW]
            Spec2_elev_min_mhw = self._Spec2_elev_min + self._MHW  # [m MHW]
            self._spec1[self._topo <= Spec1_elev_min_mhw] = 0  # Remove species where below elevation minimum
            self._spec2[self._topo <= Spec2_elev_min_mhw] = 0  # Remove species where below elevation minimum

            # Limit to geomorphological range
            spec1_geom = copy.deepcopy(self._spec1)
            spec1_geom[self._spec1 < 0] = 0
            spec1_geom[self._spec1 > 1] = 1
            spec2_geom = copy.deepcopy(self._spec2)
            spec2_geom[self._spec2 < 0] = 0
            spec2_geom[self._spec2 > 1] = 1
            self._veg = spec1_geom + spec2_geom  # Update vegmap
            self._veg[self._veg > self._maxvegeff] = self._maxvegeff  # Limit to effective range
            self._veg[self._veg < 0] = 0

            # Determine effective vegetation cover by smoothing; represents effect of nearby vegetation on local wind
            self._effective_veg = scipy.ndimage.filters.gaussian_filter(self._veg, [1.5, 1.5], mode='constant')

            self._vegcount = self._vegcount + 1  # Update counter

        # --------------------------------------
        # RECORD VARIABLES PERIODICALLY

        if (it + 1) % (self._save_frequency * self._iterations_per_cycle) == 0:
            moment = int((it + 1) / self._save_frequency / self._iterations_per_cycle)
            self._topo_TS[:, :, moment] = self._topo
            self._spec1_TS[:, :, moment] = self._spec1
            self._spec2_TS[:, :, moment] = self._spec2
            self._veg_TS[:, :, moment] = self._veg

        self._erosmap_sum = self._erosmap_sum + aeolian_erosion_prob
        self._deposmap_sum = self._deposmap_sum + aeolian_deposition_prob
        self._sedimentation_balance_sumA = self._sedimentation_balance_sumA + balance_init
        self._sedimentation_balance_sumB = self._sedimentation_balance_sumB + self._sedimentation_balance
        self._topographic_change_sumA = self._topographic_change_sumA + stability_init
        self._topographic_change_sumB = self._topographic_change_sumB + self._topographic_change
        self._avalanched_cells[it] = aval

        # --------------------------------------
        # RESET DOMAINS

        self._sedimentation_balance[:] = 0  # [m] Reset the balance map
        self._topographic_change[:] = 0  # [m] Reset the balance map


    @property
    def name(self):
        return self._name

    @property
    def iterations(self):
        return self._iterations

    @property
    def iterations_per_cycle(self):
        return self._iterations_per_cycle

    @property
    def topo(self):
        return self._topo

    @property
    def topo_TS(self):
        return self._topo_TS

    @property
    def slabheight(self):
        return self._slabheight

    @property
    def veg(self):
        return self._veg

    @property
    def veg_TS(self):
        return self._veg_TS

    @property
    def spec1_TS(self):
        return self._spec1_TS

    @property
    def spec2_TS(self):
        return self._spec2_TS

    @property
    def save_data(self):
        return self._save_data

    @property
    def save_frequency(self):
        return self._save_frequency

    @property
    def outputloc(self):
        return self._outputloc

    @property
    def simnum(self):
        return self._simnum

    @property
    def simulation_time_yr(self):
        return self._simulation_time_yr

    @property
    def RSLR(self):
        return self._RSLR

    @property
    def MHW(self):
        return self._MHW

    @property
    def StormRecord(self):
        return self._StormRecord

    @property
    def x_s_TS(self):
        return self._x_s_TS

    @property
    def storm_iterations_per_year(self):
        return self._storm_iterations_per_year


# # __________________________________________________________________________________________________________________________________
# # RUN MODEL
#
# start_time = time.time()  # Record time at start of simulation
#
# # Create an instance of the BMI class
# meeb = MEEB(
#     name="SLR 3 mm/yr, 2009-2012 Hindcast",
#     simulation_time_yr=3,
#     alongshore_domain_boundary_min=6000,
#     alongshore_domain_boundary_max=6500,
#     RSLR=0.003,
#     seeded_random_numbers=True,
#     p_dep_sand=0.05,  # 0.25 = 10 m^3/m/yr, 0.5 = 5 m^m/3/yr, 0.75 = 3.333 m^m/3/yr, 1 = 2.5 m^m/3/yr
#     p_ero_sand=0.25,  # if p_dep = 0.5, p_ero of 0.5 = 5 m^m/3/yr, 0.25 = 2.5 m^m/3/yr, 0.1 = 1 m^m/3/yr
#     direction2=2,
#     direction4=4,
#     wave_asymetry=0.5,
#     init_filename="Init_NCB-NewDrum-Ocracoke_2009_PreIrene.npy",
#     hindcast=True,
#     hindcast_start=1530,
#     storm_timeseries_filename='StormTimeSeries_1980-2020_NCB-CE_Beta0pt039_BermEl2pt03.npy',
# )
#
# print(meeb.name)
#
# # Loop through time
# for time_step in range(int(meeb.iterations)):
#     # Print time step to screen
#     print("\r", "Time Step: ", time_step / meeb.iterations_per_cycle, "years", end="")
#
#     # Run time step
#     meeb.update(time_step)
#
# # Print elapsed time of simulation
# print()
# SimDuration = time.time() - start_time
# print()
# print("Elapsed Time: ", SimDuration, "sec")
#
# # Save Results
# if meeb.save_data:
#     filename = meeb.outputloc + "Sim_" + str(meeb.simnum)
#     dill.dump_module(filename)  # To re-load data: dill.load_session(filename)
#
#
# # __________________________________________________________________________________________________________________________________
# # PLOT RESULTS
#
# # Final Elevation & Vegetation
# Fig = plt.figure(figsize=(14, 9.5))
# Fig.suptitle(meeb.name, fontsize=13)
# MHW = meeb.RSLR * meeb.simulation_time_yr
# topo = meeb.topo * meeb.slabheight
# topo = np.ma.masked_where(topo <= MHW, topo)  # Mask cells below MHW
# cmap1 = routine.truncate_colormap(copy.copy(plt.cm.get_cmap("terrain")), 0.5, 0.9)  # Truncate colormap
# cmap1.set_bad(color='dodgerblue', alpha=0.5)  # Set cell color below MHW to blue
# ax1 = Fig.add_subplot(211)
# cax1 = ax1.matshow(topo, cmap=cmap1, vmin=0, vmax=5.0)
# cbar = Fig.colorbar(cax1)
# cbar.set_label('Elevation [m]', rotation=270, labelpad=20)
# ax2 = Fig.add_subplot(212)
# veg = meeb.veg
# veg = np.ma.masked_where(topo <= MHW, veg)  # Mask cells below MHW
# cmap2 = copy.copy(plt.cm.get_cmap("YlGn"))
# cmap2.set_bad(color='dodgerblue', alpha=0.5)  # Set cell color below MHW to blue
# cax2 = ax2.matshow(veg, cmap=cmap2, vmin=0, vmax=1)
# cbar = Fig.colorbar(cax2)
# cbar.set_label('Vegetation [%]', rotation=270, labelpad=20)
# plt.tight_layout()
#
# # Animation: Elevation and Vegetation Over Time
# for t in range(0, int(meeb.simulation_time_yr / meeb.writeyear) + 1):
#     Fig = plt.figure(figsize=(14, 8))
#
#     MHW = meeb.RSLR * t
#     topo = meeb.topo_TS[:, :, t] * meeb.slabheight  # [m]
#     topo = np.ma.masked_where(topo <= MHW, topo)  # Mask cells below MHW
#     cmap1 = routine.truncate_colormap(copy.copy(plt.cm.get_cmap("terrain")), 0.5, 0.9)  # Truncate colormap
#     cmap1.set_bad(color='dodgerblue', alpha=0.5)  # Set cell color below MHW to blue
#     ax1 = Fig.add_subplot(211)
#     cax1 = ax1.matshow(topo, cmap=cmap1, vmin=0, vmax=5.0)
#     cbar = Fig.colorbar(cax1)
#     cbar.set_label('Elevation [m]', rotation=270, labelpad=20)
#     timestr = "Year " + str(t * meeb.writeyear)
#     plt.text(2, meeb.topo.shape[0] - 2, timestr, c='white')
#
#     veg = meeb.veg_TS[:, :, t]
#     veg = np.ma.masked_where(topo <= MHW, veg)  # Mask cells below MHW
#     cmap2 = copy.copy(plt.cm.get_cmap("YlGn"))
#     cmap2.set_bad(color='dodgerblue', alpha=0.5)  # Set cell color below MHW to blue
#     ax2 = Fig.add_subplot(212)
#     cax2 = ax2.matshow(veg, cmap=cmap2, vmin=0, vmax=1)
#     cbar = Fig.colorbar(cax2)
#     cbar.set_label('Vegetation [%]', rotation=270, labelpad=20)
#     timestr = "Year " + str(t * meeb.writeyear)
#     plt.text(2, meeb.veg.shape[0] - 2, timestr, c='darkblue')
#     plt.tight_layout()
#     if not os.path.exists("Output/SimFrames/"):
#         os.makedirs("Output/SimFrames/")
#     name = "Output/SimFrames/meeb_elev_" + str(t)
#     plt.savefig(name)  # dpi=200
#     plt.close()
#
# frames = []
# for filenum in range(0, int(meeb.simulation_time_yr / meeb.writeyear) + 1):
#     filename = "Output/SimFrames/meeb_elev_" + str(filenum) + ".png"
#     frames.append(imageio.imread(filename))
# imageio.mimwrite("Output/SimFrames/meeb_elev.gif", frames, fps=3)
# print()
# print("[ * GIF successfully generated * ]")
#
# plt.show()
