"""__________________________________________________________________________________________________________________________________

Main Model Script for MEEB

Mesoscale Explicit Ecogeomorphic Barrier model

IRB Reeves

Last update: 16 October 2024

__________________________________________________________________________________________________________________________________"""

import numpy as np
import math
import matplotlib.pyplot as plt
import scipy
import copy
import gc
from datetime import datetime, timedelta

import routines_meeb as routine


class MEEB:
    def __init__(
            self,

            # GENERAL
            name="default",
            simnum=1,  # Reference number of the simulation. Used for personal reference.
            MHW=0.39,  # [m NAVD88] Initial mean high water
            RSLR=0.000,  # [m/yr] Relative sea-level rise rate
            aeolian_iterations_per_year=50,  # Number of aeolian updates in 1 model year
            storm_iterations_per_year=25,  # Number of storm and shoreline change updates in 1 model year
            vegetation_iterations_per_year=1,  # Number of vegetation updates in 1 model year
            save_frequency=0.5,  # [years] Save results every n years
            simulation_time_yr=15.0,  # [yr] Length of the simulation time
            cellsize=1,  # [m] Cell length and width
            alongshore_domain_boundary_min=0,  # [m] Alongshore minimum boundary location for model domain
            alongshore_domain_boundary_max=10e7,  # [m] Alongshore maximum boundary location for model domain; if left to this default value, it will automatically adjust to the actual full length of the domain
            crossshore_domain_boundary_min=0,  # [m] Cross-shore minimum boundary location for model domain
            crossshore_domain_boundary_max=10e7,  # [m] Cross-shore maximum boundary location for model domain; if left to this default value, it will automatically adjust to the actual full length of the domain
            inputloc="Input/",  # Input file directory (end string with "/")
            outputloc="Output/",  # Output file directory (end string with "/")
            init_by_file=True,  # [bool] Whether to initialize model by providing filenames for numpy arrays that are read (True), or directly input arrays into model (False); the latter is MUCH better for parallel sims
            init_filename="Init_NCB-NewDrum-Ocracoke_2018_PostFlorence_18400-23400.npy",  # [m NVD88] Name of initial topography and vegetation input file; requires input_by_file to be TRUE
            init_elev_array=np.array(np.nan),  # [m NAVD88] Numpy array of initial elevation; requires init_by_file to be False
            init_spec1_array=np.array(np.nan),  # [0-1] Numpy array of initial spec1 density; requires init_by_file to be False
            init_spec2_array=np.array(np.nan),  # [0-1] Numpy array of initial spec2 density; requires init_by_file to be False
            hindcast=False,  # [bool] Determines whether the model is run with the default stochastisity generated storms [hindcast=False], or an empirical storm, wind, wave, temp timeseries [hindcast=True]
            simulation_start_date='20181007',  # [date] Date from which to start hindcast; must be string in format 'yyyymmdd'
            hindcast_timeseries_start_date='19790101',  # [date] Start date of hindcast timeseries input data; format 'yyyymmdd'
            seeded_random_numbers=True,  # [bool] Determines whether to use seeded random number generator for reproducibility
            save_data=False,

            # AEOLIAN
            slabheight=0.02,  # Height of slabs for aeolian transport, proportion of cell dimension (0.02, Teixeira et al. 2023)
            saltation_length=5,  # [cells] Hop length for saltating slabs of sand (5 m, Teixeira et al. 2023); note units of cells (e.g., if cellsize = 2 m and saltation_length = 5 cells, slabs will hop 10 m)
            saltation_length_rand_deviation=2,  # [cells] Deviation around saltation_length for random uniform distribution of saltation lengths. Must be at lest 1 cell smaller than saltation_length.
            groundwater_depth=0.4,  # Proportion of the smoothed topography used to set groundwater profile
            wind_rose=(0.81, 0.04, 0.06, 0.09),  # Proportion of wind TOWARDS (right, down, left, up)
            p_dep_sand=0.22,  # [0-1] Probability of deposition in sandy cells with 0% vegetation cover
            p_dep_sand_VegMax=0.54,  # [0-1] Probability of deposition in sandy cells with 100% vegetation cover; must be greater than or equal to p_dep_sand/p_dep_basesaltation_length_rand_deviation
            p_dep_base=0.1,  # [0-1] Probability of deposition of base cells
            p_ero_sand=0.10,  # [0-1] Probability of erosion of bare/sandy cells
            entrainment_veg_limit=0.10,  # [0-1] Percent of vegetation cover beyond which aeolian sediment entrainment is no longer possible
            saltation_veg_limit=0.35,  # Threshold vegetation effectiveness needed for a cell along a slab saltation path to be considered vegetated
            shadowangle=12,  # [deg]
            repose_bare=20,  # [deg]
            repose_veg=30,  # [deg]
            repose_threshold=0.3,  # [0-1] Vegetation threshold for applying repose_veg
            eq_backbarrier_depth=1.5,  # [m] Equilibrium depth of back-barrier bay/lagoon

            # SHOREFACE & SHORELINE
            shoreface_flux_rate=5000,  # [m3/m/yr] Shoreface flux rate coefficient
            shoreface_equilibrium_slope=0.02,  # Equilibrium slope of the shoreface
            shoreface_depth=10,  # [m] Depth to shoreface toe (i.e. depth of ‘closure’)
            shoreface_length_init=500,  # [m] Initial length of shoreface
            wave_asymmetry=0.6,  # [0-1] Fraction of waves approaching from the left (when looking offshore)
            wave_high_angle_fraction=0.39,  # Fraction of waves approaching at angles higher than 45 degrees from shore normal
            mean_wave_height=0.98,  # [m] Mean offshore significant wave height
            mean_wave_period=6.6,  # [s] Mean wave period
            alongshore_section_length=25,  # [m] Distance alongshore between shoreline positions used in the shoreline diffusion calculations
            average_dune_toe_height=1.67,  # [m] Time- and space-averaged dune toe height above MHW
            estimate_shoreface_parameters=True,  # [bool] Turn on to estimate shoreface parameters as function of specific wave and sediment characteristics
            shoreface_grain_size=2e-4,  # [m] Median grain size (D50) of ocean shoreface; used for optional shoreface parameter estimations
            shoreface_transport_efficiency=0.01,  # Shoreface suspended sediment transport efficiency factor; used for optional shoreface parameter estimations
            shoreface_friction=0.01,  # Shoreface friction factor; used for optional shoreface parameter estimations
            specific_gravity_submerged_sed=1.65,  # Submerged specific gravity of sediment; used for optional shoreface parameter estimations

            # VEGETATION
            sp1_a=-1.5,  # Vertice a, spec1. vegetation growth based on Nield and Baas (2008)
            sp1_b=-0.05,  # Vertice b, spec1. vegetation growth based on Nield and Baas (2008)
            sp1_c=0.5,  # Vertice c, spec1. vegetation growth based on Nield and Baas (2008)
            sp1_d=1.5,  # Vertice d, spec1. vegetation growth based on Nield and Baas (2008)
            sp1_e=2.2,  # Vertice e, spec1. vegetation growth based on Nield and Baas (2008)
            sp2_a=-1.6,  # Vertice a, spec2. vegetation growth based on Nield and Baas (2008)
            sp2_b=-0.7,  # Vertice b, spec2. vegetation growth based on Nield and Baas (2008)
            sp2_c=0.0,  # Vertice c, spec2. vegetation growth based on Nield and Baas (2008)
            sp2_d=0.2,  # Vertice d, spec2. vegetation growth based on Nield and Baas (2008)
            sp2_e=2.1,  # Vertice e, spec2. vegetation growth based on Nield and Baas (2008)
            sp1_peak=0.2,  # Growth peak, spec1
            sp2_peak=0.05,  # Growth peak, spec2
            VGR=0,  # [%] Growth reduction by end of period
            sp1_lateral_probability=0.2,  # [0-1] Probability of lateral expansion of existing vegetation
            sp2_lateral_probability=0.2,  # [0-1] Probability of lateral expansion of existing vegetation
            sp1_pioneer_probability=0.05,  # [0-1] Probability of occurrence of new pioneering vegetation
            sp2_pioneer_probability=0.05,  # [0-1] Probability of occurrence of new pioneering vegetation
            maxvegeff=1.0,  # [0-1] Value of maximum vegetation effectiveness allowed
            Spec1_elev_min=0.25,  # [m MHW] Minimum elevation (relative to MHW) for species 1
            Spec2_elev_min=0.25,  # [m MHW] Minimum elevation (relative to MHW) for species 2
            flow_reduction_max_spec1=0.02,  # [0-1] Proportion of overwash flow reduction through a cell populated with species 1 at full density
            flow_reduction_max_spec2=0.05,  # [0-1] Proportion of overwash flow reduction through a cell populated with species 2 at full density
            effective_veg_sigma=3,  # Standard deviation for Gaussian filter of vegetation cover

            # STORM OVERWASH AND BEACH-DUNE CHANGE
            storm_list_filename="SyntheticStorms_NCB-CE_10k_1979-2020_Beta0pt039_BermEl1pt78.npy",
            storm_timeseries_filename="StormTimeSeries_1979-2020_NCB-CE_Beta0pt039_BermEl1pt78.npy",  # Only needed if running hindcast simulations (i.e., without stochastic storms)
            Rin=249,  # [m^3/hr] Flow infiltration and drag parameter, run-up overwash regime
            Cs=0.0283,  # Constant for representing flow momentum for sediment transport in overwash
            nn=0.5,  # Flow routing constant
            MaxUpSlope=1.5,  # Maximum slope water can flow uphill
            marine_flux_limit=1,  # [m/hr] Maximum elevation change allowed per time step (prevents instabilities)
            overwash_min_discharge=1.0,  # [m^3/hr] Minimum discharge out of cell needed to transport sediment
            Kow=0.0001684,  # Sediment transport coefficient for run-up overwash regime
            mm=1.04,  # Inundation overwash constant
            Cbb=0.7,  # [0-1] Coefficient for exponential decay of sediment load entering back-barrier bay, run-up regime
            overwash_min_subaqueous_discharge=1,  # [m^3/hr] Minimum discharge out of subaqueous back-barrier cell needed to transport sediment
            overwash_substeps=25,  # Number of substeps to run for each hour in run-up overwash regime (e.g., 3 substeps means discharge/elevation updated every 20 minutes)
            beach_equilibrium_slope=0.022,  # Equilibrium slope of the beach
            swash_erosive_timescale=1.48,  # Non-dimensional erosive timescale coefficient for beach/duneface sediment transport (Duran Vinent & Moore, 2015)
            beach_substeps=25,  # Number of substeps per iteration of beach/duneface model; instabilities will occur if too low
            shift_mean_storm_intensity=0,  # [%/yr] Linear yearly percent shift in mean storm TWL (as proxy for intensity) in stochastic storm model; use 0 for no shift
    ):
        """MEEB: Mesoscale Explicit Ecogeomorphic Barrier model.


        Examples
        --------
        >>> meeb = MEEB(simulation_time_yr=3, RSLR=0.003)

        Create an instance of the BMI class.

        >>> for time_step in range(int(meeb.iterations)):
        ...     meeb.update(time_step)

        Loop through time.

        >>> ElevFig = plt.figure(figsize=(14, 7.5))
        ... plt.matshow(meeb.topo, cmap='terrain', vmin=-1, vmax=6)

        Plot elevation from final timestep.
        """

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
        self._crossshore_domain_boundary_min = crossshore_domain_boundary_min
        self._crossshore_domain_boundary_max = crossshore_domain_boundary_max
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
        self._saltation_length = saltation_length
        self._saltation_length_rand_deviation = saltation_length_rand_deviation
        self._beach_equilibrium_slope = beach_equilibrium_slope
        self._swash_erosive_timescale = swash_erosive_timescale
        self._beach_substeps = beach_substeps
        self._k_sf = shoreface_flux_rate
        self._s_sf_eq = shoreface_equilibrium_slope
        self._DShoreface = shoreface_depth
        self._LShoreface = shoreface_length_init
        self._wave_asymmetry = wave_asymmetry
        self._wave_high_angle_fraction = wave_high_angle_fraction
        self._mean_wave_height = mean_wave_height
        self._mean_wave_period = mean_wave_period
        self._alongshore_section_length = alongshore_section_length
        self._average_dune_toe_height = average_dune_toe_height
        self._eq_backbarrier_depth = eq_backbarrier_depth
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
        self._sp1_lateral_probability = sp1_lateral_probability
        self._sp2_lateral_probability = sp2_lateral_probability
        self._sp1_pioneer_probability = sp1_pioneer_probability
        self._sp2_pioneer_probability = sp2_pioneer_probability
        self._maxvegeff = maxvegeff
        self._Spec1_elev_min = Spec1_elev_min
        self._Spec2_elev_min = Spec2_elev_min
        self._flow_reduction_max_spec1 = flow_reduction_max_spec1
        self._flow_reduction_max_spec2 = flow_reduction_max_spec2
        self._effective_veg_sigma = effective_veg_sigma
        self._Rin = Rin
        self._Cs = Cs
        self._nn = nn
        self._MaxUpSlope = MaxUpSlope
        self._marine_flux_limit = marine_flux_limit
        self._overwash_min_discharge = overwash_min_discharge
        self._Kow = Kow
        self._mm = mm
        self._Cbb = Cbb
        self._overwash_min_subaqueous_discharge = overwash_min_subaqueous_discharge
        self._overwash_substeps = overwash_substeps
        self._shift_mean_storm_intensity = shift_mean_storm_intensity

        # __________________________________________________________________________________________________________________________________
        # SET INITIAL CONDITIONS

        # SEEDED RANDOM NUMBER GENERATOR
        if seeded_random_numbers:
            self._RNG = np.random.default_rng(seed=13)  # Seeded random numbers for reproducibility (e.g., model development/testing)
            self._RNG_storm = np.random.default_rng(seed=14)  # Separate seeded RNG for storms so that the storm sequence can always stay the same despite any parameterization changes
        else:
            self._RNG = np.random.default_rng()  # Non-seeded random numbers (e.g., model simulations)
            self._RNG_storm = np.random.default_rng()

        # TIME
        self._iterations_per_cycle = self._aeolian_iterations_per_year  # [iterations/year] Number of iterations in 1 model year
        self._vegetation_update_frequency = round(self.iterations_per_cycle / self._vegetation_iterations_per_year)  # Frequency of vegetation updates (i.e., every n iterations)
        self._storm_update_frequency = round(self.iterations_per_cycle / self._storm_iterations_per_year)  # Frequency of storm updates (i.e., every n iterations)
        self._iterations = int(self._iterations_per_cycle * self._simulation_time_yr)  # Total number of iterations
        self._simulation_start_date = datetime.strptime(self._simulation_start_date, '%Y%m%d').date()  # Convert to datetime
        if hindcast:
            self._hindcast_timeseries_start_date = datetime.strptime(self._hindcast_timseries_start_date, '%Y%m%d').date()  # Convert to datetime
            self._simulation_start_iteration = (((self._simulation_start_date.year - self._hindcast_timeseries_start_date.year) * self._iterations_per_cycle) +
                                                math.floor(self._simulation_start_date.timetuple().tm_yday / 365 * self._iterations_per_cycle))  # Iteration, realtive to timeseries start, from which to begin hindcast
            if self._simulation_start_iteration % 2 != 0:
                self._simulation_start_iteration -= 1  # Round simulation start iteration to even number
        else:
            self._simulation_start_iteration = 0
        self._iteration_dates = [self._simulation_start_date + timedelta(minutes=10512 * x) for x in range(self._iterations)]  # List of dates corresponding to each model iteration

        # TOPOGRAPHY
        if init_by_file:
            Init = np.float32(np.load(inputloc + init_filename))
            self._alongshore_domain_boundary_max = min(self._alongshore_domain_boundary_max, Init[0, :, :].shape[0])
            self._crossshore_domain_boundary_max = min(self._crossshore_domain_boundary_max, Init[0, :, :].shape[1])
            self._topo = Init[0, self._alongshore_domain_boundary_min: self._alongshore_domain_boundary_max, self._crossshore_domain_boundary_min: self._crossshore_domain_boundary_max]  # [m NAVD88] 2D array of initial topography
            self._spec1 = Init[1, self._alongshore_domain_boundary_min: self._alongshore_domain_boundary_max, self._crossshore_domain_boundary_min: self._crossshore_domain_boundary_max]  # [0-1] 2D array of vegetation effectiveness for spec1
            self._spec2 = Init[2, self._alongshore_domain_boundary_min: self._alongshore_domain_boundary_max, self._crossshore_domain_boundary_min: self._crossshore_domain_boundary_max]  # [0-1] 2D array of vegetation effectiveness for spec2
        else:
            if np.logical_or(np.isnan(np.sum(init_elev_array)), np.logical_or(np.isnan(np.sum(init_spec1_array)), np.isnan(np.sum(init_spec2_array)))):
                raise ValueError("Initial elevation and vegetation numpy arrays must be provided as input to MEEB object if init_by_file is False.")
            else:
                self._topo = copy.deepcopy(init_elev_array)  # [m NAVD88] 2D array of initial topography
                self._spec1 = copy.deepcopy(init_spec1_array)  # [0-1] 2D array of vegetation effectiveness for spec1
                self._spec2 = copy.deepcopy(init_spec2_array)  # [0-1] 2D array of vegetation effectiveness for spec2
        self._longshore, self._crossshore = self._topo.shape  # [cells] Cross-shore/alongshore size of domain
        self._groundwater_elevation = np.zeros(self._topo.shape)  # [m NAVD88] Initialize

        # SHOREFACE & SHORELINE
        if estimate_shoreface_parameters:  # Option to estimate shoreface parameter values from wave and sediment charactersitics; Follows Nienhuis & Lorenzo-Trueba (2019)
            w_s = (specific_gravity_submerged_sed * 9.81 * shoreface_grain_size ** 2) / ((18 * 1e-6) + np.sqrt(0.75 * specific_gravity_submerged_sed * 9.81 * (shoreface_grain_size ** 3)))  # [m/s] Settling velocity (Church & Ferguson, 2004)
            z0 = 2 * self._mean_wave_height / 0.78  # [m] Minimum depth of integration (simple approximation of breaking wave depth based on offshore wave height)
            self._DShoreface = 0.018 * self._mean_wave_height * self._mean_wave_period * math.sqrt(9.81 / (specific_gravity_submerged_sed * shoreface_grain_size))  # [m] Shoreface depth
            self._s_sf_eq = (3 * w_s / 4 / np.sqrt(self._DShoreface * 9.81) * (5 + 3 * self._mean_wave_period ** 2 * 9.81 / 4 / (np.pi ** 2) / self._DShoreface))  # Equilibrium shoreface slope
            self._k_sf = ((3600 * 24 * 365) * ((shoreface_transport_efficiency * shoreface_friction * 9.81 ** (11 / 4) * self._mean_wave_height ** 5 * self._mean_wave_period ** (5 / 2)) / (960 * specific_gravity_submerged_sed * math.pi ** (7 / 2) * w_s ** 2)) *
                          (((1 / (11 / 4 * z0 ** (11 / 4))) - (1 / (11 / 4 * self._DShoreface ** (11 / 4)))) / (self._DShoreface - z0)))  # [m^3/m/yr] Shoreface response rate
            self._LShoreface = int(self._DShoreface / self._s_sf_eq)  # [m] Initialize length of shoreface such that initial shoreface slope equals equilibrium shoreface slope
        self._alongshore_section_length = int(self._alongshore_section_length / self._cellsize)  # [cells]
        self._x_s = routine.init_ocean_shoreline(self._topo, self._MHW, self._alongshore_section_length)  # [m] Start locations of shoreline according to initial topography and MHW
        self._x_t = self._x_s - self._LShoreface  # [m] Start locations of shoreface toe

        self._coast_diffusivity, self._di, self._dj, self._ny = routine.init_AST_environment(self._wave_asymmetry,
                                                                                             self._wave_high_angle_fraction,
                                                                                             self._mean_wave_height,
                                                                                             self._mean_wave_period,
                                                                                             self._DShoreface,
                                                                                             self._alongshore_section_length,
                                                                                             self._longshore)

        # VEGETATION
        self._veg = self._spec1 + self._spec2  # Determine the initial cumulative vegetation effectiveness
        self._veg[self._veg > self._maxvegeff] = self._maxvegeff  # Cumulative vegetation effectiveness cannot be negative or larger than one
        self._veg[self._veg < 0] = 0
        self._effective_veg = scipy.ndimage.gaussian_filter(self._veg, [self._effective_veg_sigma / self._cellsize, self._effective_veg_sigma / self._cellsize], mode='constant')  # Effective vegetation cover represents effect of nearby vegetation on local wind
        self._growth_reduction_timeseries = np.linspace(0, self._VGR / 100, int(np.ceil(self._simulation_time_yr)))

        # STORMS
        self._StormList = np.float32(np.load(inputloc + storm_list_filename))
        self._mean_stochastic_storm_TWL = np.mean(self._StormList[:, 2])  # np.mean(self._StormList[:, 2][self._StormList[:, 2] >= self._average_dune_toe_height])
        self._storm_timeseries = np.load(inputloc + storm_timeseries_filename)
        self._pstorm = [0.333, 0.333, 0.167, 0.310, 0.381, 0.310, 0.310, 0.310, 0.286, 0, 0.119, 0.024, 0.048, 0.048, 0.048, 0.071, 0.333, 0.286, 0.214,
                        0.190, 0.190, 0.262, 0.214, 0.262, 0.238]  # Empirical probability of storm occurance for each 1/25th (~biweekly) iteration of the year, from 1979-2021 NCB storm record (1.78 m NAVD88 Berm Elev.)
        # self._pstorm = [0.738, 0.667, 0.571, 0.786, 0.833, 0.643, 0.643, 0.762, 0.476, 0.167, 0.238, 0.095, 0.214, 0.167, 0.119, 0.119, 0.357, 0.476, 0.357,
        #                 0.405, 0.524, 0.524, 0.738, 0.548, 0.619]  # Empirical probability of storm occurance for each 1/25th (~biweekly) iteration of the year, from 1979-2021 NCB storm record (1.56 m NAVD88 Berm Elev.)

        if self._hindcast and self._iterations > (self._storm_timeseries[-1, 0] - self._simulation_start_iteration):
            raise ValueError("Simulation length is greater than hindcast timeSeries length.")

        # MODEL PARAMETERS
        self._MHW_init = copy.deepcopy(self._MHW)
        self._wind_direction = np.zeros([self._iterations], dtype=int)
        self._slabheight = round(self._slabheight, 2)  # Round slabheight to 2 decimals
        self._sedimentation_balance = np.zeros(self._topo.shape, dtype=np.float32)  # [m] Initialize map of the sedimentation balance: difference between erosion and deposition for 1 model year; (+) = net deposition, (-) = net erosion
        self._topographic_change = self._topo * 0  # [m] Map of the absolute value of topographic change over 1 model year (i.e., a measure of if the topography is changing or stable)
        self._x_s_TS = [self._x_s]  # Initialize storage array for shoreline position
        self._x_t_TS = [self._x_t]  # Initialize storage array for shoreface toe position
        self._sp1_peak_at0 = copy.deepcopy(self._sp1_peak)  # Store initial peak growth of sp. 1
        self._sp2_peak_at0 = copy.deepcopy(self._sp2_peak)  # Store initial peak growth of sp. 2
        self._vegcount = 0
        self._shoreline_change_aggregate = np.zeros([self._longshore])
        self._OWflux = np.zeros([self._longshore])  # [m^3]
        self._StormRecord = np.zeros([5])  # Record of each storm that occurs in model: Year, iteration, Rhigh, Rlow, duration

        # __________________________________________________________________________________________________________________________________
        # MODEL OUPUT CONFIGURATION

        self._topo_TS = np.empty([self._longshore, self._crossshore, int(np.floor(self._simulation_time_yr / self._save_frequency)) + 1], dtype=np.float32)  # Array for saving each topo map at specified frequency
        self._topo_TS[:, :, 0] = self._topo
        self._spec1_TS = np.empty([self._longshore, self._crossshore, int(np.floor(self._simulation_time_yr / self._save_frequency)) + 1], dtype=np.float32)  # Array for saving each spec1 map at specified frequency
        self._spec1_TS[:, :, 0] = self._spec1
        self._spec2_TS = np.empty([self._longshore, self._crossshore, int(np.floor(self._simulation_time_yr / self._save_frequency)) + 1], dtype=np.float32)  # Array for saving each spec2 map at specified frequency
        self._spec2_TS[:, :, 0] = self._spec2
        self._veg_TS = np.empty([self._longshore, self._crossshore, int(np.floor(self._simulation_time_yr / self._save_frequency)) + 1], dtype=np.float32)  # Array for saving each veg map at specified frequency
        self._veg_TS[:, :, 0] = self._veg
        self._storm_inundation_TS = np.zeros([self._longshore, self._crossshore, int(np.floor(self._simulation_time_yr / self._save_frequency)) + 1], dtype=np.float32)  # Array for saving each veg map at specified frequency
        self._inundated_output_aggregate = np.zeros([self._longshore, self._crossshore], dtype=np.float32)

        if init_by_file:
            del Init
            gc.collect()

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
        self._groundwater_elevation = (scipy.ndimage.gaussian_filter(self._topo, sigma=12 / self._cellsize) - self._MHW) * self._groundwater_depth + self._MHW  # [m NAVD88] Groundwater elevation based on smoothed topographic height above SL
        self._groundwater_elevation[self._groundwater_elevation < self._MHW] = self._MHW  # [m NAVD88]

        # Find subaerial and shadow cells
        subaerial = self._topo > self._MHW  # [bool] True for subaerial cells
        wind_shadows = routine.shadowzones(self._topo, self._shadowangle, direction=int(self._wind_direction[it]), MHW=self._MHW, cellsize=self._cellsize)  # [bool] Map of True for in shadow, False not in shadow

        # Erosion/Deposition Probabilities
        aeolian_erosion_prob = routine.erosprobs(self._effective_veg, wind_shadows, subaerial, self._topo, self._groundwater_elevation, self._p_ero_sand, self._entrainment_veg_limit, self._slabheight, self._MHW)  # Returns map of erosion probabilities
        aeolian_deposition_prob = routine.depprobs(self._effective_veg, wind_shadows, subaerial, self._p_dep_base, self._p_dep_sand, self._p_dep_sand_VegMax, self._topo, self._groundwater_elevation)  # Returns map of deposition probabilities

        # Move sand slabs
        if self._wind_direction[it] == 1 or self._wind_direction[it] == 3:  # Left or Right wind direction
            aeolian_elevation_change = routine.shiftslabs(aeolian_erosion_prob, aeolian_deposition_prob, self._saltation_length, self._saltation_length_rand_deviation, self._effective_veg, self._saltation_veg_limit, int(self._wind_direction[it]), True, self._topo, self._MHW, self._RNG)  # Returns map of height changes in units of slabs
        else:  # Up or Down wind direction
            aeolian_elevation_change = routine.shiftslabs(aeolian_erosion_prob, aeolian_deposition_prob, self._saltation_length, self._saltation_length_rand_deviation, self._effective_veg, self._saltation_veg_limit, int(self._wind_direction[it]), True, self._topo, self._MHW, self._RNG)  # Returns map of height changes in units of slabs

        # Apply changes, make calculations
        self._topo += aeolian_elevation_change * self._slabheight  # [m NAVD88] Changes applied to the topography; convert aeolian_elevation_change from slabs to meters
        self._sedimentation_balance = self._sedimentation_balance + (self._topo - topo_iteration_start)  # [m] Update the sedimentation balance map
        self._topographic_change = self._topographic_change + abs(self._topo - topo_iteration_start)  # [m]

        # --------------------------------------
        # STORMS

        if it % self._storm_update_frequency == 0:
            iteration_year = np.floor(it % self._iterations_per_cycle / 2).astype(int)  # Iteration of the year (e.g., if there's 50 iterations per year, this represents the week of the year)

            # Beach slopes
            foredune_crest_loc, not_gap = routine.foredune_crest(self._topo, self._MHW, self._cellsize)
            beach_slopes = routine.calculate_beach_slope(self._topo, self._x_s, foredune_crest_loc, self._average_dune_toe_height, self._MHW, self._cellsize)

            # Generate Storms Stats
            if self._hindcast:  # Empirical storm time series
                storm, Rhigh, Rlow, dur = routine.get_storm_timeseries(self._storm_timeseries, it, self._longshore, self._MHW, self._simulation_start_iteration)  # [m NAVD88]
            else:  # Stochastic storm model
                storm, Rhigh, Rlow, dur = routine.stochastic_storm(self._pstorm, iteration_year, self._StormList, beach_slopes, self._longshore, self._MHW, self._RNG_storm)  # [m NAVD88]

                # # Account for change in mean sea-level on synthetic storm elevations by adding aggregate RSLR since simulation start, and any linear storm climate shift in intensity
                # TWL_climate_shift = self._mean_stochastic_storm_TWL * ((self._shift_mean_storm_intensity / 100) * (it / self._aeolian_iterations_per_year))  # This version shifts TWL of all storms equally
                TWL_climate_shift = Rhigh * ((self._shift_mean_storm_intensity / 100) * (it / self._aeolian_iterations_per_year))  # This version shifts TWL more for bigger storms
                Rhigh += (self._MHW - self._MHW_init) + TWL_climate_shift  # [m NAVD88] Add change in sea level to storm water levels, which were in elevation relative to initial sea level, and shift intensity
                Rlow += (self._MHW - self._MHW_init) + TWL_climate_shift  # [m NAVD88] Add change in sea level to storm water levels, which were in elevation relative to initial sea level, and shift intensity

            if storm:

                # Storm Processes: Beach/duneface change, overwash
                self._StormRecord = np.vstack((self._StormRecord, [year, iteration_year, np.max(Rhigh), np.max(Rlow), dur]))
                self._topo, topo_change, self._OWflux, inundated, Qbe = routine.storm_processes(
                    topof=self._topo,
                    Rhigh=Rhigh,
                    dur=dur,
                    Rin=self._Rin,
                    Cs=self._Cs,
                    nn=self._nn,
                    MaxUpSlope=self._MaxUpSlope,
                    fluxLimit=self._marine_flux_limit,
                    Qs_min=self._overwash_min_discharge,
                    Kow=self._Kow,
                    mm=self._mm,
                    MHW=self._MHW,
                    Cbb=self._Cbb,
                    Qs_bb_min=self._overwash_min_subaqueous_discharge,
                    substep=self._overwash_substeps,
                    beach_equilibrium_slope=self._beach_equilibrium_slope,
                    swash_erosive_timescale=self._swash_erosive_timescale,
                    beach_substeps=self._beach_substeps,
                    x_s=self._x_s,
                    cellsize=self._cellsize,
                    spec1=self._spec1,
                    spec2=self._spec2,
                    flow_reduction_max_spec1=self._flow_reduction_max_spec1,
                    flow_reduction_max_spec2=self._flow_reduction_max_spec2,
                )

                # Update vegetation from storm effects
                inundated = np.round(scipy.ndimage.gaussian_filter(inundated.astype(float), 2 / self._cellsize, mode='reflect'))
                self._spec1[inundated > 0] = 0  # Remove species where beach is inundated
                self._spec2[inundated > 0] = 0  # Remove species where beach is inundated
                self._veg = self._spec1 + self._spec2  # Update

                # Aggregate inundation [boolean] for the period between the previous and next time step at which output is saved (save_frequency)
                self._inundated_output_aggregate += inundated

                # Check for nans in topo
                if np.isnan(np.sum(self._topo)):
                    self._topo = routine.replace_nans_infs(self._topo)

            else:
                self._OWflux = np.zeros([self._longshore])  # [m^3] No overwash if no storm
                topo_change = np.zeros(self._topo.shape)  # [m NAVD88]
                Qbe = np.zeros([self._longshore])  # [m^3] No overwash if no storm

            self._sedimentation_balance += topo_change  # [m]
            self._topographic_change += abs(topo_change)  # [m]

            # --------------------------------------
            # SHORELINE CHANGE

            # Update Shoreline Position from Cross-Shore Sediment Transport (i.e., RSLR, overwash, beach/dune change)
            self._x_s, self._x_t, shoreface_slope = routine.shoreline_change_from_CST(
                self._DShoreface,
                self._k_sf,
                self._s_sf_eq,
                self._RSLR,
                Qbe,  # [m^3] Volume of sediment imported from (+) or exported to (-) the upper shoreface by storm beach/duneface change
                self._OWflux,
                self._x_s,
                self._x_t,
                self._alongshore_section_length,
                self._storm_iterations_per_year,
                self._cellsize,
            )

            self._x_s = routine.shoreline_change_from_AST(self._x_s,
                                                          self._coast_diffusivity,
                                                          self._di,
                                                          self._dj,
                                                          self._alongshore_section_length,
                                                          self._storm_update_frequency / self._iterations_per_cycle,
                                                          self._ny,
                                                          )

            prev_shoreline = self._x_s_TS[-1]  # Shoreline positions from previous time step

            # Adjust topography domain to according to shoreline change
            topo_pre_shoreline_change = self._topo.copy()
            self._topo = routine.adjust_ocean_shoreline(
                self._topo,
                self._x_s,
                prev_shoreline,
                self._MHW,
                shoreface_slope,
                self._RSLR,
                self._storm_iterations_per_year,
            )

            # Enforce angles of repose
            """IR 25Apr24: Ideally, angles of repose would be enforced after avery aeolian iteration and every storm. However, to significantly increase model speed, I now enforce AOR only at the end of each
            shoreline iteration (i.e., every 2 aeolian iterations). The morphodynamic effects of this are apparently negligible, while run time is much quicker."""
            self._topo = routine.enforceslopes(self._topo, self._veg, self._slabheight, self._repose_bare, self._repose_veg, self._repose_threshold, self._MHW, self._cellsize, self._RNG)[0]  # [m NAVD88]

            # Update sedimentation balance after adjusting the shoreline and enforcing AOR
            self._sedimentation_balance += self._topo - topo_pre_shoreline_change  # [m] Update the sedimentation balance map
            self._topographic_change += abs(self._topo - topo_pre_shoreline_change)

            # Store shoreline and shoreface toe locations
            self._x_s_TS = np.vstack((self._x_s_TS, self._x_s))  # Store
            self._x_t_TS = np.vstack((self._x_t_TS, self._x_t))  # Store

            # Maintain equilibrium back-barrier depth
            self._topo = routine.maintain_equilibrium_backbarrier_depth(self._topo, self._eq_backbarrier_depth, self._MHW)

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
            lateral1 = routine.lateral_expansion(spec1_prev, 1, self._sp1_lateral_probability * veg_multiplier, self._RNG)
            lateral2 = routine.lateral_expansion(spec2_prev, 1, self._sp2_lateral_probability * veg_multiplier, self._RNG)
            lateral1[self._topo <= self._MHW] = False  # Constrain to subaerial
            lateral2[self._topo <= self._MHW] = False

            # Pioneer Establishment
            pioneer1 = routine.establish_new_vegetation(self._topo, self._MHW, self._sp1_pioneer_probability * veg_multiplier, self._RNG) * (spec1_prev <= 0)
            pioneer2 = routine.establish_new_vegetation(self._topo, self._MHW, self._sp2_pioneer_probability * veg_multiplier, self._RNG) * (spec2_prev <= 0) * (self._topographic_change == 0)
            pioneer1[self._topo <= self._MHW] = False  # Constrain to subaerial
            pioneer2[self._topo <= self._MHW] = False

            # Update Vegetation Cover
            spec1_diff = self._spec1 - spec1_prev  # Determine changes in vegetation cover
            spec2_diff = self._spec2 - spec2_prev  # Determine changes in vegetation cover
            spec1_growth = spec1_diff * (spec1_diff > 0)  # Split cover changes into into gain and loss
            spec1_loss = spec1_diff * (spec1_diff < 0)
            spec2_growth = spec2_diff * (spec2_diff > 0)  # Split cover changes into into gain and loss
            spec2_loss = spec2_diff * (spec2_diff < 0)

            spec1_change_allowed = np.minimum(1 - self._veg, spec1_growth) * np.logical_or(spec1_prev > 0, np.logical_or(lateral1, pioneer1))  # Only allow growth in adjacent or pioneer cells
            spec2_change_allowed = np.minimum(1 - self._veg, spec2_growth) * np.logical_or(spec2_prev > 0, np.logical_or(lateral2, pioneer2))  # Only allow growth in adjacent or pioneer cells
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
            self._effective_veg = scipy.ndimage.gaussian_filter(self._veg, [self._effective_veg_sigma / self._cellsize, self._effective_veg_sigma / self._cellsize], mode='constant')

            self._vegcount = self._vegcount + 1  # Update counter

        # --------------------------------------
        # RECORD VARIABLES PERIODICALLY

        if (it + 1) % (self._save_frequency * self._iterations_per_cycle) == 0:
            moment = int((it + 1) / self._save_frequency / self._iterations_per_cycle)
            self._topo_TS[:, :, moment] = self._topo
            self._spec1_TS[:, :, moment] = self._spec1
            self._spec2_TS[:, :, moment] = self._spec2
            self._veg_TS[:, :, moment] = self._veg
            self._storm_inundation_TS[:, :, moment] = self._inundated_output_aggregate
            self._inundated_output_aggregate *= False  # Reset for next output period

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
    def cellsize(self):
        return self._cellsize

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
    def storm_inundation_TS(self):
        return self._storm_inundation_TS

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
    def MHW_init(self):
        return self._MHW_init

    @property
    def StormRecord(self):
        return self._StormRecord

    @property
    def x_s_TS(self):
        return self._x_s_TS

    @property
    def storm_iterations_per_year(self):
        return self._storm_iterations_per_year
