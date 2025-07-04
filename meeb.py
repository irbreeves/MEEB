"""__________________________________________________________________________________________________________________________________

Main Model Script for MEEB

Mesoscale Explicit Ecogeomorphic Barrier model

IRB Reeves

Last update: 3 July 2025

__________________________________________________________________________________________________________________________________"""

import numpy as np
from math import ceil, floor, sqrt, pi
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
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
            save_frequency=0.5,  # [years] Save results every n years
            simulation_time_yr=15.0,  # [yr] Length of the simulation time
            cellsize=2,  # [m] Cell length and width
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
            saltation_length=2,  # [cells] Hop length for saltating slabs of sand (5 m, Teixeira et al. 2023); note units of cells (e.g., if cellsize = 2 m and saltation_length = 5 cells, slabs will hop 10 m)
            saltation_length_rand_deviation=1,  # [cells] Deviation around saltation_length for random uniform distribution of saltation lengths. Must be at lest 1 cell smaller than saltation_length.
            groundwater_depth=0.4,  # Proportion of the smoothed topography used to set groundwater profile
            wind_rose=(0.91, 0.04, 0.01, 0.04),  # Proportion of wind TOWARDS (right, down, left, up)
            p_dep_sand=0.09,  # [0-1] Probability of deposition in sandy cells with 0% vegetation cover
            p_dep_sand_VegMax=0.17,  # [0-1] Probability of deposition in sandy cells with 100% vegetation cover; must be greater than or equal to p_dep_sand/p_dep_basesaltation_length_rand_deviation
            p_dep_base=0.1,  # [0-1] Probability of deposition of base cells
            p_ero_sand=0.08,  # [0-1] Probability of erosion of bare/sandy cells
            entrainment_veg_limit=0.09,  # [0-1] Percent of vegetation cover beyond which aeolian sediment entrainment is no longer possible
            saltation_veg_limit=0.37,  # Threshold vegetation effectiveness needed for a cell along a slab saltation path to be considered vegetated
            shadowangle=12,  # [deg]
            repose_bare=20,  # [deg] Angle of repose for unvegetated cells
            repose_veg=30,  # [deg] Angle of repose for vegetation cells
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
            shoreline_diffusivity_coefficient=0.07,  # [m^(3/5) s^(-6/5)] Alongshore transport diffusion coefficient
            average_dune_toe_height=1.67,  # [m] Time- and space-averaged dune toe height above MHW
            estimate_shoreface_parameters=True,  # [bool] Turn on to estimate shoreface parameters as function of specific wave and sediment characteristics
            shoreface_grain_size=2e-4,  # [m] Median grain size (D50) of ocean shoreface; used for optional shoreface parameter estimations
            shoreface_transport_efficiency=0.01,  # Shoreface suspended sediment transport efficiency factor; used for optional shoreface parameter estimations
            shoreface_friction=0.01,  # Shoreface friction factor; used for optional shoreface parameter estimations
            specific_gravity_submerged_sed=1.65,  # Submerged specific gravity of sediment; used for optional shoreface parameter estimations

            # STORM OVERWASH AND BEACH-DUNE CHANGE
            storm_list_filename="SyntheticStorms_NCB-CE_10k_1979-2020_Beta0pt039_BermEl1pt78.npy",
            storm_timeseries_filename="StormTimeSeries_1979-2020_NCB-CE_Beta0pt039_BermEl1pt78.npy",  # Only needed if running hindcast simulations (i.e., without stochastic storms)
            Rin=245,  # [m^3/hr] Flow infiltration and drag parameter, run-up overwash regime
            Cs=0.0235,  # Constant for representing flow momentum for sediment transport in overwash
            nn=0.5,  # Flow routing constant
            MaxUpSlope=1.5,  # Maximum slope water can flow uphill
            marine_flux_limit=1,  # [m/hr] Maximum elevation change allowed per time step (prevents instabilities)
            overwash_min_discharge=1.0,  # [m^3/hr] Minimum discharge out of cell needed to transport sediment
            Kow=0.0003615,  # Sediment transport coefficient for run-up overwash regime
            mm=1.05,  # Inundation overwash constant
            Cbb=0.7,  # [0-1] Coefficient for exponential decay of sediment load entering back-barrier bay, run-up regime
            overwash_min_subaqueous_discharge=1,  # [m^3/hr] Minimum discharge out of subaqueous back-barrier cell needed to transport sediment
            overwash_substeps=25,  # Number of substeps to run for each hour in run-up overwash regime (e.g., 3 substeps means discharge/elevation updated every 20 minutes)
            beach_equilibrium_slope=0.021,  # Equilibrium slope of the beach
            swash_erosive_timescale=1.51,  # Non-dimensional erosive timescale coefficient for beach/duneface sediment transport (Duran Vinent & Moore, 2015)
            beach_substeps=1,  # Number of substeps per iteration of beach/duneface model; instabilities will occur if too low
            shift_mean_storm_intensity_end=0,  # [%/yr] Linear yearly percent shift in mean storm TWL (as proxy for intensity) in stochastic storm model; use 0 for no shift
            shift_mean_storm_intensity_start=0,  # [%] Percent change in storm intensity at start of simulation
            storm_twl_duration_correlation=0,  # Correlation factor (slope of linear regression) between observed/modeled storm total water levels and storm durations

            # -------------------------------
            # VEGETATION

            # Species: 2 Herbaceous, 1 Woody
            # States: Bare, H1_seed, H1_adult, H2_seed, H2_adult, W_seed, W_adult, W_dead

            # Dispersal and Flow
            sp1_lateral_probability=0.008,  # [0-1] Probability of lateral expansion of existing vegetation
            sp2_lateral_probability=0.008,  # [0-1] Probability of lateral expansion of existing vegetation
            sp1_pioneer_probability=0.002,  # [0-1] Probability of occurrence of new pioneering vegetation
            sp2_pioneer_probability=0.0012,  # [0-1] Probability of occurrence of new pioneering vegetation
            flow_reduction_max_spec1=0.002,  # [0-1] Proportion of overwash flow reduction through a cell populated with species 1 at full density
            flow_reduction_max_spec2=0.02,  # [0-1] Proportion of overwash flow reduction through a cell populated with species 2 at full density
            effective_veg_sigma=3,  # Standard deviation for Gaussian filter of vegetation cover

            # Transition Probabilities
            H1_germ_Pmax_tempC=0.30,  # Herbaceous germination at optimal temperature, species 1
            H2_germ_Pmax_tempC=0.30,  # Herbaceous germination at optimal temperature, species 2
            W_germ_Pmax_tempC=0.14,  # Woody germination at optimal temperature
            W_germ_Pmin_herbaceous_facil=0.8,  # Woody germination at least optimal (i.e., zero) herbaceous cover
            H1_s_mort_Pmax_tempC=0.6,  # Herbaceous seedling mortality at optimal temperature, species 1
            H2_s_mort_Pmax_tempC=0.6,  # Herbaceous seedling mortality at optimal temperature, species 2
            W_s_mort_Pmax_tempC=0.4,  # Woody seedling mortality at optimal temperature
            H1_growth_Pmax_tempC=0.95,  # Herbaceous seedling to adult at optimal temperature, species 1
            H1_growth_Pmax_elev=0.8,  # Herbaceous seedling to adult at optimal elevation, species 1
            H1_growth_Pmin_stim=0.7,  # Herbaceous seedling to adult with no deposition stimulation, species 1
            H2_growth_Pmax_tempC=0.95,  # Herbaceous seedling to adult at optimal temperature, species 2
            H2_growth_Pmax_elev=0.8,  # Herbaceous seedling to adult at optimal elevation, species 2
            H2_growth_Pmin_stim=0.7,  # Herbaceous seedling to adult with no deposition stimulation, species 2
            W_growth_Pmax_tempC=0.7,  # Woody seedling to adult at optimal temperature
            W_growth_Pmax_elev=0.6,  # Woody seedling to adult at optimal elevation
            W_growth_Pmin_stim=0.9,  # Woody seedling to adult with no deposition stimulation
            H1_a_senesce_Pmin_tempC=0.02,  # Herbaceous adult senescence to dead at optimal temperature for survival, species 1
            H1_a_senesce_Pmax_tempC=0.06,  # Herbaceous adult senescence to dead at least optimal temperatures for survival, species 1
            H2_a_senesce_Pmin_tempC=0.02,  # Herbaceous adult senescence to dead at optimal temperature for survival, species 2
            H2_a_senesce_Pmax_tempC=0.06,  # Herbaceous adult senescence to dead at least optimal temperatures for survival, species 2
            W_a_senesce_Pmin_tempC=0.004,  # Woody adult senescence at optimal temperature for survival
            W_a_senesce_Pmax_tempC=0.006,  # Woody adult senescence at least optimal temperatures for survival
            W_d_loss_Pmin=0.1,  # Woody dead loss (breakdown) minimum
            W_d_loss_Pmax_submerged_frozen=0.15,  # Woody dead loss (breakdown) to bare when submerged (below MHW) or frozen (temp < 0)
            W_d_loss_Pmax_discharge=0.6,  # Woody dead loss (breakdown) to bare at optimal HWE discharge
            W_d_loss_Pmax_twl=0.5,  # Woody dead loss (breakdown) to bare at optimum HWE TWL (as proxy for wind strength)

            # HWE Discharge Thresholds
            H1_QHWE_min=40,  # [m^3] Minimum HWE discharge below which mortality probability is 0%, herbaceous species 1
            H1_QHWE_max=80,  # [m^3] Maximum HWE discharge above which mortality probability is 100%, herbaceous species 1
            H2_QHWE_min=40,  # [m^3] Minimum HWE discharge below which mortality probability is 0%, herbaceous species 2
            H2_QHWE_max=80,  # [m^3] Maximum HWE discharge above which mortality probability is 100%, herbaceous species 2
            W_QHWE_min=40,  # [m^3] Minimum HWE discharge below which mortality probability is 0%, woody species
            W_QHWE_max=60,  # [m^3] Maximum HWE discharge above which mortality probability is 100%, woody species

            # HWE TWL Thresholds
            W_TWL_min=1,  # [m MHW] Minimum HWE total water level below which woody loss to bare is 0%
            W_TWL_max=3,  # [m MHW] Maximum HWE total water level below which woody loss to bare is 100%

            # Erosion/Deposition Thresholds
            germination_erosion_limit=0.06,  # [m] Maximum depth of erosion beyond which germination probability is 0%, all species
            germination_burial_limit=0.06,  # [m] Maximum depth of burial beyond which germination probability is 0%, all species
            seedling_erosion_limit=-0.1,  # [m] Maximum depth of erosion beyond which seedling mortality probability is 100%, all species
            seedling_burial_limit=0.1,  # [m] Maximum depth of deposition beyond which seedling mortality probability is 100%, all species
            H1_uproot_limit=-0.4,  # [m] Maximum depth of erosion beyond which mortality probability is 100%, herbaceous species 1
            H1_burial_limit=0.8,  # [m] Maximum depth of deposition beyond which mortality probability is 100%, herbaceous species 1
            H2_uproot_limit=-0.4,  # [m] Maximum depth of erosion beyond which mortality probability is 100%, herbaceous species 2
            H2_burial_limit=0.8,  # [m] Maximum depth of deposition beyond which mortality probability is 100%, herbaceous species 2
            W_uproot_limit=-0.3,  # [m] Maximum depth of erosion beyond which mortality probability is 100%, woody species
            W_burial_limit=2.0,  # [m] Maximum depth of deposition beyond which mortality probability is 100%, woody species

            # Elevation Thresholds & Parameters
            H1_elev_gamma_a=1.63,  # Alpha of gamma probability density function for elevation, herbaceous species 1
            H1_elev_gamma_loc=0.45,  # [m MHW] Location parameter (shift) of gamma probability density function for elevation, herbaceous species 1
            H1_elev_gamma_scale=3,  # Scale parameter of gamma probability density function for elevation, herbaceous species 1
            H2_elev_gamma_a=1.63,  # Alpha of gamma probability density function for elevation, herbaceous species 2
            H2_elev_gamma_loc=0.25,  # [m MHW] Location parameter (shift) of gamma probability density function for elevation, herbaceous species 2
            H2_elev_gamma_scale=3,  # Scale parameter of gamma probability density function for elevation, herbaceous species 2
            W_elev_gamma_a=3.979586,  # Alpha of gamma probability density function for elevation, woody species
            W_elev_gamma_loc=0.5,  # [m MHW] Location parameter (shift) of gamma probability density function for elevation, woody species
            W_elev_gamma_scale=0.182173,  # Scale parameter of gamma probability density function for elevation, woody species

            # Temperature Thresholds and Parameters
            H1_germ_tempC_min=24,  # [C] Minimum temperature for germination, herbaceous species 1
            H1_germ_tempC_max=40,  # [C] Maximum temperature for germination, herbaceous species 1
            H2_germ_tempC_min=26,  # [C] Minimum temperature for germination, herbaceous species 2
            H2_germ_tempC_max=46,  # [C] Maximum temperature for germination, herbaceous species 2
            W_germ_tempC_min=20,  # [C] Minimum temperature for germination, woody species
            W_germ_tempC_max=48,  # [C] Maximum temperature for germination, woody species
            H1_growth_tempC_min=16,  # [C] Minimum temperature for growth, herbaceous species 1
            H1_growth_tempC_max=32,  # [C] Maximum temperature for growth, herbaceous species 1
            H2_growth_tempC_min=18,  # [C] Minimum temperature for growth, herbaceous species 2
            H2_growth_tempC_max=38,  # [C] Maximum temperature for growth, herbaceous species 2
            W_growth_tempC_min=10,  # [C] Minimum temperature for growth, woody species
            W_growth_tempC_max=48,  # [C] Maximum temperature for growth, woody species
            H1_mort_tempC_min=-10,  # [C] Threshold temperature below which mortality is 100%, herbaceous species 1
            H1_mort_tempC_max=50,  # [C] Threshold temperature above which mortality is 100%, herbaceous species 1
            H2_mort_tempC_min=-10,  # [C] Threshold temperature below which mortality is 100%, herbaceous species 2
            H2_mort_tempC_max=50,  # [C] Threshold temperature above which mortality is 100%, herbaceous species 2
            W_s_mort_tempC_min=-8,  # [C] Threshold temperature below which seedling mortality is 100%, woody species
            W_s_mort_tempC_max=50,  # [C] Threshold temperature above which seedling mortality is 100%, woody species
            W_a_mort_tempC_min=-15,  # [C] Threshold temperature below which adult mortality is 100%, woody species
            W_a_mort_tempC_max=50,  # [C] Threshold temperature above which adult mortality is 100%, woody species
            standard_dev_temperature=7.5,  # [C] Standard deviation (spread) of temperature anomolies around the mean

            # Growth Stimulation From Deposition Thresholds
            H1_stim_min=0,  # [m] Minimum deposition for stimulation from deposition, herbaceous species 1
            H1_stim_max=0.4,  # [m] Maximum deposition for stimulation from deposition, herbaceous species 1
            H2_stim_min=0,  # [m] Minimum deposition for stimulation from deposition, herbaceous species 2
            H2_stim_max=0.4,  # [m] Maximum deposition for stimulation from deposition, herbaceous species 2
            W_stim_min=0,  # [m] Minimum deposition for stimulation from deposition, woody species
            W_stim_max=0.1,  # [m] Maximum deposition for stimulation from deposition, woody species

            # Woody Fronting Dune Elevation and Shoreline Distance Thresholds
            W_dune_elev_min=1.85,  # [m MHW] Frontingf dune elevation below which woody establishment (germination and growth) is 0%
            W_dune_elev_max=2.25,  # [m MHW] Fronting dune elevation above which woody establishment (germination and growth) is 100%
            W_shoreline_distance_min=170,  # [m MHW] Distance from ocean shoreline below which woody establishment (germination and growth) is 0% in absence of sufficiently tall dune
            W_shoreline_distance_max=200,  # [m MHW] Distance from ocean shoreline above which woody establishment (germination and growth) is 100% in absence of sufficiently tall dune

            # Competition/Facilitation Thresholds and Parameters
            H1_growth_woody_comp_max=0.5,  # Maximum woody fractional cover beyond which germination/growth of herbaceous species 1 is 0%
            H2_growth_woody_comp_max=0.5,  # Maximum woody fractional cover beyond which germination/growth of herbaceous species 2 is 0%
            W_germ_herbaceous_facil_max=0.5,  # Maximum herbaceous fractional cover beyond which woody germination is maximized

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
        self._shoreline_diffusivity_coefficient = shoreline_diffusivity_coefficient
        self._average_dune_toe_height = average_dune_toe_height
        self._eq_backbarrier_depth = eq_backbarrier_depth
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
        self._shift_mean_storm_intensity_end = shift_mean_storm_intensity_end / 100
        self._shift_mean_storm_intensity_start = shift_mean_storm_intensity_start / 100
        self._storm_twl_duration_correlation = storm_twl_duration_correlation
        self._sp1_lateral_probability = sp1_lateral_probability
        self._sp2_lateral_probability = sp2_lateral_probability
        self._sp1_pioneer_probability = sp1_pioneer_probability
        self._sp2_pioneer_probability = sp2_pioneer_probability
        self._flow_reduction_max_spec1 = flow_reduction_max_spec1
        self._flow_reduction_max_spec2 = flow_reduction_max_spec2
        self._effective_veg_sigma = effective_veg_sigma
        # ---------------
        self._H1_germ_Pmax_tempC = H1_germ_Pmax_tempC
        self._H2_germ_Pmax_tempC = H2_germ_Pmax_tempC
        self._W_germ_Pmax_tempC = W_germ_Pmax_tempC
        self._W_germ_Pmin_herbaceous_facil = W_germ_Pmin_herbaceous_facil
        self._H1_s_mort_Pmax_tempC = H1_s_mort_Pmax_tempC
        self._H2_s_mort_Pmax_tempC = H2_s_mort_Pmax_tempC
        self._W_s_mort_Pmax_tempC = W_s_mort_Pmax_tempC
        self._H1_growth_Pmax_tempC = H1_growth_Pmax_tempC
        self._H1_growth_Pmax_elev = H1_growth_Pmax_elev
        self._H1_growth_Pmin_stim = H1_growth_Pmin_stim
        self._H2_growth_Pmax_tempC = H2_growth_Pmax_tempC
        self._H2_growth_Pmax_elev = H2_growth_Pmax_elev
        self._H2_growth_Pmin_stim = H2_growth_Pmin_stim
        self._W_growth_Pmax_tempC = W_growth_Pmax_tempC
        self._W_growth_Pmax_elev = W_growth_Pmax_elev
        self._W_growth_Pmin_stim = W_growth_Pmin_stim
        self._H1_a_senesce_Pmin_tempC = H1_a_senesce_Pmin_tempC
        self._H1_a_senesce_Pmax_tempC = H1_a_senesce_Pmax_tempC
        self._H2_a_senesce_Pmin_tempC = H2_a_senesce_Pmin_tempC
        self._H2_a_senesce_Pmax_tempC = H2_a_senesce_Pmax_tempC
        self._W_a_senesce_Pmin_tempC = W_a_senesce_Pmin_tempC
        self._W_a_senesce_Pmax_tempC = W_a_senesce_Pmax_tempC
        self._W_d_loss_Pmin = W_d_loss_Pmin
        self._W_d_loss_Pmax_submerged_frozen = W_d_loss_Pmax_submerged_frozen
        self._W_d_loss_Pmax_discharge = W_d_loss_Pmax_discharge
        self._W_d_loss_Pmax_twl = W_d_loss_Pmax_twl
        self._H1_QHWE_min = H1_QHWE_min
        self._H1_QHWE_max = H1_QHWE_max
        self._H2_QHWE_min = H2_QHWE_min
        self._H2_QHWE_max = H2_QHWE_max
        self._W_QHWE_min = W_QHWE_min
        self._W_QHWE_max = W_QHWE_max
        self._W_TWL_min = W_TWL_min
        self._W_TWL_max = W_TWL_max
        self._germination_erosion_limit = germination_erosion_limit
        self._germination_burial_limit = germination_burial_limit
        self._seedling_erosion_limit = seedling_erosion_limit
        self._seedling_burial_limit = seedling_burial_limit
        self._H1_uproot_limit = H1_uproot_limit
        self._H1_burial_limit = H1_burial_limit
        self._H2_uproot_limit = H2_uproot_limit
        self._H2_burial_limit = H2_burial_limit
        self._W_uproot_limit = W_uproot_limit
        self._W_burial_limit = W_burial_limit
        self._H1_elev_gamma_a = H1_elev_gamma_a
        self._H1_elev_gamma_loc = H1_elev_gamma_loc
        self._H1_elev_gamma_scale = H1_elev_gamma_scale
        self._H2_elev_gamma_a = H2_elev_gamma_a
        self._H2_elev_gamma_loc = H2_elev_gamma_loc
        self._H2_elev_gamma_scale = H2_elev_gamma_scale
        self._W_elev_gamma_a = W_elev_gamma_a
        self._W_elev_gamma_loc = W_elev_gamma_loc
        self._W_elev_gamma_scale = W_elev_gamma_scale
        self._H1_germ_tempC_min = H1_germ_tempC_min
        self._H1_germ_tempC_max = H1_germ_tempC_max
        self._H2_germ_tempC_min = H2_germ_tempC_min
        self._H2_germ_tempC_max = H2_germ_tempC_max
        self._W_germ_tempC_min = W_germ_tempC_min
        self._W_germ_tempC_max = W_germ_tempC_max
        self._H1_growth_tempC_min = H1_growth_tempC_min
        self._H1_growth_tempC_max = H1_growth_tempC_max
        self._H2_growth_tempC_min = H2_growth_tempC_min
        self._H2_growth_tempC_max = H2_growth_tempC_max
        self._W_growth_tempC_min = W_growth_tempC_min
        self._W_growth_tempC_max = W_growth_tempC_max
        self._H1_mort_tempC_min = H1_mort_tempC_min
        self._H1_mort_tempC_max = H1_mort_tempC_max
        self._H2_mort_tempC_min = H2_mort_tempC_min
        self._H2_mort_tempC_max = H2_mort_tempC_max
        self._W_s_mort_tempC_min = W_s_mort_tempC_min
        self._W_s_mort_tempC_max = W_s_mort_tempC_max
        self._W_a_mort_tempC_min = W_a_mort_tempC_min
        self._W_a_mort_tempC_max = W_a_mort_tempC_max
        self._standard_dev_temperature = standard_dev_temperature
        self._H1_stim_min = H1_stim_min
        self._H1_stim_max = H1_stim_max
        self._H2_stim_min = H2_stim_min
        self._H2_stim_max = H2_stim_max
        self._W_stim_min = W_stim_min
        self._W_stim_max = W_stim_max
        self._W_dune_elev_min = W_dune_elev_min
        self._W_dune_elev_max = W_dune_elev_max
        self._W_shoreline_distance_min = W_shoreline_distance_min
        self._W_shoreline_distance_max = W_shoreline_distance_max
        self._H1_growth_woody_comp_max = H1_growth_woody_comp_max
        self._H2_growth_woody_comp_max = H2_growth_woody_comp_max
        self._W_germ_herbaceous_facil_max = W_germ_herbaceous_facil_max

        # __________________________________________________________________________________________________________________________________
        # SET INITIAL CONDITIONS

        # SEEDED RANDOM NUMBER GENERATOR
        if seeded_random_numbers:
            self._RNG = np.random.Generator(np.random.SFC64(seed=13))  # Seeded random numbers for reproducibility (e.g., model development/testing)
            self._RNG_storm = np.random.Generator(np.random.SFC64(seed=14))  # Separate seeded RNG for storms so that the storm sequence can always stay the same despite any parameterization changes
        else:
            self._RNG = np.random.Generator(np.random.SFC64())  # Non-seeded random numbers (e.g., model simulations)
            self._RNG_storm = np.random.Generator(np.random.SFC64())

        # TIME
        self._iterations_per_cycle = self._aeolian_iterations_per_year  # [iterations/year] Number of iterations in 1 model year
        self._storm_update_frequency = round(self.iterations_per_cycle / self._storm_iterations_per_year)  # Frequency of storm updates (i.e., every n iterations)
        self._iterations = int(self._iterations_per_cycle * self._simulation_time_yr)  # Total number of iterations
        self._simulation_start_date = datetime.strptime(self._simulation_start_date, '%Y%m%d').date()  # Convert to datetime
        if hindcast:
            self._hindcast_timeseries_start_date = datetime.strptime(self._hindcast_timseries_start_date, '%Y%m%d').date()  # Convert to datetime
            self._simulation_start_iteration = (((self._simulation_start_date.year - self._hindcast_timeseries_start_date.year) * self._iterations_per_cycle) +
                                                floor(self._simulation_start_date.timetuple().tm_yday / 365 * self._iterations_per_cycle))  # Iteration, realtive to timeseries start, from which to begin hindcast
            if self._simulation_start_iteration % 2 != 0:
                self._simulation_start_iteration -= 1  # Round simulation start iteration to even number
        else:
            self._simulation_start_iteration = floor(self._simulation_start_date.timetuple().tm_yday / 365 * self._iterations_per_cycle)
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
                self._topo = init_elev_array.copy().astype(np.float32)  # [m NAVD88] 2D array of initial topography
                self._spec1 = init_spec1_array.copy().astype(np.float32)  # [0-1] 2D array of vegetation effectiveness for spec1
                self._spec2 = init_spec2_array.copy().astype(np.float32)  # [0-1] 2D array of vegetation effectiveness for spec2
        self._longshore, self._crossshore = self._topo.shape  # [cells] Cross-shore/alongshore size of domain
        self._groundwater_elevation = np.zeros(self._topo.shape, dtype=np.float32)  # [m NAVD88] Initialize

        # SHOREFACE & SHORELINE
        if estimate_shoreface_parameters:  # Option to estimate shoreface parameter values from wave and sediment charactersitics; Follows Nienhuis & Lorenzo-Trueba (2019)
            w_s = (specific_gravity_submerged_sed * 9.81 * shoreface_grain_size ** 2) / ((18 * 1e-6) + np.sqrt(0.75 * specific_gravity_submerged_sed * 9.81 * (shoreface_grain_size ** 3)))  # [m/s] Settling velocity (Church & Ferguson, 2004)
            z0 = 2 * self._mean_wave_height / 0.78  # [m] Minimum depth of integration (simple approximation of breaking wave depth based on offshore wave height)
            self._DShoreface = 0.018 * self._mean_wave_height * self._mean_wave_period * sqrt(9.81 / (specific_gravity_submerged_sed * shoreface_grain_size))  # [m] Shoreface depth
            self._s_sf_eq = (3 * w_s / 4 / np.sqrt(self._DShoreface * 9.81) * (5 + 3 * self._mean_wave_period ** 2 * 9.81 / 4 / (np.pi ** 2) / self._DShoreface))  # Equilibrium shoreface slope
            self._k_sf = ((3600 * 24 * 365) * ((shoreface_transport_efficiency * shoreface_friction * 9.81 ** (11 / 4) * self._mean_wave_height ** 5 * self._mean_wave_period ** (5 / 2)) / (960 * specific_gravity_submerged_sed * pi ** (7 / 2) * w_s ** 2)) *
                          (((1 / (11 / 4 * z0 ** (11 / 4))) - (1 / (11 / 4 * self._DShoreface ** (11 / 4)))) / (self._DShoreface - z0)))  # [m^3/m/yr] Shoreface response rate
            self._LShoreface = int(self._DShoreface / self._s_sf_eq)  # [m] Initialize length of shoreface such that initial shoreface slope equals equilibrium shoreface slope
        self._alongshore_section_length = int(self._alongshore_section_length / self._cellsize)  # [cells]
        self._x_s = routine.init_ocean_shoreline(self._topo, self._MHW, self._alongshore_section_length).astype(np.float32)  # [m] Start locations of shoreline according to initial topography and MHW
        self._x_t = self._x_s - self._LShoreface  # [m] Start locations of shoreface toe
        self._coast_diffusivity, self._di, self._dj, self._ny = routine.init_AST_environment(self._wave_asymmetry,
                                                                                             self._wave_high_angle_fraction,
                                                                                             self._mean_wave_height,
                                                                                             self._mean_wave_period,
                                                                                             self._DShoreface,
                                                                                             self._alongshore_section_length,
                                                                                             self._longshore,
                                                                                             self._shoreline_diffusivity_coefficient)
        self._coast_diffusivity = self._coast_diffusivity.astype(np.float32)

        # STORMS
        self._StormList = np.float32(np.load(inputloc + storm_list_filename))
        self._storm_timeseries = np.float32(np.load(inputloc + storm_timeseries_filename))
        self._pstorm = [0.333, 0.333, 0.167, 0.310, 0.381, 0.310, 0.310, 0.310, 0.286, 0, 0.119, 0.024, 0.048, 0.048, 0.048, 0.071, 0.333, 0.286, 0.214,
                        0.190, 0.190, 0.262, 0.214, 0.262, 0.238]  # Empirical probability of storm occurance for each 1/25th (~biweekly) iteration of the year, from 1979-2021 NCB storm record (1.78 m NAVD88 Berm Elev.)
        # self._pstorm = [0.738, 0.667, 0.571, 0.786, 0.833, 0.643, 0.643, 0.762, 0.476, 0.167, 0.238, 0.095, 0.214, 0.167, 0.119, 0.119, 0.357, 0.476, 0.357,
        #                 0.405, 0.524, 0.524, 0.738, 0.548, 0.619]  # Empirical probability of storm occurance for each 1/25th (~biweekly) iteration of the year, from 1979-2021 NCB storm record (1.56 m NAVD88 Berm Elev.)

        if self._hindcast and self._iterations > (self._storm_timeseries[-1, 0] - self._simulation_start_iteration):
            raise ValueError("Simulation length is greater than hindcast timeSeries length.")

        # VEGETATION
        # self._temperatureC_average_daily_max = [8.3, 7.8, 7.8, 7.8, 8.16, 8.68, 9.33, 10.09, 10.85, 11.76, 12.7, 13.94, 15.41, 16.74, 18.26, 19.59, 20.66, 21.6, 22.5, 23.56,
        #                                         24.59, 25.93, 27.23, 28.4, 29.26, 30, 30, 30, 30, 29.66, 29.4, 28.9, 28.39, 27.79, 26.95, 25.89, 24.66, 23.49, 22.16, 20.64,
        #                                         19.27, 17.84, 16.51, 15.17, 13.87, 12.7, 11.6, 10.66, 9.7, 8.89]  # [C] Average observed daily maximum high for 1/50-yr increments, Wallops Island area, 1991-2020

        self._temperatureC_average_daily_max = [8.05, 7.8, 8.42, 9.71, 11.305, 13.32, 16.075, 18.925, 21.13, 23.03, 25.26, 27.815, 29.63, 30, 29.83, 29.15, 28.09, 26.42, 24.075,
                                                21.4, 18.555, 15.84, 13.285, 11.13, 9.295]  # [C] Average observed daily maximum high for 1/25-yr increments, Wallops Island area, 1991-2020

        init_veg = self._spec1 + self._spec2  # Determine the initial cumulative vegetation effectiveness

        self._veg_fraction = np.zeros([self._longshore, self._crossshore, 8], dtype=np.float32)  # Vector of initial states [Bare, H1_seed, H1_adult, H2_seed, H2_adult, W_seed, W_adult, W_dead]
        self._veg_fraction[:, :, 0] = 1 - init_veg  # Set initial Bare
        self._veg_fraction[:, :, 1] = (self._spec1 / 2) * 0.15  # Set initial H1 Seedling
        self._veg_fraction[:, :, 2] = (self._spec1 / 2) * 0.85  # Set initial H1 Adult
        self._veg_fraction[:, :, 3] = (self._spec1 / 2) * 0.15  # Set initial H2 Seedling
        self._veg_fraction[:, :, 4] = (self._spec1 / 2) * 0.85  # Set initial H2 Adult
        self._veg_fraction[:, :, 5] = self._spec2 * 0.15  # Set initial W Seedling
        self._veg_fraction[:, :, 6] = self._spec2 * 0.80  # Set initial W Adult
        self._veg_fraction[:, :, 7] = self._spec2 * 0.05  # Set initial W Dead

        self._effective_veg = gaussian_filter(init_veg, [self._effective_veg_sigma / self._cellsize, self._effective_veg_sigma / self._cellsize], mode='constant')  # Effective vegetation cover represents effect of nearby vegetation on local wind

        # MODEL PARAMETERS
        self._MHW_init = self._MHW
        self._wind_direction = np.zeros([self._iterations], dtype=np.int32)
        self._slabheight = round(self._slabheight, 2)  # Round slabheight to 2 decimals
        self._sedimentation_balance_year = np.zeros(self._topo.shape, dtype=np.float32)  # [m] Initialize map of the sedimentation balance: difference between erosion and deposition for 1 model year; (+) = net deposition, (-) = net erosion
        self._sedimentation_balance_biweekly = np.zeros(self._topo.shape, dtype=np.float32)  # [m] Initialize map of the sedimentation balance: difference between erosion and deposition for 1/25 model year; (+) = net deposition, (-) = net erosion
        self._x_s_TS = [self._x_s]  # Initialize storage array for shoreline position
        self._x_t_TS = [self._x_t]  # Initialize storage array for shoreface toe position
        self._x_bb_TS = [routine.backbarrier_shoreline_nonjitted(self._topo, self._MHW)]  # Initialize storage array for back-barrier shoreline position
        self._shoreline_change_aggregate = np.zeros([self._longshore], dtype=np.float32)
        self._OWflux = np.zeros([self._longshore], dtype=np.float32)  # [m^3]
        self._StormRecord = np.zeros([5], dtype=np.float32)  # Record of each storm that occurs in model: Year, iteration, Rhigh, Rlow, duration

        # __________________________________________________________________________________________________________________________________
        # MODEL OUPUT CONFIGURATION

        self._topo_TS = np.empty([self._longshore, self._crossshore, int(np.floor(self._simulation_time_yr / self._save_frequency)) + 1], dtype=np.float16)  # Array for saving each topo map at specified frequency
        self._topo_TS[:, :, 0] = self._topo.astype(np.float16)
        self._storm_inundation_TS = np.zeros([self._longshore, self._crossshore, int(np.floor(self._simulation_time_yr / self._save_frequency)) + 1], dtype=np.float16)  # Array for saving each veg map at specified frequency
        self._inundated_output_aggregate = np.zeros([self._longshore, self._crossshore], dtype=np.float16)
        self._MHW_TS = np.zeros([int(np.floor(self._simulation_time_yr / self._save_frequency)) + 1])  # Array for saving each MHW at specified frequency
        self._veg_fraction_TS = np.zeros([self._longshore, self._crossshore, self._veg_fraction.shape[2], int(np.floor(self._simulation_time_yr / self._save_frequency)) + 1], dtype=np.float16)  # Array for storing fraction of veg carrying capacity over time
        self._veg_fraction_TS[:, :, :, 0] = self._veg_fraction

        if init_by_file:
            del Init
            gc.collect()
        else:
            del init_elev_array, init_spec1_array, init_spec2_array
            gc.collect()

    # __________________________________________________________________________________________________________________________________
    # MAIN ITERATION LOOP

    def update(self, it):
        """Update MEEB by a single time step"""

        year = ceil(it / self._iterations_per_cycle)
        iteration_year = np.floor((it + self._simulation_start_iteration) % self._iterations_per_cycle / 2).astype(int)  # Storm iteration of the year (i.e., time of the year)

        # Update sea level for this iteration
        self._MHW += self._RSLR / self._iterations_per_cycle  # [m NAVD88]

        # --------------------------------------
        # AEOLIAN
        self._wind_direction[it] = self._RNG.choice(np.arange(1, 5), p=self._wind_rose).astype(int)  # Randomly select and record wind direction for this iteration
        topo_copy_pre = self._topo.copy()

        # Get present groundwater elevations
        self._groundwater_elevation = (gaussian_filter(self._topo, sigma=12 / self._cellsize) - self._MHW) * self._groundwater_depth + self._MHW  # [m NAVD88] Groundwater elevation based on smoothed topographic height above SL
        self._groundwater_elevation[self._groundwater_elevation < self._MHW] = self._MHW  # [m NAVD88]

        # Find subaerial and shadow cells
        subaerial = self._topo > self._MHW  # [bool] True for subaerial cells
        wind_shadows = routine.shadowzones(self._topo, self._shadowangle, direction=int(self._wind_direction[it]), MHW=self._MHW, cellsize=self._cellsize)  # [bool] Map of True for in shadow, False not in shadow

        # Erosion/Deposition Probabilities
        aeolian_erosion_prob = routine.erosprobs(self._effective_veg, wind_shadows, subaerial, self._topo, self._groundwater_elevation, self._p_ero_sand, self._entrainment_veg_limit, self._slabheight, self._MHW)  # Returns map of erosion probabilities
        aeolian_deposition_prob = routine.depprobs(self._effective_veg, wind_shadows, subaerial, self._p_dep_base, self._p_dep_sand, self._p_dep_sand_VegMax, self._topo, self._groundwater_elevation)  # Returns map of deposition probabilities

        # Move sand slabs
        aeolian_elevation_change = routine.shiftslabs(
            aeolian_erosion_prob,
            aeolian_deposition_prob,
            self._saltation_length,
            self._saltation_length_rand_deviation,
            self._effective_veg,
            self._saltation_veg_limit,
            int(self._wind_direction[it]),
            True,
            self._topo,
            self._repose_bare,
            self._MHW,
            self._cellsize,
            self._RNG)  # Returns map of height changes in units of slabs

        # Apply changes, make calculations
        self._topo += aeolian_elevation_change * self._slabheight  # [m NAVD88] Changes applied to the topography; convert aeolian_elevation_change from slabs to meters

        # --------------------------------------
        # STORMS

        if it % self._storm_update_frequency == 0:

            # Generate Storms Stats
            if self._hindcast:  # Empirical storm time series
                storm, Rhigh, Rlow, dur = routine.get_storm_timeseries(self._storm_timeseries, it, self._longshore, self._MHW, self._simulation_start_iteration)  # [m NAVD88]
            else:  # Stochastic storm model
                # Calculate current beach slopes
                foredune_crest_loc, not_gap = routine.foredune_crest(self._topo, self._MHW, self._cellsize)
                beach_slopes = routine.calculate_beach_slope(self._topo, foredune_crest_loc, self._average_dune_toe_height, self._MHW, self._cellsize)

                storm, Rhigh, Rlow, dur = routine.stochastic_storm(self._pstorm, iteration_year, self._StormList, beach_slopes, self._longshore, self._MHW, self._RNG_storm)  # [m NAVD88]

                # # Account for change in mean sea-level on synthetic storm elevations by adding aggregate RSLR since simulation start, and any linear storm climate shift in intensity
                TWL_climate_shift = Rhigh * (self._shift_mean_storm_intensity_start + ((self._shift_mean_storm_intensity_end - self._shift_mean_storm_intensity_start) * (it / self._iterations)))  # This version shifts TWL more for bigger storms
                Dur_climate_shift = np.round(np.mean(TWL_climate_shift * self._storm_twl_duration_correlation))
                Rhigh += (self._MHW - self._MHW_init) + TWL_climate_shift  # [m NAVD88] Add change in sea level to storm water levels, which were in elevation relative to initial sea level, and shift intensity
                Rlow += (self._MHW - self._MHW_init) + TWL_climate_shift  # [m NAVD88] Add change in sea level to storm water levels, which were in elevation relative to initial sea level, and shift intensity
                dur = int(dur + Dur_climate_shift)  # [hr] Modify duration from climate shift

            if storm:

                # Storm Processes: Beach/duneface change, overwash
                self._StormRecord = np.vstack((self._StormRecord, np.array([year, iteration_year, np.max(Rhigh), np.max(Rlow), dur], dtype=np.float32)))
                self._topo, self._OWflux, inundated, Qbe = routine.storm_processes(
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
                    herbaceous_cover=self._veg_fraction[:, :, 2] + self._veg_fraction[:, :, 4],
                    woody_cover=self._veg_fraction[:, :, 6] + self._veg_fraction[:, :, 7],
                    flow_reduction_max_spec1=self._flow_reduction_max_spec1,
                    flow_reduction_max_spec2=self._flow_reduction_max_spec2,
                )

                # Aggregate inundation [boolean] for the period between the previous and next time step at which output is saved (save_frequency)
                self._inundated_output_aggregate += np.round(inundated)

                # Check for nans in topo
                if np.isnan(np.sum(self._topo)):
                    self._topo = routine.replace_nans_infs(self._topo)

            else:
                self._OWflux = np.zeros([self._longshore], dtype=np.float32)  # [m^3] No overwash if no storm
                Qbe = np.zeros([self._longshore], dtype=np.float32)  # [m^3] No overwash if no storm
                inundated = np.zeros([self._longshore, self._crossshore])
                Rhigh = np.zeros([self._longshore])

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
            topo_copy_pre = self._topo.copy()
            self._topo = routine.adjust_ocean_shoreline(
                self._topo,
                self._x_s,
                prev_shoreline,
                self._MHW,
                shoreface_slope,
                self._RSLR,
                self._storm_iterations_per_year,
                self._cellsize,
            )

            # Enforce angles of repose
            """IR 25Apr24: Ideally, angles of repose would be enforced after avery aeolian iteration and every storm. However, to significantly increase model speed, I now enforce AOR only at the end of each
            shoreline iteration (i.e., every 2 aeolian iterations). The morphodynamic effects of this are apparently negligible, while run time is much quicker."""
            veg_cover = self._veg_fraction[:, :, 2] + self._veg_fraction[:, :, 4] + self._veg_fraction[:, :, 6] + self._veg_fraction[:, :, 7]
            self._topo = routine.enforceslopes(self._topo, veg_cover, self._slabheight, self._repose_bare, self._repose_veg, self._repose_threshold, self._MHW, self._cellsize, self._RNG)  # [m NAVD88]

            # Store shoreline and shoreface toe locations
            self._x_s_TS = np.vstack((self._x_s_TS, self._x_s))  # Store
            self._x_t_TS = np.vstack((self._x_t_TS, self._x_t))  # Store
            self._x_bb_TS = np.vstack((self._x_bb_TS, routine.backbarrier_shoreline(self._topo, self._MHW)))  # Store

            # Maintain equilibrium back-barrier depth
            self._topo = routine.maintain_equilibrium_backbarrier_depth(self._topo, self._eq_backbarrier_depth, self._MHW)

            # Update sedimentation balance
            self._sedimentation_balance_year += self._topo - topo_copy_pre  # [m] Update the sedimentation balance map
            self._sedimentation_balance_biweekly += self._topo - topo_copy_pre  # [m] Update the sedimentation balance map

            # --------------------------------------
            # NEW VEGETATION
            it_veg = int(iteration_year)
            temperature_C = self._temperatureC_average_daily_max[it_veg]  # [C] Find temperature for this iteration
            extreme_temperature_C = temperature_C - 5.5 + self._RNG.normal(loc=0, scale=self._standard_dev_temperature)  # -5.5 is the average daily mean relative to max
            hwe_q = inundated * 300  # TEMP!
            x_b = routine.backbarrier_shoreline(self._topo, self._MHW)

            # Dispersal
            H1_currently_vegetated = self._veg_fraction[:, :, 1] + self._veg_fraction[:, :, 2] > 0.02  # Cells that are currently vegetated
            H2_currently_vegetated = self._veg_fraction[:, :, 3] + self._veg_fraction[:, :, 4] > 0.02  # Cells that are currently vegetated
            W_currently_vegetated = self._veg_fraction[:, :, 5] + self._veg_fraction[:, :, 6] + self._veg_fraction[:, :, 7] > 0.02  # Cells that are currently vegetated

            # Pioneer Colonization via Seeds & Rhizome Fragments
            H1_pioneer = self._RNG.random(self._topo.shape) < self._sp1_pioneer_probability
            H2_pioneer = self._RNG.random(self._topo.shape) < self._sp1_pioneer_probability
            W_pioneer = self._RNG.random(self._topo.shape) < self._sp1_pioneer_probability

            # Lateral Expansion
            H1_lateral = routine.lateral_expansion(self._veg_fraction[:, :, 1] + self._veg_fraction[:, :, 2], 1, self._sp1_lateral_probability, self._RNG)
            H2_lateral = routine.lateral_expansion(self._veg_fraction[:, :, 3] + self._veg_fraction[:, :, 4], 1, self._sp1_lateral_probability, self._RNG)
            W_lateral = routine.lateral_expansion(self._veg_fraction[:, :, 5] + self._veg_fraction[:, :, 6] + self._veg_fraction[:, :, 7], 1, self._sp2_lateral_probability, self._RNG)

            # Determine Where Disperal is Allowed
            H1_germ_allowed = np.logical_or(np.logical_or(H1_pioneer, H1_lateral), H1_currently_vegetated)
            H2_germ_allowed = np.logical_or(np.logical_or(H2_pioneer, H2_lateral), H2_currently_vegetated)
            W_germ_allowed = np.logical_or(np.logical_or(W_pioneer, W_lateral), W_currently_vegetated)

            # Calculate Transition Probabilities
            H1_germ, H2_germ, W_germ = routine.germination_prob(temperature_C,
                                                                self._topo,
                                                                self._MHW,
                                                                storm,
                                                                hwe_q,
                                                                self._x_s,
                                                                x_b,
                                                                self._cellsize,
                                                                self._veg_fraction,
                                                                self._sedimentation_balance_biweekly,
                                                                self._germination_erosion_limit,
                                                                self._germination_burial_limit,
                                                                self._H1_germ_tempC_max,
                                                                self._H1_germ_tempC_min,
                                                                self._H2_germ_tempC_max,
                                                                self._H2_germ_tempC_min,
                                                                self._W_germ_tempC_max,
                                                                self._W_germ_tempC_min,
                                                                self._H1_growth_woody_comp_max,
                                                                self._H2_growth_woody_comp_max,
                                                                self._W_germ_Pmin_herbaceous_facil,
                                                                self._W_germ_herbaceous_facil_max,
                                                                self._W_dune_elev_min,
                                                                self._W_dune_elev_max,
                                                                self._W_shoreline_distance_min,
                                                                self._W_shoreline_distance_max,
                                                                self._H1_germ_Pmax_tempC,
                                                                self._H2_germ_Pmax_tempC,
                                                                self._W_germ_Pmax_tempC,
                                                                )

            # Constrain Germination to Cells Where Dispersal is Allowed
            H1_germ *= H1_germ_allowed
            H2_germ *= H2_germ_allowed
            W_germ *= W_germ_allowed

            H1_s_mort, H2_s_mort, W_s_mort = routine.seedling_mortality_prob(self._topo,
                                                                             self._MHW,
                                                                             self._x_s,
                                                                             x_b,
                                                                             self._cellsize,
                                                                             self._sedimentation_balance_biweekly,
                                                                             temperature_C,
                                                                             extreme_temperature_C,
                                                                             storm,
                                                                             hwe_q,
                                                                             self._seedling_erosion_limit,
                                                                             self._seedling_burial_limit,
                                                                             self._H1_growth_tempC_min,
                                                                             self._H1_growth_tempC_max,
                                                                             self._H2_growth_tempC_min,
                                                                             self._H2_growth_tempC_max,
                                                                             self._W_growth_tempC_min,
                                                                             self._W_growth_tempC_max,
                                                                             self._W_dune_elev_min,
                                                                             self._W_dune_elev_max,
                                                                             self._W_shoreline_distance_min,
                                                                             self._W_shoreline_distance_max,
                                                                             self._H1_s_mort_Pmax_tempC,
                                                                             self._H2_s_mort_Pmax_tempC,
                                                                             self._W_s_mort_Pmax_tempC,
                                                                             self._H1_QHWE_min,
                                                                             self._H1_QHWE_max,
                                                                             self._H2_QHWE_min,
                                                                             self._H2_QHWE_max,
                                                                             self._W_QHWE_min,
                                                                             self._W_QHWE_max,
                                                                             self._H1_mort_tempC_min,
                                                                             self._H1_mort_tempC_max,
                                                                             self._H2_mort_tempC_min,
                                                                             self._H2_mort_tempC_max,
                                                                             self._W_s_mort_tempC_min,
                                                                             self._W_s_mort_tempC_max,
                                                                             self._RNG,
                                                                             )

            H1_growth, H2_growth, W_growth = routine.growth_prob(self._topo,
                                                                 self._MHW,
                                                                 self._x_s,
                                                                 x_b,
                                                                 self._sedimentation_balance_biweekly,
                                                                 temperature_C,
                                                                 storm,
                                                                 hwe_q,
                                                                 self._veg_fraction,
                                                                 self._H1_growth_tempC_min,
                                                                 self._H1_growth_tempC_max,
                                                                 self._H2_growth_tempC_min,
                                                                 self._H2_growth_tempC_max,
                                                                 self._W_growth_tempC_min,
                                                                 self._W_growth_tempC_max,
                                                                 self._H1_stim_min,
                                                                 self._H1_stim_max,
                                                                 self._H2_stim_min,
                                                                 self._H2_stim_max,
                                                                 self._W_stim_min,
                                                                 self._W_stim_max,
                                                                 self._H1_elev_gamma_a,
                                                                 self._H1_elev_gamma_scale,
                                                                 self._H1_elev_gamma_loc,
                                                                 self._H2_elev_gamma_a,
                                                                 self._H2_elev_gamma_scale,
                                                                 self._H2_elev_gamma_loc,
                                                                 self._W_elev_gamma_a,
                                                                 self._W_elev_gamma_scale,
                                                                 self._W_elev_gamma_loc,
                                                                 self._H1_growth_woody_comp_max,
                                                                 self._H2_growth_woody_comp_max,
                                                                 self._H1_growth_Pmax_tempC,
                                                                 self._H2_growth_Pmax_tempC,
                                                                 self._W_growth_Pmax_tempC,
                                                                 self._H1_growth_Pmin_stim,
                                                                 self._H2_growth_Pmin_stim,
                                                                 self._W_growth_Pmin_stim,
                                                                 self._H1_growth_Pmax_elev,
                                                                 self._H2_growth_Pmax_elev,
                                                                 self._W_growth_Pmax_elev,
                                                                 )

            W_a_removal = routine.woody_removal_prob(self._sedimentation_balance_biweekly,
                                                     self._W_burial_limit,
                                                     self._W_uproot_limit,
                                                     self._RNG
                                                     )

            H1_a_senesce, H2_a_senesce, W_a_senesce = routine.senescence_prob(self._topo,
                                                                              self._MHW,
                                                                              self._x_s,
                                                                              x_b,
                                                                              self._sedimentation_balance_biweekly,
                                                                              temperature_C,
                                                                              extreme_temperature_C,
                                                                              storm,
                                                                              hwe_q,
                                                                              W_a_removal,
                                                                              self._H1_growth_tempC_min,
                                                                              self._H1_growth_tempC_max,
                                                                              self._H2_growth_tempC_min,
                                                                              self._H2_growth_tempC_max,
                                                                              self._W_growth_tempC_min,
                                                                              self._W_growth_tempC_max,
                                                                              self._H1_a_senesce_Pmin_tempC,
                                                                              self._H1_a_senesce_Pmax_tempC,
                                                                              self._H2_a_senesce_Pmin_tempC,
                                                                              self._H2_a_senesce_Pmax_tempC,
                                                                              self._W_a_senesce_Pmin_tempC,
                                                                              self._W_a_senesce_Pmax_tempC,
                                                                              self._H1_QHWE_min,
                                                                              self._H1_QHWE_max,
                                                                              self._H2_QHWE_min,
                                                                              self._H2_QHWE_max,
                                                                              self._W_QHWE_min,
                                                                              self._W_QHWE_max,
                                                                              self._H1_uproot_limit,
                                                                              self._H2_uproot_limit,
                                                                              self._H1_burial_limit,
                                                                              self._H2_burial_limit,
                                                                              self._H1_mort_tempC_min,
                                                                              self._H1_mort_tempC_max,
                                                                              self._H2_mort_tempC_min,
                                                                              self._H2_mort_tempC_max,
                                                                              self._W_a_mort_tempC_min,
                                                                              self._W_a_mort_tempC_max,
                                                                              self._RNG,
                                                                              )

            W_d_loss = routine.woody_dead_loss(self._topo,
                                               self._MHW,
                                               self._x_s,
                                               x_b,
                                               self._sedimentation_balance_biweekly,
                                               extreme_temperature_C,
                                               storm,
                                               hwe_q,
                                               np.mean(Rhigh),
                                               self._W_uproot_limit,
                                               self._W_burial_limit,
                                               self._W_d_loss_Pmax_submerged_frozen,
                                               self._W_QHWE_min,
                                               self._W_QHWE_max,
                                               self._W_TWL_min,
                                               self._W_TWL_max,
                                               self._W_d_loss_Pmin,
                                               self._W_d_loss_Pmax_discharge,
                                               self._W_d_loss_Pmax_twl,
                                               )

            self._veg_fraction = routine.veg_matrix_mult(self._veg_fraction,
                                                         H1_germ,
                                                         H2_germ,
                                                         W_germ,
                                                         H1_s_mort,
                                                         H2_s_mort,
                                                         W_s_mort,
                                                         H1_growth,
                                                         H2_growth,
                                                         W_growth,
                                                         W_a_removal,
                                                         H1_a_senesce,
                                                         H2_a_senesce,
                                                         W_a_senesce,
                                                         W_d_loss
                                                         )

            # Reset sedimentation balance
            self._sedimentation_balance_biweekly *= 0

            # Determine effective vegetation cover by smoothing; represents effect of nearby vegetation on local wind
            veggie = self._veg_fraction[:, :, 2] + self._veg_fraction[:, :, 4] + self._veg_fraction[:, :, 6] + self._veg_fraction[:, :, 7]  # Adults and dead shrubs
            self._effective_veg = gaussian_filter(veggie, [self._effective_veg_sigma / self._cellsize, self._effective_veg_sigma / self._cellsize], mode='constant')

        # --------------------------------------
        # RECORD VARIABLES PERIODICALLY

        if (it + 1) % (self._save_frequency * self._iterations_per_cycle) == 0:
            moment = int((it + 1) / self._save_frequency / self._iterations_per_cycle)
            self._topo_TS[:, :, moment] = self._topo.astype(np.float16)
            self._storm_inundation_TS[:, :, moment] = self._inundated_output_aggregate.astype(np.float16)
            self._MHW_TS[moment] = self._MHW
            self._inundated_output_aggregate *= False  # Reset for next output period
            self._veg_fraction_TS[:, :, :, moment] = self._veg_fraction

        # --------------------------------------
        # RESET DOMAINS

        self._sedimentation_balance_year[:] = 0  # [m] Reset the balance map

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
    def MHW_TS(self):
        return self._MHW_TS

    @property
    def StormRecord(self):
        return self._StormRecord

    @property
    def x_s_TS(self):
        return self._x_s_TS

    @property
    def x_bb_TS(self):
        return self._x_bb_TS

    @property
    def veg_fraction_TS(self):
        return self._veg_fraction_TS

    @property
    def veg_fraction(self):
        return self._veg_fraction

    @property
    def storm_iterations_per_year(self):
        return self._storm_iterations_per_year
