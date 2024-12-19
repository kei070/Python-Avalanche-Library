#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This presents an attempt to implement the seNorge snow model in Python.
The model was originally written in R by Tuomo Saloranta. Most of the documentation is adapted from the R scripts.

The model R-code was downloaded from https://ars.els-cdn.com/content/image/1-s2.0-S0022169416301755-mmc1.zip

Here the function "seNorge_snowmodel_vo111" will be implemented representing the main function for execution of the
seNorge 1-D snowmodel (v.1.1.1).

The documentation was copied directly from seNorge_snowmodel_vo111.r
"""


# imports
import sys
import numpy as np
import warnings

from .solar_norm_vo111 import solar_norm_vo111
from .sca_funk_vo111 import sca_funk_vo111
from .Func_DatetimeSimple import date_dt

# function
def seNorge_snowmodel_vo111(DMY_TP_Mx, stat_info, params, init_cond=np.array([0, 0, 0, 0, 0])):

    """
    Restructured "official" version (vo)
    (Differences to "v111", 1) simplified parameter file, 2) precipitation correction factors removed and no elevation
    information needed, 3) UTM to dec.degrees units)

    Inputs:
          DMY_TP_Mx:  a (N x 6) matrix where rows are continuous dates, and columns are
                      1) day, 2) month, 3) year, 4) air temperature (degC), 5) precipitation (mm/d)
          stat_info:  a vector (1 x 2) of station information  1) latitude (decimal degrees), 2) vegetation type
                      (0=below treeline; 1=above treeline)
          params:     the model parameter matrix (10 x 2), were 1.col is parameter value, and 2.col an alternative value
                      for grid cells above the treeline (=NA if same value applies for all grid cells)
          init.cond:  the model initial condition vector (SWE_ice, SWE_liq, snowdepth, SWEi_max, SWEi_buff) all in units
                      [mm]

    Output:
          outDMY_TP_snow_Mx: an (N x 17) matrix where rows are continuous dates, and columns are
                             0-4) as in InputMx,
                             5) SWE (mm),
                             6) SD (mm),
                             7) density (kg/L),
                             8) melting/refreezing (mm/d), 10) runoff from snowpack (mm/d),
                             9) runoff from snowpack
                             10) ratio of liquid water to ice (mm/mm),
                             11) grid cell fraction of snow-covered area (SCA)
                             12-16) variables that can be used to define new initial conditions.
                                    can be used to define new initial conditions.

    Using subfunctions: "sca_funk_vo111.r" and "solar_norm_vo111.r"

    by Tuomo Saloranta, Norwegian Water Resources and Energy Directorate (NVE), January 2015
    based on previous seNorge model code, among others by Zelalem Mengistu, Jess Andersen, Thomas Skaugen and Stefano
    Endrizzi last modified by TUS 26.02.2015 (corrected stat.info description in the header)
    """

    inDim = np.shape(DMY_TP_Mx) # dimensions of the input

    outDMY_TP_snow_Mx = np.zeros((inDim[0], 17))  # initialize output matrix
    outDMY_TP_snow_Mx[:, 0:5] = DMY_TP_Mx  # copy input as columns 1-5 of the output

    # === pick values from station information vector
    lat_deg = stat_info[0]  # NB! in decimal degrees here
    veg_ind = int(stat_info[1])  # must be read as integer since it is used as index!

    # === pick values from parameter matrix, depending on the vegetation cover (below/above treeline)
    rmax = params[0, 0]  # max. allowed fraction (WL/WI) of liquid water in snowpack

    TM = 0              # fixed melting poiunt temperature
    TS = params[1, 0]  # threshold air temperature for rain/snow
    if (TM > TS):
        sys.exit("The snow/rain threshold temperature parameter TS must be set above zero melting point")
    # end if

    Crf = params[2, 0]  # degree-day refreezing factor

    b0 = params[3, veg_ind]  # coefficient #1 in melt rate algorithm
    c0 = params[4, veg_ind]  # coefficient #2 in melt rate algorithm

    fSCA = params[5, veg_ind] # subgrid SWE variation factor (plus/minus around the mean SWE)

    rho_nsmin = params[6, veg_ind] # minimum allowed new snow density at -18 deg C (= 0 deg F)
    ny0 = params[7, 0]   # initial snow viscosity at 0 deg C and 0 kg/L
    C5 = params[8, 0]   # coefficient for snow viscosity change with temperature
    C6 = params[9, 0]   # coefficient for snow viscosity change with snow density

    MaxDensity = 0.55  # maximum value for density (kg/L)
    MaxChange = 0.5   # maximum allowed factor of change for snowdepth per time step (0.5 = max. doubling of density)

    #initialization
    swe_ice = init_cond[0]       # ice fraction of SWE [mm]  (snow pack ice content)
    swe_liq = init_cond[1]       # liquid water fraction of SWE [mm] (snow pack water content)
    snowdepth = init_cond[2]     # snowdepth [mm]

    swe_tot = swe_ice + swe_liq       # SWE [mm]

    SWEi_max = init_cond[3]      # maximum swe_ice so far (used in the SCA-routine)
    SWEi_buff = init_cond[4]     # new snow layer (used in the SCA-routine)

    for i in np.arange(0, inDim[0], 1): # >>>> Loop for the entire time series >>>>

        # === === Step 1: SWE module === ===
        swe_ice_x = swe_ice  # snow pack ice content from the previous time step [mm]
        swe_liq_x = swe_liq  # snow pack water content from the previous time step [mm]
        snowdepth_x = snowdepth  # snow depth from the previous time step [mm]
        swe_tot_x = swe_ice_x + swe_liq_x  # SWE from the previous time step [mm]

        # daily temperature [degC]  and  precipitation [mm]
        temp = DMY_TP_Mx[i, 3]
        prcp = DMY_TP_Mx[i, 4]

        # day number
        year = int(DMY_TP_Mx[i, 2])
        month = int(DMY_TP_Mx[i, 1])
        day = int(DMY_TP_Mx[i, 0])
        DN = date_dt(year, month, day).timetuple().tm_yday

        # === liquid or solid precipitation (both cannot occur at the same time step)
        if (temp > TS) :
            PR = prcp
            PS = 0.0
        else:
            PS = prcp
            PR = 0.0
        # end if else

        # === refreezing or melting (both potential values here)
        if (temp <= TM):  # refreezing
            MW_pot = Crf * (temp-TM)   # potential refreezing, negative values
        else:  # melting

            # print(f"date: {year}-{month}-{day}")
            # print(f"DoY = {DN}")

            # potential short-wave radiation at top of atmosphere, normalized to maximum value at 60 N
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                S_star = solar_norm_vo111(lat_deg, DN)
            # end with

            MW_pot = b0 * (temp-TM) + c0 * S_star # potential melting, positive values

            if temp < 2:  # melting goes linearly towards zero from 2 to 0 degrees C
                # linearly decrasing from melting at 2degC to no melting at zero degC
                MW_pot = (temp/2)*(b0*(2-TM) + c0 * S_star)
            # end if

        # end if else

        # === If NOT "no snow at the previous time step AND no new snow at the current time step"
        if (~((swe_tot_x == 0.0) & (PS == 0.0))):

          	# === restrict potential refreezing to available liquid water  (required for "sca.funk" input)
            if ((-MW_pot) > swe_liq_x):
                MW_unsc = -swe_liq_x
            else:
                MW_unsc = MW_pot  # "MW.unsc" is refreezing or melting, not yet scaled in case SCA<1
            # end if else

            # === Call SCA-algorithm
            SCA_output = sca_funk_vo111(-MW_unsc, PS, swe_ice_x, SWEi_max, SWEi_buff, fSCA)

            # update SCA variables
            SWEi_max = SCA_output[0]  # maximum swe_ice so far [mm]
            SWEi_buff = SCA_output[1] # new snow layer [mm]
            SCA = SCA_output[2]       # fraction of snow-covered area (0-1)
            f_red_M = SCA_output[4]   # reduction factor (0-1) for melting or refreezing in case SCA<1

            # this is to avoid bad values due to 0/0 divisions in the SCA-routine
            if (np.abs(MW_unsc) > 0):
                # reduce melting or refreezing in case SCA < 1. Limit also actual melting to available ice
                MW = np.min([swe_ice_x + PS, f_red_M * MW_unsc])
            else:
                MW = 0.0
            # end if else

            # === update snow pack ice and water contents
            # update swe_ice with new snow, and refreezing (negative) or melting (positive)
            swe_ice = swe_ice_x + PS - MW
            # update potential swe_liq with rain, and refreezing (negative) or melting (positive)
            swe_liq_pot = swe_liq_x + PR + MW
            # restrict actual swe_liq with maximum allowed water content, and ...
            swe_liq = np.min([swe_liq_pot, swe_ice * rmax])

            # ... redirect the rest to runoff [mm]
            isoil = swe_liq_pot - swe_liq

        else: # If "no snow at the previous time step AND no new snow at the current time step"

            swe_ice = 0.0
            swe_liq = 0.0

            # runoff [mm]
            isoil = PR

            MW = 0

            SWEi_max = 0.0
            SWEi_buff = 0.0
            SCA = 0.0
            f_red_M = 1   #set for output puposes
        # end if else

        # total SWE
        swe_tot = swe_ice + swe_liq

        # === === Step 2: Snow density and compaction module === ===

        # The different processes are assumed to occur in the following order: i) melting of old snow pack, ii) new snow
        # fall (and melting of new snow fall, if any), iii) viscous compaction

        if (swe_tot > 0): # if there is some snow left

            if (temp > TS):  #calculate the amount of new snow remaining on the ground [mm]
                newsnow = 0.0
            else:
                # the amount of new snow cannot exceed the SWE remaining on the ground, if the whole old snow pack and
                # part of new snow fall is melted.
                newsnow = np.min([PS, swe_ice])
            # end if else

            # Substep 1: Change of snowdepth due to snowmelt
            # swe_ice_x/snowdepth_x can be 0/0 = NaN, therefore:
            if snowdepth_x == 0:
                swe_ice_x_snowdepth_x = np.nan
            else:
                swe_ice_x_snowdepth_x = swe_ice_x/snowdepth_x
            # end if else
            dSD_1 = - np.nanmin([snowdepth_x, np.max([0, MW])/(swe_ice_x_snowdepth_x)])

            # Substep 2: Change of snowdepth due to newsnow
            T_fahr = (temp * 9.0/5.0) + 32.0 #Celsius to Fahrenheit
            # density of new snow (kg/L)
            rho_newsnow = np.max([0.050, (rho_nsmin + (np.max([0, T_fahr])/100)**2)])

            snowdepth_1 = snowdepth_x + dSD_1 + newsnow/rho_newsnow  # snowdepth [mm] after Substeps 1 & 2
            rho_1 = swe_tot/snowdepth_1 # density [L/kg] after Substeps 1 & 2

            # Substep 3: Viscous compaction
            # gravitation constant [m/s2]
            G = 9.81
            # viscosity of snow at 0 degC and density of zero(?!) [Ns/m2 = kg/m/s]
            ETA0 = ny0*1e6
            # coefficient (this means that half of the snow mass is considered as effective mass for bulk compaction)
            kcomp = 0.5
            # seconds per day
            secperday = 86400

            # overburden force [N/m2 = kg/s2/m]; swe_tot in units mm = kg/m2
            overburden = kcomp * G * swe_tot
            # the liquid water in snowpack (for compaction algorithm) cannot exceed snow pore volume
            liquid_w = np.min([swe_liq_pot, snowdepth_1 - 1.1*swe_ice])

            # factor decrease in viscosity due to presence of liquid water
            f1 = 1/(1 + 60*liquid_w/snowdepth_1)
            # viscosity [Ns/m2] (effective snow temperature is assumed to be half of the negative air temperature)
            viscosity = f1*ETA0*(rho_1/0.250)*np.exp(-C5*np.min([0.5*temp, 0.0]) + C6*rho_1)

            # [1/s] * [mm] * [s/d]
            dSD_2 = -(overburden/viscosity)*snowdepth_1*secperday

            # restriction to max compaction rate
            if ((-dSD_2) > MaxChange*snowdepth_1):
                dSD_2 = -MaxChange*snowdepth_1
            # end if

            # snow depth after Substep 3
            snowdepth = snowdepth_1 + dSD_2
            # ===

            if ((swe_tot/snowdepth) > MaxDensity):
                # restriction to max density
                snowdepth = swe_tot/MaxDensity
            # end if

        else:  # if there is no snow left
            snowdepth = 0.0
        # end if else

        # catch Nan values
        if snowdepth == 0:
            dens = np.nan
        else:
            dens = swe_tot/snowdepth
        # end if

        if ((swe_ice <= 0) | (swe_liq < 0)):
            swe_liq_ice = np.nan
        else:
            swe_liq_ice = swe_liq/swe_ice
        # end if else

        # === Output
        outDMY_TP_snow_Mx[i, 5] = swe_tot # SWE [mm]
        outDMY_TP_snow_Mx[i, 6] = snowdepth # snowdepth [mm]
        outDMY_TP_snow_Mx[i, 7] = dens # density [kg/L]
        outDMY_TP_snow_Mx[i, 8] = MW  # melting/refreezing [mm]
        outDMY_TP_snow_Mx[i, 9] = isoil # runoff [mm]
        outDMY_TP_snow_Mx[i, 10] = swe_liq_ice # liquid water to ice mass fraction in the snow pack [mm/mm]
        outDMY_TP_snow_Mx[i, 11] = SCA # fraction of snow-covered area (0-1)

        # for initial condition vector purposes
        outDMY_TP_snow_Mx[i, 12] = swe_ice
        outDMY_TP_snow_Mx[i, 13] = swe_liq
        outDMY_TP_snow_Mx[i, 14] = snowdepth
        outDMY_TP_snow_Mx[i, 15] = SWEi_max
        outDMY_TP_snow_Mx[i, 16] = SWEi_buff

    # END of for-loop

    return outDMY_TP_snow_Mx
# END function

# === Notes: ===
# f_red_M can be NaN or out of 0-1 range in case of extremely low melt values (0/0 divisions)



