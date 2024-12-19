#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Function for applying the seNorge model.
"""

#%% imports
import numpy as np
import pandas as pd

from .seNorge_snowmodel_vo111 import seNorge_snowmodel_vo111
from .Func_Find_Close_2d import find_closest_2d
from .Func_Progressbar import print_progress_bar

from Lists_and_Dictionaries.Paths import path_seNorge


#%% function
def apply_seNorge(at2m, prec, days, months, years, lats, lons, zs, lats_to, lons_to, return_rest=False, rest_in=None,
                  srcpath=f"/{path_seNorge}/IMPETUS/seNorge/seNorge_Model_R/parameter_forcing_files/"):

    # generate the input or take it from the restart file
    if rest_in is None:
        rest_in = np.zeros((5, np.shape(at2m)[-2], np.shape(at2m)[-1]))
    # end if

    # set up the restart array
    rest_out = np.zeros((5, np.shape(at2m)[-2], np.shape(at2m)[-1]))

    # set up the three arrays to store the relevant seNorge data
    swe = np.zeros(np.shape(at2m))
    sdepth = np.zeros(np.shape(at2m))
    sdens = np.zeros(np.shape(at2m))
    melt_ref = np.zeros(np.shape(at2m))

    # read parameter file and paste the values to "params" matrix
    params = np.array(pd.read_table(srcpath + "seNorge_vo111_param.csv", sep=",", index_col=0, na_values=-9999))


    print("\nRunning seNorge...\n")
    l = np.shape(at2m)[1] * np.shape(at2m)[2]
    print_progress_bar(0, l, suffix="x=0, y=0")
    count = 0
    for x in np.arange(np.shape(at2m)[2]):
        for y in np.arange(np.shape(at2m)[1]):

            # set the suffix for the progressbar
            suff = f"x={x:4}  y={y:4}"

            # get the lat and lon of the gridcell
            lat = float(lats[y, x])
            lon = float(lons[y, x])

            # find the grid cell in the topography file
            topo_ind = find_closest_2d(lat, lon, lats_to, lons_to)
            elev = zs.values.squeeze()[topo_ind]

            # if the elevation is lower than 5m continue to next iteration
            # --> basically all grid cells under 5m elevation are ocean
            if elev < 5:
                suff = suff + "   elev < 5m => jumping to next cell..."
                count += 1
                print_progress_bar(count, l, suffix=suff)
                continue
            else:
                suff = suff + "   elev > 5m => running seNorge...     "
            # end if

            # set the treeline according to the elevation
            # 0 --> below  and  1 --> above treeline
            # here we simply assume the treeline is a 500m
            # (see https://en.wikipedia.org/wiki/Tree_line for Finnmarksvidda)
            treel = 0
            if elev > 500:
                treel = 1
                # print(f"\nGrid cell elevation is {r_2(elev):.1f} m, i.e., above the treeline.\n")
            # else:
                # print(f"\nGrid cell elevation is {r_2(elev):.1f} m, i.e., below the treeline.\n")
            # end if else

            # set up the input arrays
            inp_data = np.array([days, months, years, at2m[:, y, x], prec[:, y, x]]).T

            stat_info = np.array([lat, treel])

            # start from zero initial conditions for the five variables (SWE_ice, SWE_liq, snowdepth, SWEi_max,
            #                                                                                                 SWEi_buff)
            # all in units [mm]
            initial_cond = rest_in[:, y, x]  # np.array([0, 0, 0, 0, 0])

            # === Run the seNorge snow model ===
            OutputMx = seNorge_snowmodel_vo111(inp_data, stat_info, params, initial_cond)
            # 5) SWE (mm),
            # 6) SD (mm),
            # 7) density (kg/L),
            # 8) melting/refreezing (mm/d)

            # add the seNorge results to the output arrays
            swe[:, y, x] = OutputMx[:, 5]
            sdepth[:, y, x] = OutputMx[:, 6]
            sdens[:, y, x] = OutputMx[:, 7]
            melt_ref[:, y, x] = OutputMx[:, 8]

            rest_out[:, y, x] = OutputMx[-1, 12:]

            count += 1
            print_progress_bar(count, l, suffix=suff)
        # end for y
    # end for x
    print("\n...done.\n")

    if return_rest:
        return swe, sdepth, sdens, melt_ref, rest_out
    else:
        return swe, sdepth, sdens, melt_ref
    # end if else

# end def