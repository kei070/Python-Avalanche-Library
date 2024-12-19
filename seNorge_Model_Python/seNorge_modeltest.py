#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This represents an attempt to rewrite the seNorge_modeltest.r script into Python.

Copied from the R-script:

This R-script demonstrates a simple test run of the seNorge snow model (v.1.1.1)
Remember to modify the file paths below first!
by Tuomo Saloranta, Norwegian Water Resources and Energy Directorate (NVE), 13.02.-15
"""


#%% imports
import numpy as np
import pandas as pd
import pylab as pl

from Functions.seNorge_snowmodel_vo111 import seNorge_snowmodel_vo111
from Lists_and_Dictionaries.Paths import path_seNorge


#%% set correct paths to code and forcing
srcpath = f"/{path_seNorge}/IMPETUS/seNorge/seNorge_Model_R/parameter_forcing_files/"
codepath = f"/{path_seNorge}/IMPETUS/seNorge/seNorge_Model_R/model_R_code/"


#%% read parameter file and paste the values to "params" matrix
params = np.array(pd.read_table(srcpath + "seNorge_vo111_param.csv", sep=",", index_col=0, na_values=-9999))


#%% set initial conditions and load test forcing file

# describe site latitude and vegetation type
# 1) latitude (decimal.degrees), 2) vegetation type (0=below treeline; 1=above treeline)
stat_info = np.array([61.325, 1])

# load forcing data matrix (InputMx)
InputMx = np.array(pd.read_csv(srcpath + "seNorge_testforcing.csv", sep=",", skiprows=0, index_col=0))
# InputMx: an (N x 5) matrix where rows are continuous dates (daily) and columns are:
#                            0) year, 1) month, 3) day,
#                            4) air temperature
#                            5) precipitation


# start from zero initial conditions for the five variables (SWE_ice, SWE_liq, snowdepth, SWEi_max, SWEi_buff) all
# in units [mm]
initial_cond = np.array([0, 0, 0, 0, 0])


#%% plot the input data
fig = pl.figure(figsize=(8, 4))

axes = [fig.add_subplot(i) for i in np.arange(121, 122+1, 1)]

axes[0].plot(InputMx[:, -2])
axes[1].plot(InputMx[:, -1])

axes[0].set_ylabel("SAT in $\degree$C")
axes[1].set_ylabel("Precipitation in mm")
axes[0].set_xlabel("Days")
axes[1].set_xlabel("Days")

pl.show()
pl.close()


#%% === Run the seNorge snow model ===
OutputMx = seNorge_snowmodel_vo111(InputMx, stat_info, params, initial_cond)

# OutputMx:  an (N x 17) matrix where rows are continuous dates, and columns are
#            1-5) as in InputMx,
#            6) SWE (mm),
#            7) SD (mm),
#            8) density (kg/L),
#            9) melting/refreezing (mm/d), 10) runoff from snowpack (mm/d),
#            11) ratio of liquid water to ice (mm/mm),
#            12) grid cell fraction of snow-covered area (SCA)
#            13-17) variables that can be used to define new initial conditions.
# === === === === === ===


#%% plot a figure of the test model output

lbl_lst = ["SWE", "Snow depth", "Snow density", "melt/refreeze", "runoff from snowpack", "ratio of liquid water to ice",
           "fraction snow-covered area"]
ylbl_lst = ["mm", "mm", "kg/L", "mm/d", "mm/d", "mm/mm",  "fraction"]

# set up the plot
fig = pl.figure(figsize=(10, 6))

axes = [fig.add_subplot(i) for i in np.arange(231, 236+1, 1)]

l_ind = 0
for i, ax in zip([5, 6, 7, 8, 10, 11], axes):
    ax.plot(OutputMx[:, i])

    # ax.set_xlabel("DoY")
    ax.set_ylabel(ylbl_lst[l_ind])
    ax.set_title(lbl_lst[l_ind])

    l_ind += 1
# end i, ax

axes[3].set_xlabel("Days")
axes[4].set_xlabel("Days")
axes[5].set_xlabel("Days")

fig.subplots_adjust(hspace=0.3, wspace=0.3)

pl.show()
pl.close()
