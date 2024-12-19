#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Call all the necessary scripts to perform the find gridcells procedure, including the conversion to the .csv that is
necessary to perform the extraction on the climate server.
"""


#%% imports
import subprocess
import numpy as np

from Lists_and_Dictionaries.Paths import py_path_par


#%% set the paths to the scripts
py_path = f"{py_path_par}/"


#%% set height thresholds
h_low = 700
h_hi = 1200


#%% send the scripts

# calculate the predictors only for those days with available risk assessment
for reg_code in np.arange(3009, 3013+1):
    # print(reg_code)
    subprocess.call(["python", py_path + "Find_GridCells_Above_or_Between_CertainHeight.py",
                     str(reg_code), str(h_low), str(h_hi)])
    subprocess.call(["python", py_path + "Convert_Height_shps_to_DataFrame.py",
                     str(reg_code), str(h_low), str(h_hi)])
# end for reg_code
