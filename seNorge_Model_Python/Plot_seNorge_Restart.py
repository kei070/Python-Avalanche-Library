#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate some test plots for the restart files.
"""

#%% imports
from joblib import load
import pylab as pl
import cartopy.crs as ccrs

from Lists_and_Dictionaries.Paths import path_par


#%% set parameters for the scenario selection
model = "EC-Earth"
scen = "historical"
period = ""  # either "", "_MC", or "_LC"
sta_year = 1985
end_year = 1990


#%% load the restart data
rest_path = f"{path_par}/IMPETUS/seNorge/Restart/"

# rest_name = "seNorge_NorCP_NorthNorway_{model}_{scen}{period}_{sta_year}_{end_year}_Restart.joblib"
rest_name = "seNorge_NORA3_NorthNorway_1970_1975_Restart.joblib"

rest_data = load(rest_path + rest_name)


#%% plot
fig = pl.figure(figsize=(8, 5))
ax00 = fig.add_subplot(111)

p00 = ax00.pcolormesh(rest_data[2, :, :] / 10)
cb00 = fig.colorbar(p00, ax=ax00)


pl.show()
pl.close()