#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Apply seNorge on NORA3 data.
"""


#%% imports
import glob
import numpy as np
import pandas as pd
import xarray as xr
import warnings
from netCDF4 import Dataset, date2num
from joblib import dump

from Functions.Particular_Functions.Func_Apply_seNorge import apply_seNorge
from Lists_and_Dictionaries.Paths import path_par, path_seNorge


#%% import the NORA3 elevation grid
topo_path = f"{path_par}/IMPETUS/NORA3/"
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    topo_nc = xr.open_dataset(topo_path + "nora3_topo.nc")
# end with
lats_to = topo_nc.latitude.values
lons_to = topo_nc.longitude.values


#%% paths
data_path = f"/{path_par}/IMPETUS/NORA3/NORA3_NorthNorway_Sub/Annual_Files_DailyMean/"


#%% prepare loading of nc files
sta_year = 1970
end_year = 1975

fn_list = np.array(sorted(glob.glob(data_path + "*.nc")))

years = np.array([fn[-7:-3] for fn in fn_list]).astype(int)


#%% load the nc files
nc = xr.open_mfdataset(fn_list[(years >= sta_year) & (years <= end_year)], data_vars="minimal", coords="minimal",
                       compat="override")


#%% get the time axis but suppress conversion to datetime
time_info = Dataset(fn_list[0]).variables["time"]
time = date2num(nc.time.values.astype('M8[ms]').astype('O'), units=time_info.units, calendar=time_info.calendar)


#%% extract the subset-grid variables
at2m = np.array(nc.air_temperature_2m.squeeze().load()) - 273.15  # convert to degC
prec = np.array(nc.precipitation_amount_hourly.squeeze().load())


#%% find the indices in the NORA3 subset
lats_sub = nc.latitude.values.squeeze()
lons_sub = nc.longitude.values.squeeze()
xs_sub = nc.x.values
ys_sub = nc.y.values


#%% read parameter file and paste the values to "params" matrix
srcpath = f"/{path_seNorge}/IMPETUS/seNorge/seNorge_Model_R/parameter_forcing_files/"
params = np.array(pd.read_table(srcpath + "seNorge_vo111_param.csv", sep=",", index_col=0, na_values=-9999))


#%% get days, month, and years
days, months, years = np.array(nc.time.dt.day), np.array(nc.time.dt.month), np.array(nc.time.dt.year)


#%% apply the seNorge model to the first 5 years to generate (somewhat) realisitc initial conditions to restart the
#   model again in 1970
swe, sdepth, sdens, melt_ref, rest_data = apply_seNorge(at2m, prec, days, months, years,
                                                        lats=nc.latitude, lons=nc.longitude,
                                                        zs=topo_nc.ZS, lats_to=lats_to, lons_to=lons_to,
                                                        return_rest=True)


#%% store the restart data
rest_path = f"{path_par}/IMPETUS/seNorge/Restart/seNorge_NORA3_NorthNorway_{sta_year}_{end_year}_Restart.joblib"
dump(rest_data, rest_path)


