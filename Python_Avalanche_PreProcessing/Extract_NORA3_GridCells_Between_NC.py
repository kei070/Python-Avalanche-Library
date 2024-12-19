#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Do the NORA3 gridcell extraction year for year but for all regions in one script.
"""


#%% imports
import os
import sys
import glob
import time
import argparse
import numpy as np
import xarray as xr
from netCDF4 import Dataset
from datetime import datetime

import pylab as pl

from Functions.Func_Progressbar import print_progress_bar
from Lists_and_Dictionaries.Paths import path_par
from Lists_and_Dictionaries.Variable_Name_NORA3_NorCP import varn_nora3


#%% parser
parser = argparse.ArgumentParser(
                    description="""Extracts the grid cells from the downloaded NORA3 subset for each of the avalanche
                    regions for a given range of years""")

parser.add_argument('--start_year', default=1970, type=int,
                    help='The start year of the time series to extract from the NORA3 subset.')
parser.add_argument('--end_year', default=2024, type=int,
                    help='The start year of the time series to extract from the NORA3 subset.')
parser.add_argument('--low', default=400, type=int, help='The lower threshold for the altitude band to extract.')
parser.add_argument('--high', default=900, type=int, help='The upper threshold for the altitude band to extract.')

args = parser.parse_args()


#%% get the arguments from the parser
start_year = args.start_year
end_year = args.end_year
h_low = args.low
h_hi = args.high


#%% set the years
yrs = list(np.arange(start_year, end_year+1))
print("\nYears:")
print(yrs)
print()


#%% set region code
reg_codes = [3009, 3010, 3011, 3012, 3013]


#%% set the list of variables to be stored --> now in a list in Lists_and_Dictionaries
var = varn_nora3

# remove snow and rain as these are not used here
var.pop("snow")
var.pop("rain")

# ["air_temperature_2m", "precipitation_amount_hourly", "wind_speed", "wind_direction", "relative_humidity_2m",
#         "surface_net_longwave_radiation", "surface_net_shortwave_radiation"]


#%% load the nc file for the chosen year
data_path = f"{path_par}/IMPETUS/NORA3/NORA3_NorthNorway_Sub/Annual_Files/"


#%% loop over the years
nora3_time = []
nc_vals = {var[key]: [] for key in var}
for yr in yrs:

    # list the nc files
    fn_list = sorted(glob.glob(data_path + f"*{yr}*.nc"))


    # open the first file to extract the lon-lat info
    nc = Dataset(fn_list[0])
    nc_xr = xr.open_dataset(fn_list[0],  decode_times=False)  # also open with xarray to make the processing of the time
    #                                                           dimension easier


    #% extract a get the time
    nora3_time.append(nc_xr.time)


    #% load the nc dataset
    for vark in var:
        varn = var[vark]
        nc_vals[varn].append(np.squeeze(nc.variables[varn][:]))
    # end for varn
# end for yr


#%% extract time units and calendar
ti_units = nc_xr['time'].attrs['units']
ti_calendar = nc_xr['time'].attrs.get('calendar', 'standard')


#%% extract NORA3 lat lon info
lon = nc.variables["longitude"][:]
lat = nc.variables["latitude"][:]


#%% concatenate the lists into arrays along the time axis
nora3_time = np.concatenate(nora3_time)
print(np.shape(nora3_time))
for vark in var:
    varn = var[vark]
    nc_vals[varn] = np.concatenate(nc_vals[varn], axis=0)
# end for varn

# get the time length
t_len = len(nora3_time)

for vark in var:
    varn = var[vark]
    print(np.shape(nc_vals[varn]))
# end for varn


#%% loop over the region codes to extract the indices
indx_d = {}
indy_d = {}
not_avail = []  # set up a list for the regions that do not have grid cells in the given range
print("\nExtracting indices...\n")
for reg_code in reg_codes:

    # set region name according to region code
    if reg_code == 3009:
        region = "NordTroms"
    elif reg_code == 3010:
        region = "Lyngen"
    elif reg_code == 3011:
        region = "Tromsoe"
    elif reg_code == 3012:
        region = "SoerTroms"
    elif reg_code == 3013:
        region = "IndreTroms"
    # end if elif


    #% set data path
    shp_path = f"{path_par}/IMPETUS/NORA3/Cells_Between_Thres_Height/NorthernNorway_Subset/{region}/"


    # load the shp file to get the coordinates
    print(shp_path + f"*between{h_low}_and_{h_hi}m*.csv")
    shp = np.loadtxt(glob.glob(shp_path + f"*between{h_low}_and_{h_hi}m*.csv")[0], delimiter=",", skiprows=1)

    if len(shp) == 0:
        print(f"No gridcells for {reg_code}. Continuing...")
        not_avail.append(reg_code)
        continue
    # end if

    # extract NORA3 lat and lon above threshold
    indx_l = []
    indy_l = []
    for lo, la in zip(shp[:, 1], shp[:, 2]):
        ind = np.where((lo == lon) & (la == lat))
        indx_l.append(ind[0][0])
        indy_l.append(ind[1][0])
    # end for lo, la

    indx_d[reg_code] = indx_l
    indy_d[reg_code] = indy_l

# end for reg_code


#%% set up a dicitionary to store all data
print("\nStoring data...\n")
for reg_code in reg_codes:

    """
    # test plot of indices
    pl.plot(indx_d[reg_code], label="x index")
    pl.plot(indy_d[reg_code], label="y index")
    pl.legend()
    pl.show()
    pl.close()
    """

    if reg_code in not_avail:
        continue
    # end if

    # set region name according to region code
    if reg_code == 3009:
        region = "NordTroms"
    elif reg_code == 3010:
        region = "Lyngen"
    elif reg_code == 3011:
        region = "Tromsoe"
    elif reg_code == 3012:
        region = "SoerTroms"
    elif reg_code == 3013:
        region = "IndreTroms"
    # end if elif

    out_path = (f"{path_par}/IMPETUS/NORA3/Avalanche_Region_Data/" +
                f"/Between{h_low}_and_{h_hi}m/{'-'.join([str(yrs[0]), str(yrs[-1])])}/")
    os.makedirs(out_path, exist_ok=True)

    count = 0
    l = len(indx_d[reg_code])
    print(f"\n{l} locations.\n")
    print_progress_bar(0, l, prefix='Progress:', suffix=f'Complete of {region}', length=50)

    temp_dic = {var[vark]:[] for vark in var}
    temp_dic["date"] = nora3_time
    temp_dic["x_loc"] = []
    temp_dic["y_loc"] = []
    for loc1, loc2 in zip(indx_d[reg_code], indy_d[reg_code]):

        # print([loc1, loc2])

        # add the locations to the dictionary
        temp_dic["x_loc"].append(loc1)
        temp_dic["y_loc"].append(loc2)

        # start a timer
        start_time = time.time()

        for vark in var:
            varn = var[vark]
            # print(varn)
            temp_dic[varn].append(nc_vals[varn][:, loc1, loc2])
        # end for varn

        count += 1
        print_progress_bar(count, l, prefix='Progress:', suffix=f'Complete of {region}', length=50)

    # end for loc1, loc2

    # convert lists to arrays
    for vark in var:
        varn = var[vark]
        temp_dic[varn] = np.stack(temp_dic[varn])
    # end for varn

    # generate a nc dataset from the dictionary
    print(f"\nGenerating {region} nc-file.\n")
    ncfile = Dataset(out_path + f"NORA3_{region}_{'-'.join([str(yrs[0]), str(yrs[-1])])}.nc", 'w', format='NETCDF4')

    # Define dimensions
    ncfile.createDimension('time', t_len)  # unlimited
    ncfile.createDimension('loc', l)

    # Create variables
    nc_times = ncfile.createVariable('time', np.float64, ('time'))
    nc_locx = ncfile.createVariable("loc_x", np.float64, ("loc"))
    nc_locy = ncfile.createVariable("loc_y", np.float64, ("loc"))
    nc_vars = [ncfile.createVariable(var[vark], np.float64, ('loc', 'time')) for vark in var]

    # Assign data to variables
    for vark, nc_var in zip(var, nc_vars):
        varn = var[vark]
        nc_var[:] = temp_dic[varn]
    # end for varn, nc_var

    nc_times[:] = nora3_time
    nc_locx[:] = temp_dic["x_loc"]
    nc_locy[:] = temp_dic["y_loc"]

    # set the origin of the time coordinate
    nc_times.units = ti_units
    nc_times.calendar = ti_calendar

    # Add global attributes
    ncfile.description = f'{region} NORA3 data for gridcells between {h_low} and {h_hi} m.'
    ncfile.history = 'Created ' + datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Close the file
    ncfile.close()

# end for reg_code


