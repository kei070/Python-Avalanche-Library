#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Do the seNorge gridcell extraction year for year but for all regions in one script.
"""


#%% imports
import os
import glob
import time
import argparse
import numpy as np
import xarray as xr
from netCDF4 import Dataset
from datetime import datetime

from Functions.Func_Progressbar import print_progress_bar
from Lists_and_Dictionaries.Paths import path_par


#%% parse the arguments
parser = argparse.ArgumentParser(description="""Extracts the grid cells from the seNorge output based on the NORA3
                                 subset for each of the avalanche regions for a given range of years""")

parser.add_argument('--start_year', type=int, default=1970,
                    help='The start year of the time series to extract from the NORA3 subset.')
parser.add_argument('--end_year', type=int, default=2024,
                    help='The start year of the time series to extract from the NORA3 subset.')
parser.add_argument('--low', type=int, default=300, help='The lower threshold for the altitude band to extract.')
parser.add_argument('--high', type=int, default=1300, help='The upper threshold for the altitude band to extract.')
parser.add_argument('--dataset', type=str, default="NORA3", choices=["NORA3", "NorCP"],
                    help='The dataset on which the seNorge calculations were based. Either NORA3 or NorCP.')
parser.add_argument('--model', default="EC-Earth", choices=["EC-Earth", "GFDL-CM3"], type=str,
                    help='Model used in the NorCP downscaling. Only used if dataset is NorCP.')
parser.add_argument('--scen', default="historical", choices=["historical", "rcp45", "rcp85"], type=str,
                    help='Scenario used in the downscaling. Only used if dataset is NorCP.')
parser.add_argument('--period', default="", choices=["", "MC", "LC"], type=str,
                    help='Period used in the downscaling. Only used if dataset is NorCP.')
args = parser.parse_args()


#%% set region code
reg_codes = [3009, 3010, 3011, 3012, 3013]


#%% set the list of variables to be stored

# seNorge
var = ["SWE", "SnowDepth", "SnowDens", "Melt_Refreeze"]


#%% load the nc file for the chosen year

# hard drive path
data_path = f"/{path_par}/IMPETUS/seNorge/Data/"


#%% get the arguments from the parser
h_low = args.low
h_hi = args.high
dataset = args.dataset

model = ""
scen = ""
period = ""
out_add = ""
if dataset == "NorCP":
    model = f"_{args.model}"
    scen = f"_{args.scen}"
    if scen != "_historical":
        period = f"_{args.period}"
    # end if
    out_add = f"/{model[1:]}{scen}{period}/"

    # set the start and end years based on the period
    print("\nDataset is NorCP, setting start and end year based on period...\n")
    if args.period == "":
        start_year = 1985
        end_year = 2005
    elif period == "_MC":
        start_year = 2040
        end_year = 2060
    elif period == "_LC":
        start_year = 2080
        end_year = 2100
    # end if elif
else:
    start_year = args.start_year
    end_year = args.end_year
# end if

print(f"\nstart: {start_year}  |  end: {end_year}\n")


#%% set the years
yrs = list(np.arange(start_year, end_year+1))


#%% load the seNorge file
nc = Dataset(data_path + f"seNorge_NorthNorway_{dataset}{model}{scen}{period}_{start_year}_{end_year}.nc")
nc_xr = xr.open_dataset(data_path + f"seNorge_NorthNorway_{dataset}{model}{scen}{period}_{start_year}_{end_year}.nc",
                        decode_times=False)


#%% extract time units and calendar
try:
    ti_units = nc_xr['time'].attrs['units']
    ti_calendar = nc_xr['time'].attrs.get('calendar', 'standard')
except:
    ti_units = nc.variables["time"].units
    ti_calendar = nc.variables["time"].calendar
# end try except


#%% extract NORA3 lat lon info
lon = nc.variables["longitude"][:]
lat = nc.variables["latitude"][:]


#%% concatenate the lists into arrays along the time axis

nora3_time = nc_xr.time

# get the time length
t_len = len(nora3_time)


#%% load the data
nc_vals = {key: [] for key in var}
# load the nc dataset
print("\nLoading data...\n")
l = len(var)
print_progress_bar(0, l, prefix='Progress:', length=50)
count = 0
for varn in var:
    print_progress_bar(count, l, prefix='Progress:', suffix=f'{varn:15}', length=50)
    nc_vals[varn] = np.squeeze(nc.variables[varn][:])
    count += 1
# end for varn
print_progress_bar(count, l, prefix='Progress:', suffix='Done.                 ', length=50)
print("\n\n")


#%% loop over the region codes to extract the indices
indx_d = {}
indy_d = {}
not_avail = []  # set up a list for the regions that do not have grid cells in the given range
print("Extracting indices...\n")
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
    shp_path = f"/{path_par}/IMPETUS/NORA3/Cells_Between_Thres_Height/NorthernNorway_Subset/{region}/"

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
    l = len(shp[:, 1])
    print_progress_bar(0, l, prefix='Progress:', suffix=f'Complete of {region:11}', length=50)
    count = 0
    for lo, la in zip(shp[:, 1], shp[:, 2]):
        ind = np.where((lo == lon) & (la == lat))
        indx_l.append(ind[0][0])
        indy_l.append(ind[1][0])
        count += 1
        print_progress_bar(count, l, prefix='Progress:', suffix=f'Complete of {region:11}', length=50)
    # end for lo, la

    indx_d[reg_code] = indx_l
    indy_d[reg_code] = indy_l

# end for reg_code

print("\n\n")


#%% set up a dicitionary to store all data
print("Store data as nc-files...\n")
for reg_code in reg_codes:

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

    out_path = (f"/{path_par}/IMPETUS/{dataset}/Avalanche_Region_Data_seNorge/Between{h_low}_and_{h_hi}m/" + out_add)

    os.makedirs(out_path, exist_ok=True)

    count = 0
    l = len(indx_d[reg_code])
    print(f"\n{l} locations.\n")
    print_progress_bar(0, l, prefix='Progress:', suffix=f'Complete of {region:11}', length=50)

    temp_dic = {varn:[] for varn in var[:]}
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

        for varn in var[:]:
            # print(varn)
            temp_dic[varn].append(nc_vals[varn][:, loc1, loc2])
        # end for varn

        count += 1
        print_progress_bar(count, l, prefix='Progress:', suffix=f'Complete of {region:11}', length=50)

    # end for loc1, loc2

    # convert lists to arrays
    for varn in var:
        temp_dic[varn] = np.stack(temp_dic[varn])
    # end for varn

    # generate a nc dataset from the dictionary
    print(f"\nGenerating {region} nc-file.\n")
    ncfile = Dataset(out_path + f"seNorge_{dataset}{model}{scen}{period}_{region}_" +
                     f"{'-'.join([str(yrs[0]), str(yrs[-1])])}.nc", 'w', format='NETCDF4')

    # Define dimensions
    ncfile.createDimension('time', t_len)  # unlimited
    ncfile.createDimension('loc', l)

    # Create variables
    nc_times = ncfile.createVariable('time', np.float64, ('time'))
    nc_locx = ncfile.createVariable("loc_x", np.float64, ("loc"))
    nc_locy = ncfile.createVariable("loc_y", np.float64, ("loc"))
    nc_vars = [ncfile.createVariable(varn, np.float64, ('loc', 'time')) for varn in var]

    # Assign data to variables
    for varn, nc_var in zip(var, nc_vars):
        nc_var[:] = temp_dic[varn]
    # end for varn, nc_var

    nc_times[:] = nora3_time
    nc_locx[:] = temp_dic["x_loc"]
    nc_locy[:] = temp_dic["y_loc"]

    # set the origin of the time coordinate
    nc_times.units = ti_units
    nc_times.calendar = ti_calendar

    # Add global attributes
    ncfile.description = f'{region} {dataset} data for gridcells between {h_low} and {h_hi} m.'
    ncfile.history = 'Created ' + datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Close the file
    ncfile.close()

# end for reg_code


