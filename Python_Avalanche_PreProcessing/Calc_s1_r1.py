#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dedicated script for calculating the daily sums of snow and rain (s1 and r1, respectively)
"""


#%% imports
import sys
import glob
import numpy as np
import argparse
from netCDF4 import Dataset

from Functions.Func_Progressbar import print_progress_bar
from Lists_and_Dictionaries.Paths import path_par


#%% parser
parser = argparse.ArgumentParser(
                    description="""Calculate snow and rain for NORA3.""")
parser.add_argument('--reg_code', default=3013, type=int, help='Code of the region.')
parser.add_argument('--low', default=400, type=int, help='The lower threshold for the altitude band to extract.')
parser.add_argument('--high', default=900, type=int, help='The upper threshold for the altitude band to extract.')

args = parser.parse_args()


#%% get the arguments from the parser
reg_code = args.reg_code
h_low = args.low  # 400  # args.low
h_hi = args.high  # 900  # args.high


#%% set region name according to region code
regions = {3009:"NordTroms", 3010:"Lyngen", 3011:"Tromsoe", 3012:"SoerTroms", 3013:"IndreTroms"}
region = regions[reg_code]


#%% NORA3 data path
data_path = f"/{path_par}/IMPETUS/NORA3/Avalanche_Region_Data/Between{h_low}_and_{h_hi}m/"
print(data_path)

#% load data --> should be a grid cell in the Lyngen Alps
fn_list = sorted(glob.glob(data_path + f"*{region}*.nc"), key=str.casefold)

if len(fn_list) == 0:
    sys.exit(f"No predictors for {reg_code} between {h_low} and {h_hi} m. Stopping execution.")
# end if

# print(f"\nNumber of files: {len(fn_list)}.\n")


#%% load the nc file
nc = Dataset(fn_list[0], mode="r+")


#%% load the variables
prec = nc.variables["precipitation_amount_hourly"]
at2m = nc.variables["air_temperature_2m"]


#%% loop over the cells
snow = []
rain = []

print("\nCalculating snow and rain...\n")

l = np.shape(prec)[0]
print_progress_bar(0, l-1, prefix='', suffix='', decimals=1, length=50, fill='█', print_end="\r")
for i in np.arange(l):
    pr_temp = prec[i, :]
    te_temp = at2m[i, :]

    # handle rain
    temp = np.zeros(np.shape(prec)[1])
    temp[te_temp > 273.15] = pr_temp[te_temp > 273.15]
    rain.append(temp)

    # handle snow
    temp = np.zeros(np.shape(prec)[1])
    temp[te_temp <= 273.15] = pr_temp[te_temp <= 273.15]
    snow.append(temp)

    print_progress_bar(i, l-1, prefix='', suffix='', decimals=1, length=50, fill='█', print_end="\r")
# end for i

snow = np.stack(snow, axis=0)
rain = np.stack(rain, axis=0)


#%% convert NaNs and unreasonable values to 0
# --> technically there should be none, so I hope this is justified
# --> most of those are values slightly < 0
snow_nan = np.sum(np.isnan(snow))
snow_0 = np.sum(snow < 0)
snow_1e10 = np.sum(snow > 1e10)
rain_nan = np.sum(np.isnan(rain))
rain_0 = np.sum(rain < 0)
rain_1e10 = np.sum(rain > 1e10)
snow_ind = np.isnan(snow) | (snow < 0) | (snow > 1e10)
rain_ind = np.isnan(rain) | (rain < 0) | (rain > 1e10)

snow_sum = np.sum(snow_ind)
rain_sum = np.sum(rain_ind)

if np.sum(snow_ind) > 0:
    print(f"\nSnow contains {snow_sum} = {snow_sum/snow.size*100:.1f}% invalid values. Setting them to zero.")
    print("The invalid values are constituted of:")
    print(f"{snow_nan} = {snow_nan/snow.size*100:.1f}% are NaN")
    print(f"{snow_0} = {snow_0/snow.size*100:.1f}% < 0")
    print(f"{snow_1e10} = {snow_1e10/snow.size*100:.1f}% > 1e10\n")

if np.sum(rain_ind) > 0:
    print(f"\nRain contains {rain_sum} = {rain_sum/rain.size*100:.1f}% invalid values. Setting them to zero.")
    print("The invalid values are constituted of:")
    print(f"{rain_nan} = {rain_nan/rain.size*100:.1f}% are NaN")
    print(f"{rain_0} = {rain_0/rain.size*100:.1f}% < 0")
    print(f"{rain_1e10} = {rain_1e10/rain.size*100:.1f}% > 1e10\n")
# end if

snow[snow_ind] = 0
rain[rain_ind] = 0


#%% fill variables
try:
    nc.variables["snow"][:] = snow
    nc.variables["rain"][:] = rain
except:
    # add the snow and rain data to the nc as new variables
    nc_snow = nc.createVariable("snow", np.float64, ('loc', 'time'))
    nc_rain = nc.createVariable("rain", np.float64, ('loc', 'time'))
    nc_snow[:] = snow
    nc_rain[:] = rain
# end try except


#%% because they are corrected refill the total precip with the sum of snow and rain
# nc.variables["precipitation_amount_hourly"][:] = rain + snow

# --> NOT IMPLEMENTED --> because I don't want to change the original data I perform this same operation LATER in the
#                         calculation of the predictors (see calc_prec_pred)


#%% close the nc files
nc.close()