#%% imports
import sys
import glob
import numpy as np
import xarray as xr
import argparse
from netCDF4 import Dataset
from dask.diagnostics import ProgressBar

from Lists_and_Dictionaries.Paths import path_par


#%% set data path
data_path = f"{path_par}/IMPETUS/NORA3/Avalanche_Region_Data/"


#%% parser
parser = argparse.ArgumentParser(
                    description="""Concatenates that annual files of the extracted grid cells from NorCP for the
                    predictors.""")
parser.add_argument('--low', default=400, type=int, help='The lower threshold for the altitude band to extract.')
parser.add_argument('--high', default=900, type=int, help='The upper threshold for the altitude band to extract.')
parser.add_argument('--sta_yr1', default=1970, type=int, help="""Start year 1. First year of start interval. The default
                    start and end years here given set up a series of one-year periods: 1970-1970, 1971-1971, ...,
                    2023-2023, 2024-2024.""")
parser.add_argument('--sta_yr2', default=1970, type=int, help='Start year 2. Last year of start interval.')
parser.add_argument('--end_yr1', default=2024, type=int, help='End year 1. First year of end (or last) interval.')
parser.add_argument('--end_yr2', default=2024, type=int, help='End year 2. End year of end (or last) interval.')

args = parser.parse_args()


#%% get the arguments from the parser
h_low = args.low  # 400  # args.low
h_hi = args.high  # 900  # args.high
sta_yr1 = args.sta_yr1
sta_yr2 = args.sta_yr2
end_yr1 = args.end_yr1
end_yr2 = args.end_yr2


#%% set the extracted time periods
dy1 = sta_yr2 - sta_yr1
dy2 = end_yr2 - end_yr1
# t_pers = [f"{yr1}-{yr2}" for yr1, yr2 in zip(np.arange(sta_yr1, end_yr1+1, dy1), np.arange(sta_yr2, end_yr2+1, dy2))]
t_pers = [f"{yr1}-{yr2}" for yr1, yr2 in zip(np.arange(sta_yr1, end_yr1+1, 1), np.arange(sta_yr2, end_yr2+1, 1))]


#%% set the region list
reg_codes = [3009, 3010, 3011, 3012, 3013]


#%% loop over the regions
for reg_code in reg_codes:

    print(f"\n{reg_code}\n")

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

    # loop over the time periods to load the necessary file names
    fn_list = []
    for t_per in t_pers:
        fn_list.append(sorted(glob.glob(data_path + f"/Between{h_low}_and_{h_hi}m/" + t_per + f"/*{region}*.nc"))[0])
    # end for t_per

    if len(fn_list) == 0:
        print(f"No files available for {reg_code}. Continuing...")
        continue
    # end if
    print()
    print(f"Number of year intervals for {region}: {np.shape(fn_list)}\n")

    # get start and end year
    sta_yr = t_pers[0].split("-")[0]
    end_yr = t_pers[-1].split("-")[1]


    # open the nc-files
    nc = xr.open_mfdataset(fn_list, combine="by_coords", data_vars="minimal")
    with ProgressBar():
        nc.to_netcdf(data_path + f"/Between{h_low}_and_{h_hi}m/NORA3_{region}_Between{h_low}_and_{h_hi}m.nc")
    # end with
# end for reg_code


#%% test
# from netCDF4 import Dataset
# nc = Dataset(data_path + f"/Between{h_low}_and_{h_hi}m/NORA3_{region}_Between{h_low}_and_{h_hi}m.nc")
# print(nc)