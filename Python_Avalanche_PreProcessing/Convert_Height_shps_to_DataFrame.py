#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert the shp files with the gridcells above threshold info to usual dataframes.
"""


#%% imports
import glob
import geopandas as gpd
import argparse

from Lists_and_Dictionaries.Paths import path_par


#%% set up the parser
parser = argparse.ArgumentParser(
                    prog="Convert height shapes to dataframe",
                    description="""Convert the height shape to a dataframe so that one does not need geopandas.""")

# ...and add the arguments
parser.add_argument('--reg_code', default=3009, type=int, help='Code of the region.')
parser.add_argument("--h_low", type=int, default=400, help="The lower threshold of the grid cell altitude.")
parser.add_argument("--h_hi", type=int, default=1300, help="The upper threshold of the grid cell altitude.")
args = parser.parse_args()


#%% get the parameters from the parser
reg_code = args.reg_code
h_low = args.h_low
h_hi = args.h_hi


#%% loop over the regions
# for reg_code in reg_codes:

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
f_name = glob.glob(shp_path + f"*{h_low}_and_{h_hi}m*.shp")[0]
shp = gpd.read_file(f_name)

shp[["lon", "lat"]].to_csv(f_name[:-3] + "csv", float_format='%.17g')

