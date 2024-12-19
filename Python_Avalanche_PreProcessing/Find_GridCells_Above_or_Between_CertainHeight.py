"""
Find the grid cells on the NORA3 grid between given elevation thresholds.
If the NorCP data is regreidded to the NORA3 grid this can of course also be used for the NorCP perdictor generation.
"""


#%% imports
import os
from netCDF4 import Dataset
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import argparse

from Lists_and_Dictionaries.Paths import path_par, obs_path


#%% set up the parser
parser = argparse.ArgumentParser(
                    prog="Find Grid Cells NORA3 Grid",
                    description="""Find the grid cells on the NORA3 grid between the given elevation thresholds.""")

# ...and add the arguments
parser.add_argument('--reg_code', default=3009, type=int, help='Code of the region.')
parser.add_argument("--h_low", type=int, default=400, help="The lower threshold of the grid cell altitude.")
parser.add_argument("--h_hi", type=int, default=1300, help="The upper threshold of the grid cell altitude.")
args = parser.parse_args()


#%% get the parameters from the parser
reg_code = args.reg_code
h_low = args.h_low
h_hi = args.h_hi


#%% set the x and y indices to extract the Northern Norway subset
# x1, y1, x2, y2 = 390, 800, 530, 1050  # OLD
x1, y1, x2, y2 = 360, 830, 460, 950


#% connect region code and name
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


#%% set paths and filenames
nora3_path = f"/{path_par}/IMPETUS/NORA3/Cells_Above_Thres_Height/NorthernNorway_Subset/{region}/"
betw_path = f"/{path_par}/IMPETUS/NORA3/Cells_Between_Thres_Height/NorthernNorway_Subset/{region}/"
shp_path = f"/{obs_path}/IMPETUS/Debmita/AvalancheRegion_shapefiles/"
data_path = f"/{path_par}/IMPETUS/NORA3/"
f_name = "fc2021030100_003_fp.nc"


#% load the shape file and the NORA3 file for the grid
shp = gpd.read_file(shp_path + "avalanche_regions.shp")
n_and_r = []
for i, j in zip(shp.omrNavn, shp.region):
    element = f"{i} ({j})"
    n_and_r.append(element)
# end for i, j
shp["N&R"] = n_and_r
shp_north = shp[shp.region < 3017][["geometry", "region"]]

tromsoe_shp = gpd.read_file(shp_path + "Tromsoe_Shape.shp")
tromsoe_shp["region"] = 3011


#%% concatenate the tromsoe_shp with the shp
shp = pd.concat([shp[["geometry", "region"]], tromsoe_shp[["geometry", "region"]]], axis=0)


#%% load the nc file
nc = Dataset(data_path + f_name)


#%% check the variables
# print(nc)


#%% extract the projection
crs = nc.variables["projection_lambert"].proj4


#%% try to reproject the shape files
shp = shp.to_crs(crs)


#%% load the surface geopotential
print(nc.variables["surface_geopotential"])
sgz = np.squeeze(nc.variables["surface_geopotential"][:]) / 9.81

# extract the subset
sgz = sgz[y1:y2, x1:x2]


#% load lat and lon
lons = nc.variables["longitude"][y1:y2, x1:x2]
lats = nc.variables["latitude"][y1:y2, x1:x2]


#% extract the x and y coordinates as flat arrays
arr1x = np.ravel(lons)
arr1y = np.ravel(lats)

xs, ys = np.meshgrid(nc.variables["x"][x1:x2], nc.variables["y"][y1:y2])

arr1x = xs.ravel()
arr1y = ys.ravel()


#%% using the X and Y columns, build a dataframe, then the geodataframe
df1 = pd.DataFrame({'X':arr1x, 'Y':arr1y, "lon":lons.ravel(), "lat":lats.ravel(), "sgz":np.ravel(sgz)})
df1['coords'] = list(zip(df1['X'], df1['Y']))
df1['coords'] = df1['coords'].apply(Point)
gdf1 = gpd.GeoDataFrame(df1, geometry='coords')
gdf1 = gdf1.set_crs(crs)


#%%  try to find the intersect of the NORA3 grid with one of the avalanche region shapes
df_join = gpd.sjoin(left_df=shp[shp["region"] == reg_code], right_df=gdf1, how="left", predicate="intersects")


#%% try to convert this dataframe to one that can be plotted
df2 = pd.DataFrame({'X':np.array(df_join.X), 'Y':np.array(df_join.Y), "lon":np.array(df_join.lon),
                    "lat":np.array(df_join.lat), "sgz":np.array(df_join.sgz)})

df2['coords'] = list(zip(df2['X'], df2['Y']))
df2['coords'] = df2['coords'].apply(Point)
gdf2 = gpd.GeoDataFrame(df2, geometry='coords')
gdf2 = gdf2.set_crs(crs)


#%% attempt a plot
# gdf2.plot("sgz", cmap="Reds", legend=True)


#%% extract those coordinates for which the elevation is above Xm
# df_sgz_hi = gdf2[gdf2.sgz > h_thres]
df_sgz_betw = gdf2[(gdf2.sgz > h_low) & (gdf2.sgz < h_hi)]


#%% test plot
# df_sgz_hi.plot("sgz", cmap="Reds", legend=True)
# df_sgz_betw.plot("sgz", cmap="Reds", legend=True)


#%% store the dataset
os.makedirs(nora3_path, exist_ok=True)

gdf2.to_file(nora3_path + f"NORA3_Coords_in_{region}_Shape.shp")

# df_sgz_hi.to_file(nora3_path + f"NORA3_above{h_thres}m_Coords_in_{region}_Shape.shp")

os.makedirs(betw_path, exist_ok=True)
df_sgz_betw.to_file(betw_path + f"NORA3_between{h_low}_and_{h_hi}m_Coords_in_{region}_Shape.shp")

