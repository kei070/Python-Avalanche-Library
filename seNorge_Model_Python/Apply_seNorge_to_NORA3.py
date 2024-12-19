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
import pylab as pl
import cartopy.crs as ccrs
import warnings
from netCDF4 import Dataset, date2num
from joblib import dump, load

from Functions.Func_Apply_seNorge import apply_seNorge
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
end_year = 2024

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


#%% load the restart data
rest_path = f"{path_par}/IMPETUS/seNorge/Restart/"
rest_in = load(rest_path + "seNorge_NORA3_NorthNorway_1970_1975_Restart.joblib")


#%% apply the seNorge model --> use the restart data generated from 1970-1975
swe, sdepth, sdens, melt_ref, rest_data = apply_seNorge(at2m, prec, days, months, years,
                                                        lats=nc.latitude, lons=nc.longitude, zs=topo_nc.ZS,
                                                        lats_to=lats_to, lons_to=lons_to,
                                                        return_rest=True, rest_in=rest_in)


#%% store the restart data
dump(rest_data, rest_path + f"seNorge_NORA3_NorthNorway_{sta_year}_{end_year}_Restart.joblib")


#%% store the data

# Create xarray DataArrays
da_A = xr.DataArray(swe, coords=[time, ys_sub, xs_sub], dims=["time", "y", "x"], name="SWE",
                    attrs={"units":"mm", "long_name":"snow water equivalent"})
da_B = xr.DataArray(sdepth, coords=[time, ys_sub, xs_sub], dims=["time", "y", "x"], name="SnowDepth",
                    attrs={"units":"mm", "long_name":"snow depth"})
da_C = xr.DataArray(sdens, coords=[time, ys_sub, xs_sub], dims=["time", "y", "x"], name="SnowDens",
                    attrs={"units":"kg/L", "long_name":"snow density"})
da_D = xr.DataArray(melt_ref, coords=[time, ys_sub, xs_sub], dims=["time", "y", "x"], name="Melt_Refreeze",
                    attrs={"units":"mm/d", "long_name":"melt or refreeze"})
da_lat = xr.DataArray(lats_sub, coords=[ys_sub, xs_sub], dims=["y", "x"], name="latitude",
                      attrs={"units":"degrees North"})
da_lon = xr.DataArray(lons_sub, coords=[ys_sub, xs_sub], dims=["y", "x"], name="longitude",
                      attrs={"units":"degrees East"})
da_time = xr.DataArray(time, coords=[time], dims=["time"], name="time",
                      attrs={"units":time_info.units})

# Combine into a Dataset
ds = xr.Dataset({
    "SWE": da_A,
    "SnowDepth": da_B,
    "SnowDens": da_C,
    "Melt_Refreeze": da_D,
    "latitude":da_lat,
    "longitude":da_lon,
    "time":time  # probably won't work
})

ds.attrs["calendar"] = time_info.calendar

ds.attrs["title"] = "seNorge output based on NORA3 1970-2023 Troms-Subset"

ds.attrs["description"] = """Output (or rather part of the output) of the seNorge model run on NORA3 data. The seNorge
model was originally written in R by Tuomo Saloranta and translated to Python by Kai-Uwe Eiselt. The R-code is kindly
made publically available at https://ars.els-cdn.com/content/image/1-s2.0-S0022169416301755-mmc1.zip

Relevant publications for seNorge include:
    http://www.the-cryosphere.net/6/1323/2012/tc-6-1323-2012.pdf
    http://www.the-cryosphere-discuss.net/8/1973/2014/tcd-8-1973-2014.pdf
    """

# Save to NetCDF
out_name = f"{path_par}/IMPETUS/seNorge/Data/seNorge_NorthNorway_NORA3_{sta_year}_{end_year}.nc"
ds.to_netcdf(out_name)


#%% open the nc file with netCDF4 and add some attributes
nc = Dataset(out_name, mode="r+")

nc.variables["time"].units = time_info.units
nc.variables["time"].calendar = time_info.calendar
nc.variables["SWE"].long_name = "snow water equivalent"
nc.variables["SWE"].units = "mm"
nc.variables["SnowDepth"].long_name = "snow depth"
nc.variables["SnowDepth"].units = "mm"
nc.variables["SnowDens"].long_name = "snow density"
nc.variables["SnowDens"].units = "kg/L"
nc.variables["Melt_Refreeze"].long_name = "melt or refreeze"
nc.variables["Melt_Refreeze"].units = "mm/d"
nc.variables["latitude"].units = "degrees North"
nc.variables["longitude"].units = "degrees East"

nc.description = """Output (or rather part of the output) of the seNorge model run on NORA3 data. The seNorge model
was originally written in R by Tuomo Saloranta and translated to Python by Kai-Uwe Eiselt. The R-code is kindly made
publically available at https://ars.els-cdn.com/content/image/1-s2.0-S0022169416301755-mmc1.zip

Relevant publications for seNorge include:
    http://www.the-cryosphere.net/6/1323/2012/tc-6-1323-2012.pdf
    http://www.the-cryosphere-discuss.net/8/1973/2014/tcd-8-1973-2014.pdf
    """

nc.title = f"seNorge output based on NORA3 {sta_year}-{end_year} Troms-Subset"

nc.close()


#%% test plots of the results
xs, ys = lons_sub, lats_sub

crs = ccrs.Orthographic(central_longitude=16.0, central_latitude=65.0, globe=None)
for i in [-1]:  # np.arange(2):

    fig = pl.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection=crs)

    p00 = ax.pcolormesh(xs, ys, sdepth[i, :, :]/10, cmap="Blues", transform=ccrs.PlateCarree(), vmin=0, vmax=400)
    cb00 = fig.colorbar(p00, ax=ax)
    cb00.set_label("Snow depth in cm")

    ax.coastlines(resolution='50m', linewidth=0.5)

    ax.set_title(f"{days[i]}-{months[i]}-{years[i]}")

    pl.show()
    pl.close()

# end for


#%% attempt a video animation
"""
ani_path = "/media/kei070/One Touch/IMPETUS/seNorge/Animations/"

fig = pl.figure(figsize=(8, 5))

ax = fig.add_subplot(111, projection=crs)

cax = ax.pcolormesh(xs, ys, sdepth[0, :, :], cmap="Blues", transform=ccrs.PlateCarree(), vmin=0, vmax=250)
cb00 = fig.colorbar(cax, ax=ax)
cb00.set_label("Snow depth in cm")

ax.coastlines(resolution='50m', linewidth=0.5)

title = ax.set_title(f"{days[0]}-{months[0]}-{years[0]}")

# Define the animation update function
def update(frame):
    cax.set_array(sdepth[frame, :, :]/10)
    title.set_text(f"{days[frame]}-{months[frame]}-{years[frame]}")
    return cax,
# end def

# Create and save the animation
ani = FuncAnimation(fig, update, frames=len(time), blit=True)
ani.save(ani_path + 'animation.mp4', writer='ffmpeg', fps=10)
"""