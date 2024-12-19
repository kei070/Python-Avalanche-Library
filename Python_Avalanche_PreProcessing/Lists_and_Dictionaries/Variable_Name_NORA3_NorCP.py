#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lists/dictionaries for the NORA3 and NorCP variable names
"""

# general keys
gen_var_name = ["t2m", "prec", "snow", "rain", "rh", "w10m", "wdir", "nlw", "nsw"]

# NORA3 names
varn_nora3_l = ["air_temperature_2m", "precipitation_amount_hourly", "snow", "rain", "relative_humidity_2m",
                "wind_speed", "wind_direction", "surface_net_longwave_radiation", "surface_net_shortwave_radiation"]

# NorCP names
varn_norcp_l = ["tas", "pr", "snow", "rain", "hurs", "wspeed", "wdir", "rlns", "rsns"]

# NORA3 dictionary
varn_nora3 = {k:v for k, v in zip(gen_var_name, varn_nora3_l)}

# NorCP dictionary
varn_norcp = {k:v for k, v in zip(gen_var_name, varn_norcp_l)}

# predictor names
pred_names = {"t2m":{"mean":"t", "min":"tmin", "max":"tmax"}, "rh":{"mean":"rh"},
              "w10m":{"mean":"w", "min":"wmin", "max":"wmax"},
              "wdir":{"mean":"wdir"}, "nlw":{"mean":"nlw"}, "nsw":{"mean":"nsw"}}

# set the 3h variables and the 1h variables
var1h = ["tas", "pr", "snow", "rain", "wspeed", "wdir"]
var3h = ["rlns", "rsns", "hurs"]