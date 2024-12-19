#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This represents and attempt to implement the helper function solar_norm_vo111 from the seNorge model.

The model R-code was downloaded from https://ars.els-cdn.com/content/image/1-s2.0-S0022169416301755-mmc1.zip

Copied from the orginial R-script:

Daily normalized (to 60 deg N) (0, 1) potential solar radiation
from Walter et al. (2005)
by Tuomo Saloranta, Norwegian Water Resources and Energy Directorate (NVE), 290612
"""

# imports
import numpy as np

# function
def solar_norm_vo111(lat_deg, DoY):  #  DoY so that Jan 1 = 1

    """
    Calculate the daily normalized (to 60 deg N) (0, 1) potential solar radiation

    Parameters:
        lat_deg  Float. Latitude of the location for which the snow will be modelled.
        DoY      Integer. Day of year such that January 1st is 1.
    """

    # take pi from numpy to use it later on; note that this may lead to differences in the results if the original
    # R-code uses a different value for pi
    pi = np.pi
    # pi = 3.141593  # this is the value that R reports when one accesses the predefined variable pi

    # solar declination [rad]
    s_dec = 0.4102 * np.sin(2 * pi * (DoY-80) /365)

    # convert latitude into radiants
    lat = pi * lat_deg / 180

    # solar radiation on a horizontal plane at the top of atmosphere in MJ/day/m2
    SW_top = (117.5/pi)*(np.arccos(-np.tan(s_dec) * np.tan(lat)) * np.sin(s_dec) * np.sin(lat) +
                         np.cos(s_dec) * np.cos(lat) * np.sin(np.arccos(-np.tan(s_dec) * np.tan(lat))))
    # print(SW_top)

    # = Exceptions as polar night midningth sun gives NA

    # === polar night NA --> 0
    pol_night_inx = ( (~np.isfinite(SW_top)) & (np.sin(2 * pi * (DoY-81) / 365) < 0) )

    if pol_night_inx:
        SW_top = 0
    # end if

    # === midnight sun NA --> SW_top at 66 degN (rough assumption!!!!)
    midnight_sun_inx = ( (~np.isfinite(SW_top)) & (np.sin(2 * pi * (DoY-80) / 365) >= 0) )

    lat_66 = pi * 66 / 180
    SW_top_66 = (117.5/pi)*( np.arccos(-np.tan(s_dec)*np.tan(lat_66))*np.sin(s_dec)*np.sin(lat_66) +
                            np.cos(s_dec)*np.cos(lat_66)*np.sin(np.arccos(-np.tan(s_dec)*np.tan(lat_66))) )

    if midnight_sun_inx:
        SW_top = SW_top_66
    # end if

    # daily normalized (to 60 deg N) (0, 1) potential solar radiation
    SW_norm = SW_top/42.58778

    return SW_norm
# end def