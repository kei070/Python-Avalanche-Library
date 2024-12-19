#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The features.
"""

#%% imports
import sys
import glob
import pandas as pd
from .Paths import path_par


#%% load one of the files to and get all columns
path = f"{path_par}/IMPETUS/NORA3/Avalanche_Predictors_Full_TimeSeries/Mean/Between400_and_900m/"


#%% single line
try:
    features = list(pd.read_csv(glob.glob(path + "*.csv")[0], index_col=0).columns)
except IndexError:
    print("\nIndexError in the Features.py script:\n")
    print("No file was found to load the features from. Make sure that there are full time-series predictor files in:")
    print(path + "\n")
    print("Stopping execution.")
    sys.exit()
# end try except


#%% set the plot names for the features
feats_pl = {'t_mean':"t1", 't_max':"tmax", 't_min':"tmin", 't_range':"t_range",
            'ftc':"ftc",
            't2':"t2", 't3':"t3",'t4':"t4", 't5':"t5", 't6':"t6", 't7':"t7",
            'tmax2':'tmax2', 'tmax3':'tmax3', 'tmax4':'tmax4', 'tmax5':'tmax5', 'tmax6':'tmax6', 'tmax7':'tmax7',
            'dtemp1':'dtemp1', 'dtemp2':'dtemp2', 'dtemp3':'dtemp3',
            'dtempd1':'dtempd1', 'dtempd2':'dtempd2', 'dtempd3':'dtempd3',
            'pdd':"pdd",
            'ws_mean':"w1", 'ws_max':"wmax", 'ws_min':"wmin", "ws_range":"dws",
            "dws1":"dws1", "dws2":"dws2", "dws3":"dws3",
            "dwsd1":"dwsd1", "dwsd2":"dwsd2", "dwsd3":"dwsd3",
            'wind_direction':"w_dir", "dwdir":"dwdir",
            "dwdir1":"dwdir1", "dwdir2":"dwdir2", "dwdir3":"dwdir3",
            'w2':"w2", 'w3':"w3", 'w4':"w4", 'w5':"w5", 'w6':"w6", 'w7':"w7",
            'wmax2':'wmax2', 'wmax3':'wmax3', 'wmax4':'wmax4', 'wmax5':'wmax5', 'wmax6':'wmax6', 'wmax7':'wmax7',
            'total_prec_sum':"Ptot",
            's1':'s1', 'r1':'r1',
            'r2':'r2', 'r3':'r3', 'r4':'r4', 'r5':'r5', 'r6':'r6', 'r7':'r7',
            's2':'s2', 's3':'s3', 's4':'s4', 's5':'s5', 's6':'s6', 's7':'s7',
            'wdrift':'wdrift', 'wdrift_2':'wdrift_2', 'wdrift_3':'wdrift_3',
            'wdrift3':'wdrift3', 'wdrift3_2':'wdrift3_2', 'wdrift3_3':'wdrift3_3',
            'RH':"rh",
            'NLW':"nlw", 'NSW':"nsw",
            'RH2':"rh2", 'RH3':"rh3", 'RH4':"rh4", 'RH5':"rh5", 'RH6':"rh6", 'RH7':"rh7",
            'NLW2':"nlw2", 'NLW3':"nlw3", 'NLW4':"nlw4",'NLW5':"nlw5",'NLW6':"nlw6",'NLW7':"nlw7",
            'NSW2':"nsw2", 'NSW3':"nsw3", 'NSW4':"nsw4", 'NSW5':"nsw5", 'NSW6':"nsw6", 'NSW7':"nsw7",
            'SWE':"SWE", 'SnowDepth':"SDP", 'SnowDens':"SD", 'MeltRefr':"MR",
            'SWE2':"SWE2", 'SWE3':"SWE3", 'SWE4':"SWE4", 'SWE5':"SWE5", 'SWE6':"SWE6", 'SWE7':"SWE7",
            'SnowDepth2':"SDP2", 'SnowDepth3':"SDP3", 'SnowDepth4':"SDP4", 'SnowDepth5':"SDP5",
            'SnowDepth6':"SDP6", 'SnowDepth7':"SDP7",
            'SnowDens2':"SD2", 'SnowDens3':"SD3", 'SnowDens4':"SD4", 'SnowDens5':"SD5",
            'SnowDens6':"SD6", 'SnowDens7':"SD7",
            'MeltRefr2':"MR2", 'MeltRefr3':"MR3", 'MeltRefr4':"MR4", 'MeltRefr5':"MR5",
            'MeltRefr6':"MR6", 'MeltRefr7':"MR7", "ava_clim":"aci"}


#%% old
"""
features = ['t_mean', 't_max', 't_min', 't_range',
            'ftc',
            't2', 't3','t4', 't5', 't6', 't7',
            'tmax2', 'tmax3', 'tmax4', 'tmax5', 'tmax6', 'tmax7',
            'dtemp1', 'dtemp2', 'dtemp3',
            'dtempd1', 'dtempd2', 'dtempd3',
            'pdd',
            'ws_mean', 'ws_max', 'wind_direction', "ws_range",
            'w2', 'w3', 'w4', 'w5', 'w6', 'w7',
            'wmax2', 'wmax3', 'wmax4', 'wmax5', 'wmax6', 'wmax7',
            'total_prec_sum',
            's1', 'r1',
            'r2', 'r3', 'r4', 'r5', 'r6', 'r7',
            's2', 's3', 's4', 's5', 's6', 's7',
            'wdrift', 'wdrift_2', 'wdrift_3', 'wdrift3', 'wdrift3_2', 'wdrift3_3',
            'RH',
            'NLW', 'NSW',
            'RH2', 'RH3', 'RH4', 'RH5', 'RH6', 'RH7',
            'NLW2', 'NLW3', 'NLW4','NLW5', 'NLW6', 'NLW7',
            'NSW2', 'NSW3', 'NSW4', 'NSW5', 'NSW6', 'NSW7',
            'SWE', 'SnowDepth', 'SnowDens', 'MeltRefr',
            'SWE2', 'SWE3', 'SWE4', 'SWE5', 'SWE6', 'SWE7',
            'SnowDepth2', 'SnowDepth3', 'SnowDepth4', 'SnowDepth5', 'SnowDepth6', 'SnowDepth7',
            'SnowDens2', 'SnowDens3', 'SnowDens4', 'SnowDens5', 'SnowDens6', 'SnowDens7',
            'MeltRefr2', 'MeltRefr3', 'MeltRefr4', 'MeltRefr5', 'MeltRefr6', 'MeltRefr7']
"""
