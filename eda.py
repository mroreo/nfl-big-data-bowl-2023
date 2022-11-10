# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 22:40:04 2022

@author: ben
"""
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from data_gen import NFLDataGenerator

week_1 = NFLDataGenerator(week_ids=[1])

pass_rusher_df = week_1.week_df.query('pff_role == "Pass Rush"')
pass_rusher_df = gpd.GeoDataFrame(pass_rusher_df, geometry=gpd.points_from_xy(pass_rusher_df['x'], pass_rusher_df['y']))
pass_rusher_df = pass_rusher_df.merge(week_1.pocket_poly_df, how='left',on=['gameId', 'playId', 'frameId'])
pass_rusher_df['pocket_penetration'] = gpd.GeoSeries(pass_rusher_df['geometry']).within(gpd.GeoSeries(pass_rusher_df['pocket_polygon']))

idx = 6
gameId = week_1.plays_df.iloc[idx]['gameId']
playId = week_1.plays_df.iloc[idx]['playId']
playDescription = week_1.plays_df.iloc[idx]['playDescription']

pressure_info_df = (pass_rusher_df
     .query('pocket_penetration')
     .groupby(['gameId','playId'])
     .apply(lambda x: pd.Series({'time_of_penetration': x['time'].min(),
                                 'number_of_pressures': len(x['nflId'].unique())}))
     .reset_index())

snap_events = ['ball_snap', 'autoevent_ballsnap']
snap_df = (week_1.week_df
           .query('event.isin(@snap_events)')
           .groupby(['gameId','playId'])
           ['time'].min()
           .reset_index())

nonsnap_events = ['ball_snap', 'autoevent_ballsnap','None']
nonsnap_df = (week_1.week_df
           .query('~event.isin(@snap_events)')
           .groupby(['gameId','playId'])
           ['time'].min()
           .reset_index()
           .rename({'time':'nonsnap_time'},axis=1))

pressure_df = (snap_df
               .merge(pressure_info_df, how='left', on=['gameId','playId'])
               .merge(nonsnap_df, how='left', on=['gameId','playId'])
               .assign(
                   time_til_penetration = lambda x: (pd.to_datetime(x['time_of_penetration']) - pd.to_datetime(x['time'])).dt.total_seconds))
