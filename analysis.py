# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 22:17:39 2022

@author: ben
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from data_gen import NFLDataGenerator
from plots import plot_game_play_id, plot_rusher_distances
from pocket_area import extract_pocket_area
from time_until_pressure import extract_pressure_creator, extract_pressure_time

all_plays_df = []
# all_pressure_times_df = []
# all_pressure_creator_df = []
all_pocket_area_df = []
for w in range(1, 9):
    nfl_data_gen = NFLDataGenerator(week_ids=[w])
    # pressure_times_df = extract_pressure_time(nfl_data_gen)
    # pressure_creator_df = extract_pressure_creator(nfl_data_gen, pressure_times_df)
    pocket_area_df = extract_pocket_area(nfl_data_gen)
    # pressure_times_df['week'] = w
    # pressure_creator_df['week'] = w
    pocket_area_df['week'] = w
    
    # all_pressure_times_df.append(pressure_times_df)
    # all_pressure_creator_df.append(pressure_creator_df)
    all_plays_df.append(nfl_data_gen.plays_df.query('pff_playAction == 0'))
    all_pocket_area_df.append(pocket_area_df)
    
# all_pressure_times_df = pd.concat(all_pressure_times_df)
# all_pressure_creator_df = pd.concat(all_pressure_creator_df)
all_plays_df = pd.concat(all_plays_df)
all_pocket_area_df = pd.concat(all_pocket_area_df)

pocket_area_df = (all_pocket_area_df.merge(
    all_pocket_area_df.groupby(['gameId', 'playId'])['_pocket_area'].max().reset_index(),
    how='inner',on=['gameId', 'playId', '_pocket_area'])