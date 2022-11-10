# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 23:47:09 2022

@author: ben
"""

import pandas as pd
import numpy as np
from data_gen import NFLDataGenerator
from plots import plot_game_play_id, plot_rusher_distances
from distances import calc_dist

pd.set_option('display.max_columns', None)

BALLSNAP_EVENTS = ['autoevent_ballsnap', 'ball_snap']

def double_team_pressure_ind(df_grp):
    
    def _agg_pressure_ind(df_grp2):
    
        if np.min(df_grp2['pb_dist_to_qb']) <= np.min(df_grp2['pr_dist_to_qb']):
            return pd.Series({'pressure_ind': 0})
        else:
            return pd.Series({'pressure_ind': 1})
    
    return df_grp.groupby(['playId','frameId']).apply(lambda x: _agg_pressure_ind(x))

def extract_pressure_time(nfl_data_gen, ballsnap_events=BALLSNAP_EVENTS):
    merged_week_df = nfl_data_gen.week_df

    qb_df = (merged_week_df
             .query('pff_role == "Pass"')
             [['gameId', 'playId', 'nflId','frameId', 'x', 'y']]
             .rename({'x': 'qb_x', 'y': 'qb_y', 'nflId': 'qbnflId'}, axis = 1))
    pass_rusher_df = (merged_week_df
                      .query('pff_role == "Pass Rush"')
                      [['gameId','playId','nflId','frameId','x','y']]
                      .rename({'x': 'pr_x', 'y': 'pr_y','nflId': 'pr_nflId'}, axis = 1))
    pass_blocker_df = (merged_week_df
                       .query('pff_role == "Pass Block"')
                       [['gameId','playId','nflId','frameId','x','y', 'pff_nflIdBlockedPlayer']]
                       .rename({'x': 'pb_x', 'y': 'pb_y', 'nflId': 'pb_nflId', 'pff_nflIdBlockedPlayer':'pr_nflId'}, axis = 1))

    pressure_df = (pass_blocker_df
                   .merge(pass_rusher_df, how='inner', on=['gameId','playId','frameId','pr_nflId'])
                   .merge(qb_df, how='inner', on=['gameId','playId','frameId'])
                   .assign(
                       pr_dist_to_qb = lambda x: calc_dist(x['pr_x'], x['pr_y'], x['qb_x'], x['qb_y']),
                       pb_dist_to_qb = lambda x: calc_dist(x['pb_x'], x['pb_y'], x['qb_x'], x['qb_y'])
                       )
                   .groupby(['gameId','pr_nflId'])
                   .apply(lambda x: double_team_pressure_ind(x))
                   .reset_index())

    pressure_times_df = pressure_df.query('pressure_ind == 1').groupby(['gameId','playId','pr_nflId'])['frameId'].min().reset_index()
    
    snap_df = nfl_data_gen.week_df[['gameId', 'playId', 'frameId', 'time', 'event']].query('event.isin(@ballsnap_events)').drop_duplicates()
    snap_df = snap_df.groupby(['gameId', 'playId'])['frameId', 'time'].min().reset_index().rename({'frameId':'snapFrameId', 'time':'snapTime'},axis=1)
    frame_df = nfl_data_gen.week_df[['gameId', 'playId', 'frameId', 'time']].drop_duplicates().rename({'time':'frameTime'}, axis=1)
    
    pressure_times_df = pressure_times_df.merge(snap_df, how='left', on=['gameId', 'playId'])
    pressure_times_df = pressure_times_df.merge(frame_df, how='left', on=['gameId', 'playId', 'frameId'])
    pressure_times_df = pressure_times_df.query('snapTime < frameTime')
    pressure_times_df['pressure_time_s'] = (pd.to_datetime(pressure_times_df['frameTime']) - pd.to_datetime(pressure_times_df['snapTime'])).dt.total_seconds()
    
    return pressure_times_df


def extract_pressure_creator(nfl_data_gen, pressure_times_df):
    time_until_pressure_df = pressure_times_df.groupby(['gameId','playId'])['pressure_time_s'].min().reset_index()
    pressure_creator_df = time_until_pressure_df.merge(pressure_times_df, how='left',on=['gameId', 'playId','pressure_time_s'])
    pressure_creator_df = pressure_creator_df.groupby('pr_nflId').apply(lambda x: pd.Series({'pressure_cts': len(x),
                                                                       'avg_pressure_time_s': x['pressure_time_s'].mean()})).reset_index()
    pressure_creator_df = pressure_creator_df.merge(nfl_data_gen.players_df[['displayName', 'nflId']], how='left', left_on='pr_nflId', right_on='nflId')
    return pressure_creator_df

if __name__ == '__main__':
    nfl_data_gen = NFLDataGenerator(week_ids=[1])
    pressure_times_df = extract_pressure_time(nfl_data_gen)
    pressure_creator_df = extract_pressure_creator(nfl_data_gen, pressure_times_df)
    
    gameId = 2021090900	
    playId = 3406
    plot_game_play_id(nfl_data_gen.week_df, gameId, playId)
    
    play_df = nfl_data_gen.get_play(gameId, playId)
    
    plot_rusher_distances
    
