# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 22:01:31 2022

@author: ben
"""
import numpy as np
import pandas as pd

def calc_dist(x1, y1, x2, y2):
    return np.sqrt(np.square(x1 - x2) + np.square(y1-y2))

def dist_to_ball(play_df):
    
    football_df = play_df.query('team == "football"')[['gameId', 'playId','frameId','x','y']].rename({'x':'fb_x','y':'fb_y'}, axis=1)
    
    play_df = play_df.merge(football_df, how='left', on=['gameId', 'playId','frameId'])
    play_df['dist_to_fb'] = np.round(calc_dist(play_df['x'],play_df['y'],play_df['fb_x'],play_df['fb_y']),2)
    return play_df

def dist_to_qb(play_df):
    
    qb_df = play_df.query('officialPosition == "QB"')[['gameId', 'playId','frameId','x','y']].rename({'x':'qb_x','y':'qb_y'}, axis=1)
    
    play_df = play_df.merge(qb_df, how='left', on=['gameId', 'playId','frameId'])
    play_df['dist_to_qb'] = np.round(calc_dist(play_df['x'],play_df['y'],play_df['qb_x'],play_df['qb_y']),2)
    return play_df