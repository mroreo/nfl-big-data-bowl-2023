# -*- coding: utf-8 -*-
"""
Created on Sat Dec 31 13:29:18 2022

@author: ben
"""

import pandas as pd
import numpy as np

GAMES_FPATH = './data/games.csv'
PFF_SCOUTING_DATA_FPATH='./data/pffScoutingdata.csv'
PLAYERS_FPATH='./data/players.csv'
PLAYS_FPATH='./data/plays.csv'
WEEK_FPATH='./data/week{}.csv'

SNAP_EVENTS = ['ball_snap', 'autoevent_ballsnap']
NONSNAP_EVENTS = ['autoevent_passforward', 'pass_forward', 'qb_sack', 'autoevent_passinterrupted',
                  'pass_tipped', 'qb_strip_sack', 'autoevent_passinterrupted']

def filter_plays(plays_df,
                 pass_results=['I','C','IN'], 
                 drop_back_types=['TRADITIONAL']):
    """
    Filter plays specific to the analysis we are interested in looking at.

    Parameters
    ----------
    plays_df : df
        the plays dataframe to filter for specific play types.
    pass_results : list, optional
        keep only plays that have these pass results. The default is ['I','C','IN'].
    drop_back_types : TYPE, optional
        keep only plays that have these drop back types. The default is ['TRADITIONAL'].

    Returns
    -------
    plays_df : TYPE
        DESCRIPTION.

    """
    plays_df = (plays_df
        .query('(pff_playAction == 0) & (passResult.isin(@pass_results))')
        .query('foulName1.isna()')
        .query('dropBackType.isin(@drop_back_types)'))
    return plays_df

def gen_snap_df(week_df_merged, 
                snap_events=SNAP_EVENTS):
    """

    Generate a dataframe describing the frame id and the time of the snap.
    
    Parameters
    ----------
    week_df_merged : df
        the weekly frame by frame dataframe merged with players_df, pff_scouting_df, plays_df
    snap_events : list, optional
        the events that determine the frame of the ball snap 
        default: ['ball_snap', 'autoevent_ballsnap']

    Returns
    -------
    snap_df : df
        a dataframe determining when the snap occurs

    """
    snap_df = (week_df_merged
               .query('event.isin(@snap_events)')
               .groupby(['gameId','playId'])
               [['frameId', 'time']].min()
               .reset_index()
               .rename({'frameId': 'snap_frameId',
                        'time': 'snap_time'}, axis=1))
    return snap_df

def gen_nonsnap_df(week_df_merged, 
                   nonsnap_events=NONSNAP_EVENTS):
    """
    
    Generate a dataframe describing the frame id and the time of the next event after the snap.

    Parameters
    ----------
    week_df_merged : df
        the weekly frame by frame dataframe merged with players_df, pff_scouting_df, plays_df
    nonsnap_events : list, optional
        the events that determine the frame of the ball snap. 
        default: ['autoevent_passforward', 'pass_forward', 'qb_sack', 'autoevent_passinterrupted',
                  'pass_tipped', 'qb_strip_sack', 'autoevent_passinterrupted']

    Returns
    -------
    nonsnap_df : df
        a dataframe determining when the next event occurs after the snap

    """
    nonsnap_df = (week_df_merged
               .query('event.isin(@nonsnap_events)')
               .groupby(['gameId','playId'])
               [['frameId','time']].min()
               .reset_index()
               .rename({'frameId': 'nonsnap_frameId',
                        'time': 'nonsnap_time'},axis=1))
    return nonsnap_df


def gen_passer_df(week_df_merged):
    """
    
    Generate a dataframe regarding the qb information.

    Parameters
    ----------
    week_df_merged :df
        the weekly frame by frame dataframe merged with players_df, pff_scouting_df, plays_df

    Returns
    -------
    passer_df : df
        a dataframe indicating the qb's location

    """
    passer_df = (week_df_merged.query('pff_role == "Pass"')
                 [['gameId', 'playId','frameId','time', 'x', 'y', 's','a','dis','o','dir']]
                 .rename({'x': 'qb_x',
                          'y': 'qb_y',
                          's': 'qb_s',
                          'a': 'qb_a',
                          'o': 'qb_o',
                          'dis': 'qb_dis',
                          'dir': 'qb_dir'}, axis=1))
    return passer_df

class NFLData:
    
    def __init__(self, 
                 games_fpath=GAMES_FPATH, 
                 pff_scouting_data_fpath=PFF_SCOUTING_DATA_FPATH, 
                 players_fpath=PLAYERS_FPATH, 
                 plays_fpath=PLAYS_FPATH, 
                 week_fpath=WEEK_FPATH):
        
        
        self.games_fpath = games_fpath
        self.pff_scouting_data_fpath = pff_scouting_data_fpath
        self.players_fpath = players_fpath
        self.plays_fpath = plays_fpath
        self.week_fpath = week_fpath
        
    def load_data(self, week_ids):
        
        games_df = pd.read_csv(self.games_fpath)
        plays_df = pd.read_csv(self.plays_fpath)
        players_df = pd.read_csv(self.players_fpath)
        pff_scouting_df = pd.read_csv(self.pff_scouting_data_fpath)
        
        games_df = games_df.query('week.isin(@week_ids)')
        plays_df = plays_df.merge(games_df, how='inner', on=['gameId'])
        pff_scouting_df = pff_scouting_df.merge(games_df['gameId'], how='inner', on='gameId', validate='m:1')
        
        week_df = []
        for week_id in week_ids:
            week_df.append(pd.read_csv(self.week_fpath.format(week_id)))
        week_df = pd.concat(week_df)
        
        plays_df = filter_plays(plays_df)
        
        week_df_merged = (week_df
                   .merge(players_df, how='left', on='nflId', validate='m:1')
                   .merge(pff_scouting_df, how='left', on=['nflId', 'gameId', 'playId'], validate='m:1')
                   .merge(plays_df, how='inner', on=['gameId', 'playId'], validate='m:1')
                   )
        
        snap_df = gen_snap_df(week_df_merged)
        nonsnap_df = gen_nonsnap_df(week_df_merged)
        passer_df = gen_passer_df(week_df_merged)
        
        self.games_df = games_df
        self.plays_df = plays_df
        self.players_df = players_df
        self.pff_scouting_df = pff_scouting_df
        self.week_df_merged = week_df_merged
        self.snap_df = snap_df
        self.nonsnap_df = nonsnap_df
        self.passer_df = passer_df
        
if __name__ == '__main__':
    
    NFL = NFLData()
    NFL.load_data(week_ids=[1,2,3,4])
        