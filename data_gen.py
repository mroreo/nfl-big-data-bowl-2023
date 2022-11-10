# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 21:51:14 2022

@author: ben
"""
import pandas as pd
import numpy as np
from pocket_area import gen_pocket_poly
from plots import plot_game_play_id, plot_rusher_distances

GAMES_PATH='./data/games.csv'
PFF_SCOUTING_DATA_PATH='./data/pffScoutingdata.csv'
PLAYERS_PATH='./data/players.csv'
PLAYS_PATH='./data/plays.csv'
WEEK_PATH='./data/week{}.csv'

class NFLDataGenerator():
    
    def __init__(self, week_ids, games_path=GAMES_PATH, pff_scouting_data_path=PFF_SCOUTING_DATA_PATH,
                 players_path=PLAYERS_PATH, plays_path=PLAYS_PATH, week_path=WEEK_PATH):
        self.week_ids = week_ids
        self.games_path = games_path
        self.pff_scouting_data_path = pff_scouting_data_path
        self.players_path = players_path
        self.plays_path = plays_path
        self.week_path = week_path
        
        self.process_dfs()
        
    def process_dfs(self):
        
        games_df = pd.read_csv(self.games_path)
        plays_df = pd.read_csv(self.plays_path)
        players_df = pd.read_csv(self.players_path)
        pff_scouting_df = pd.read_csv(self.pff_scouting_data_path)
        
        games_df = games_df.query('week.isin(@self.week_ids)')
        plays_df = plays_df.merge(games_df, how='inner', on=['gameId'])
        pff_scouting_df = pff_scouting_df.merge(games_df['gameId'], how='inner', on='gameId', validate='m:1')
        
        week_df = []
        for week_id in self.week_ids:
            week_df.append(pd.read_csv(self.week_path.format(week_id)))
        week_df = pd.concat(week_df)
        
        week_df = (week_df
                   .merge(players_df, how='left', on='nflId', validate='m:1')
                   .merge(pff_scouting_df, how='left', on=['nflId', 'gameId', 'playId'], validate='m:1')
                   )
        
        pocket_poly_df = gen_pocket_poly(week_df)
        
        self.games_df = games_df
        self.plays_df = plays_df
        self.players_df = players_df
        self.pff_scouting_df = pff_scouting_df
        self.week_df = week_df
        self.pocket_poly_df = pocket_poly_df
    
    def get_play(self, gameId, playId):
        return self.plays_df.query('(gameId == @gameId) & (playId == @playId)')
    
    def animate_play_id(self, gameId, playId):
        plot_game_play_id(self.week_df, gameId, playId)
