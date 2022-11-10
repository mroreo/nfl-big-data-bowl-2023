# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 22:16:15 2022

@author: ben
"""
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer

import pandas as pd
import numpy as np
import tensorflow as tf
import math
from tensorflow.keras.utils import Sequence, pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import sequence
from keras.layers import LSTM, Dropout, Dense, Embedding, LeakyReLU
from keras.metrics import binary_accuracy
from keras.callbacks import ModelCheckpoint

def calc_dist(x1, y1, x2, y2):
    return np.sqrt(np.square(x1 - x2) + np.square(y1-y2))

class PressureDataGenerator(Sequence):
    
    def __init__(self, week_ids, games_path, pff_scouting_data_path, players_path, plays_path, week_path, timesteps, batch_size, shuffle=True):
        self.week_ids = week_ids
        self.games_path = games_path
        self.pff_scouting_data_path = pff_scouting_data_path
        self.players_path = players_path
        self.plays_path = plays_path
        self.week_path = week_path
        self.timesteps = timesteps
        self.batch_size = batch_size
        self.shuffle = shuffle
        
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
        
        pff_scouting_df['pressure_ind'] = np.where((pff_scouting_df['pff_hit'] == 1) |
                                                   (pff_scouting_df['pff_hurry'] == 1) |
                                                   (pff_scouting_df['pff_sack'] == 1), 1, 0)
        
        pressure_df = pff_scouting_df.groupby(['gameId', 'playId'])['pressure_ind'].max().reset_index()   
        
        self.games_df = games_df
        self.plays_df = plays_df
        self.players_df = players_df
        self.pff_scouting_df = pff_scouting_df
        self.week_df = week_df
        self.pressure_df = pressure_df
    
    def on_epoch_end(self):
        if self.shuffle:
            self.pressure_df = self.pressure_df.sample(frac=1).reset_index(drop=True)
    
    def __len__(self):
        return int(np.floor(len(self.pressure_df) / self.batch_size))
    
    def __gen_features(self, gameId, playId):
        
        play_df = self.week_df.query('(gameId == @gameId) & (playId == @playId)')
        
        ballsnap_frame_id = play_df.query('event.isin(["ball_snap", "autoevent_ballsnap"])')['frameId'].min()
        pass_frame_id = play_df.query('event.isin(["autoevent_passforward", "pass_forward", "qb_sack", "run", "fumble", "qb_strip_sack"])')['frameId'].min()
        
        if math.isnan(ballsnap_frame_id):
            ballsnap_frame_id = min(pass_frame_id - self.timesteps, 5)
        
        ballsnapped_df = play_df.query('(frameId >= @ballsnap_frame_id) & (frameId <= @pass_frame_id)')
        merged_ballsnapped_df = (ballsnapped_df
                                     .merge(self.players_df, how='inner', on='nflId', validate='m:1')
                                     .merge(self.pff_scouting_df, how='inner', on=['nflId','gameId','playId'], validate='m:1'))
        
        qb_df = merged_ballsnapped_df.query('pff_positionLinedUp == "QB"').copy(deep=True)
        qb_df.rename({'x':'qb_x', 'y': 'qb_y'}, axis=1, inplace=True)
        qb_team = qb_df['team'].unique()[0]
        opponents_df = merged_ballsnapped_df.query('team != @qb_team')
        opponents_df = opponents_df.merge(qb_df[['frameId','qb_x', 'qb_y']], how='left', on='frameId', validate='m:1')
        opponents_df['dist_to_qb'] = calc_dist(opponents_df['x'].values, opponents_df['y'].values, opponents_df['qb_x'].values, opponents_df['qb_y'].values)
        opponents_df = pd.concat([opponents_df,pd.get_dummies(opponents_df['pff_role'])], axis=1)
        opponent_dist_df = opponents_df.pivot(index='frameId',columns='nflId',values='dist_to_qb')
        opponents_role_df = opponents_df.pivot(index='frameId',columns='nflId',values=['Coverage', 'Pass Rush'])
        opponent_feat_df = pd.concat([opponent_dist_df, opponents_role_df], axis=1).values
        X = opponent_feat_df[:self.timesteps,:]
        
        no_of_num_feats = 11
        X[:,:no_of_num_feats] = X[:,:no_of_num_feats]/ 10
        
        return X
    
    def __get_data(self, batches):
        
        X_batches = []
        Y_batches = []
        for i, row in batches.iterrows():
            gameId = row['gameId']
            playId = row['playId']
            try:
                X = self.__gen_features(gameId, playId)
            except Exception as e:
                print('gameId: {} playId: {}'.format(gameId, playId))
            X_batches.append(X)
            Y_batches.append(row['pressure_ind'])
        
        return np.array(X_batches), np.array(Y_batches).reshape(-1,1) 
        
    
    def __getitem__(self, index):
        
        batches = self.pressure_df[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__get_data(batches)        
        return X, y
    
if __name__ == '__main__':
    
    batch_size = 32
    timesteps = 10
    data_path = './model'
    
    pressure_train_gen = PressureDataGenerator(week_ids=[1], 
                                       games_path='./data/games.csv',
                                       pff_scouting_data_path='./data/pffScoutingdata.csv', 
                                       players_path='./data/players.csv',
                                       plays_path='./data/plays.csv', 
                                       week_path='./data/week{}.csv', 
                                       timesteps=timesteps, 
                                       batch_size=batch_size)
    
    pressure_val_gen = PressureDataGenerator(week_ids=[2], 
                                       games_path='./data/games.csv',
                                       pff_scouting_data_path='./data/pffScoutingdata.csv', 
                                       players_path='./data/players.csv',
                                       plays_path='./data/plays.csv', 
                                       week_path='./data/week{}.csv', 
                                       timesteps=timesteps, 
                                       batch_size=batch_size)
    
    x_shape = pressure_train_gen[0][0].shape

    tf.random.set_seed(7)
    model = Sequential()
    model.add(LSTM(16, batch_input_shape = x_shape, return_sequences=True))
    model.add(LeakyReLU(alpha=0.5))
    model.add(Dropout(0.5))
    # model.add(LSTM(32,return_sequences=True))
    # model.add(Dropout(0.5))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", metrics=[binary_accuracy], optimizer="adam")

    model.summary()
    checkpointer = ModelCheckpoint(filepath=data_path + '/model-{epoch:02d}.hdf5', verbose=1)
    model.fit(pressure_train_gen, epochs=30, verbose=2, validation_data = pressure_val_gen)