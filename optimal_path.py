# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 20:30:30 2022

@author: ben
"""

import pandas as pd
import numpy as np
import itertools
import keras.backend as K
from sklearn.preprocessing import StandardScaler
from keras.layers import Dense, Input, Layer, InputSpec, Flatten
from keras.models import Model
from keras.optimizers import Adam
from keras import callbacks
from keras import optimizers, Sequential
from keras.initializers import VarianceScaling
from keras.layers import Dense, LSTM, RepeatVector, TimeDistributed
from sklearn.model_selection import GroupShuffleSplit 

pd.set_option('display.max_columns', None)

GAMES_PATH='./data/games.csv'
PFF_SCOUTING_DATA_PATH='./data/pffScoutingdata.csv'
PLAYERS_PATH='./data/players.csv'
PLAYS_PATH='./data/plays.csv'
WEEK_PATH='./data/week{}.csv'
WEEK_IDS = [1,2,3,4]
SNAP_EVENTS = ['ball_snap', 'autoevent_ballsnap']
NONSNAP_EVENTS = ['autoevent_passforward', 'pass_forward', 'qb_sack', 'autoevent_passinterrupted',
                  'pass_tipped', 'qb_strip_sack', 'autoevent_passinterrupted']
PASS_RESULTS = ['I','C','IN']
TIME_STEPS = 10
SEED = 42
DATA_SPLIT_PCT = 0.3

games_df = pd.read_csv(GAMES_PATH)
plays_df = pd.read_csv(PLAYS_PATH)
players_df = pd.read_csv(PLAYERS_PATH)
pff_scouting_df = pd.read_csv(PFF_SCOUTING_DATA_PATH)

games_df = games_df.query('week.isin(@WEEK_IDS)')
plays_df = plays_df.merge(games_df, how='inner', on=['gameId'])
pff_scouting_df = pff_scouting_df.merge(games_df['gameId'], how='inner', on='gameId', validate='m:1')

week_df = []
for week_id in WEEK_IDS:
    week_df.append(pd.read_csv(WEEK_PATH.format(week_id)))
week_df = pd.concat(week_df)

week_df = (week_df
           .merge(players_df, how='left', on='nflId', validate='m:1')
           .merge(pff_scouting_df, how='left', on=['nflId', 'gameId', 'playId'], validate='m:1')
           )

snap_df = (week_df
           .query('event.isin(@SNAP_EVENTS)')
           .groupby(['gameId','playId'])
           ['frameId'].min()
           .reset_index()
           .rename({'frameId': 'snap_frameId'}, axis=1))

nonsnap_df = (week_df
           .query('event.isin(@NONSNAP_EVENTS)')
           .groupby(['gameId','playId'])
           [['frameId']].min()
           .reset_index()
           .rename({'frameId': 'nonsnap_frameId'},axis=1))

pass_blockers_df = (week_df
                    [['nflId', 'gameId', 'playId','frameId', 'pff_nflIdBlockedPlayer', 'x', 'y', 's','a','dis','o','dir']]
                    .query('~pff_nflIdBlockedPlayer.isna()')
                    .rename({'nflId': 'blockerNflId',
                             'x': 'passBlocker_x',
                             'y': 'passBlocker_y',
                             's': 'passBlocker_s',
                             'a': 'passBlocker_a',
                             'o': 'passBlocker_o',
                             'dis': 'passBlocker_dis',
                             'dir': 'passBlocker_dir'}, axis=1))

passer_df = (week_df.query('pff_role == "Pass"')
             [['nflId', 'gameId', 'playId','frameId', 'x', 'y', 's','a','dis','o','dir']]
             .rename({'nflId': 'qbNflId',
                      'x': 'qb_x',
                      'y': 'qb_y',
                      's': 'qb_s',
                      'a': 'qb_a',
                      'o': 'qb_o',
                      'dis': 'qb_dis',
                      'dir': 'qb_dir'}, axis=1))
                        

def convert_to_inches(string_series):
    
    split_df = string_series.str.split('-', expand=True)
    return split_df[0].astype(int) * 12 + split_df[1].astype(int)


optimal_play_df = (week_df
                   .query('(pff_hurry == 1) | (pff_sack == 1)')
                   .query('pff_role == "Pass Rush"')
                   .merge(snap_df, how='left', on=['gameId', 'playId'])
                   .merge(nonsnap_df, how='left', on=['gameId', 'playId'])
                   .query('(frameId >= snap_frameId) & (frameId <= nonsnap_frameId)')
                   .drop('pff_nflIdBlockedPlayer', axis=1)
                   .merge(pass_blockers_df, how='inner', left_on=['gameId', 'playId', 'frameId','nflId'], 
                          right_on=['gameId','playId','frameId','pff_nflIdBlockedPlayer'])
                   .merge(passer_df, how='inner', on=['gameId', 'playId', 'frameId'])
                   .merge(games_df, how='inner', on=['gameId'])
                   .assign(
                       playDirection_ind = lambda x: np.where(x['playDirection']=='right',1,0),
                       height_inches = lambda x: convert_to_inches(x['height'])
                       ))

non_optimal_play_df = (week_df
                   .query('(pff_hurry == 0) & (pff_sack == 0)')
                   .query('pff_role == "Pass Rush"')
                   .merge(snap_df, how='left', on=['gameId', 'playId'])
                   .merge(nonsnap_df, how='left', on=['gameId', 'playId'])
                   .query('(frameId >= snap_frameId) & (frameId <= nonsnap_frameId)')
                   .drop('pff_nflIdBlockedPlayer', axis=1)
                   .merge(pass_blockers_df, how='inner', left_on=['gameId', 'playId', 'frameId','nflId'], 
                          right_on=['gameId','playId','frameId','pff_nflIdBlockedPlayer'])
                   .merge(passer_df, how='inner', on=['gameId', 'playId', 'frameId'])
                   .merge(games_df, how='inner', on=['gameId'])
                   .assign(
                       playDirection_ind = lambda x: np.where(x['playDirection']=='right',1,0),
                       height_inches = lambda x: convert_to_inches(x['height'])
                       ))


def temporalize(x, y, lookback):
    
    output_X = []
    output_y = []
    for i in range(len(x) - lookback - 1):
        t = []
        for j in range(1, lookback + 1):
            # Gather the past records upto the lookback period
            t.append(x[[(i + j + 1)], :])
        output_X.append(t)
        output_y.append(y[i + lookback + 1])
    return np.squeeze(np.array(output_X)), np.array(output_y)
    
    
def gen_dataset(df):
    player_pos = ['x','y']
    target_player_pos = [f'target_{c}' for c in player_pos]
    features = ['x', 'y', 's', 'a', 'dis','o','dir']
    passblocker_pos_feats = [f'passBlocker_{c}' for c in features]
    qb_pos_feats = [f'qb_{c}' for c in features]
    other_features = ['playDirection_ind', 'weight', 'height_inches']
    all_features = list(itertools.chain(features, passblocker_pos_feats, qb_pos_feats, other_features))
    all_ids, all_X, all_y = [], [], []
    for k, df_grp in df.groupby(['gameId', 'playId', 'nflId']):
        df_grp[target_player_pos] = df_grp[player_pos].shift(-1)
        df_grp = df_grp[~df_grp[target_player_pos[0]].isna()]
        
        input_x = df_grp[all_features].values
        input_y = df_grp[target_player_pos].values
        
        X, y = temporalize(input_x, input_y, lookback=10)
        try:
            all_ids.append(np.array(list(k)*len(X)).reshape(len(X),-1))
            all_X.append(X)
            all_y.append(y)
        except Exception as e:
            pass
    all_ids = np.concatenate(np.array(all_ids))
    all_X = np.concatenate(np.array(all_X))
    all_y = np.concatenate(np.array(all_y))
    return all_ids, all_X, all_y

def flatten(X):
    '''
    Flatten a 3D array.
    Input
    X            A 3D array for lstm, where the array is sample x timesteps x features.
    Output
    flattened_X  A 2D array, sample x features.
    '''
    flattened_X = np.empty((X.shape[0], X.shape[2]))  # sample x features array.
    for i in range(X.shape[0]):
        flattened_X[i] = X[i, (X.shape[1]-1), :]
    return(flattened_X)

def scale(X, scaler):
    '''
    Scale 3D array.
    Inputs
    X            A 3D array for lstm, where the array is sample x timesteps x features.
    scaler       A scaler object, e.g., sklearn.preprocessing.StandardScaler, sklearn.preprocessing.normalize
    Output
    X            Scaled 3D array.
    '''
    for i in range(X.shape[0]):
        X[i, :, :] = scaler.transform(X[i, :, :])

    return X

train_ids, train_X, train_y = gen_dataset(optimal_play_df.query('week < 4'))
valid_ids, valid_X, valid_y = gen_dataset(optimal_play_df.query('week == 4'))
test_ids, test_X, test_y = gen_dataset(non_optimal_play_df.query('week == 4'))

scaler = StandardScaler().fit(flatten(train_X))
train_X_scaled = scale(train_X, scaler)
valid_X_scaled = scale(valid_X, scaler)
test_X_scaled = scale(test_X, scaler)

epochs = 200
batch = 64
lr = 0.0005
patience = 5
timesteps =  train_X_scaled.shape[1] 
n_features =  train_X_scaled.shape[2]

lstm_autoencoder = Sequential()
# Encoder
lstm_autoencoder.add(LSTM(64, activation='relu', input_shape=(timesteps, n_features), return_sequences=True))
lstm_autoencoder.add(LSTM(32, activation='relu', return_sequences=False))
lstm_autoencoder.add(RepeatVector(timesteps))
# # Decoder
lstm_autoencoder.add(LSTM(16, activation='relu', return_sequences=True))
lstm_autoencoder.add(LSTM(32, activation='relu', return_sequences=True))
lstm_autoencoder.add(TimeDistributed(Dense(2)))
lstm_autoencoder.add(Flatten())
lstm_autoencoder.add(Dense(2))
lstm_autoencoder.summary()

opt = Adam(learning_rate=lr)
lstm_autoencoder.compile(loss='mse', optimizer=opt)

# cp = ModelCheckpoint("checkpoints/weights.h5", save_best_only=True)
# tb = TensorBoard()
# es = EarlyStopping(patience=patience)

history = lstm_autoencoder.fit(
    train_X_scaled, train_y,
    epochs=epochs,
    batch_size=batch,
    validation_data=(valid_X_scaled, valid_y),
    shuffle=True,
    verbose=2).history

test_id_df = pd.DataFrame(test_ids, columns=['gameId', 'playId', 'nflId'])
test_id_df_no_dupes = test_id_df.drop_duplicates().reset_index(drop=True)

record = test_id_df_no_dupes.iloc[0]
gameId = record['gameId']
playId = record['playId']
nflId = record['nflId']
all_idx = test_id_df.query('(gameId == @gameId) & (playId == @playId) & (nflId == @nflId)').index.values
pred_xy = lstm_autoencoder.predict(test_X_scaled[all_idx,:,:].reshape(-1,10,24))
pred_opt_play_df = non_optimal_play_df.query('(gameId == @gameId) & (playId == @playId) & (nflId == @nflId)').reset_index(drop=True)
pred_opt_play_df['pred_x'] = pred_opt_play_df['x']
pred_opt_play_df['pred_y'] = pred_opt_play_df['y']

for i, j in enumerate(range(len(pred_opt_play_df) - len(pred_xy), len(pred_opt_play_df))):
    
    pred_opt_play_df.at[j,'pred_x'] = pred_xy[i,0]
    pred_opt_play_df.at[j,'pred_y'] = pred_xy[i,1]

import matplotlib.pyplot as plt

fig, ax = plt.subplots(1,1, figsize=(16,8))
plt.plot(pred_opt_play_df['x'],pred_opt_play_df['y'], label='pass rusher', marker='o')
plt.plot(pred_opt_play_df['qb_x'],pred_opt_play_df['qb_y'], label='qb', marker='o')
plt.plot(pred_opt_play_df['pred_x'],pred_opt_play_df['pred_y'], label='pred pass rusher', marker='o')
plt.plot(pred_opt_play_df['passBlocker_x'],pred_opt_play_df['passBlocker_y'], label='pass blocker', marker='o')
ax.legend()
plt.show()









