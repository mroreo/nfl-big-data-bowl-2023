# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 22:16:15 2022

@author: ben
"""
import pandas as pd
import numpy as np
import keras.backend as K
from sklearn.preprocessing import StandardScaler
from keras.layers import Dense, Input, Layer, InputSpec
from keras.models import Model
from keras.optimizers import SGD
from keras import callbacks
from keras import optimizers, Sequential
from keras.initializers import VarianceScaling
from keras.layers import Dense, LSTM, RepeatVector, TimeDistributed

pd.set_option('display.max_columns', None)

GAMES_PATH='./data/games.csv'
PFF_SCOUTING_DATA_PATH='./data/pffScoutingdata.csv'
PLAYERS_PATH='./data/players.csv'
PLAYS_PATH='./data/plays.csv'
WEEK_PATH='./data/week{}.csv'
WEEK_IDS = [1,2,3,4]
SNAP_EVENTS = ['ball_snap', 'autoevent_ballsnap']
NONSNAP_EVENTS = ['autoevent_passforward', 'pass_forward']
PASS_RESULTS = ['I','C','IN']
TIME_STEPS = 10

# Generated training sequences for use in the model.
def create_sequences(values, time_steps=TIME_STEPS):
    output = []
    for i in range(len(values) - time_steps + 1):
        output.append(values[i : (i + time_steps)])
    return np.stack(output)

def gen_dataset(week_df):
    all_x = []
    for k, df_grp in week_df.groupby(['gameId','playId','nflId']):
        values = df_grp[['x', 'y', 's', 'a', 'dis', 'o', 'dir']].values
        x_values = create_sequences(values)
        all_x.append(x_values)
    all_x = np.concatenate(all_x)
    return all_x

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

class ClusteringLayer(Layer):
    """
    Clustering layer converts input sample (feature) to soft label.
    # Example
    ```
        model.add(ClusteringLayer(n_clusters=10))
    ```
    # Arguments
        n_clusters: number of clusters.
        weights: list of Numpy array with shape `(n_clusters, n_features)` witch represents the initial cluster centers.
        alpha: degrees of freedom parameter in Student's t-distribution. Default to 1.0.
    # Input shape
        2D tensor with shape: `(n_samples, n_features)`.
    # Output shape
        2D tensor with shape: `(n_samples, n_clusters)`.
    """

    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=3)

    def build(self, input_shape):
        assert len(input_shape) == 3
        input_dim = (None, input_shape[1], input_shape[2])
        self.input_spec = InputSpec(dtype=K.floatx(), shape=input_dim)
        self.clusters = self.add_weight(shape=(self.n_clusters, input_shape[1], input_shape[2]), initializer='glorot_uniform', name='clusters')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        """ student t-distribution, as same as used in t-SNE algorithm.        
                 q_ij = 1/(1+dist(x_i, µ_j)^2), then normalize it.
                 q_ij can be interpreted as the probability of assigning sample i to cluster j.
                 (i.e., a soft assignment)
        Arguments:
            inputs: the variable containing data, shape=(n_samples, n_features)
        Return:
            q: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
        """
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1)) # Make sure each sample's 10 values add up to 1.
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def LSTMAutoEncoder(timesteps, n_features, act='relu'):
    
    input_ts = Input(shape=(timesteps, n_features), name='input')
    x = input_ts
    x = LSTM(32, activation='relu', return_sequences=True, name="encoder_0")(x)
    x = LSTM(16, activation='relu', return_sequences=False, name="encoder_1")(x)
    
    encoded = RepeatVector(timesteps, name='encoder_2')(x)
    
    # Decoder
    x = encoded
    x = LSTM(16, activation='relu', return_sequences=True, name="decoder_1")(x)
    x = LSTM(32, activation='relu', return_sequences=True, name="decoder_0")(x)
    x = TimeDistributed(Dense(n_features))(x)
    decoded = x
    return Model(inputs=input_ts, outputs=decoded, name='LSTMAE'), Model(inputs=input_ts, outputs=encoded, name='encoder')

def main():
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


    pass_df = (plays_df
                   .query('passResult == @PASS_RESULTS')
                   .query('pff_playAction == 0')
                   .query('foulName1.isna()')
                   .query('dropBackType == "TRADITIONAL"'))

    week_df = (week_df
                   .merge(snap_df, how='inner', on=['gameId','playId'])
                   .merge(nonsnap_df, how='inner', on=['gameId','playId'])
                   .merge(pass_df, how='inner', on=['gameId','playId'])
                   .query('(frameId >= snap_frameId) & (frameId <= nonsnap_frameId)')
                   .query('pff_role == "Pass Rush"')
                   .query('(nflId == 33131) | (nflId == 34777)'))
    
    train_x = gen_dataset(week_df.query('week < 4'))
    valid_x = gen_dataset(week_df.query('week == 4'))
    
    scaler = StandardScaler().fit(flatten(train_x))
    train_x_scaled = scale(train_x, scaler)
    valid_x_scaled = scale(valid_x, scaler)
    
    timesteps =  train_x.shape[1] # equal to the lookback
    n_features =  train_x.shape[2] # 59
    
    epochs = 200
    batch = 64
    lr = 0.0001
    
    from keras.utils.vis_utils import plot_model
    
    autoencoder, encoder = LSTMAutoEncoder(timesteps, n_features)
    
    adam = optimizers.Adam(lr)
    autoencoder.compile(optimizer=adam, loss='mse')
    autoencoder.fit(train_x_scaled, train_x_scaled, 
                    epochs=epochs, batch_size=batch, validation_data=(valid_x_scaled, valid_x_scaled), verbose=2)
    
    n_clusters = 2
    clustering_layer = ClusteringLayer(n_clusters, name='clustering')(encoder.output)
    model = Model(inputs=encoder.input, outputs=clustering_layer)
    model.compile(optimizer=SGD(0.01, 0.9), loss='kld')
    
    
    