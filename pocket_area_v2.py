# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 02:39:58 2023

@author: ben
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import warnings
import plotly.io as pio
import plotly.graph_objects as go
import glob
import os
import re
from itertools import chain
from plots import plot_game_play_id
from distances import calc_dist
from shapely import geometry
from sklearn.preprocessing import StandardScaler
from shapely.ops import unary_union
from shapely.validation import make_valid
from shapely.errors import ShapelyDeprecationWarning
from PIL import Image

pio.renderers.default = 'browser'

pd.set_option('display.max_columns', None)
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning) 
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

GAMES_PATH='./data/games.csv'
PFF_SCOUTING_DATA_PATH='./data/pffScoutingdata.csv'
PLAYERS_PATH='./data/players.csv'
PLAYS_PATH='./data/plays.csv'
WEEK_PATH='./data/week{}.csv'
WEEK_IDS = [1,2,3,4,5,6,7,8]
PASS_RESULTS = ['I','C','IN']
SNAP_EVENTS = ['ball_snap', 'autoevent_ballsnap']
NONSNAP_EVENTS = ['autoevent_passforward', 'pass_forward', 'qb_sack', 'autoevent_passinterrupted',
                  'pass_tipped', 'qb_strip_sack', 'autoevent_passinterrupted']
OLINE_POS_lINEDUP= ['C','LT','LG','RT','RG']

def ccworder(A):
    A= A- np.mean(A, 1)[:, None]
    return np.argsort(np.arctan2(A[1, :], A[0, :]))

def gen_snap_df(week_df, snap_events=SNAP_EVENTS):
    snap_df = (week_df
               .query('event.isin(@snap_events)')
               .groupby(['gameId','playId'])
               [['frameId', 'time']].min()
               .reset_index()
               .rename({'frameId': 'snap_frameId',
                        'time': 'snap_time'}, axis=1))
    return snap_df

def gen_nonsnap_df(week_df, nonsnap_events=NONSNAP_EVENTS):
    nonsnap_df = (week_df
               .query('event.isin(@nonsnap_events)')
               .groupby(['gameId','playId'])
               [['frameId','time']].min()
               .reset_index()
               .rename({'frameId': 'nonsnap_frameId',
                        'time': 'nonsnap_time'},axis=1))
    return nonsnap_df

def gen_passer_df(week_df):
    passer_df = (week_df.query('pff_role == "Pass"')
                 [['gameId', 'playId','frameId','time', 'x', 'y', 's','a','dis','o','dir']]
                 .rename({'x': 'qb_x',
                          'y': 'qb_y',
                          's': 'qb_s',
                          'a': 'qb_a',
                          'o': 'qb_o',
                          'dis': 'qb_dis',
                          'dir': 'qb_dir'}, axis=1))
    return passer_df

def gen_line_df(week_df, position, id_cols = ['gameId','nflId', 'playId','frameId'], feats_cols = ['x', 'y', 's','a','dis','o','dir'], oline_pos_lineup=OLINE_POS_lINEDUP, lookback_pd=5, min_speed=0.01, t=0.1):
    
    all_cols = list(chain(*[id_cols, feats_cols]))
    
    rename_cols = {}
    for f in feats_cols:
        renamed_col = f'{position}_{f}'
        rename_cols[f] = renamed_col
    
    if position == 'pb':
        line_df = (week_df
                    .query('pff_role == "Pass Block"'))
                    
        if oline_pos_lineup:
            line_df = (line_df.query('pff_positionLinedUp.isin(@oline_pos_lineup)'))
    else:
        line_df = (week_df
                    .query('pff_role == "Pass Rush"'))
    
    line_df = (line_df[all_cols].rename(rename_cols, axis=1))
    
    line_id_df = line_df[['gameId','playId','nflId']].drop_duplicates().groupby(['gameId', 'playId']).apply(lambda x: x.reset_index(drop=True).reset_index()).reset_index(drop=True).rename({'index': f'{position}_id'}, axis=1)

    line_df = (line_df.merge(line_id_df, how='left', on=['gameId', 'playId', 'nflId']))
    line_df[f'{position}_displacement'] = line_df.apply(lambda x: max(x[f'{position}_s'], min_speed) * t + 0.5*(x[f'{position}_a']) * t**2, axis=1)
    line_df[f'{position}_point'] = line_df.apply(lambda x: geometry.Point(x[f'{position}_x'], x[f'{position}_y']).buffer(x[f'{position}_displacement']), axis=1)
    
#     all_movement_paths = []
#     for k, df_grp in line_df.groupby(['gameId', 'playId', 'nflId']):
        
#         all_paths = []
#         for i in range(1, len(df_grp)+1):
#             all_paths.append(unary_union(df_grp[f'{position}_point'].values[max(0, i - lookback_pd):i]))
        
#         df_grp[f'{position}_movement_path'] = all_paths
#         all_movement_paths.append(df_grp)
#     line_df_poly = pd.concat(all_movement_paths)
    return line_id_df, line_df

def pivot_line_df(line_df, position):
    line_df = line_df.pivot(index=['gameId', 'playId', 'frameId'], columns=[f'{position}_id'], values=[f'{position}_point', f'{position}_x', f'{position}_y', f'{position}_o', f'{position}_dir', f'{position}_s', f'{position}_a', f'{position}_dis'])
    line_df.columns = ['{}_{}'.format(c[0], c[1]) for c in line_df.columns]
    line_df.reset_index(inplace=True)
    return line_df

def gen_pocket_polygon(row):
    
    # row = pocket_area_df.iloc[0]
    
    all_coords = []
    for c in row.index.values:
        
        if 'pb_x' in c:
            if not(pd.isnull(row[c])):
                all_coords.append((row[c], row[c.replace('_x', '_y')]))
    
    all_coords2 = np.array(all_coords)[np.argsort(np.array(all_coords)[:,1])]
    oline_coords2 = np.vstack([all_coords2,[row['qb_x'], row['qb_y']]])
    
    all_coords.append((row['qb_x'], row['qb_y']))
    
    A = np.array(all_coords).T
    sorted_order = ccworder(A)
    oline_coords = A.T[sorted_order]
    
    geom1 = geometry.Polygon(oline_coords)
    geom2 = geometry.Polygon(oline_coords2)
    
    if geom1.area > geom2.area:
        return geom1
    else:
        return geom2
    
def gen_pocket_area_df(passer_df, pass_blockers_df, snap_df, nonsnap_df):
    
    pass_blockers_df = pivot_line_df(pass_blockers_df, position='pb')
    
    pocket_area_df = (passer_df
                          .merge(pass_blockers_df, how='left', on=['gameId', 'playId', 'frameId'])
                          .merge(snap_df, how='left', on=['gameId', 'playId'])
                          .merge(nonsnap_df, how='left', on=['gameId', 'playId'])
                          .query('(frameId >= snap_frameId) & (frameId <= nonsnap_frameId)')
                          .assign(
                              time_since_snap = lambda x: (pd.to_datetime(x['time']) - pd.to_datetime(x['snap_time'])).dt.total_seconds()
                              ))
    
    pocket_area_df['pocket_polygon'] = pocket_area_df.apply(lambda x: gen_pocket_polygon(x), axis=1)
    pocket_area_df['pocket_area'] = pocket_area_df['pocket_polygon'].apply(lambda x: x.area)
    return pocket_area_df

def gen_pocket_adj_risk_poly(row):
    pocket_poly = row['pocket_polygon']
    pocket_poly = make_valid(pocket_poly)
    
    # pb_paths = [p for p in row.index.values if 'pb_point' in p]
    pr_paths = [p for p in row.index.values if 'pr_point' in p]
    
    # pb_paths_poly = [p for p in row[pb_paths].values if not(pd.isnull(p))]
    pr_paths_poly = [p for p in row[pr_paths].values if not(pd.isnull(p))]
    
    pr_combined = unary_union(pr_paths_poly)
    # pb_combined = unary_union(pb_paths_poly)
    
    # attack_polys = (pr_combined.symmetric_difference(pb_combined)).difference(pb_combined)
    attack_polys = make_valid(pr_combined)
    
    # attack_in_pocket_poly = pocket_poly.intersection(attack_polys)
    
    return attack_polys

def gen_pocket_collapse_rate(df_grp):
    
    df_grp = df_grp.sort_values('frameId', ascending=True)
    df_grp['pocket_collapse_rate'] = (df_grp['pocket_area'] - df_grp['pocket_area'].shift(1)) / 0.1
    
    return df_grp

def gen_all_pr_pocket_influence_areas(pocket_area_df, pass_blockers_poly_df, pass_rushers_poly_df, gameId=None, playId=None):
    
    if not(gameId):
        gameId = pocket_area_df.name[0]
    if not(playId):
        playId = pocket_area_df.name[1]

    pocket_area_df_subset = pocket_area_df.query('(gameId == @gameId) & (playId == @playId)')
    
    all_pr_stats_df = []
    for i, row in pocket_area_df_subset.iterrows():
        frameId = row['frameId']
        pocket_poly = row['pocket_polygon']
        pocket_poly = make_valid(pocket_poly)
        
        pass_blockers_poly_df_subset = pass_blockers_poly_df.query('(gameId == @gameId) & (playId == @playId) & (frameId == @frameId)')
        pass_rushers_poly_df_subset = pass_rushers_poly_df.query('(gameId == @gameId) & (playId == @playId) & (frameId == @frameId)')
        
        all_pb_influence_polys = unary_union(pass_blockers_poly_df_subset['pb_point'].values)
        all_pr_influence_polys = unary_union(pass_rushers_poly_df_subset['pr_point'].values)
        
        all_pb_influence_polys = make_valid(all_pb_influence_polys)
        all_pr_influence_polys = make_valid(all_pr_influence_polys)
        
        all_pr_x_pb_influence_poly = (all_pr_influence_polys.symmetric_difference(all_pb_influence_polys)).difference(all_pb_influence_polys)
        all_pr_x_pb_influence_poly = make_valid(all_pr_x_pb_influence_poly)
        
        all_pr_x_pb_influence_pocket_poly = pocket_poly.intersection(all_pr_x_pb_influence_poly)
        
        row['all_pr_x_pb_influence_pocket_poly'] = all_pr_x_pb_influence_pocket_poly
        row['all_pr_x_pb_influence_pocket_poly_area'] = all_pr_x_pb_influence_pocket_poly.area
        all_pr_stats_df.append(pd.DataFrame(row).T)
    
    all_pr_stats_df = pd.concat(all_pr_stats_df, axis=0)
    return all_pr_stats_df

def get_pr_influences(pocket_area_df, pr_poly_df, gameId = None, playId= None):
    all_pb_cols =  [p for p in pocket_area_df.columns if 'pb_point' in p]
    if not(gameId):
        gameId = pocket_area_df.name[0]
    if not(playId):
        playId = pocket_area_df.name[1]
    
    pr_stats_df = []
    for i, row in pocket_area_df.iterrows():
        
        frameId = row['frameId']
        pocket_poly = row['pocket_polygon']
        pocket_poly = make_valid(pocket_poly)
        
        all_pb_poly = [row[p] for p in all_pb_cols if not(pd.isnull(row[p]))]
        
        oline_influence_poly = unary_union(all_pb_poly)
        
        pr_poly_df_subset = pass_rushers_poly_df.query('(gameId == @gameId) & (playId == @playId) & (frameId == @frameId)')
        
        for j, row2 in pr_poly_df_subset.iterrows():
            pr_point = row2['pr_point']
            pr_influence_poly = (pr_point.symmetric_difference(oline_influence_poly)).difference(oline_influence_poly)
            pr_influence_poly = make_valid(pr_influence_poly)
            pr_influence_in_pocket_poly = pocket_poly.intersection(pr_influence_poly)
            pr_influence_in_pocket_poly = make_valid(pr_influence_in_pocket_poly)
            
            row2['pr_influence_poly'] = pr_influence_poly
            row2['pr_influence_in_pocket_poly'] = pr_influence_in_pocket_poly
            row2['pr_influence_in_pocket_area'] = pr_influence_in_pocket_poly.area
            pr_stats_df.append(pd.DataFrame(row2).T)
    
    pr_stats_df = pd.concat(pr_stats_df, axis=0)
    return pr_stats_df

def calc_influence_growth_rate(df_grp, window=3):
    df_grp['pr_influence_rate'] = np.where(df_grp['pr_influence_in_pocket_area'].shift(1) == 0, 0, df_grp['pr_influence_in_pocket_area'].astype(float) / df_grp['pr_influence_in_pocket_area'].shift(1).astype(float))
    return  df_grp

def get_pocket_penetration_frame(df_grp):
    pocket_penetration_df = df_grp.query('pr_influence_in_pocket_area > 0')
    return pd.Series({'pocket_penetration_frameId': pocket_penetration_df['frameId'].min()})

def make_gif(frame_folder, fname):
    all_fnames = glob.glob(f"{frame_folder}/*.png")
    l = sorted(all_fnames, key=get_key)
    frames = [Image.open(image) for image in l]
    frame_one = frames[0]
    frame_one.save(fname, format="GIF", append_images=frames, save_all=True, duration=200, loop=0)
    
def get_key(fp):
    filename = os.path.splitext(os.path.basename(fp))[0]
    int_part = filename[filename.find("(")+1:filename.find(")")]
    return int(int_part)

games_df = pd.read_csv(GAMES_PATH)
plays_df = pd.read_csv(PLAYS_PATH)
players_df = pd.read_csv(PLAYERS_PATH)
pff_scouting_df = pd.read_csv(PFF_SCOUTING_DATA_PATH)

games_df = games_df.query('week.isin(@WEEK_IDS)')
plays_df = plays_df.merge(games_df, how='inner', on=['gameId'])
pff_scouting_df = pff_scouting_df.merge(games_df['gameId'], how='inner', on='gameId', validate='m:1')

#Filter for pass plays only
plays_df = (plays_df
            .query('(pff_playAction == 0) & (passResult.isin(@PASS_RESULTS))')
            .query('foulName1.isna()')
            .query('dropBackType == "TRADITIONAL"'))

#Load the weekly data
week_df = []
for week_id in WEEK_IDS:
    week_df.append(pd.read_csv(WEEK_PATH.format(week_id)))
week_df = pd.concat(week_df)

week_df = (week_df
           .merge(players_df, how='left', on='nflId', validate='m:1')
           .merge(pff_scouting_df, how='left', on=['nflId', 'gameId', 'playId'], validate='m:1')
           .merge(plays_df, how='inner', on=['gameId', 'playId'], validate='m:1')
           )

#Generate the snap event dataframe
snap_df = gen_snap_df(week_df)
nonsnap_df = gen_nonsnap_df(week_df)

#Get the QB dataframe
passer_df = gen_passer_df(week_df)

#Get the pass blockers dataframe
pass_blockers_id_df, pass_blockers_poly_df = gen_line_df(week_df, position='pb')
all_pass_blockers_id_df, all_pass_blockers_poly_df = gen_line_df(week_df, position='pb', oline_pos_lineup=None)
pass_rushers_id_df, pass_rushers_poly_df = gen_line_df(week_df, position='pr')

pocket_area_df = gen_pocket_area_df(passer_df, pass_blockers_poly_df, snap_df, nonsnap_df)
pocket_area_df = pocket_area_df.drop([c for c in pocket_area_df.columns if 'pb_' in c], axis=1)
pocket_area_df = pocket_area_df.merge(pivot_line_df(all_pass_blockers_poly_df, 'pb'), how='left', on=['gameId', 'playId', 'frameId'])

pb_movement_cols = [c for c in pocket_area_df.columns if 'pb_point_' in c]
all_cols = list(chain(*[['gameId', 'playId', 'frameId', 'time', 'time_since_snap','snap_frameId','nonsnap_frameId', 'pocket_polygon', 'pocket_area'], pb_movement_cols]))

pocket_area_df = pocket_area_df[all_cols]
pocket_area_df = pocket_area_df.groupby(['gameId', 'playId'],group_keys=True).apply(lambda x: gen_pocket_collapse_rate(x))
pocket_area_df = pocket_area_df.reset_index(drop=True)

#Generate All Pass Rusher Influences and differences
all_pr_influence_stats_df = pocket_area_df.groupby(['gameId', 'playId']).apply(lambda x: gen_all_pr_pocket_influence_areas(x, pass_blockers_poly_df, pass_rushers_poly_df))
all_pr_influence_stats_df_calced = all_pr_influence_stats_df.reset_index(drop=True)
all_pr_influence_stats_df_calced['all_pr_x_pb_influence_pocket_poly_area'] = all_pr_influence_stats_df_calced['all_pr_x_pb_influence_pocket_poly_area'].astype(float)
all_pr_influence_stats_df_calced = all_pr_influence_stats_df_calced.merge(snap_df[['gameId', 'playId', 'snap_time']], how='left', on=['gameId', 'playId'])
all_pr_influence_stats_df_calced['all_pr_influence_vs_pocket_area'] = all_pr_influence_stats_df_calced['all_pr_x_pb_influence_pocket_poly_area'] / all_pr_influence_stats_df_calced['pocket_area']
all_pr_influence_stats_df_calced['all_pr_influence_vs_pocket_area_discounted'] = all_pr_influence_stats_df_calced['all_pr_influence_vs_pocket_area'] / (1.3)**(all_pr_influence_stats_df_calced['frameId'] - all_pr_influence_stats_df_calced['snap_frameId'])
all_pr_influence_stats_df_calced['time_since_snap_s'] = (pd.to_datetime(all_pr_influence_stats_df_calced['time']) - pd.to_datetime(all_pr_influence_stats_df_calced['snap_time'])).dt.total_seconds()

all_pr_feats_df = all_pr_influence_stats_df_calced.groupby(['gameId', 'playId'])['all_pr_influence_vs_pocket_area_discounted'].mean()
all_pr_feats_df = all_pr_feats_df.reset_index().rename({'all_pr_influence_vs_pocket_area_discounted': 'PRIS'}, axis=1)

penetration_feat_df = all_pr_influence_stats_df_calced.query('all_pr_influence_vs_pocket_area > 0').groupby(['gameId', 'playId'])['time_since_snap_s'].min().reset_index().rename({'time_since_snap_s': 'penetration_time_since_snap_s'}, axis=1)
all_pr_feats_df = all_pr_feats_df.merge(penetration_feat_df, how='left', on=['gameId', 'playId'])
all_pr_feats_df['pocket_penetration_ind'] = np.where(pd.isnull(all_pr_feats_df['penetration_time_since_snap_s']), 0, 1)
all_pr_feats_df['penetration_time_since_snap_s'] = np.where(all_pr_feats_df['penetration_time_since_snap_s'] > 5, 5, all_pr_feats_df['penetration_time_since_snap_s'])
all_pr_feats_df['penetration_time_since_snap_s'] = all_pr_feats_df['penetration_time_since_snap_s'].fillna(6)
all_pr_feats_df = all_pr_feats_df.merge(pff_scouting_df.groupby(['gameId', 'playId'])['pff_hurry'].max().reset_index(), how='left', on=['gameId', 'playId'])

all_pr_feats_df = all_pr_feats_df.merge(games_df[['gameId', 'week']], how='left', on='gameId')
all_pr_feats_df = all_pr_feats_df.merge(plays_df[['gameId', 'playId', 'passResult']], how='left', on=['gameId', 'playId'])
all_pr_feats_df['pass_incomplete_ind'] = np.where(all_pr_feats_df['passResult'] == 'C', 0, 1) 

train_df = all_pr_feats_df.query('week <= 6')
valid_df = all_pr_feats_df.query('week > 6')

all_feats = ['pff_hurry','pocket_penetration_ind', 'penetration_time_since_snap_s', 'PRIS']
train_X = train_df[all_feats].values
valid_X = valid_df[all_feats].values

train_y = train_df['pass_incomplete_ind'].values
valid_y = valid_df['pass_incomplete_ind'].values

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

scaler = MinMaxScaler()
scaler.fit(train_X)

scaled_train_x = scaler.transform(train_X)
scaled_valid_X = scaler.transform(valid_X)

model = LogisticRegression(random_state=42)
model.fit(scaled_train_x, train_y)

pred_train_y = model.predict_proba(scaled_train_x)[:,1]
pred_valid_y = model.predict_proba(scaled_valid_X)[:,1]

roc_auc_score(valid_y, pred_valid_y)

import scikitplot as skplt
import matplotlib.pyplot as plt

skplt.metrics.plot_roc_curve(valid_y, model.predict_proba(scaled_valid_X))
plt.show()

feat_imp_df = pd.DataFrame({'variables':all_feats,
                            'coefficients': model.coef_[0]})

sns.barplot(feat_imp_df, y='variables', x='coefficients')

all_pr_influence_stats_df_calced = all_pr_influence_stats_df_calced.groupby(['nflId', 'displayName']).apply(lambda x: pd.Series({'play_ct': len(x),
                                                                                                               'non_zero_influence_cts': len(x[x['pr_influence_vs_pocket_area_discounted'] > 0]),
                                                                                                               'pct_of_influenced_plays': len(x[x['pr_influence_vs_pocket_area_discounted'] > 0]) / len(x),
                                                                                                               'avg_penetration_time': x['time_since_snap_s'].mean(),
                                                                                                               'avg_influence_score': x[x['pr_influence_vs_pocket_area_discounted'] > 0]['pr_influence_vs_pocket_area_discounted'].mean()}))


#Generate the pass rusher influences and differences
pr_influence_stats_df = pocket_area_df.groupby(['gameId', 'playId']).apply(lambda x: get_pr_influences(x, pass_rushers_poly_df))
pr_influence_stats_df_calced = pr_influence_stats_df.reset_index(drop=True) 
pr_influence_stats_df_calced = pr_influence_stats_df_calced.merge(players_df[['nflId', 'displayName']], how='left', on='nflId')
pr_influence_stats_df_calced = pr_influence_stats_df_calced.merge(pocket_area_df[['gameId', 'playId', 'frameId', 'pocket_area']], how='left', on=['gameId', 'playId', 'frameId'])
pr_influence_stats_df_calced = pr_influence_stats_df_calced.merge(snap_df[['gameId', 'playId', 'snap_frameId', 'snap_time']], how='left', on=['gameId', 'playId'])
pr_influence_stats_df_calced = pr_influence_stats_df_calced.merge(week_df[['gameId', 'playId', 'frameId', 'time']].drop_duplicates(), how='left', on=['gameId', 'playId', 'frameId'])
pr_influence_stats_df_calced['pr_influence_vs_pocket_area'] = pr_influence_stats_df_calced['pr_influence_in_pocket_area'] / pr_influence_stats_df_calced['pocket_area']
pr_influence_stats_df_calced['pr_influence_vs_pocket_area_discounted'] = pr_influence_stats_df_calced['pr_influence_vs_pocket_area'] / (1.3)**(pr_influence_stats_df_calced['frameId'] - pr_influence_stats_df_calced['snap_frameId'])
pr_influence_stats_df_calced['time_since_snap_s'] = (pd.to_datetime(pr_influence_stats_df_calced['time']) - pd.to_datetime(pr_influence_stats_df_calced['snap_time'])).dt.total_seconds()

#Merge pocket collapse rate
pr_influence_stats_df_calced = pr_influence_stats_df_calced.merge(pocket_area_df[['gameId', 'playId','frameId', 'pocket_collapse_rate']], how='left', on=['gameId', 'playId', 'frameId'])

#Calc influence growth rate
pr_influence_stats_df_calced = pr_influence_stats_df_calced.groupby(['gameId', 'nflId', 'playId'], group_keys=True).apply(lambda x: calc_influence_growth_rate(x)).reset_index(drop=True)

#Calc pocket penetration frame
pocket_penetration_df = pr_influence_stats_df_calced.groupby(['gameId', 'playId']).apply(lambda x: get_pocket_penetration_frame(x))
pocket_penetration_df = pocket_penetration_df.reset_index()

all_games = all_gameplays_df = pocket_area_df[['gameId', 'playId']].drop_duplicates().reset_index(drop=True)

idx = 0
gameId = all_gameplays_df['gameId'].values[idx]
playId = all_gameplays_df['playId'].values[idx]

pr_stats_df = pr_influence_stats_df_calced.query('(gameId == @gameId) & (playId == @playId)')
pocket_area_stats_df = pocket_area_df.query('(gameId == @gameId) & (playId == @playId)')
pocket_penetration_df_subset = pocket_penetration_df.query('(gameId == @gameId) & (playId == @playId)')

pocket_penetration_frameId = pocket_penetration_df_subset['pocket_penetration_frameId'].values[0]

fig, ax = plt.subplots(4,1, figsize=(16,8))
sns.lineplot(pocket_area_stats_df, x='frameId', y='pocket_area', ax=ax[0])
sns.lineplot(pocket_area_stats_df, x='frameId', y='pocket_collapse_rate', ax=ax[1])
ax[1].axhline(0, linestyle='--', alpha=0.5, color='gray')
if pocket_penetration_frameId:
    ax[1].axvline(pocket_penetration_frameId, linestyle='--', alpha=0.5, color='gray')
sns.lineplot(pr_stats_df, x='frameId', y='pr_influence_vs_pocket_area', hue='displayName', ax=ax[2])
sns.lineplot(pr_stats_df, x='frameId', y='pr_influence_vs_pocket_area_discounted', hue='displayName', ax=ax[3])
plt.show()

player_pocket_pen_df = pr_influence_stats_df_calced.query('pr_influence_in_pocket_area > 0')
player_pocket_pen_df = player_pocket_pen_df.groupby(['gameId', 'playId', 'nflId'])['time_since_snap_s'].min()
player_pocket_pen_df = player_pocket_pen_df.reset_index()


player_influence_scores = pr_influence_stats_df_calced.groupby(['gameId', 'playId', 'nflId', 'displayName'])['pr_influence_vs_pocket_area_discounted'].sum()
player_influence_scores = player_influence_scores.reset_index()
player_influence_scores['pr_influence_vs_pocket_area_discounted'] = player_influence_scores['pr_influence_vs_pocket_area_discounted']* 100
player_influence_scores = player_influence_scores.merge(player_pocket_pen_df, how='left', on=['gameId', 'playId', 'nflId'])
player_influence_scores = player_influence_scores.groupby(['nflId', 'displayName']).apply(lambda x: pd.Series({'play_ct': len(x),
                                                                                                               'non_zero_influence_cts': len(x[x['pr_influence_vs_pocket_area_discounted'] > 0]),
                                                                                                               'pct_of_influenced_plays': len(x[x['pr_influence_vs_pocket_area_discounted'] > 0]) / len(x),
                                                                                                               'avg_penetration_time': x['time_since_snap_s'].mean(),
                                                                                                               'avg_influence_score': x[x['pr_influence_vs_pocket_area_discounted'] > 0]['pr_influence_vs_pocket_area_discounted'].mean()}))
player_influence_scores.reset_index(inplace=True)
player_influence_scores = player_influence_scores.query('play_ct > 30')
player_influence_scores.sort_values(['avg_influence_score'], ascending=False, inplace=True)

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

x = player_influence_scores[['pct_of_influenced_plays', 'avg_penetration_time', 'avg_influence_score']].astype(float).values
scaler = MinMaxScaler()
scaled_x = scaler.fit_transform(x)

kmeans = KMeans(n_clusters=10, random_state=0).fit(scaled_x)

player_influence_scores['cluster'] = kmeans.predict(scaled_x)

fig = go.Figure(data=[go.Scatter3d(
    x=player_influence_scores.pct_of_influenced_plays,
    y=player_influence_scores.avg_penetration_time,
    z=player_influence_scores.avg_influence_score,
    hovertemplate='<b>%{text}</b><extra></extra>',
    text = [name for name in player_influence_scores.displayName],
    mode='markers',
    marker=dict(
        size=8,
        color=player_influence_scores.cluster,
        opacity=0.8
    )
)])

fig.update_layout(scene = dict(
                    xaxis_title='Percentage of Influenced Plays',
                    yaxis_title='Avg Penetration Time',
                    zaxis_title='Discounted Influence Score'))

fig.show()

influenced_plays = player_influence_scores[player_influence_scores['pr_influence_vs_pocket_area_discounted'] > 0]

influenced_plays = influenced_plays.reindex(influenced_plays['time_since_snap_s'].median().sort_values().index, axis=1)

px.box(influenced_plays, x='displayName', y='time_since_snap_s')

import plotly.express as px
fig = px.scatter_3d(player_influence_scores, x='pct_of_influenced_plays', y='avg_penetration_time', z='avg_influence_score')
fig.show()


pass_rushers_append_polygon = pass_rushers_poly_df[['gameId', 'playId', 'frameId', 'pr_point']].rename({'pr_point': 'player_polygon'}, axis=1)
pass_rushers_append_polygon['team'] = 'player_polygon'
pass_rushers_append_polygon['pff_role'] = 'Pass Rusher'
pass_rusher_appended = pd.concat([week_df, pass_rushers_append_polygon], axis=0)
all_gameplays_df = pass_rusher_appended[['gameId', 'playId']].drop_duplicates().reset_index(drop=True)
idx = 0
gameId = all_gameplays_df['gameId'].values[idx]
playId = all_gameplays_df['playId'].values[idx]
plot_game_play_id(pass_rusher_appended, gameId, playId, size=(1200, 800))

pass_blockers_append_polygon = pass_blockers_poly_df[['gameId', 'playId', 'frameId', 'pb_point']].rename({'pb_point': 'player_polygon'}, axis=1)
pass_blockers_append_polygon['team'] = 'player_polygon'
pass_blockers_append_polygon['pff_role'] = 'Pass Blocker'
pass_blockers_appended = pd.concat([week_df, pass_blockers_append_polygon], axis=0)
all_gameplays_df = pass_blockers_appended[['gameId', 'playId']].drop_duplicates().reset_index(drop=True)
idx = 0
gameId = all_gameplays_df['gameId'].values[idx]
playId = all_gameplays_df['playId'].values[idx]
plot_game_play_id(pass_blockers_appended, gameId, playId, size=(1200, 800))

pb_and_pr_appended = pd.concat([week_df, pass_blockers_append_polygon, pass_rushers_append_polygon], axis=0)
all_gameplays_df = pb_and_pr_appended[['gameId', 'playId']].drop_duplicates().reset_index(drop=True)
idx = 0
gameId = all_gameplays_df['gameId'].values[idx]
playId = all_gameplays_df['playId'].values[idx]
plot_game_play_id(pb_and_pr_appended, gameId, playId, size=(1200, 800))

pass_rusher_influence_append_polygon = all_pr_influence_stats_df_calced[['gameId', 'playId', 'frameId', 'all_pr_x_pb_influence_pocket_poly']].rename({'all_pr_x_pb_influence_pocket_poly': 'pr_pocket_influence_polygon'}, axis=1)
pass_rusher_influence_append_polygon['team'] = 'pr_pocket_influence_polygon'
pass_rusher_influence_append_polygon['pff_role'] = 'Pass Rusher Pocket Influence'

pocket_influence_appended = pd.concat([week_df, pass_blockers_append_polygon, pass_rushers_append_polygon, pass_rusher_influence_append_polygon], axis=0)
all_gameplays_df = pb_and_pr_appended[['gameId', 'playId']].drop_duplicates().reset_index(drop=True)
idx = 0
gameId = all_gameplays_df['gameId'].values[idx]
playId = all_gameplays_df['playId'].values[idx]
plot_game_play_id(pocket_influence_appended, gameId, playId, size=(1200, 800))



