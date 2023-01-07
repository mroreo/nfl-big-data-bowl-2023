# -*- coding: utf-8 -*-
"""
Created on Sat Dec 31 14:38:28 2022

@author: ben
"""
import pandas as pd
import numpy as np
from itertools import chain
from shapely import geometry
from shapely.ops import unary_union

OLINE_POS_lINEDUP= ['C','LT','LG','RT','RG']

def ccworder(A):
    A= A- np.mean(A, 1)[:, None]
    return np.argsort(np.arctan2(A[1, :], A[0, :]))

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
    
def gen_line_df(week_df_merged,
                position, 
                id_cols = ['gameId','nflId', 'playId','frameId'], 
                feats_cols = ['x', 'y', 's','a','dis','o','dir'], 
                oline_pos_lineup=OLINE_POS_lINEDUP, 
                lookback_pd=5, 
                min_speed=0.01):
    """
    
    Generate two dataframes that provide a visual of the pass rushers' or pass blockers' path and the polygons.

    Parameters
    ----------
    week_df_merged :df
        the weekly frame by frame dataframe merged with players_df, pff_scouting_df, plays_df
    position : str
        generate the line df for pass blockers or pass rushers. pb or pr
    id_cols : list, optional
        id columns for doing the pivot. The default is ['gameId','nflId', 'playId','frameId'].
    feats_cols : list, optional
        feature columns to generate for each id. The default is ['x', 'y', 's','a','dis','o','dir'].
    oline_pos_lineup : list, optional
        a list of oline pos to generate the passer pocket. The default is OLINE_POS_lINEDUP.
    lookback_pd : TYPE, optional
        the number of frames to look back when generating pass rusher path or pass blocker path. The default is 5.
    min_speed : TYPE, optional
        minimum speed of each player. This is to prevent errors when generating buffer areas. The default is 0.01.

    Returns
    -------
    line_id_df : df
        dataframe with the lineman positions 
    line_df_poly : df
        dataframe with the lineman positions and the polygons 

    """
    all_cols = list(chain(*[id_cols, feats_cols]))
    
    rename_cols = {}
    for f in feats_cols:
        renamed_col = f'{position}_{f}'
        rename_cols[f] = renamed_col
    
    if position == 'pb':
        line_df = (week_df_merged
                    .query('pff_role == "Pass Block"'))
                    
        if oline_pos_lineup:
            line_df = (line_df.query('pff_positionLinedUp.isin(@oline_pos_lineup)'))
    else:
        line_df = (week_df_merged
                    .query('pff_role == "Pass Rush"'))
    
    line_df = (line_df[all_cols].rename(rename_cols, axis=1))
    
    line_id_df = line_df[['gameId','playId','nflId']].drop_duplicates().groupby(['gameId', 'playId']).apply(lambda x: x.reset_index(drop=True).reset_index()).reset_index(drop=True).rename({'index': f'{position}_id'}, axis=1)

    line_df = (line_df.merge(line_id_df, how='left', on=['gameId', 'playId', 'nflId']))
    line_df[f'{position}_point'] = line_df.apply(lambda x: geometry.Point(x[f'{position}_x'], x[f'{position}_y']).buffer((max(x[f'{position}_s'], min_speed) + x[f'{position}_a'])/10), axis=1)
    
    all_movement_paths = []
    for k, df_grp in line_df.groupby(['gameId', 'playId', 'nflId']):
        
        all_paths = []
        for i in range(1, len(df_grp)+1):
            all_paths.append(unary_union(df_grp[f'{position}_point'].values[max(0, i - lookback_pd):i]))
        
        df_grp[f'{position}_movement_path'] = all_paths
        all_movement_paths.append(df_grp)
    line_df_poly = pd.concat(all_movement_paths)
    return line_id_df, line_df_poly
    
def pivot_line_df(line_df, position):
    line_df = line_df.pivot(index=['gameId', 'playId', 'frameId'], columns=[f'{position}_id'], values=[f'{position}_point', f'{position}_movement_path', f'{position}_x', f'{position}_y', f'{position}_o', f'{position}_dir', f'{position}_s', f'{position}_a', f'{position}_dis'])
    line_df.columns = ['{}_{}'.format(c[0], c[1]) for c in line_df.columns]
    line_df.reset_index(inplace=True)
    return line_df
    
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

def calc_pr_disruption_area(x):
    
    pb_paths = [p for p in x.index.values if 'pb_movement_path' in p]
    pb_paths_poly = [p for p in x[pb_paths].values if not(pd.isnull(p))]
    
    pb_combined = unary_union(pb_paths_poly)
    
    attack_poly = x['pr_movement_path'].symmetric_difference(pb_combined).difference(pb_combined)
    attack_poly = make_valid(attack_poly)
    
    attack_in_pocket_poly = make_valid(x['pocket_polygon']).intersection(attack_poly)
    return attack_in_pocket_poly