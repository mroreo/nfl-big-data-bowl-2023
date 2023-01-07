# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 23:23:06 2023

@author: ben
"""
import numpy as np
import pandas as pd
from data import NFLData
from pocket_area import gen_line_df, gen_pocket_area_df

def gen_area_stats(df):
    
    feat_dict = {}
    feat_dict['weighted_affected_area'] = np.sum(df['weighted_area'])
    feat_dict['time_until_penetration'] = df[df['weighted_area'] > 0]['time_since_snap'].min()
    feat_dict['max_affected_area'] = np.max(df['weighted_area'])
    return pd.Series(feat_dict)
    
def main():
    
    NFL = NFLData()
    NFL.load_data(week_ids=[1,2,3,4])
    
    pass_blockers_id_df, pass_blockers_poly_df = gen_line_df(NFL.week_df_merged, position='pb')
    all_pass_blockers_id_df, all_pass_blockers_poly_df = gen_line_df(NFL.week_df_merged, position='pb', oline_pos_lineup=None)
    pass_rushers_id_df, pass_rushers_poly_df = gen_line_df(NFL.week_df_merged, position='pr')
    
    pocket_area_df = gen_pocket_area_df(NFL.passer_df, pass_blockers_poly_df, NFL.snap_df, NFL.nonsnap_df)
    
    pocket_area_df = pocket_area_df.drop([c for c in pocket_area_df.columns if 'pb_' in c], axis=1)
    pocket_area_df = pocket_area_df.merge(pivot_line_df(all_pass_blockers_poly_df, 'pb'), how='left', on=['gameId', 'playId', 'frameId'])
    
    pb_movement_cols = [c for c in pocket_area_df.columns if 'pb_movement_path' in c]
    all_cols = list(chain(*[['gameId', 'playId', 'frameId', 'time', 'time_since_snap','snap_frameId','nonsnap_frameId', 'pocket_polygon', 'pocket_area'], pb_movement_cols]))
    
    pocket_area_df = pocket_area_df[all_cols]
    pocket_area_df = pocket_area_df.merge(pivot_line_df(pass_rushers_poly_df, 'pr')[['gameId', 'playId', 'frameId', 'pr_movement_path_0', 'pr_movement_path_1', 'pr_movement_path_2',
                                                           'pr_movement_path_3', 'pr_movement_path_4', 'pr_movement_path_5', 'pr_movement_path_6']], how='left', on=['gameId', 'playId', 'frameId'])

    pocket_area_df['affected_pocket_polygon'] = pocket_area_df.apply(lambda x: gen_pocket_adj_risk_poly(x), axis=1)
    pocket_area_df['affected_pocket_area'] = pocket_area_df.apply(lambda x: x['affected_pocket_polygon'].area, axis=1)
    
    pocket_area_df['affected_pocket_area_pct'] = pocket_area_df['affected_pocket_area']/ pocket_area_df['pocket_area']
    
    #Plot week append
    week_df_append_polygon = pocket_area_df[['gameId', 'playId', 'frameId', 'pocket_polygon']]
    week_df_append_polygon['team'] = 'pocket_polygon'
    
    week_df_append_affected_polygon = pocket_area_df[['gameId', 'playId', 'frameId', 'affected_pocket_polygon']]
    week_df_append_affected_polygon['team'] = 'affected_pocket_polygon'
    
    #Create the appended data for plotting the polygons later
    week_df_appended = pd.concat([week_df, week_df_append_affected_polygon, week_df_append_polygon], axis=0)
    
    pocket_poly_df = pocket_area_df[['gameId', 'playId', 'frameId','snap_frameId','nonsnap_frameId', 'pocket_polygon', 'time_since_snap']]
    pocket_poly_df = (pocket_poly_df
                      .merge(pivot_line_df(all_pass_blockers_poly_df,'pb'), how='left', on=['gameId', 'playId', 'frameId'])
                      .rename({'nflId': 'pb_nflId'}, axis=1)
                      .merge(pass_rushers_poly_df[['gameId', 'nflId', 'playId','frameId', 'pr_movement_path']], how='left', on=['gameId', 'playId', 'frameId'])
                      .rename({'nflId': 'pr_nflId'}, axis=1))
    drop_cols = [c for c in pocket_poly_df.columns if 'pb_' in c and 'pb_movement_path' not in c]
    pocket_poly_df = pocket_poly_df.drop(drop_cols, axis=1)
    pocket_poly_df['pr_pocket_intersection'] = pocket_poly_df.apply(lambda x : calc_pr_disruption_area(x), axis=1)
    pocket_poly_df.head()
    
    pocket_poly_df['pocket_area'] = pocket_poly_df['pocket_polygon'].apply(lambda x: x.area)
    pocket_poly_df['pr_pocket_intersection_area'] = pocket_poly_df['pr_pocket_intersection'].apply(lambda x: x.area)
    pocket_poly_df['frame_counts'] = pocket_poly_df['nonsnap_frameId'] - pocket_poly_df['snap_frameId']
    pocket_poly_df['frameId_rebased'] = pocket_poly_df['frameId'] - pocket_poly_df['snap_frameId']
    haromonic_dict = {}
    for i in range(int(pocket_poly_df['frameId_rebased'].max())):
        haromonic_dict[i] = nthHarmonic(i)
    
    pocket_poly_df['max_weights'] = pocket_poly_df['frame_counts'].map(haromonic_dict)
    pocket_poly_df['weights'] = pocket_poly_df['frameId_rebased'].map(haromonic_dict)
    pocket_poly_df['inverse_weights'] = 1 - pocket_poly_df['weights'] / pocket_poly_df['max_weights']
    pocket_poly_df['weighted_area'] = pocket_poly_df['inverse_weights'] * (pocket_poly_df['pr_pocket_intersection_area'] / pocket_poly_df['pocket_area'])
    
    avg_pocket_disruption_df = pocket_poly_df.groupby(['gameId', 'playId', 'pr_nflId']).apply(lambda x: gen_area_stats(x)).reset_index()
    avg_pocket_disruption_df.sort_values('weighted_affected_area', ascending=False, inplace=True)
    avg_pocket_disruption_df = avg_pocket_disruption_df.merge(players_df[['nflId','displayName']], how='left', left_on='pr_nflId', right_on='nflId')
    avg_pocket_disruption_df.head()
    
    
    
