# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 17:26:10 2022

@author: ben

To Do:
    - Need to handle the use case where TEs are blocking as well. 
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from plots import plot_game_play_id, plot_rusher_distances
from distances import calc_dist
from shapely import geometry

pd.set_option('display.max_columns', None)

def calc_pocket_area(qb_oline_coords_df):
    
    all_pocket_poly = []
    for i, row in qb_oline_coords_df.iterrows():
        
        all_xy_cords = row[~row.isna()]
        sorted_y_cords = all_xy_cords['y'].sort_values()
        
        oline_coords = []
        for pos in sorted_y_cords.index:
            if pos != 'QB':
                coords = (all_xy_cords['x'][pos], all_xy_cords['y'][pos])
                oline_coords.append(coords)
        
        oline_coords.append((all_xy_cords['x']['QB'], all_xy_cords['y']['QB']))
            
        pocket_poly = geometry.Polygon(oline_coords)
        all_pocket_poly.append(pocket_poly)
    return np.array(all_pocket_poly)

def gen_pocket_poly(merged_week_df):
    
    qb_pass_blocker_df = (merged_week_df
                          .query('(pff_role == "Pass Block") | (pff_role == "Pass") | (pff_positionLinedUp == "QB")')
                          )

    qb_oline_coords_df = qb_pass_blocker_df.pivot(index=['gameId', 'playId', 'frameId'], 
                                              columns=['pff_positionLinedUp'],values=['x','y'])
    
    qb_oline_coords_df['pocket_polygon'] = calc_pocket_area(qb_oline_coords_df)
    pocket_poly_df = gpd.GeoDataFrame(qb_oline_coords_df)
    pocket_poly_df = pocket_poly_df['pocket_polygon'].reset_index()
    pocket_poly_df['pocket_area'] = gpd.GeoSeries(pocket_poly_df['pocket_polygon']).area
    return pocket_poly_df
    
    


               