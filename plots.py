# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.io as pio
import plotly.graph_objects as go
from distances import dist_to_qb, calc_dist

#pio.renderers.default = 'browser'

COLORS = {
    'ARI':"#97233F", 
    'ATL':"#A71930", 
    'BAL':'#241773', 
    'BUF':"#00338D", 
    'CAR':"#0085CA", 
    'CHI':"#C83803", 
    'CIN':"#FB4F14", 
    'CLE':"#311D00", 
    'DAL':'#003594',
    'DEN':"#FB4F14", 
    'DET':"#0076B6", 
    'GB':"#203731", 
    'HOU':"#03202F", 
    'IND':"#002C5F", 
    'JAX':"#9F792C", 
    'KC':"#E31837", 
    'LA':"#003594", 
    'LAC':"#0080C6", 
    'LV':"#000000",
    'MIA':"#008E97", 
    'MIN':"#4F2683", 
    'NE':"#002244", 
    'NO':"#D3BC8D", 
    'NYG':"#0B2265", 
    'NYJ':"#125740", 
    'PHI':"#004C54", 
    'PIT':"#FFB612", 
    'SEA':"#69BE28", 
    'SF':"#AA0000",
    'TB':'#D50A0A', 
    'TEN':"#4B92DB", 
    'WAS':"#5A1414", 
    'football':'#CBB67C'
}

def gen_hover_txt(frame_df):
    hover_txt_arr = []
    for i, row in frame_df.iterrows():
        hover_txt_arr.append("nflId:{}<br>displayName:{}<br>Position:{}<br>Role:{}<br>DistToQb:{}".format(row["nflId"], row["displayName"], row["pff_positionLinedUp"], row["pff_role"], row['dist_to_qb']))
    return hover_txt_arr

def plot_play_df(fig, play_df, colors=COLORS):

    frame_id_dict = {}
    counter = 0
    for i, frameId in enumerate(play_df['frameId'].unique()):
        frame_df = play_df.query('frameId == @frameId').reset_index(drop=True)
        j_idx = []
        for team, group_df in frame_df.groupby('team'):
            
            hover_txt_arr = gen_hover_txt(group_df)
            
            if team != 'football' and team != 'pocket_polygon' and team != 'affected_pocket_polygon' and team != 'player_polygon':
                fig.add_trace(go.Scatter(x=group_df["x"], y=group_df["y"],mode = 'markers',marker_color=colors[team],name=team,
                                         hovertemplate=hover_txt_arr, visible=False))
                j_idx.append(counter)
                counter += 1
            elif team == 'football':
                fig.add_trace(go.Scatter(x=group_df["x"], y=group_df["y"],mode = 'markers',marker_color=colors[team],name=team,
                                         hovertemplate="", visible=False))
                j_idx.append(counter)
                counter += 1
            elif team == 'pocket_polygon':
                pocket_polygon = group_df['pocket_polygon'].values[0]
                xx, yy = pocket_polygon.exterior.coords.xy

                fig.add_trace(go.Scatter(x=list(xx), y=list(yy),fill='toself',hovertemplate="pocketarea: {}<br>".format(round(pocket_polygon.area,2)), fillcolor='rgba(255, 0, 0, 0.1)', visible=False))
                j_idx.append(counter)
                counter += 1
            elif team == 'affected_pocket_polygon':
                pocket_polygon = group_df['affected_pocket_polygon'].values[0]
                
                if pocket_polygon:
                    
                    if pocket_polygon.type =='MultiPolygon':
                        
                        xx, yy = [], []
                        for geom in pocket_polygon.geoms:
                            sub_xx, sub_yy = geom.exterior.coords.xy
                            xx.append(sub_xx), yy.append(sub_yy)
                            
                        xx = np.concatenate(np.array(xx))
                        yy = np.concatenate(np.array(yy))
                    
                    else:
                        xx, yy = pocket_polygon.exterior.coords.xy
                    
                    fig.add_trace(go.Scatter(x=list(xx), y=list(yy),fill='toself',line_color='pink', visible=False))
                
                else:
                    fig.add_trace(go.Scatter(x=[], y=[],fill='toself',line_color='pink', visible=False))
                
                
                j_idx.append(counter)
                counter += 1
            elif team == 'player_polygon':
                
                for i, row in group_df.iterrows():
                    
                    pff_role = row['pff_role']
                    player_polygon = row['player_polygon']
                    
                    if pff_role == 'Pass Blocker':
                        color = 'lightgreen'
                    else:
                        color = 'pink'
                
                    if player_polygon:
                        
                        if player_polygon.type =='MultiPolygon':
                            
                            xx, yy = [], []
                            for geom in pocket_polygon.geoms:
                                sub_xx, sub_yy = geom.exterior.coords.xy
                                xx.append(sub_xx), yy.append(sub_yy)
                                
                            xx = np.concatenate(np.array(xx))
                            yy = np.concatenate(np.array(yy))
                        
                        else:
                            xx, yy = player_polygon.exterior.coords.xy
                        
                        fig.add_trace(go.Scatter(x=list(xx), y=list(yy),fill='toself',line_color=color, visible=False, name=pff_role))
                    
                    else:
                        fig.add_trace(go.Scatter(x=[], y=[],fill='toself',line_color=color, visible=False, name=pff_role))
                    
                    j_idx.append(counter)
                    counter += 1
        frame_id_dict[frameId] = j_idx
    
    steps = []
    for i, frameId in enumerate(play_df['frameId'].unique()):
        step = dict(
            method="update",
            args=[{"visible": np.zeros((len(fig.data)), dtype=bool)},
                  {"title": "Frame: " + str(i)}],  # layout attribute
        )
        step["args"][0]["visible"][frame_id_dict[frameId]] = True  # Toggle i'th trace to "visible"
        steps.append(step)
       
    
    sliders = [{
            "active": 0,
            "yanchor": "top",
            "xanchor": "left",
            "steps": steps
        }]
    
    fig.update_layout(
        sliders=sliders
    )
    
def plot_game_play_id(game_df, gameId, playId, size=(800, 400)):
    play_df = game_df.query('(gameId == @gameId) & (playId == @playId)')
    play_df['displayName'] = np.where(play_df['team'].isin(['football','pocket_polygon','affected_pocket_polygon', 'player_polygon']), '',play_df['displayName'])
    
    fig = go.Figure( layout_yaxis_range=[0,53.3], layout_xaxis_range=[0,120])
    if 'dist_to_qb' not in play_df.columns:
        mod_play_df = dist_to_qb(play_df)
    else:
        mod_play_df = play_df
    
    plot_play_df(fig, mod_play_df)
    fig.update_layout(
        autosize=False,
        width=size[0],
        height=size[1])
    fig.show()
    
def plot_rusher_distances(game_df, gameId, playId):
    
    play_df = game_df.query('(gameId == @gameId) & (playId == @playId)')
    play_df['displayName'] = np.where(play_df['team'] == 'football', '',play_df['displayName'])
    mod_play_df = dist_to_qb(play_df)
    
    qb_df = play_df.query('officialPosition == "QB"')
    qb_df['calc_dis'] = calc_dist(qb_df['x'], qb_df['y'], qb_df['x'].shift(1), qb_df['y'].shift(1))
    qb_df['dis_cumsum'] = qb_df['calc_dis'].cumsum()
    qb_df['dis_cumsum_diff'] = np.gradient(qb_df['dis_cumsum'])
    
    pass_rushers_df = mod_play_df.query('pff_role == "Pass Rush"')
    fig, ax = plt.subplots(2,1, figsize=(12,8))
    sns.lineplot(pass_rushers_df, hue='displayName', x='frameId', y='dist_to_qb', ax=ax[0])
    sns.lineplot(qb_df, hue='displayName', x='frameId', y='dis_cumsum', ax=ax[0])
    sns.lineplot(qb_df, hue='displayName', x='frameId', y='dis', ax=ax[1])
    plt.show() 
    
if __name__ == '__main__':
    games_df = pd.read_csv('./data/games.csv')
    pff_scounting_df = pd.read_csv('./data/pffScoutingData.csv')
    players_df = pd.read_csv('./data/players.csv')
    plays_df = pd.read_csv('./data/plays.csv')
    week_df = pd.read_csv('./data/week1.csv')

    gameId = 2021090900
    playId = 2279
    play_df = week_df.query('(gameId == @gameId) & (playId == @playId)')

    merge_df = play_df.merge(players_df, how='left', on='nflId', validate='m:1')
    merge_df = merge_df.merge(pff_scounting_df, how='left', on=['nflId', 'gameId', 'playId'], validate='m:1')
    merge_df['displayName'] = np.where(merge_df['team'] == 'football', '',merge_df['displayName'])
    fig = go.Figure( layout_yaxis_range=[0,53.3], layout_xaxis_range=[0,120])
    plot_play_df(fig, merge_df)
    fig.show()