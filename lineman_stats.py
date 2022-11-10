# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 21:50:34 2022

@author: ben
"""
import pandas as pd
import numpy as np
from data_gen import NFLDataGenerator
from plots import plot_game_play_id, plot_rusher_distances

pd.set_option('display.max_columns', None)

all_top_pass_blockers_df = []
for i in range(1, 9):
    wi_data = NFLDataGenerator(week_ids=[i])

    top_pass_blockers_df = (wi_data
         .pff_scouting_df
         .query('pff_role == "Pass Block"')
         .groupby('nflId').apply(lambda x: pd.Series({'total_beats': np.sum(x['pff_beatenByDefender']),
                                                      'total_pass_blocks': len(x),
                                                      'pass_block_perc': (1 - (np.sum(x['pff_beatenByDefender']) / len(x)))}))
         .reset_index()
         .merge(wi_data.players_df, how='left', on='nflId')
         .assign(week = 'week_{}'.format(i))
         )
    
    all_top_pass_blockers_df.append(top_pass_blockers_df)
all_top_pass_blockers_df = pd.concat(all_top_pass_blockers_df)
wtd_pass_block_df = (all_top_pass_blockers_df
     .fillna(0)
     .groupby('nflId', group_keys=False)
     .apply(lambda x: 
            (x
             .sort_values('week', ascending=True)
             .assign(wtd_pass_block_perc = lambda y: 1 - (y['total_beats'].cumsum() / y['total_pass_blocks'].cumsum())))))

top_c = wtd_pass_block_df.query('officialPosition == "C"')
pivoted_c_df = top_c.pivot(index=['nflId', 'displayName', 'officialPosition'],columns='week', values=['wtd_pass_block_perc'])