# -*- coding: utf-8 -*-
"""
The functions below will add derived features
"""

import numpy as np
import pandas as pd

"""
Below functions that add "atomic" features, ie extra columns for the original index/dataframe
(where every row is a given tennis match basically)
"""

def compute_games_played_and_won(df, players_names):
    """
    Will add a pair of boolean columns for every player specified in players_names

    eg. "Federer R._Played", "Federer R.__Won"
    """
    temp_df = df.copy()
    for p in players_names:
        temp_df[p + "__Played"] = ((temp_df.P1_Name == p) | (temp_df.P2_Name == p)).astype(np.int)
        
    for p in players_names:
        temp_df[p + "__Won"] =  (
            (temp_df[p + "__Played"]) &  (
                ((temp_df.P1_Name == p) & (temp_df.Player1Wins)) |
                ((temp_df.P2_Name == p) & (~temp_df.Player1Wins))
            )
        ).astype(np.int)
         
    return temp_df


def compute_win_round_type(df, players_names):
    """
    This will add booleans columns for the types of Round wins, ie "Federer R._Won_1st Round"

    The dataframe has to have the "XXX__Won" columns already for players in players_names
    """
    temp_df = df.copy()
    round_counts = df.Round.value_counts()

    # Filtering out odd values like "0th Round"
    round_types = [round_t for round_t, round_c in round_counts.iteritems() if round_c > 10]

    for p in players_names:
        for round_t in round_types:
            temp_df[p + "__Won_" + round_t ] = (
                (temp_df[p + "__Won"]) &
                (temp_df.Round == round_t)
            ).astype(np.int)

    return temp_df
