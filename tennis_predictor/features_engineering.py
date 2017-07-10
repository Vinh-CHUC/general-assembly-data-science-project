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
    This will create a new dataframe with basic derived features for players

    The new dataframe will have columns over multiple levels:
    - Player name
    -- Played, Won
    """
    cols = pd.MultiIndex.from_product([players_names, ["Played", "Won"]], names=["Player", "Stat"])
    new_df = pd.DataFrame(index=df.index, columns=cols)
    new_df.sort_index(axis=1, inplace=True)

    for p in players_names:
        new_df.loc(axis=1)[p, "Played"] = (
                (df.P1_Name == p) | (df.P2_Name == p)
        )
        
    for p in players_names:
        new_df.loc(axis=1)[p, "Won"] =  (
                new_df.loc(axis=1)[p, "Played"] & (
                ((df.P1_Name == p) & (df.Player1Wins)) |
                ((df.P2_Name == p) & (~df.Player1Wins))
            )
        ).astype(np.int)
         
    return new_df


def compute_win_round_type(df, rounds, players_names):
    """
    This will add booleans columns to df for the types of Round wins, ie Federer R+Won_1st Round

    Args:
        df (DataFrame): a dataframe with two levels of columns. First being the player's name, the
            second level has to have a boolean column called "Won"
        rounds (Series): a series that describes the rounds of the tennis matches ("1st Round", ..)
            it has to be indexed the same as df
        players_names: an iterable of player names
    """
    temp_df = df.copy()
    round_counts = rounds.value_counts()
    # Filtering out odd values like "0th Round"
    round_types = [round_t for round_t, round_c in round_counts.iteritems() if round_c > 10]

    for p in players_names:
        for round_t in round_types:
            temp_df.loc(axis=1)[p, "Won_" + round_t] = (
                (temp_df.loc(axis=1)[p, "Won"]) &
                (rounds == round_t)
            ).astype(np.int)

    return temp_df
