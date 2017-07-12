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
        ).astype(np.int)
        
    for p in players_names:
        new_df.loc(axis=1)[p, "Won"] =  (
                new_df.loc(axis=1)[p, "Played"] & (
                ((df.P1_Name == p) & (df.Player1Wins)) |
                ((df.P2_Name == p) & (~df.Player1Wins))
            )
        ).astype(np.int)
         
    return new_df


def compute_games_played_and_won__fast(df, players_names):
    """
    This is the same as the function above, but we compute things using Numpy directly for more
    efficiency
    Keeping both versions around to double check that they are the same...
    """
    df_copy = df.copy()
    players_names == np.sort(np.array(players_names)).reshape(1, -1)
    P1_Name = df_copy.P1_Name.values.reshape(-1, 1)
    P2_Name = df_copy.P2_Name.values.reshape(-1, 1)

    # We're relying on broadcasting here
    players_played = (players_names == P1_Name) | (players_names == P2_Name)
    """
            A   B
    A   B   x   x
    A   C   x
    D   B       x
    C   D   
    """
    players_won = (
        ((players_names == P1_Name) & (df_copy.Player1Wins.values.reshape(-1, 1))) |
        ((players_names == P2_Name) & (~df_copy.Player1Wins.values.reshape(-1, 1)))
    )

    # Interleaving the columns
    shape = players_played.shape
    played_and_won = np.empty((shape[0], shape[1] * 2), dtype=players_won.dtype)
    played_and_won[:, 0::2] = players_played
    played_and_won[:, 1::2] = players_won

    # Assigning the numpy array to the dataframe and returning it
    cols = pd.MultiIndex.from_product([players_names, ["Played", "Won"]], names=["Player", "Stat"])
    new_df = pd.DataFrame(index=df_copy.index, columns=cols)
    new_df.sort_index(axis=1, inplace=True)
    new_df.loc[:,:] = played_and_won.astype(int)

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
    round_types = sorted([round_t for round_t, round_c in round_counts.iteritems() if round_c > 10])

    for p in players_names:
        for round_t in round_types:
            temp_df.loc(axis=1)[p, "Won_" + round_t] = (
                (temp_df.loc(axis=1)[p, "Won"]) &
                (rounds == round_t)
            ).astype(np.int)

    return temp_df.sort_index(axis=1)


def compute_win_round_type__fast(df, rounds, players_names):
    """
    Same as above but implemented in pure Numpy for more performance
    """
    df_copy = df.copy()

    players_names == np.sort(np.array(players_names)).reshape(1, -1)
    round_counts = rounds.value_counts()
    # Filtering out odd values like "0th Round"
    round_types = sorted([round_t for round_t, round_c in round_counts.iteritems() if round_c > 10])

    # The "Won" columns are the odd columns
    players_won = df_copy.values[:, 1::2].astype(bool)

    # List of matrixes for winning a specific round
    players_won_rounds = []
    for round_t in round_types:
        players_won_specific_round = players_won & (rounds.values.reshape(-1, 1) == round_t)
        players_won_rounds.append(players_won_specific_round)

    # Now we have the original matrix of whether players won, as well as len(round_t) matrixes for
    # whether players won a specific rounds
    # Let's interleave all of those
    shape = players_won.shape
    won_rounds = np.empty((shape[0], shape[1] * len(round_types)), dtype=players_won.dtype)
    for idx, m in enumerate(players_won_rounds):
        won_rounds[:, idx::len(round_types)] = m

    # Assigning the numpy array to a new dataframe and returning it
    cols = pd.MultiIndex.from_product(
            [players_names, ["Won_" + rt for rt in round_types]],
    names=["Player", "Stat"])
    new_df = pd.DataFrame(index=df_copy.index, columns=cols)
    new_df.sort_index(axis=1, inplace=True)
    new_df.loc[:,:] = won_rounds.astype(int)

    return pd.concat([new_df, df_copy], axis=1).sort_index(axis=1)
