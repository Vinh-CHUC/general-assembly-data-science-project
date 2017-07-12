# -*- coding: utf-8 -*-
"""
The functions below will add derived features
"""
from itertools import product

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
    new_df = pd.DataFrame(data=played_and_won.astype(int), index=df_copy.index, columns=cols)

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


class FeatureEngineer(object):
    def __init__(self, player_names):
        self.player_names = np.sort(np.array(player_names)).reshape(1, -1)

    @staticmethod
    def _get_players_won_matrix(df):
        column_group = df.columns.levels[1]
        won_column_idx, column_group_l = list(column_group).index("Won"), len(column_group)

        return df.values[:, won_column_idx::column_group_l].astype(bool)

    @staticmethod
    def _get_players_stats_columns_count(df):
        return df.columns.levels[1].size

    def compute_games_played_and_won(self, df):
        """
        Args:
            df (pd.Dataframe): Dataframe with P1_Name, P2_Name and Player1Wins

        Returns:
            pd.Dataframe: With same index as df and two levels of columns:
                - Player name
                -- Played, Won (boolean saying if the player played and won the tennis game of that
                row)
        """
        P1_Name = df.P1_Name.values.reshape(-1, 1)
        P2_Name = df.P2_Name.values.reshape(-1, 1)
        Player1Wins = df.Player1Wins.values.reshape(-1, 1)

        # We're relying on broadcasting here
        players_played = (self.player_names == P1_Name) | (self.player_names == P2_Name)
        """
                A   B
        A   B   x   x
        A   C   x
        D   B       x
        C   D
        """
        players_won = (
            ((self.player_names == P1_Name) & Player1Wins) |
            ((self.player_names == P2_Name) & ~Player1Wins)
        )

        # Interleaving the columns
        shape = players_played.shape
        played_and_won = np.empty((shape[0], shape[1] * 2), dtype=players_won.dtype)
        played_and_won[:, 0::2] = players_played
        played_and_won[:, 1::2] = players_won

        # Assigning the numpy array to the dataframe and returning it
        cols = pd.MultiIndex.from_product(
            # It's important the "Played" < "Won" (lex order)
            [self.player_names.ravel(), ["Played", "Won"]],
            names=["Player", "Stat"]
        )
        new_df = pd.DataFrame(data=played_and_won.astype(int), index=df.index, columns=cols)

        return new_df

    def compute_win_round_type(self, df, rounds):
        """
        Args:
            df (pd.Dataframe): df with two levels of columns:
                - Player name
                -- .., Won, .. There has to be a subcolumn called "Won"
            rounds (pd.Series): series (same index as df) indicating the type of round of tennis
                games

        Returns:
            pd.Dataframe: With same index as df and two levels of columns:
                - Player name
                -- All the previous columns in df + "Won_The Final", "Won_.."
        """
        df_copy = df.sort_index(axis=1)

        # Filtering out odd values like "0th Round"
        round_types = sorted([
            round_t for round_t, round_c in rounds.value_counts().iteritems()
            if round_c > 10
        ])
        rounds = rounds.values.reshape(-1, 1)

        players_won = self._get_players_won_matrix(df_copy)

        # The output as a np matrix
        players_won_rounds = np.empty(
            (players_won.shape[0], players_won.shape[1] * len(round_types)),
            dtype=players_won.dtype
        )
        for idx, round_t in enumerate(round_types):
            # More broadcasting: (nbmatches, nbplayers) & (nbmatches, 1) --> (nbmatches, nbplayers)
            players_won_specific_round = players_won & (rounds == round_t)
            players_won_rounds[:, idx::len(round_types)] = players_won_specific_round

        # Assigning the numpy array to a new dataframe and returning it
        cols = pd.MultiIndex.from_product(
                [self.player_names.ravel(), ["Won_" + rt for rt in round_types]],
        names=["Player", "Stat"])  # Important players and rounds are already sorted here
        new_df = pd.DataFrame(data=players_won_rounds.astype(int), index=df.index, columns=cols)

        return pd.concat([new_df, df], axis=1)

    def compute_series_stats(self, df, series):
        """
        Args:
            df (pd.Dataframe): df with two levels of columns:
                - Player name
                -- [.., .., ..]: All the columns should be boolean
            rounds (pd.Series): series (same index as df) indicating the ATP Series of the tennis
                game played (eg. ATP250, Grand Slam...)

        Returns:
            df (pd.Dataframe): df with two levels of columns:
                - Player name
                -- [.., .., ..]: Copy of the data in the original boolean columns + copies of
                    those for each element in "Series"
        """
        df_copy = df.sort_index(axis=1)

        # The original stats: We're going to duplicate this + boolean mask for every series
        players_won = self._get_players_won_matrix(df_copy)

        series_type = np.sort(series.unique())
        series = series.values.reshape(-1, 1)  # Match id --> ATP series

        # The output as a np matrix
        player_stats_per_atp_series = np.empty(
            (players_won.shape[0], players_won.shape[1] * series_type.size),
            dtype=players_won.dtype
        )

        # Looping over "ATP250", "Grand Slam", ..
        for series_idx, series_t in enumerate(series_type):
            stat_per_series_type = players_won & (series == series_t)
            # Inserting that column in the right place
            player_stats_per_atp_series[:,series_idx::series_type.size] = stat_per_series_type

        # Assigning the numpy array to a new dataframe and returning it
        cols = pd.MultiIndex.from_product(
            [
                self.player_names.ravel(),  # Player names
                ["Won_" + s_t for s_t in series_type]
            ],
            names=["Player", "Stat"]  # Important players and rounds are already sorted here
        )
        new_df = pd.DataFrame(
            data=player_stats_per_atp_series.astype(int), index=df.index, columns=cols
        )

        return pd.concat([new_df, df], axis=1)


def compute_win_round_type__fast(df, rounds, players_names):
    """
    Same as above but implemented in pure Numpy for more performance
    """
    df_copy = df.sort_index(axis=1)

    players_names == np.sort(np.array(players_names)).reshape(1, -1)
    round_counts = rounds.value_counts()
    # Filtering out odd values like "0th Round"
    round_types = sorted([round_t for round_t, round_c in round_counts.iteritems() if round_c > 10])

    # The "Won" columns are the odd columns
    players_won = df_copy.values[:, 1::2].astype(bool)

    # List of matrixes for winning a specific round
    players_won_rounds = []
    for round_t in round_types:
        # More broadcasting: (nbmatches, nbplayers) & (nbmatches, 1) --> (nbmatches, nbplayers)
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
    names=["Player", "Stat"])  # Important players and rounds are already sorted here
    new_df = pd.DataFrame(data=won_rounds.astype(int), index=df_copy.index, columns=cols)

    return pd.concat([new_df, df_copy], axis=1, copy=False)
