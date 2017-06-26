# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

WINNER_COLS = ["Winner", "WRank", "W1", "W2", "W3", "W4", "W5", "Wsets"]
LOSER_COLS = ["Loser", "LRank", "L1", "L2", "L3", "L4", "L5", "Lsets"]

P1_COLS = ["P1_Name", "P1_Rank", "P1_1", "P1_2", "P1_3", "P1_4", "P1_5", "P1_Sets"]
P2_COLS = ["P2_Name", "P2_Rank", "P2_1", "P2_2", "P2_3", "P2_4", "P2_5", "P2_Sets"]


def number_players(df, choose_winner_l):
    """
    We need to somehow decide who will be "Player1" and "Player2". This function will "rename" all
    the player-specific columns to P1_{SomeCol} and P2_{SomeCol}

    choose_winner_l (lambda): When evaluated on a row of the dataframe, if it yields True then the
    winner will be labeled as Player1, and the loser as Player2. And vice-versa
    """

    new_df = pd.DataFrame(index=df.index)

    # If choose_winner_l yields True for a given row, then Winner becomes Player1, otherwise Loser
    # becomes Player1
    for idx, new_col in enumerate(P1_COLS):
        new_df[new_col] = np.where(
            choose_winner_l(df),
            df[WINNER_COLS[idx]],
            df[LOSER_COLS[idx]],
        )

    # And vice-versa
    for idx, new_col in enumerate(P2_COLS):
        new_df[new_col] = np.where(
            choose_winner_l(df),
            df[LOSER_COLS[idx]],
            df[WINNER_COLS[idx]],
        )

    return new_df
