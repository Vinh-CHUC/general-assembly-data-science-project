# -*- coding: utf-8 -*-
import pandas as pd


def convert_to_numeric(df, cols):
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    return df


def drop_nas(df, cols, *, verbose=True):
    if verbose:
        print(df[cols].isnull().sum())
    return df.dropna(subset=cols)
