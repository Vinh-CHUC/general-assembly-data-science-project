# -*- coding: utf-8 -*-
import pandas as pd, numpy as np


def format_feature_coefs(feature_names, coefs):
    """
    This will return a dataframe:
    * Features | Coef
    """

    feature_and_coef = list(zip(feature_names, coefs))
    feature_and_coef = pd.DataFrame({
        "Feature":  [t[0] for t in feature_and_coef],
        "Coef":  [t[1] for t in feature_and_coef]
    })

    return feature_and_coef.set_index("Feature")


def sort_feature_coefs_by_bas(feature_coefs_df):
    abs_order = feature_coefs_df.Coef.abs().sort_values(ascending=False)
    return feature_coefs_df.copy().loc[abs_order.index]
