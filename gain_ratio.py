from typing import Dict
import pandas as pd
from math import log2


__all__ = ["calculate_gain_ratio"]


def calculate_gain_ratio(df: pd.DataFrame, target: str) -> Dict[str, float]:
    result: Dict[str, float] = {}

    target_entropy: float = _calculate_entropy(df[target])

    for c in df:
        entropy_before: float = target_entropy
        entropy_after: float = _calculate_child_entropy(df=df, target=target, feature=c)
        split_information: float = _calculate_entropy(df[c])
        result[c] = (entropy_before - entropy_after) / split_information

    return result


def _calculate_child_entropy(df: pd.DataFrame, target: str, feature: str) -> float:
    result: float = 0

    df_length = df.shape[0]

    for feature_value in df[feature].unique():
        tmp: pd.DataFrame = df[df[feature] == feature_value]
        p: float = tmp.shape[0] / df_length
        result += p * _calculate_entropy(tmp[target])

    return result


def _calculate_entropy(series: pd.Series) -> float:
    result: float = 0
    size: float = float(series.size)

    for it in series.unique():
        pi: float = series[series == it].shape[0] / size
        result -= pi * log2(pi)

    return result
