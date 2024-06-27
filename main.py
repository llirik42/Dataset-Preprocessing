from math import log2, floor
from typing import Dict

import numpy as np
import opendatasets as od
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression

from drawing import draw_heat_map, show_all
from gain_ratio import calculate_gain_ratio
from preprocessing import preprocess_features


def print_gain_ratio(df: pd.DataFrame, target: str) -> None:
    gain_ratio: Dict[str, float] = calculate_gain_ratio(df=df, target=target)

    print(target)
    for key, value in sorted(gain_ratio.items(), key=lambda x: x[1], reverse=True):
        print(f"\t{key}: {value}")
    print()


def classify(df: pd.DataFrame) -> pd.DataFrame:
    n: int = 1 + floor(log2(len(df)))
    q: np.ndarray = np.linspace(start=0, stop=1, num=n)

    cut_df: pd.DataFrame = pd.DataFrame()
    cut_df["Rain"] = pd.cut(x=df["Rain"], bins=[0, 0.5, 1], include_lowest=True)
    cut_df["Snow"] = pd.cut(x=df["Snow"], bins=[0, 0.5, 1], include_lowest=True)
    cut_df["Year"] = pd.cut(x=df["Year"], bins=range(2005, 2018), include_lowest=False)
    cut_df["Month"] = pd.cut(x=df["Month"], bins=range(0, 13), include_lowest=False)
    cut_df["Day"] = pd.cut(x=df["Day"], bins=range(0, 32), include_lowest=False)
    cut_df["Weekday"] = pd.cut(x=df["Weekday"], bins=range(-1, 7), include_lowest=False)
    cut_df["Hour"] = pd.cut(x=df["Hour"], bins=range(-1, 24), include_lowest=False)
    cut_df["Visibility (km)"] = pd.cut(x=df["Visibility (km)"], bins=[0, 9.5, 12, 17], include_lowest=True)

    manually_cut: list[str] = ["Rain", "Snow", "Year", "Month", "Day", "Weekday", "Hour", "Visibility (km)"]

    for c in df:
        if c not in manually_cut:
            cut_df[c] = pd.qcut(x=df[c], q=q)

    return cut_df


def select_best_k_features_sklearn(df: pd.DataFrame, k: int, target: str) -> pd.DataFrame:
    x: pd.DataFrame = df[[col for col in df.columns if col != target]]
    y: pd.DataFrame = df[target]
    reg = SelectKBest(k=k, score_func=f_regression).fit(X=x, y=y)
    return x[[val for i, val in enumerate(x.columns) if reg.get_support()[i]]]


def main() -> None:
    od.download("https://www.kaggle.com/datasets/budincsevity/szeged-weather")
    df: pd.DataFrame = pd.read_csv("szeged-weather/weatherHistory.csv")
    target: str = "Temperature (C)"

    preprocess_features(df)
    draw_heat_map(df.corr().abs().round(2))
    cut_df: pd.DataFrame = classify(df)
    print_gain_ratio(df=cut_df, target=target)

    best_k_features_df: pd.DataFrame = select_best_k_features_sklearn(df=df, k=5, target=target)
    for column in best_k_features_df.columns:
        print(column)

    show_all()


if __name__ == "__main__":
    main()
