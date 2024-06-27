from dateutil import tz
from datetime import datetime

import pandas as pd


__all__ = ['preprocess_features']


def preprocess_features(df: pd.DataFrame) -> None:
    _preprocess_formatted_date(df)
    _preprocess_summary_columns(df)
    _preprocess_recip_type(df)
    _preprocess_loud_cover(df)
    _handle_remissions(df)


def _preprocess_formatted_date(df: pd.DataFrame) -> None:
    to_zone: tz.tzlocal = tz.tzlocal()
    dates: pd.Series = df['Formatted Date'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S.%f %z").astimezone(tz=to_zone))
    df.drop(['Formatted Date'], axis=1, inplace=True)
    df['Year'] = dates.dt.year
    df['Month'] = dates.dt.month
    df['Day'] = dates.dt.day
    df['Weekday'] = dates.dt.weekday
    df['Hour'] = dates.dt.hour


def _preprocess_summary_columns(df: pd.DataFrame) -> None:
    df.drop('Summary', axis=1, inplace=True)
    df.drop('Daily Summary', axis=1, inplace=True)


def _preprocess_recip_type(df: pd.DataFrame) -> None:
    df['Rain'] = df['Precip Type'].apply(lambda x: float(x == 'rain'))
    df['Snow'] = df['Precip Type'].apply(lambda x: float(x == 'snow'))
    df.drop(['Precip Type'], axis=1, inplace=True)


def _preprocess_loud_cover(df: pd.DataFrame) -> None:
    df.drop('Loud Cover', axis=1, inplace=True)


def _handle_remissions(df: pd.DataFrame) -> None:
    df.drop(df.loc[df['Pressure (millibars)'] <= 0].index, inplace=True)
