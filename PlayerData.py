import nfl_data_py as nfl
import pandas as pd
import numpy as  np


def get_weekly_data() -> pd.DataFrame:
    years = [2024]

    # Info from each game
    cols = ['player_id', 'player_name', 'position', 'position_group', 'week',
             'passing_yards', 'passing_tds', 'interceptions', 'sack_fumbles_lost',
             'rushing_yards', 'rushing_tds', 'rushing_fumbles_lost', 'rushing_2pt_conversions',
             'receptions', 'receiving_yards', 'receiving_tds', 'receiving_fumbles_lost',
             'receiving_2pt_conversions', 'fantasy_points']
    

    df = nfl.import_weekly_data(years=years, columns=cols, downcast=True)

    return df


def get_player_ppg() -> pd.DataFrame:

    SCORING = {
        'passing_yards' : 1/25,
        'passing_tds' : 4,
        'interceptions' : -1,
        'rushing_yards' : 1/10,
        'rushing_tds': 6,
        'receiving_yards' : 1/10,
        'receiving_tds' : 6,
        'receptions' : 0.5,
        'rushing_2pt_conversions' : 2,
        'receiving_2pt_conversions' : 2,
        'sack_fumbles_lost' : -2,
        'rushing_fumbles_lost' : -2,
        'receiving_fumbles_lost' : -2
    }

    games_df = get_weekly_data()

    # Thanks @chatGPT
    games_df["fantasy_points"] = sum(
        games_df[col] * pts
        for col, pts in SCORING.items()
        if col in games_df.columns
        ).fillna(0)

    player_stats = (games_df
        .groupby(["player_id", "player_name", "position"])
        .agg(
            games_played=("week", "nunique"),
            ppg=("fantasy_points", "mean")
        )
        .reset_index())


    return player_stats


def query_player_ppg(ppg_df: pd.DataFrame, position: str | None = None, min_games: int | None = None) -> pd.DataFrame:
    mask = pd.Series(True, index=ppg_df.index)

    if position is not None:
        mask &= ppg_df["position"] == position

    if min_games is not None:
        mask &= ppg_df["games_played"] >= min_games

    return (
        ppg_df.loc[mask]
              .sort_values("ppg", ascending=False)
              .reset_index(drop=True)
    )