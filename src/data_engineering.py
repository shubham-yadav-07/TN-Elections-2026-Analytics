"""
data_engineering.py
Tamil Nadu 2026 Elections — Data Loading, Cleaning & Feature Engineering
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# 1. LOADERS

def load_details() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "Tamil_Nadu_State_Elections_2026_Details.csv")

def load_metadata() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "Tamil_Nadu_State_Elections_2026_Constituency_Metadata.csv")

def load_alliance() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "Tamil_Nadu_State_Elections_2026_Alliance.csv")


# 2. CLEANING

def clean_details(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip().str.replace(" ", "_")
    df["Candidate"] = df["Candidate"].str.strip().str.title()
    df["Party"]     = df["Party"].str.strip().str.upper()
    df["Constituency"] = df["Constituency"].str.strip().str.title()

    # Numeric coercion
    for col in ["EVM_Votes", "Postal_Votes", "Total_Votes",
                "Tot_Constituency_votes_polled", "Winning_votes"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    df["%_of_Votes"] = pd.to_numeric(df["%_of_Votes"], errors="coerce").fillna(0.0)
    df["Win_Lost_Flag"] = df["Win_Lost_Flag"].astype(bool)
    return df

def clean_metadata(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip().str.replace(" ", "_")
    df["Constituency"] = df["Constituency"].str.strip().str.title()
    df["District"]     = df["District"].str.strip().str.title()
    df["Reserved"]     = df["Reserved"].str.strip().str.upper().fillna("GEN")
    df["Lok_sabha_constituency"] = df["Lok_sabha_constituency"].str.strip().str.title()
    df["Cons_vote_pct"] = pd.to_numeric(df["Cons_vote_pct"], errors="coerce")
    return df

def clean_alliance(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip().str.replace(" ", "_")
    df["PARTY_ABBR"]    = df["PARTY_ABBR"].str.strip().str.upper()
    df["ALLIANCE_NAME"] = df["ALLIANCE_NAME"].str.strip().str.title()
    return df

# 3. FEATURE ENGINEERING

def build_master(details: pd.DataFrame,
                 metadata: pd.DataFrame,
                 alliance: pd.DataFrame) -> pd.DataFrame:
    """
    Merge all three tables and engineer derived features.
    """
    master = details.merge(metadata, on="Constituency", how="left", suffixes=("", "_meta"))
    master = master.merge(
        alliance[["PARTY_ABBR", "ALLIANCE_NAME"]],
        left_on="Party", right_on="PARTY_ABBR", how="left"
    )
    master["ALLIANCE_NAME"] = master["ALLIANCE_NAME"].fillna("Independent / Others")

    # Winning margin %
    master["Win_Margin_Pct"] = np.where(
        master["Win_Lost_Flag"],
        (master["Winning_votes"] / master["Tot_Constituency_votes_polled"].replace(0, np.nan)) * 100,
        np.nan
    )

    # Postal vote ratio
    master["Postal_Ratio"] = master["Postal_Votes"] / (master["Total_Votes"].replace(0, np.nan))

    # Competitiveness score (lower margin → more competitive)
    master["Competitiveness"] = 100 - master["Win_Margin_Pct"].fillna(50)

    # Is reserved constituency flag
    master["Is_Reserved"] = master["Reserved"].isin(["SC", "ST"])

    # Candidate rank within constituency
    master["Rank_In_Const"] = master.groupby("Constituency")["Total_Votes"] \
                                    .rank(method="dense", ascending=False).astype(int)

    return master


def get_winners(master: pd.DataFrame) -> pd.DataFrame:
    return master[master["Win_Lost_Flag"]].reset_index(drop=True)


def get_party_summary(master: pd.DataFrame) -> pd.DataFrame:
    winners = get_winners(master)
    grp = master.groupby("Party").agg(
        Seats_Won    = ("Win_Lost_Flag", "sum"),
        Total_Votes  = ("Total_Votes",   "sum"),
        Constituencies_Contested = ("Constituency", "nunique"),
    ).reset_index()
    grp["Win_Rate_%"] = (grp["Seats_Won"] / grp["Constituencies_Contested"] * 100).round(2)
    grp["Vote_Share_%"] = (grp["Total_Votes"] / grp["Total_Votes"].sum() * 100).round(2)
    return grp.sort_values("Seats_Won", ascending=False)


def get_alliance_summary(master: pd.DataFrame) -> pd.DataFrame:
    winners = get_winners(master)
    seats = winners.groupby("ALLIANCE_NAME")["Constituency"].count().reset_index()
    seats.columns = ["Alliance", "Seats_Won"]
    votes = master.groupby("ALLIANCE_NAME")["Total_Votes"].sum().reset_index()
    votes.columns = ["Alliance", "Total_Votes"]
    summary = seats.merge(votes, on="Alliance")
    summary["Vote_Share_%"] = (summary["Total_Votes"] / summary["Total_Votes"].sum() * 100).round(2)
    return summary.sort_values("Seats_Won", ascending=False)


def get_district_summary(master: pd.DataFrame) -> pd.DataFrame:
    winners = get_winners(master)
    dist = winners.groupby(["District", "ALLIANCE_NAME"]).size().reset_index(name="Seats")
    meta = master.groupby("District").agg(
        Avg_Turnout  = ("Cons_vote_pct", "mean"),
        Total_Constituencies = ("Constituency", "nunique")
    ).reset_index()
    return dist.merge(meta, on="District")


def swing_zone_clustering(master: pd.DataFrame, n_clusters: int = 4) -> pd.DataFrame:
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans

    winners = get_winners(master)[["Constituency", "Win_Margin_Pct",
                                   "Cons_vote_pct", "Tot_parties_competed"]].dropna()
    X = StandardScaler().fit_transform(winners[["Win_Margin_Pct",
                                                 "Cons_vote_pct",
                                                 "Tot_parties_competed"]])
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    winners["Swing_Zone"] = km.fit_predict(X)
    labels = {0: "Safe Seat", 1: "Competitive", 2: "High Turnout Swing", 3: "Multi-Cornered"}
    winners["Swing_Label"] = winners["Swing_Zone"].map(labels)
    return winners


if __name__ == "__main__":
    det  = clean_details(load_details())
    meta = clean_metadata(load_metadata())
    all_ = clean_alliance(load_alliance())
    master = build_master(det, meta, all_)
    print(f"Master shape: {master.shape}")
    print(master.head(3))
    print("\nParty Summary:")
    print(get_party_summary(master).head(10))