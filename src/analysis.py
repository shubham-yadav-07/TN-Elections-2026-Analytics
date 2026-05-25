"""
analysis.py
Tamil Nadu 2026 Elections — Statistical Analysis & Pattern Mining
"""

import pandas as pd
import numpy as np
from scipy import stats
from src.data_engineering import (
    load_details, load_metadata, load_alliance,
    clean_details, clean_metadata, clean_alliance,
    build_master, get_winners, get_party_summary,
    get_alliance_summary, get_district_summary
)


def build_all():
    det   = clean_details(load_details())
    meta  = clean_metadata(load_metadata())
    alli  = clean_alliance(load_alliance())
    master = build_master(det, meta, alli)
    return master

# DESCRIPTIVE STATS

def turnout_stats(master: pd.DataFrame) -> dict:
    meta_unique = master.drop_duplicates("Constituency")
    turnout = meta_unique["Cons_vote_pct"].dropna()
    return {
        "mean":   round(turnout.mean(), 2),
        "median": round(turnout.median(), 2),
        "std":    round(turnout.std(), 2),
        "min":    round(turnout.min(), 2),
        "max":    round(turnout.max(), 2),
        "skew":   round(float(stats.skew(turnout)), 3),
        "kurtosis": round(float(stats.kurtosis(turnout)), 3),
    }


def margin_stats(master: pd.DataFrame) -> dict:
    winners = get_winners(master)
    m = winners["Win_Margin_Pct"].dropna()
    return {
        "mean":   round(m.mean(), 2),
        "median": round(m.median(), 2),
        "std":    round(m.std(), 2),
        "closest_margin": round(m.min(), 4),
        "largest_margin": round(m.max(), 2),
    }

# PATTERN DETECTION

def top_competitive_seats(master: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    winners = get_winners(master)[["Constituency", "Candidate", "Party",
                                   "ALLIANCE_NAME", "Win_Margin_Pct",
                                   "Tot_parties_competed", "Cons_vote_pct"]]
    return winners.nsmallest(n, "Win_Margin_Pct")


def top_dominant_seats(master: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    winners = get_winners(master)[["Constituency", "Candidate", "Party",
                                   "ALLIANCE_NAME", "Win_Margin_Pct",
                                   "Cons_vote_pct"]]
    return winners.nlargest(n, "Win_Margin_Pct")


def postal_vs_evm_analysis(master: pd.DataFrame) -> pd.DataFrame:
    party = master.groupby("Party").agg(
        EVM_Votes    = ("EVM_Votes",    "sum"),
        Postal_Votes = ("Postal_Votes", "sum"),
        Total_Votes  = ("Total_Votes",  "sum"),
    ).reset_index()
    party["Postal_Pct"] = (party["Postal_Votes"] / party["Total_Votes"] * 100).round(2)
    return party.sort_values("Total_Votes", ascending=False).head(20)


def reserved_vs_general(master: pd.DataFrame) -> pd.DataFrame:
    df = master.drop_duplicates("Constituency")
    summary = df.groupby("Reserved").agg(
        Avg_Turnout         = ("Cons_vote_pct",      "mean"),
        Avg_Parties_Competed = ("Tot_parties_competed", "mean"),
        Count               = ("Constituency",        "count"),
    ).reset_index()
    summary["Avg_Turnout"]          = summary["Avg_Turnout"].round(2)
    summary["Avg_Parties_Competed"] = summary["Avg_Parties_Competed"].round(2)
    return summary


def ttest_reserved_vs_general(master: pd.DataFrame):
    df = master.drop_duplicates("Constituency")
    reserved = df[df["Is_Reserved"]]["Cons_vote_pct"].dropna()
    general  = df[~df["Is_Reserved"]]["Cons_vote_pct"].dropna()
    t, p = stats.ttest_ind(reserved, general)
    return {"t_stat": round(t, 4), "p_value": round(p, 6),
            "significant": p < 0.05}


def district_dominance(master: pd.DataFrame) -> pd.DataFrame:
    winners = get_winners(master)
    # Which alliance dominates each district
    dom = winners.groupby(["District", "ALLIANCE_NAME"]).size().reset_index(name="Seats")
    idx = dom.groupby("District")["Seats"].idxmax()
    return dom.loc[idx].sort_values("Seats", ascending=False)


def multi_cornered_contests(master: pd.DataFrame) -> pd.DataFrame:
    df = master.drop_duplicates("Constituency")
    return df[df["Tot_parties_competed"] >= 10] \
             [["Constituency", "District", "Tot_parties_competed",
               "Cons_vote_pct"]].sort_values("Tot_parties_competed", ascending=False)


def vote_share_concentration(master: pd.DataFrame) -> pd.DataFrame:
    """Herfindahl index per constituency — measures vote fragmentation"""
    df = master.copy()
    df["vote_sq"] = (df["%_of_Votes"] / 100) ** 2
    hhi = df.groupby("Constituency")["vote_sq"].sum().reset_index()
    hhi.columns = ["Constituency", "HHI"]
    hhi["Fragmentation"] = 1 - hhi["HHI"]   # 0=monopoly, 1=max fragmented
    return hhi.sort_values("Fragmentation", ascending=False)


if __name__ == "__main__":
    master = build_all()
    print("Turnout Stats:", turnout_stats(master))
    print("Margin Stats:", margin_stats(master))
    print("\nTop 5 Competitive Seats:")
    print(top_competitive_seats(master, 5))
    print("\nReserved vs General t-test:")
    print(ttest_reserved_vs_general(master))