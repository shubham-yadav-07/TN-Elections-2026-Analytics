"""
visualizations.py
Tamil Nadu 2026 Elections — All Plotly Charts (2D + 3D)
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


PALETTE = px.colors.qualitative.Vivid
BG      = "#0d1117"
PAPER   = "#161b22"
FONT    = "#e6edf3"
GRID    = "#30363d"


def base_layout(title: str) -> dict:
    return dict(
        title=dict(text=title, font=dict(size=20, color=FONT)),
        paper_bgcolor=PAPER,
        plot_bgcolor=BG,
        font=dict(color=FONT),
        xaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
        yaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
        margin=dict(l=40, r=20, t=60, b=40),
    )


# 1. ALLIANCE SEAT SHARE — Sunburst

def fig_alliance_sunburst(alliance_summary: pd.DataFrame) -> go.Figure:
    fig = px.sunburst(
        alliance_summary,
        path=["Alliance"],
        values="Seats_Won",
        color="Vote_Share_%",
        color_continuous_scale="RdYlGn",
        title="Alliance Seat & Vote Share",
    )
    fig.update_layout(paper_bgcolor=PAPER, font_color=FONT,
                      title_font_size=20, margin=dict(t=60, b=10))
    return fig


# 2. PARTY SEATS BAR — Horizontal


def fig_party_seats(party_summary: pd.DataFrame, top_n: int = 15) -> go.Figure:
    df = party_summary.head(top_n).sort_values("Seats_Won")
    fig = go.Figure(go.Bar(
        x=df["Seats_Won"], y=df["Party"],
        orientation="h",
        marker=dict(color=df["Seats_Won"],
                    colorscale="Plasma",
                    showscale=True),
        text=df["Seats_Won"], textposition="outside",
    ))
    fig.update_layout(**base_layout(f"Top {top_n} Parties by Seats Won"))
    return fig


# 3. TURNOUT DISTRIBUTION — Histogram + KDE

def fig_turnout_dist(master: pd.DataFrame) -> go.Figure:
    df = master.drop_duplicates("Constituency")
    turnout = df["Cons_vote_pct"].dropna()
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=turnout, nbinsx=30,
        name="Frequency",
        marker_color="#58a6ff",
        opacity=0.75,
    ))
    # KDE overlay
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(turnout)
    x_range = np.linspace(turnout.min(), turnout.max(), 200)
    fig.add_trace(go.Scatter(
        x=x_range,
        y=kde(x_range) * len(turnout) * (turnout.max()-turnout.min())/30,
        mode="lines", name="KDE",
        line=dict(color="#f78166", width=2.5),
    ))
    fig.update_layout(**base_layout("Voter Turnout Distribution (%)"))
    return fig


# 4. WINNING MARGIN SCATTER — Bubble

def fig_margin_bubble(winners: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        winners.dropna(subset=["Win_Margin_Pct", "Cons_vote_pct"]),
        x="Cons_vote_pct",
        y="Win_Margin_Pct",
        size="Winning_votes",
        color="ALLIANCE_NAME",
        hover_name="Constituency",
        hover_data=["Party", "Candidate"],
        color_discrete_sequence=PALETTE,
        size_max=40,
        title="Winning Margin % vs Turnout (Bubble = Winning Votes)",
        labels={"Cons_vote_pct": "Turnout %", "Win_Margin_Pct": "Win Margin %"},
    )
    fig.update_layout(**base_layout("Winning Margin % vs Turnout"))
    return fig

# 5. DISTRICT HEATMAP — Turnout


def fig_district_heatmap(master: pd.DataFrame) -> go.Figure:
    df = master.drop_duplicates("Constituency") \
               .groupby("District")["Cons_vote_pct"].mean().reset_index()
    df = df.sort_values("Cons_vote_pct")
    fig = go.Figure(go.Bar(
        x=df["Cons_vote_pct"],
        y=df["District"],
        orientation="h",
        marker=dict(color=df["Cons_vote_pct"],
                    colorscale="YlOrRd",
                    showscale=True,
                    colorbar=dict(title="Turnout %")),
    ))
    fig.update_layout(**base_layout("Average Voter Turnout by District (%)"))
    fig.update_layout(height=max(400, len(df)*22))
    return fig


# 6. POSTAL vs EVM — Grouped Bar

def fig_postal_evm(postal_df: pd.DataFrame) -> go.Figure:
    df = postal_df.head(12)
    fig = go.Figure()
    fig.add_trace(go.Bar(name="EVM Votes",
                         x=df["Party"], y=df["EVM_Votes"],
                         marker_color="#58a6ff"))
    fig.add_trace(go.Bar(name="Postal Votes",
                         x=df["Party"], y=df["Postal_Votes"],
                         marker_color="#f78166"))
    fig.update_layout(**base_layout("EVM vs Postal Votes — Top 12 Parties"))
    fig.update_layout(barmode="group")
    return fig


# 7. RESERVED vs GENERAL — Box Plot

def fig_reserved_box(master: pd.DataFrame) -> go.Figure:
    df = master.drop_duplicates("Constituency")
    fig = px.box(
        df, x="Reserved", y="Cons_vote_pct",
        color="Reserved",
        points="all",
        color_discrete_sequence=PALETTE,
        title="Turnout Distribution: Reserved vs General Constituencies",
        labels={"Cons_vote_pct": "Voter Turnout %", "Reserved": "Category"},
    )
    fig.update_layout(**base_layout("Turnout: Reserved vs General Constituencies"))
    return fig

# 8. VOTE SHARE TREEMAP — Party → Constituency
def fig_treemap(master: pd.DataFrame) -> go.Figure:
    winners = master[master["Win_Lost_Flag"]].copy()
    fig = px.treemap(
        winners,
        path=[px.Constant("Tamil Nadu"), "ALLIANCE_NAME", "Party", "Constituency"],
        values="Total_Votes",
        color="Win_Margin_Pct",
        color_continuous_scale="RdYlGn",
        hover_data=["Candidate"],
        title="Alliance → Party → Constituency Seat Map",
    )
    fig.update_layout(paper_bgcolor=PAPER, font_color=FONT,
                      title_font_size=20, margin=dict(t=60))
    return fig

# 9. COMPETITIVENESS RADAR — Alliance

def fig_radar_alliance(alliance_summary: pd.DataFrame) -> go.Figure:
    categories = ["Seats_Won", "Vote_Share_%"]
    fig = go.Figure()
    for _, row in alliance_summary.iterrows():
        fig.add_trace(go.Scatterpolar(
            r=[row["Seats_Won"], row["Vote_Share_%"]],
            theta=categories,
            fill="toself",
            name=row["Alliance"],
        ))
    fig.update_layout(
        polar=dict(bgcolor=BG,
                   radialaxis=dict(visible=True, gridcolor=GRID),
                   angularaxis=dict(gridcolor=GRID)),
        paper_bgcolor=PAPER,
        font_color=FONT,
        title=dict(text="Alliance Performance Radar", font_size=20),
        showlegend=True,
    )
    return fig

# 10. 3D SCATTER — Turnout × Margin × Parties
def fig_3d_scatter(winners: pd.DataFrame) -> go.Figure:
    df = winners.dropna(subset=["Win_Margin_Pct", "Cons_vote_pct", "Tot_parties_competed"])
    fig = px.scatter_3d(
        df,
        x="Cons_vote_pct",
        y="Win_Margin_Pct",
        z="Tot_parties_competed",
        color="ALLIANCE_NAME",
        size="Winning_votes",
        hover_name="Constituency",
        hover_data=["Party", "Candidate"],
        color_discrete_sequence=PALETTE,
        size_max=20,
        title="3D: Turnout × Win Margin × Parties Contested",
        labels={
            "Cons_vote_pct":       "Turnout %",
            "Win_Margin_Pct":      "Win Margin %",
            "Tot_parties_competed":"# Parties",
        },
    )
    fig.update_layout(
        paper_bgcolor=PAPER,
        font_color=FONT,
        scene=dict(
            bgcolor=BG,
            xaxis=dict(gridcolor=GRID, title="Turnout %"),
            yaxis=dict(gridcolor=GRID, title="Win Margin %"),
            zaxis=dict(gridcolor=GRID, title="Parties Contested"),
        ),
        title_font_size=20,
        margin=dict(l=0, r=0, t=60, b=0),
    )
    return fig

# 11. 3D SURFACE — District Turnout Grid

def fig_3d_surface(master: pd.DataFrame) -> go.Figure:
    df = master.drop_duplicates("Constituency").dropna(subset=["Cons_vote_pct", "Tot_parties_competed"])
    # Bin turnout into grid
    df["turnout_bin"]  = pd.cut(df["Cons_vote_pct"],        bins=10, labels=False)
    df["parties_bin"]  = pd.cut(df["Tot_parties_competed"], bins=10, labels=False)
    pivot = df.pivot_table(values="Win_Margin_Pct",
                           index="turnout_bin",
                           columns="parties_bin",
                           aggfunc="mean").fillna(0)
    fig = go.Figure(go.Surface(
        z=pivot.values,
        colorscale="Viridis",
        colorbar=dict(title="Avg Win Margin %"),
    ))
    fig.update_layout(
        title=dict(text="3D Surface: Turnout × Parties → Win Margin",
                   font_size=20, font_color=FONT),
        paper_bgcolor=PAPER,
        font_color=FONT,
        scene=dict(
            bgcolor=BG,
            xaxis_title="Parties Contested (bin)",
            yaxis_title="Turnout % (bin)",
            zaxis_title="Avg Win Margin %",
        ),
        margin=dict(l=0, r=0, t=60, b=0),
    )
    return fig

# 12. SWING ZONE CLUSTER — 3D

def fig_cluster_3d(swing_df: pd.DataFrame) -> go.Figure:
    fig = px.scatter_3d(
        swing_df,
        x="Win_Margin_Pct",
        y="Cons_vote_pct",
        z="Tot_parties_competed",
        color="Swing_Label",
        hover_name="Constituency",
        color_discrete_sequence=PALETTE,
        title="3D KMeans Swing Zone Clustering",
        labels={
            "Win_Margin_Pct":      "Win Margin %",
            "Cons_vote_pct":       "Turnout %",
            "Tot_parties_competed":"# Parties",
        },
    )
    fig.update_layout(
        paper_bgcolor=PAPER, font_color=FONT,
        scene=dict(bgcolor=BG,
                   xaxis=dict(gridcolor=GRID),
                   yaxis=dict(gridcolor=GRID),
                   zaxis=dict(gridcolor=GRID)),
        title_font_size=20,
    )
    return fig

# 13. VOTE FRAGMENTATION — HHI Bar

def fig_fragmentation(hhi_df: pd.DataFrame, top_n: int = 20) -> go.Figure:
    df = hhi_df.head(top_n)
    fig = go.Figure(go.Bar(
        x=df["Fragmentation"],
        y=df["Constituency"],
        orientation="h",
        marker=dict(color=df["Fragmentation"],
                    colorscale="Turbo",
                    showscale=True),
    ))
    fig.update_layout(**base_layout(f"Top {top_n} Most Fragmented Constituencies (HHI)"))
    fig.update_layout(height=500)
    return fig

# 14. CORRELATION HEATMAP

def fig_corr_heatmap(master: pd.DataFrame) -> go.Figure:
    cols = ["Cons_vote_pct", "Tot_parties_competed", "Win_Margin_Pct",
            "Postal_Ratio", "Winning_votes", "%_of_Votes"]
    df = master.drop_duplicates("Constituency")[cols].dropna()
    corr = df.corr().round(2)
    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale="RdBu",
        zmid=0,
        text=corr.values,
        texttemplate="%{text}",
        showscale=True,
    ))
    fig.update_layout(**base_layout("Feature Correlation Heatmap"))
    return fig

# 15. KPI GAUGE — Overall Alliance Majority

def fig_majority_gauge(alliance_summary: pd.DataFrame,
                       total_seats: int = 234) -> go.Figure:
    top = alliance_summary.iloc[0]
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=int(top["Seats_Won"]),
        delta={"reference": total_seats // 2 + 1, "valueformat": ".0f"},
        title={"text": f"{top['Alliance']}<br>Seats Won",
               "font": {"color": FONT, "size": 18}},
        gauge={
            "axis": {"range": [0, total_seats], "tickcolor": FONT},
            "bar":  {"color": "#238636"},
            "steps": [
                {"range": [0, total_seats//2],         "color": "#21262d"},
                {"range": [total_seats//2, total_seats],"color": "#161b22"},
            ],
            "threshold": {
                "line": {"color": "#f78166", "width": 4},
                "thickness": 0.75,
                "value": total_seats // 2 + 1,
            },
        },
        number={"font": {"color": FONT}},
    ))
    fig.update_layout(paper_bgcolor=PAPER, font_color=FONT,
                      height=300, margin=dict(t=40, b=10, l=20, r=20))
    return fig