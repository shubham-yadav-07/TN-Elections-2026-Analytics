"""
app.py  —  Tamil Nadu 2026 Elections · Streamlit Dashboard
Run:  streamlit run dashboard/app.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from src.data_engineering import (
    load_details, load_metadata, load_alliance,
    clean_details, clean_metadata, clean_alliance,
    build_master, get_winners, get_party_summary,
    get_alliance_summary, get_district_summary,
    swing_zone_clustering,
)
from src.analysis import (
    turnout_stats, margin_stats,
    top_competitive_seats, top_dominant_seats,
    postal_vs_evm_analysis, reserved_vs_general,
    ttest_reserved_vs_general, district_dominance,
    multi_cornered_contests, vote_share_concentration,
)
from src.visualizations import (
    fig_alliance_sunburst, fig_party_seats, fig_turnout_dist,
    fig_margin_bubble, fig_district_heatmap, fig_postal_evm,
    fig_reserved_box, fig_treemap, fig_radar_alliance,
    fig_3d_scatter, fig_3d_surface, fig_cluster_3d,
    fig_fragmentation, fig_corr_heatmap, fig_majority_gauge,
)

st.set_page_config(
    page_title="TN Elections 2026 · Analytics",
    page_icon="🗳️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CUSTOM CSS  

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background: #0d1117;
    color: #e6edf3;
  }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #161b22 0%, #0d1117 100%);
    border-right: 1px solid #30363d;
  }
  section[data-testid="stSidebar"] .block-container { padding-top: 1rem; }

  /* Main container */
  .main .block-container {
    padding-top: 1.5rem;
    max-width: 1400px;
  }

  /* KPI Cards — 3D glass effect */
  .kpi-card {
    background: linear-gradient(135deg, rgba(36,41,47,0.9), rgba(22,27,34,0.95));
    border: 1px solid #30363d;
    border-radius: 16px;
    padding: 1.4rem 1.2rem;
    text-align: center;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4),
                inset 0 1px 0 rgba(255,255,255,0.05);
    backdrop-filter: blur(12px);
    transition: transform 0.2s, box-shadow 0.2s;
  }
  .kpi-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 16px 48px rgba(0,0,0,0.6),
                inset 0 1px 0 rgba(255,255,255,0.08);
  }
  .kpi-value {
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(135deg, #58a6ff, #79c0ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1;
  }
  .kpi-label {
    font-size: 0.78rem;
    color: #8b949e;
    margin-top: 6px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }
  .kpi-sub {
    font-size: 0.9rem;
    color: #3fb950;
    margin-top: 4px;
    font-weight: 600;
  }

  /* Section headers */
  .section-header {
    font-size: 1.35rem;
    font-weight: 700;
    color: #e6edf3;
    padding: 0.6rem 0;
    border-bottom: 2px solid #238636;
    margin-bottom: 1rem;
  }

  /* Tab active color */
  button[data-baseweb="tab"][aria-selected="true"] {
    border-bottom-color: #58a6ff !important;
    color: #58a6ff !important;
  }

  /* Streamlit metric override */
  [data-testid="stMetric"] {
    background: rgba(22,27,34,0.8);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 0.8rem;
  }
  [data-testid="stMetricValue"] { color: #79c0ff !important; }

  /* Hide Streamlit branding */
  #MainMenu, footer { visibility: hidden; }
  header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# DATA LOAD 

@st.cache_data(show_spinner="Loading election data…")
def load_all():
    det   = clean_details(load_details())
    meta  = clean_metadata(load_metadata())
    alli  = clean_alliance(load_alliance())
    master = build_master(det, meta, alli)
    return master, alli

master, alli_raw = load_all()
winners     = get_winners(master)
party_sum   = get_party_summary(master)
alli_sum    = get_alliance_summary(master)
dist_sum    = get_district_summary(master)

# SIDEBAR FILTERS

st.sidebar.image("https://upload.wikimedia.org/wikipedia/en/9/97/Seal_of_Tamil_Nadu.png",
                 width=80)
st.sidebar.markdown("## 🗳️ TN Elections 2026")
st.sidebar.markdown("---")

districts = sorted(master["District"].dropna().unique())
sel_dist  = st.sidebar.multiselect("🏙️ Filter by District",
                                   districts, default=districts)

alliances = sorted(master["ALLIANCE_NAME"].dropna().unique())
sel_alli  = st.sidebar.multiselect("🤝 Filter by Alliance",
                                   alliances, default=alliances)

reserved_opts = sorted(master["Reserved"].dropna().unique())
sel_res = st.sidebar.multiselect("📋 Reservation Category",
                                 reserved_opts, default=reserved_opts)

st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset:** Kaggle · CC BY-SA 4.0  \n**Author:** Heisenricher")

# Apply filters
filt = master[
    master["District"].isin(sel_dist) &
    master["ALLIANCE_NAME"].isin(sel_alli) &
    master["Reserved"].isin(sel_res)
]
filt_win = get_winners(filt)


st.markdown("""
<div style="background:linear-gradient(135deg,#0d1117,#161b22);
            border:1px solid #30363d; border-radius:20px;
            padding:2rem 2.5rem; margin-bottom:1.5rem;
            box-shadow:0 20px 60px rgba(0,0,0,0.5);">
  <h1 style="margin:0;font-size:2.2rem;font-weight:800;
             background:linear-gradient(135deg,#58a6ff,#79c0ff,#3fb950);
             -webkit-background-clip:text;-webkit-text-fill-color:transparent;
             background-clip:text;">
    🗳️ Tamil Nadu 2026 Legislative Assembly Elections
  </h1>
  <p style="color:#8b949e;margin:0.4rem 0 0;font-size:1rem;">
    Comprehensive Political Analytics Dashboard · Data Science Project
  </p>
</div>
""", unsafe_allow_html=True)

# KPI CARDS (Row 1)

t_stats = turnout_stats(filt)
m_stats = margin_stats(filt)
top_alli = alli_sum.iloc[0] if len(alli_sum) else {"Alliance": "—", "Seats_Won": 0, "Vote_Share_%": 0}
total_const = filt["Constituency"].nunique()

kpi_html = f"""
<div style="display:grid;grid-template-columns:repeat(5,1fr);gap:1rem;margin-bottom:1.5rem;">
  <div class="kpi-card">
    <div class="kpi-value">{total_const}</div>
    <div class="kpi-label">Constituencies</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-value">{t_stats['mean']}%</div>
    <div class="kpi-label">Avg Voter Turnout</div>
    <div class="kpi-sub">σ = {t_stats['std']}%</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-value">{m_stats['mean']}%</div>
    <div class="kpi-label">Avg Win Margin</div>
    <div class="kpi-sub">Closest: {m_stats['closest_margin']}%</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-value">{int(top_alli['Seats_Won'])}</div>
    <div class="kpi-label">Leading Alliance Seats</div>
    <div class="kpi-sub">{top_alli['Alliance'][:22]}</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-value">{top_alli['Vote_Share_%']}%</div>
    <div class="kpi-label">Leading Vote Share</div>
    <div class="kpi-sub">Alliance vote %</div>
  </div>
</div>
"""
st.markdown(kpi_html, unsafe_allow_html=True)

# TABS

tabs = st.tabs([
    "🏠 Overview",
    "🤝 Alliances",
    "🏘️ Constituencies",
    "🗺️ District Analysis",
    "🧮 Voting Patterns",
    "👥 Demographics",
    "📊 3D Analytics",
    "🔬 Statistical Tests",
    "📋 Raw Data",
])


# OVERVIEW 

with tabs[0]:
    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.markdown('<div class="section-header">Majority Gauge</div>', unsafe_allow_html=True)
        st.plotly_chart(fig_majority_gauge(alli_sum), use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Alliance Vote & Seat Distribution</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(fig_alliance_sunburst(alli_sum), use_container_width=True)

    st.markdown('<div class="section-header">Seats Won — Top Parties</div>',
                unsafe_allow_html=True)
    st.plotly_chart(fig_party_seats(get_party_summary(filt)), use_container_width=True)

    st.markdown('<div class="section-header">Alliance → Party → Constituency Treemap</div>',
                unsafe_allow_html=True)
    st.plotly_chart(fig_treemap(filt), use_container_width=True)


# ALLIANCES

with tabs[1]:
    st.markdown('<div class="section-header">Alliance Summary Table</div>',
                unsafe_allow_html=True)
    st.dataframe(get_alliance_summary(filt).style
                 .background_gradient(cmap="Greens", subset=["Seats_Won"])
                 .format({"Vote_Share_%": "{:.2f}%"}),
                 use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_radar_alliance(alli_sum), use_container_width=True)
    with col2:
        alli_bar = get_alliance_summary(filt)
        import plotly.express as px
        fig = px.bar(alli_bar, x="Alliance", y=["Seats_Won"],
                     color="Vote_Share_%", color_continuous_scale="Plasma",
                     title="Alliance Seats vs Vote Share",
                     template="plotly_dark")
        fig.update_layout(paper_bgcolor="#161b22", plot_bgcolor="#0d1117")
        st.plotly_chart(fig, use_container_width=True)


# CONSTITUENCIES 

with tabs[2]:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">Most Competitive Seats</div>',
                    unsafe_allow_html=True)
        comp = top_competitive_seats(filt, 15)
        st.dataframe(comp.style.background_gradient(cmap="Reds",
                     subset=["Win_Margin_Pct"]).format(
                     {"Win_Margin_Pct": "{:.2f}%",
                      "Cons_vote_pct":  "{:.1f}%"}),
                     use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Most Dominant Victories</div>',
                    unsafe_allow_html=True)
        dom = top_dominant_seats(filt, 15)
        st.dataframe(dom.style.background_gradient(cmap="Greens",
                     subset=["Win_Margin_Pct"]).format(
                     {"Win_Margin_Pct": "{:.2f}%",
                      "Cons_vote_pct":  "{:.1f}%"}),
                     use_container_width=True)

    st.plotly_chart(fig_margin_bubble(filt_win), use_container_width=True)

    st.markdown('<div class="section-header">Multi-Cornered Contests (10+ parties)</div>',
                unsafe_allow_html=True)
    st.dataframe(multi_cornered_contests(filt), use_container_width=True)


# DISTRICTS

with tabs[3]:
    st.plotly_chart(fig_district_heatmap(filt), use_container_width=True)

    st.markdown('<div class="section-header">District Dominance by Alliance</div>',
                unsafe_allow_html=True)
    dd = district_dominance(filt)
    import plotly.express as px
    fig = px.bar(dd.head(30), x="District", y="Seats",
                 color="ALLIANCE_NAME",
                 color_discrete_sequence=px.colors.qualitative.Vivid,
                 template="plotly_dark",
                 title="Dominant Alliance per District")
    fig.update_layout(paper_bgcolor="#161b22", plot_bgcolor="#0d1117")
    st.plotly_chart(fig, use_container_width=True)


# VOTING PATTERNS 

with tabs[4]:
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_turnout_dist(filt), use_container_width=True)
    with col2:
        postal_df = postal_vs_evm_analysis(filt)
        st.plotly_chart(fig_postal_evm(postal_df), use_container_width=True)

    st.markdown('<div class="section-header">Vote Fragmentation Index (HHI)</div>',
                unsafe_allow_html=True)
    hhi = vote_share_concentration(filt)
    st.plotly_chart(fig_fragmentation(hhi), use_container_width=True)

    st.plotly_chart(fig_corr_heatmap(filt), use_container_width=True)


# DEMOGRAPHICS 

with tabs[5]:
    st.plotly_chart(fig_reserved_box(filt), use_container_width=True)

    st.markdown('<div class="section-header">Avg Stats: Reserved vs General</div>',
                unsafe_allow_html=True)
    rv = reserved_vs_general(filt)
    st.dataframe(rv.style.background_gradient(cmap="Blues",
                 subset=["Avg_Turnout"]).format(
                 {"Avg_Turnout": "{:.2f}%",
                  "Avg_Parties_Competed": "{:.1f}"}),
                 use_container_width=True)

    # Postal behaviour by reservation category
    df_res = filt.drop_duplicates("Constituency").copy()
    import plotly.express as px
    fig = px.violin(df_res, x="Reserved", y="Cons_vote_pct",
                    color="Reserved", box=True, points="all",
                    color_discrete_sequence=px.colors.qualitative.Vivid,
                    template="plotly_dark",
                    title="Turnout Distribution Violin — Reservation Categories")
    fig.update_layout(paper_bgcolor="#161b22", plot_bgcolor="#0d1117")
    st.plotly_chart(fig, use_container_width=True)


# 3D ANALYTICS

with tabs[6]:
    st.markdown('<div class="section-header">3D Scatter — Turnout × Win Margin × Parties</div>',
                unsafe_allow_html=True)
    st.plotly_chart(fig_3d_scatter(filt_win), use_container_width=True)

    st.markdown('<div class="section-header">3D Surface — Turnout × Parties → Avg Win Margin</div>',
                unsafe_allow_html=True)
    st.plotly_chart(fig_3d_surface(filt), use_container_width=True)

    st.markdown('<div class="section-header">KMeans Swing Zone Clustering (3D)</div>',
                unsafe_allow_html=True)
    with st.spinner("Running KMeans clustering…"):
        swing_df = swing_zone_clustering(filt)
    st.plotly_chart(fig_cluster_3d(swing_df), use_container_width=True)

    st.markdown('<div class="section-header">Swing Zone Constituency Table</div>',
                unsafe_allow_html=True)
    st.dataframe(swing_df[["Constituency", "Win_Margin_Pct",
                            "Cons_vote_pct", "Tot_parties_competed",
                            "Swing_Label"]]
                 .sort_values("Swing_Label"),
                 use_container_width=True)


# STATISTICAL TESTS 

with tabs[7]:
    st.markdown('<div class="section-header">Statistical Tests & Summaries</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📐 Turnout Statistics")
        ts = turnout_stats(filt)
        for k, v in ts.items():
            st.metric(k.replace("_", " ").title(), f"{v}")

    with col2:
        st.subheader("📐 Winning Margin Statistics")
        ms = margin_stats(filt)
        for k, v in ms.items():
            st.metric(k.replace("_", " ").title(), f"{v}")

    st.markdown("---")
    st.subheader("Independent Samples t-Test: Reserved vs General Turnout")
    tt = ttest_reserved_vs_general(filt)
    col1, col2, col3 = st.columns(3)
    col1.metric("T-Statistic", tt["t_stat"])
    col2.metric("P-Value", tt["p_value"])
    col3.metric("Statistically Significant?",
                "Yes" if tt["significant"] else "No")
    if tt["significant"]:
        st.success("There IS a statistically significant difference in voter turnout between reserved and general constituencies (p < 0.05).")
    else:
        st.info("No statistically significant difference found (p ≥ 0.05).")

    st.markdown("---")
    st.subheader("Party-wise Summary")
    st.dataframe(get_party_summary(filt).style
                 .background_gradient(cmap="YlOrRd", subset=["Seats_Won"])
                 .format({"Win_Rate_%": "{:.1f}%", "Vote_Share_%": "{:.2f}%"}),
                 use_container_width=True)


# RAW DATA 

with tabs[8]:
    st.markdown('<div class="section-header">Raw Master Dataset</div>',
                unsafe_allow_html=True)
    st.info(f"Showing {len(filt):,} rows after filters.")
    st.dataframe(filt, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        csv = filt.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Filtered CSV",
                           data=csv,
                           file_name="TN_Elections_2026_filtered.csv",
                           mime="text/csv")
    with col2:
        st.markdown(f"**Total Constituencies:** {filt['Constituency'].nunique()}")
        st.markdown(f"**Total Candidates:** {filt['Candidate'].nunique()}")
        st.markdown(f"**Total Parties:** {filt['Party'].nunique()}")
        st.markdown(f"**Total Votes Cast:** {filt['Total_Votes'].sum():,}")