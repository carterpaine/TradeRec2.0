
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression

st.set_page_config(layout="wide")
st.title("MLB Trade Recommender System")

@st.cache_data
def generate_synthetic_dataset(n_players=900):
    np.random.seed(42)
    positions = ['C', '1B', '2B', '3B', 'SS', 'LF', 'CF', 'RF', 'DH', 'SP', 'RP']
    teams = ['ARI', 'LAD', 'SF', 'SD', 'COL', 'CHC', 'MIL', 'STL', 'PIT', 'CIN',
             'ATL', 'PHI', 'MIA', 'NYM', 'WAS', 'NYY', 'BOS', 'BAL', 'TB', 'TOR',
             'KC', 'DET', 'CWS', 'MIN', 'CLE', 'LAA', 'OAK', 'TEX', 'HOU', 'SEA']
    data = {
        "player_id": [f"Player{i}" for i in range(n_players)],
        "team": np.random.choice(teams, n_players),
        "position": np.random.choice(positions, n_players),
        "age": np.random.randint(20, 37, n_players),
        "WAR": np.round(np.random.normal(2.0, 1.5, n_players), 2),
        "contract_value": np.random.randint(500000, 35000000, n_players),
        "years_control": np.random.randint(1, 7, n_players),
        "sprint_speed": np.round(np.random.normal(27, 1.5, n_players), 2),
        "exit_velocity": np.round(np.random.normal(88, 5, n_players), 2),
        "defense_rating": np.round(np.random.uniform(0, 10, n_players), 2),
        "BA": np.round(np.random.uniform(.200, .330, n_players), 3),
        "OBP": np.round(np.random.uniform(.280, .400, n_players), 3),
        "SLG": np.round(np.random.uniform(.320, .600, n_players), 3),
        "OPS": np.round(np.random.uniform(.600, 1.000, n_players), 3),
        "SB": np.random.randint(0, 40, n_players),
        "HR": np.random.randint(0, 40, n_players),
        "RBI": np.random.randint(10, 120, n_players),
    }
    df = pd.DataFrame(data)
    return df[~df["position"].isin(["SP", "RP"])]

df = generate_synthetic_dataset()

with st.sidebar:
    st.header("Player Filters")
    gm_mode = st.checkbox("Enable GM Mode")
    if gm_mode:
        gm_team = st.selectbox("Select your team", sorted(df["team"].unique().tolist()))
        gm_position = st.selectbox("Filter by position", ["All"] + sorted(df["position"].unique().tolist()))
    else:
        selected_team = st.selectbox("Filter by team", ["All"] + sorted(df["team"].unique().tolist()))
        selected_position = st.selectbox("Filter by position", ["All"] + sorted(df["position"].unique().tolist()))

# FILTERING
filtered_df = df.copy()

if not gm_mode:
    if selected_team != "All":
        filtered_df = filtered_df[filtered_df["team"] == selected_team]
    if selected_position != "All":
        filtered_df = filtered_df[filtered_df["position"] == selected_position]
    top10 = filtered_df.sort_values(by="WAR", ascending=False).head(10)
    st.subheader("Top 10 Players by WAR (Traditional Stats)")
    st.dataframe(top10[["player_id", "team", "position", "BA", "OBP", "SLG", "OPS", "SB", "HR", "RBI", "WAR"]])
else:
    st.subheader(f"GM Mode: Value Over Replacement on {gm_team}")
    replacements = df[df["team"] == gm_team].groupby("position")["WAR"].max().to_dict()
    df["gm_team_replacement_WAR"] = df["position"].map(replacements)
    df["benefit"] = df["WAR"] - df["gm_team_replacement_WAR"]
    gm_filtered = df[df["benefit"] > 0]
    if gm_position != "All":
        gm_filtered = gm_filtered[gm_filtered["position"] == gm_position]
    gm_filtered = gm_filtered.sort_values(by="benefit", ascending=False)
    st.dataframe(gm_filtered[["player_id", "team", "position", "WAR", "gm_team_replacement_WAR", "benefit"]])


st.subheader("Visualizations")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**WAR Projection vs Age**")
    fig = px.scatter(filtered_df, x="age", y="predicted_WAR", color="position", hover_data=["player_id"])
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("**Team Contract Value vs WAR**")
    fig = px.scatter(filtered_df, x="contract_value", y="WAR", color="team", size="years_control")
    st.plotly_chart(fig, use_container_width=True)

if gm_mode:
    st.subheader("GM Mode Insights")
    filtered_df["benefit_score"] = (filtered_df["predicted_WAR"] / (filtered_df["contract_value"] / 1e6)) * filtered_df["years_control"]
    gm_display = filtered_df[["player_id", "team", "position", "predicted_WAR", "contract_value", "years_control", "benefit_score"]]
    st.dataframe(gm_display.sort_values("benefit_score", ascending=False))

# Download Report
st.subheader("Download Report")
st.download_button("Download Recommendations", data=filtered_df.to_csv(index=False), file_name="trade_recommendations.csv")

st.markdown("*WAR = Wins Above Replacement. Predicted WAR based on performance metrics. Trade suggestions are illustrative only.*")
