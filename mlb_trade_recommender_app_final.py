
import pandas as pd
import numpy as np

np.random.seed(42)

# Define counts and roles
total_players = 900
position_players = 600
pitchers = 300

position_roles = ['C', '1B', '2B', 'SS', '3B', 'OF']
pitcher_roles = ['RP']

# Assign roles
position_types = np.random.choice(position_roles, size=position_players)
pitcher_types = np.random.choice(pitcher_roles, size=pitchers, p=[1.0])

# Generate data
data = []

# Position players
for i in range(position_players):
    role = position_types[i]
    entry = {
        'PlayerID': f"Player{i+1:03d}",
        'Type': role,
        'BA': round(np.random.uniform(0.200, 0.330), 3),
        'OBP': round(np.random.uniform(0.280, 0.420), 3),
        'SLG': round(np.random.uniform(0.350, 0.600), 3),
        'RBI': np.random.randint(30, 120),
        'HR': np.random.randint(5, 50),
        'OPS': round(np.random.uniform(0.650, 1.000), 3),
        'WAR': round(np.random.uniform(-1.0, 8.0), 2),
        'ExitVelocity': round(np.random.uniform(85, 95), 1),
        'SprintSpeed': round(np.random.uniform(26.0, 30.0), 2),
        'DefensiveRating': round(np.random.uniform(-10, 20), 1),
        'ERA': np.nan,
        'WHIP': np.nan,
        'Ks': np.nan,
        'BB': np.nan,
        'Velocity': np.nan,
        'SpinRate': np.nan,
        'HorizontalBreak': np.nan,
        'Wins': np.nan,
        'Saves': np.nan,
        'Holds': np.nan
    }
    data.append(entry)

# Pitchers
for i in range(pitchers):
    role = pitcher_types[i]
    pid = f"Player{i+position_players+1:03d}"

    era = round(np.random.uniform(1.5, 6.0), 2)
    whip = round(np.random.uniform(0.9, 1.6), 2)
    ks = np.random.randint(40, 250)
    bb = np.random.randint(10, 80)
    velo = round(np.random.uniform(88, 100), 1)
    spin = round(np.random.uniform(2000, 2800), 1)
    hbreak = round(np.random.uniform(5.0, 20.0), 2)
    war = round(np.random.uniform(-1.0, 6.0), 2)

    wins = np.random.randint(5, 20) if role == 'SP' else np.random.randint(0, 10)
    saves = np.random.randint(20, 45) if role == 'CP' else 0
    holds = np.random.randint(10, 30) if role == 'RP' else 0

    entry = {
        'PlayerID': pid,
        'Type': role,
        'BA': np.nan,
        'OBP': np.nan,
        'SLG': np.nan,
        'RBI': np.nan,
        'HR': np.nan,
        'OPS': np.nan,
        'WAR': war,
        'ExitVelocity': np.nan,
        'SprintSpeed': np.nan,
        'DefensiveRating': np.nan,
        'ERA': era,
        'WHIP': whip,
        'Ks': ks,
        'BB': bb,
        'Velocity': velo,
        'SpinRate': spin,
        'HorizontalBreak': hbreak,
        'Wins': wins,
        'Saves': saves,
        'Holds': holds
    }
    data.append(entry)

# Create DataFrame
df = pd.DataFrame(data)



import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

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
        "defense_rating": np.round(np.random.uniform(0, 10, n_players), 2)
    }
    return pd.DataFrame(data)

df = generate_synthetic_dataset()

with st.sidebar:
    st.header("Filter Players")
    selected_team = st.selectbox("Select Team", options=["All"] + sorted(df["team"].unique().tolist()))
    selected_position = st.selectbox("Select Position", options=["All"] + sorted(df["position"].unique().tolist()))
    gm_mode = st.checkbox("Enable GM Mode (Advanced Metrics)")

    filtered_df = df.copy()
    if selected_team != "All":
        filtered_df = filtered_df[filtered_df["team"] == selected_team]
    if selected_position != "All":
        filtered_df = filtered_df[filtered_df["position"] == selected_position]

st.subheader("Filtered Players")
st.dataframe(filtered_df)

# Modeling: Predict WAR using Linear Regression (illustrative only)
X = filtered_df[["age", "exit_velocity", "sprint_speed", "defense_rating"]]
y = filtered_df["WAR"]
reg = LinearRegression().fit(X, y)
filtered_df["predicted_WAR"] = reg.predict(X)

# Trade Recommendation (Logistic)
filtered_df["should_trade"] = (filtered_df["predicted_WAR"] < 2.0).astype(int)

# Visualizations
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

# Download Trade Suggestions
st.subheader("Download Trade Report")
st.download_button("Download CSV", data=filtered_df.to_csv(index=False), file_name="trade_recommendations.csv")

st.markdown("*WAR = Wins Above Replacement. Predicted WAR based on performance metrics. Trade suggestions are illustrative only.*")
