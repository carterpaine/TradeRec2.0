
import streamlit as st
import pandas as pd
import numpy as np

# Generate artificial dataset (same as injected previously)
np.random.seed(42)
total_players = 900
position_players = 600
pitchers = 300

position_roles = ['C', '1B', '2B', 'SS', '3B', 'OF']
pitcher_roles = ['SP', 'RP', 'CP']

position_types = np.random.choice(position_roles, size=position_players)
pitcher_types = np.random.choice(pitcher_roles, size=pitchers, p=[0.5, 0.3, 0.2])

data = []
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

df = pd.DataFrame(data)

# Add Benefit Score
df["BenefitScore"] = np.where(
    df["Type"].isin(["C", "1B", "2B", "SS", "3B", "OF"]),
    df["WAR"] * 0.5 + df["ExitVelocity"].fillna(0) * 0.1 + df["SprintSpeed"].fillna(0) * 0.05 + df["DefensiveRating"].fillna(0) * 0.2,
    df["WAR"] * 0.5 + df["Velocity"].fillna(0) * 0.1 + df["SpinRate"].fillna(0) * 0.05 + df["HorizontalBreak"].fillna(0) * 0.2
)


# Add filters
teams = [f"Team{i}" for i in range(1, 31)]
df["Team"] = np.random.choice(teams, size=len(df))
positions = sorted(df["Type"].unique())

selected_team = st.selectbox("Select Team", ["All"] + teams)
selected_position = st.selectbox("Select Position/Type", ["All"] + positions)

# Apply filters
filtered_df = df.copy()
if selected_team != "All":
    filtered_df = filtered_df[filtered_df["Team"] == selected_team]
if selected_position != "All":
    filtered_df = filtered_df[filtered_df["Type"] == selected_position]

st.title("MLB Trade Recommender - Stat Leaderboard")
mode = st.radio("Select Mode", ["Regular Mode", "GM Mode"])

if mode == "Regular Mode":
    st.subheader("Top 10 WAR Leaders")
    cols = ["PlayerID", "Type", "BA", "OBP", "SLG", "RBI", "HR", "OPS", "WAR", "ERA", "WHIP", "Ks", "BB", "Velocity", "Wins", "Saves", "Holds"]
    st.dataframe(filtered_df.sort_values(by="WAR", ascending=False)[cols].head(10))
else:
    st.subheader("Top 10 GM Mode Benefit Scores")
    cols = ["PlayerID", "Type", "WAR", "BenefitScore", "ExitVelocity", "SprintSpeed", "DefensiveRating", "Velocity", "SpinRate", "HorizontalBreak"]
    st.dataframe(filtered_df.sort_values(by="BenefitScore", ascending=False)[cols].head(10))
