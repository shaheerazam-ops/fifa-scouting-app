import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="FIFA AI Ultimate Scout", layout="wide")

st.title("⚽ FIFA AI Ultimate Scout")
st.markdown("Scout players, simulate growth, and predict market value using ML 🤖")


# =========================
# CACHE DATA + MODEL
# =========================

@st.cache_resource 
def load_knn():
    model = pickle.load(open("knn_model.pkl","rb"))
    featuers = pickle.load(open("knn_features.pkl","rb"))
    scaler = pickle.load(open("knn_scaler.pkl","rb"))
    return model , featuers ,scaler

@st.cache_resource
def load_model():
    return pickle.load(open("fifa_model.pkl", "rb"))

@st.cache_data
def load_data():
    df = pd.read_csv("fifa_with_clusters.csv")  
    df["clean_name"] = df["Known As"].astype(str).str.strip().str.lower()
    return df

model = load_model()
df = load_data()
knn_model , knn_features , knn_scaler= load_knn()
cluster_names = {
    0: "Balanced Player",
    1: "Attacker",
    2: "Defender",
    3: "Physical Player",
    4: "Playmaker"
}

# =========================
# SIDEBAR NAV
# =========================
mode = st.sidebar.radio("Navigation", [
    "🏠 Player Scout",
    "⚔️ Compare",
    "🏆 Top Players",
    "🔎 Filter",
    "🏅 Best XI",
    "💰 Value Simulator"
])

# =========================
# FIFA STYLE PLAYER CARD
# =========================
def render_card(player):
    st.markdown("### 🎴 Player Card")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("OVR", player["Overall"])
        st.metric("POT", player["Potential"])

    with col2:
        st.metric("Age", player["Age"])
        st.metric("Position", player["Best Position"])

    with col3:
        st.metric("Value", f"€{player['Value(in Euro)']:,}")
        st.metric("Wage", f"€{player['Wage(in Euro)']:,}")

    st.divider()

    stats = pd.DataFrame({
        "Attribute": ["PAC", "SHO", "PAS", "DRI", "DEF", "PHY"],
        "Value": [
            player["Pace Total"],
            player["Shooting Total"],
            player["Passing Total"],
            player["Dribbling Total"],
            player["Defending Total"],
            player["Physicality Total"]
        ]
    })

    st.bar_chart(stats.set_index("Attribute"))

# =========================================================
# 🏠 PLAYER SCOUT (FIFA SEARCH + CARD)
# =========================================================
if mode == "🏠 Player Scout":

    st.subheader("🔍 Scout Players")

    query = st.text_input("Search Player (FIFA Style)").strip().lower()

    if query:
        results = df[df["clean_name"].str.contains(query, na=False)]

        if results.empty:
            st.warning("No player found")
        else:
            player = results.iloc[0]
            

            st.info(f"🧠 Playing Style: {cluster_names.get(player['cluster'], 'Unknown')}")

            st.success(f"Found: {player['Known As']}")
            render_card(player)

            st.subheader("🤝 Similar Players")

# Get player vector
            player_vector = player[knn_features].values.reshape(1, -1)
            player_vector = knn_scaler.transform(player_vector)

            # Find neighbors
            distances, indices = knn_model.kneighbors(player_vector)

            similar_players = df.iloc[indices[0][1:]]  # skip first (itself)

            for _, p in similar_players.iterrows():
                st.write(f"👉 {p['Known As']} ({p['Club Name']}) - OVR {p['Overall']}")

# =========================================================
# ⚔️ COMPARE PLAYERS
# =========================================================
elif mode == "⚔️ Compare":

    st.subheader("⚔️ Player Comparison")

    p1_name = st.text_input("Player 1").strip().lower()
    p2_name = st.text_input("Player 2").strip().lower()

    if p1_name and p2_name:

        p1 = df[df["clean_name"].str.contains(p1_name, na=False)]
        p2 = df[df["clean_name"].str.contains(p2_name, na=False)]

        if not p1.empty and not p2.empty:
            p1 = p1.iloc[0]
            p2 = p2.iloc[0]

            labels = ["PAC", "SHO", "PAS", "DRI", "DEF", "PHY"]

            fig, ax = plt.subplots()
            ax.plot(labels, [
                p1["Pace Total"], p1["Shooting Total"], p1["Passing Total"],
                p1["Dribbling Total"], p1["Defending Total"], p1["Physicality Total"]
            ], label=p1["Known As"])

            ax.plot(labels, [
                p2["Pace Total"], p2["Shooting Total"], p2["Passing Total"],
                p2["Dribbling Total"], p2["Defending Total"], p2["Physicality Total"]
            ], label=p2["Known As"])

            ax.legend()
            st.pyplot(fig)

# =========================================================
# 🏆 TOP PLAYERS
# =========================================================
elif mode == "🏆 Top Players":

    stat = st.selectbox("Select Stat", [
        "Overall", "Pace Total", "Shooting Total",
        "Passing Total", "Dribbling Total",
        "Defending Total", "Physicality Total",
        "Value(in Euro)"
    ])

    top = df.nlargest(10, stat)

    cols = ["Known As", "Club Name", "Overall"]

    if stat not in cols:
        cols.append(stat)

    st.dataframe(top[cols])

# =========================================================
# 🔎 FILTER
# =========================================================
elif mode == "🔎 Filter":

    pos = st.selectbox("Position", ["All"] + sorted(df["Best Position"].unique()))
    nation = st.selectbox("Nationality", ["All"] + sorted(df["Nationality"].unique()))
    min_ovr = st.slider("Min OVR", 50, 99, 75)

    filtered = df.copy()

    if pos != "All":
        filtered = filtered[filtered["Best Position"] == pos]

    if nation != "All":
        filtered = filtered[filtered["Nationality"] == nation]

    filtered = filtered[filtered["Overall"] >= min_ovr]

    st.dataframe(filtered[["Known As", "Club Name", "Overall", "Best Position", "Value(in Euro)"]])

# =========================================================
# 🏅 BEST XI
# =========================================================
elif mode == "🏅 Best XI":

    formation = st.selectbox("Formation", ["4-3-3", "4-4-2", "4-2-3-1"])

    if formation == "4-3-3":
        pos_map = {"GK":1,"CB":2,"LB":1,"RB":1,"CM":2,"CDM":1,"LW":1,"RW":1,"ST":1}
    elif formation == "4-4-2":
        pos_map = {"GK":1,"CB":2,"LB":1,"RB":1,"LM":1,"RM":1,"CM":2,"ST":2}
    else:
        pos_map = {"GK":1,"CB":2,"LB":1,"RB":1,"CDM":2,"CAM":1,"ST":1}

    team = []

    for pos, count in pos_map.items():
        team.append(df[df["Best Position"] == pos].nlargest(count, "Overall"))

    team_df = pd.concat(team)

    st.dataframe(team_df[["Known As", "Best Position", "Overall", "Club Name"]])

# =========================================================
# 💰 VALUE SIMULATOR (CLEAN + STABLE)
# =========================================================
elif mode == "💰 Value Simulator":

    st.subheader("💰 AI Value Simulator")

    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 15, 45, 25)
        overall = st.slider("Overall", 50, 99, 75)
        potential = st.slider("Potential", 50, 99, 80)
        wage = st.number_input("Wage", 0, 500000, 10000)
        pace = st.slider("Pace", 0, 100, 70)

    with col2:
        shooting = st.slider("Shooting", 0, 100, 70)
        passing = st.slider("Passing", 0, 100, 70)
        dribbling = st.slider("Dribbling", 0, 100, 70)
        defending = st.slider("Defending", 0, 100, 50)
        physical = st.slider("Physical", 0, 100, 70)

    input_data = pd.DataFrame([{
        "Age": age,
        "Overall": overall,
        "Potential": potential,
        "Wage(in Euro)": wage,
        "Pace Total": pace,
        "Shooting Total": shooting,
        "Passing Total": passing,
        "Dribbling Total": dribbling,
        "Defending Total": defending,
        "Physicality Total": physical
    }])

    original_value = st.number_input("Current Value (€)", 0, 200000000, 10000000)

    if st.button("🚀 Predict Value"):
        pred = model.predict(input_data)[0]

        st.success(f"Predicted Value: €{pred:,.0f}")

        diff = pred - original_value

        col1, col2 = st.columns(2)
        col1.metric("Current", f"€{original_value:,.0f}")
        col2.metric("Predicted", f"€{pred:,.0f}", delta=f"€{diff:,.0f}")

        fig, ax = plt.subplots()
        ax.bar(["Current", "Predicted"], [original_value, pred])
        st.pyplot(fig)