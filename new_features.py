import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Streamlit setup
st.set_page_config(page_title="Pot Type Predictor", layout="centered")
st.title("🎯 Next Round Pot Type Predictor (Top 2)")
st.markdown("🔮 Predict the next likely winning **two pot types**: High, Mid, or Low based on last 10 rounds.")

# Session state
if "rounds" not in st.session_state:
    st.session_state.rounds = []

# Round input form
st.subheader("➕ Enter Round Data")
with st.form("round_form"):
    pot_a = st.number_input("Pot A", min_value=0, step=1)
    pot_b = st.number_input("Pot B", min_value=0, step=1)
    pot_c = st.number_input("Pot C", min_value=0, step=1)
    winner = st.selectbox("Winning Chair", ["A", "B", "C"])
    add_btn = st.form_submit_button("Add Round")

if add_btn:
    st.session_state.rounds.append({"A": pot_a, "B": pot_b, "C": pot_c, "Winner": winner})
    if len(st.session_state.rounds) > 10:
        st.session_state.rounds.pop(0)
    st.success("✅ Round added!")

# Show data
if st.session_state.rounds:
    df = pd.DataFrame(st.session_state.rounds)
    st.subheader("🧾 Last 10 Rounds")
    st.dataframe(df, use_container_width=True)

    if len(df) == 10:
        # Convert winning chair to pot type
        def get_pot_type(row):
            pots = {"A": row["A"], "B": row["B"], "C": row["C"]}
            sorted_pots = sorted(pots.items(), key=lambda x: x[1])
            rank_map = {
                sorted_pots[0][0]: "Low",
                sorted_pots[1][0]: "Mid",
                sorted_pots[2][0]: "High"
            }
            return rank_map[row["Winner"]]

        df["WinnerPotType"] = df.apply(get_pot_type, axis=1)

        # Create training sequences
        X, y = [], []
        for i in range(len(df) - 3):
            seq = df["WinnerPotType"].iloc[i:i+3].tolist()
            target = df["WinnerPotType"].iloc[i+3]
            X.append(seq)
            y.append(target)

        if len(X) > 0:
            # Fit encoder on all pot types
            all_labels = df["WinnerPotType"].tolist()
            le = LabelEncoder()
            le.fit(all_labels)

            X_enc = [le.transform(x) for x in X]
            y_enc = le.transform(y)

            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_enc, y_enc)

            # Predict top 2 pot types from last 3
            last_seq = df["WinnerPotType"].iloc[-3:].tolist()
            last_seq_enc = le.transform(last_seq).reshape(1, -1)
            proba = model.predict_proba(last_seq_enc)[0]
            top2_idx = np.argsort(proba)[-2:][::-1]
            top2_labels = le.inverse_transform(top2_idx)

            # Format the result
            combo = f"{top2_labels[0]} & {top2_labels[1]}"
            st.subheader("🔮 Prediction")
            st.success(f"🎯 **Most likely winner pot types: {combo.upper()}**")
        else:
            st.warning("⚠️ Need at least 4 rounds of data for prediction.")
    else:
        st.info("ℹ️ Add exactly 10 rounds to enable prediction.")
else:
    st.info("👈 Start by entering round history.")